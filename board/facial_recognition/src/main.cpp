#include "httplib.h"
#include "det.h"
#include "rec.h"
#include "db.h"
#include <sys/select.h>
#include <sstream>
#include <condition_variable>

#define IMAGES_FOLDER    "/home/root/facial_recognition/images"
#define ENROLL_IMAGES    15     // dont change
#define ENROLL_DELAY_MS  800


struct StreamState {
    std::mutex              mtx;
    std::vector<uchar>      jpeg_buf;
    std::condition_variable cv;
    bool                    updated = false;
};

// shared rec state
// updated by input thread, readed by main loop for drawing

struct FaceResult {
    Rect   bbox;
    int    id   = -1;
    string name = "unknown";
};
struct RecState {
    mutex              mtx;
    bool               active = false;
    int                skip   = 0;
    vector<FaceResult> results;
    atomic<bool>       streaming{false};
};

// FILESYSTEM

// mkdir -p equivalent
static bool make_dir(const string& path)
{
    // loop for each element and create missing directory
    for (size_t pos = 1; pos < path.size(); ++pos) {
        if (path[pos] == '/') {
            string sub = path.substr(0, pos);
            if (mkdir(sub.c_str(), 0777) != 0 && errno != EEXIST) {
                cerr << "[DIR] Cannot create: " << sub
                     << " — " << strerror(errno) << endl;
                return false;
            }
        }
    }
    // create final directory
    if (mkdir(path.c_str(), 0777) != 0 && errno != EEXIST) {
        cerr << "[DIR] Cannot create: " << path
             << " — " << strerror(errno) << endl;
        return false;
    }
    return true;
}

static bool rm_rf(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return false; // path doesn't exist
    }

    // if it's a regular file, just unlink it
    if (!S_ISDIR(st.st_mode)) {
        return unlink(path.c_str()) == 0;
    }

    // if it's a directory, open it and iterate through contents
    DIR* dir = opendir(path.c_str());
    if (!dir) return false;

    struct dirent* entry;
    bool success = true;

    while ((entry = readdir(dir)) != nullptr) {
        // kkip the actual directory navigation links "." and ".."
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        std::string sub_path = path + "/" + entry->d_name;
        
        // recurse into the sub-path
        if (!rm_rf(sub_path)) {
            success = false;
        }
    }
    
    closedir(dir);

    // once empty, we can safely delete the directory itself
    if (success) {
        return rmdir(path.c_str()) == 0;
    }
    
    return false;
}


// acquires ENROLL_IMAGES facial crops from camera (during
// thread), compute avg embedding and save onto DB.
// run inside input thread | dont touch main loop.
static void cmd_enroll(int person_id, const string& person_name,
                       SharedData& sd,
                       face_recognizer& rec,
                       db& database)
{
    cout << "[ENROLL] Start id=" << person_id
         << " name=" << person_name << endl;
    cout << "[ENROLL] Frame the face. It will taken "
         << ENROLL_IMAGES << " photos." << endl;

    string id_folder = string(IMAGES_FOLDER) + "/" + to_string(person_id);
    if (!make_dir(id_folder)) {
        cerr << "[ENROLL] Impossible to create " << id_folder << endl;
        return;
    }

    vector<string> image_paths;
    int  imageCounter = 0;
    auto lastSaveTime = chrono::steady_clock::now()
                      - chrono::milliseconds(ENROLL_DELAY_MS);

    // acquisition: loop unless we dont have ENROLL_IMAGES crops
    while (imageCounter < ENROLL_IMAGES && sd.running.load()) {

        Mat          frame_full_copy;
        vector<Rect> faces_snap;

        {
            lock_guard<mutex> lk(sd.mtx_capture);
            if (sd.frame_full.empty()) {
                this_thread::sleep_for(chrono::milliseconds(5));
                continue;
            }
            frame_full_copy = sd.frame_full.clone();
        }
        {
            lock_guard<mutex> lk(sd.mtx_faces);
            faces_snap = sd.faces;
        }

        auto now = chrono::steady_clock::now();
        auto dt  = chrono::duration_cast<chrono::milliseconds>(now - lastSaveTime);

        if (!faces_snap.empty() && dt.count() >= ENROLL_DELAY_MS) {
            // biggest face
            Rect best = faces_snap[0];
            for (const auto& f : faces_snap)
                if (f.area() > best.area()) best = f;

            Rect safe = best & Rect(0, 0, frame_full_copy.cols,
                                          frame_full_copy.rows);
            if (safe.area() > 0) {
                Mat crop = frame_full_copy(safe).clone();
                string path = id_folder + "/" + to_string(imageCounter) + ".jpg";
                if (cv::imwrite(path, crop)) {
                    image_paths.push_back(path);
                    imageCounter++;
                    lastSaveTime = now;
                    cout << "[ENROLL] Photo " << imageCounter
                         << "/" << ENROLL_IMAGES
                         << " -> " << path
                         << " (" << safe.width << "x" << safe.height << ")"
                         << endl;
                }
            }
        }

        this_thread::sleep_for(chrono::milliseconds(20));
    }

    if (image_paths.empty()) {
        cerr << "[ENROLL] No acquired photos" << endl;
        return;
    }

    // Save each embedding separately (multi-sample).
    // The classifier uses all samples to build a robust decision boundary.
    cout << "[ENROLL] Computing embeddings..." << endl;
    database.remove(person_id);  // remove previous samples for this id
    int valid = 0;

    for (const auto& path : image_paths) {
        Mat img = imread(path);
        if (img.empty()) continue;
        vector<float> emb = rec.extract_embedding(img);
        if (emb.empty()) continue;
        identity ident;
        ident.id        = person_id;
        ident.name      = person_name;
        ident.embedding = emb;
        database.insert(ident);  // insert, not upsert: each photo is a sample
        valid++;
        cout << "[ENROLL] Sample " << valid << " saved: " << path << endl;
    }

    if (valid == 0) {
        cerr << "[ENROLL] No extracted embedding" << endl;
        return;
    }

    database.write();
    cout << "[ENROLL] Completed: id=" << person_id
         << " name=" << person_name
         << " (" << valid << " samples saved)" << endl;
}

static vector<Mat> augment(const Mat& src, int n = ENROLL_IMAGES-2) {
    vector<Mat> out;

    RNG rng(42);
    for (int i = 0; i < n; ++i) {
        Mat aug = src.clone();

        // slightly change brightness
        aug.convertTo(aug, -1, 1.0, rng.uniform(-20, 20));

        // horizontal flip (50%)
        if (i % 2 == 0) flip(aug, aug, 1);

        // small random resize crop
        int dx = rng.uniform(0, src.cols / 12);
        int dy = rng.uniform(0, src.rows / 12);
        Rect roi(dx, dy, src.cols - 2*dx, src.rows - 2*dy);
        resize(aug(roi), aug, Size(INPUT_W, INPUT_H));

        out.push_back(aug);
    }
    return out;
}

static void cmd_enroll_folder(int person_id, const string& person_name,
                              const string& src_folder,
                              face_recognizer& rec,
                              db& database)
{
    cout << "[ENROLL] Starting from folder: id=" << person_id
         << " name=" << person_name
         << " src=" << src_folder << endl;

    // standard folder IMAGES_FOLDER/<id>/
    string dst_folder = string(IMAGES_FOLDER) + "/" + to_string(person_id);
    if (!make_dir(dst_folder)) {
        cerr << "[ENROLL] Cannot create " << dst_folder << endl;
        return;
    }

    // list images within source folder
    vector<string> src_paths;
    DIR* dir = opendir(src_folder.c_str());
    if (!dir) {
        cerr << "[ENROLL] Impossible to open: " << src_folder << endl;
        return;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        string fname = entry->d_name;
        if (fname == "." || fname == "..") continue;
        string low = fname;
        transform(low.begin(), low.end(), low.begin(), ::tolower);
        string ext4 = low.size() >= 4 ? low.substr(low.size() - 4) : "";
        string ext5 = low.size() >= 5 ? low.substr(low.size() - 5) : "";
        if (ext4 == ".jpg" || ext4 == ".png" || ext4 == ".bmp" || ext5 == ".jpeg")
            src_paths.push_back(src_folder + "/" + fname);
    }
    closedir(dir);

    if (src_paths.empty()) {
        cerr << "[ENROLL] No image found in: " << src_folder << endl;
        return;
    }

    int num_found_images = (int)src_paths.size();
    sort(src_paths.begin(), src_paths.end());
    cout << "[ENROLL] Found " << num_found_images << " images" << endl;

    // Distribute augmentations evenly across crops.
    // augs_per_crop is the base quota; the first (remainder) crops get one extra
    // to cover the case where the division is not exact.
    // Example: 2 images, target 12 → need 10 augs → base 5 each, remainder 0.
    // Example: 3 images, target 12 → need 9 augs  → base 3 each, remainder 0.
    // Example: 3 images, target 10 → need 7 augs  → base 2 each, 1 extra on first crop.
    int total_augs   = max(0, ENROLL_IMAGES - num_found_images);
    int augs_base    = (num_found_images > 0) ? total_augs / num_found_images : 0;
    int augs_remainder = (num_found_images > 0) ? total_augs % num_found_images : 0;

    // load local cascade classifier (in order to not interfere with thread)
    CascadeClassifier local_cascade;
    if (!local_cascade.load(faceClassifier)) {
        cerr << "[ENROLL] Impossible to load classifier" << endl;
        return;
    }

    database.remove(person_id);  // remove previous samples for this id
    int valid      = 0;
    int crop_index = 0;  // index for saved crop filenames
    int aug_index  = 0;  // global index for saved augmentation filenames (avoids overwrites)
    int crop_count = 0;  // counts only successfully processed crops

    for (const auto& src_path : src_paths) {
        Mat img = imread(src_path);
        if (img.empty()) {
            cerr << "[ENROLL] Impossible to read: " << src_path << endl;
            continue;
        }

        Mat crop;
        Rect safe;

        if (img.cols < MIN_FACE_SIZE.width || img.rows < MIN_FACE_SIZE.height) {
            // direct crop due to too small image
            cout << "[ENROLL] Small image (" << img.cols << "x" << img.rows << ")" << endl;
            crop = img.clone();
        } else {
            Mat gray;
            cvtColor(img, gray, COLOR_BGR2GRAY);
            equalizeHist(gray, gray);

            vector<Rect> faces;
            local_cascade.detectMultiScale(
                gray, faces,
                SCALE_FACTOR, MIN_NEIGHBORS, 0, MIN_FACE_SIZE
            );

            if (faces.empty()) {
                cout << "[ENROLL] Cascade: no face in " << src_path << endl;
                continue;
            } else {
                // biggest face
                Rect best = faces[0];
                for (const auto& f : faces)
                    if (f.area() > best.area()) best = f;

                safe = best & Rect(0, 0, img.cols, img.rows);
                if (safe.area() == 0) continue;

                crop = img(safe).clone();
            }
        }

        // crop and save to IMAGES_FOLDER/<id>/
        string dst_path = dst_folder + "/" + to_string(crop_index) + ".jpg";
        if (!cv::imwrite(dst_path, crop)) {
            cerr << "[ENROLL] Write error: " << dst_path << endl;
            continue;
        }
        cout << "[ENROLL] Crop " << (crop_index + 1)
             << " saved: " << dst_path
             << " <- " << src_path << endl;
        crop_index++;

        // extract embedding from DPU
        vector<float> emb = rec.extract_embedding(crop);
        if (emb.empty()) {
            cerr << "[ENROLL] No extracted embedding" << endl;
            continue;
        }

        identity ident;
        ident.id        = person_id;
        ident.name      = person_name;
        ident.embedding = emb;
        database.insert(ident);
        valid++;
        cout << "[ENROLL] Sample " << valid << " saved" << endl;

        // Augmentation: this crop gets augs_base augmentations,
        // plus one extra if it falls within the remainder quota.
        int n_augs_this_crop = augs_base + (crop_count < augs_remainder ? 1 : 0);
        crop_count++;

        if (n_augs_this_crop > 0) {
            cout << "[ENROLL] Generating " << n_augs_this_crop
                 << " augmentations from crop " << crop_count << "..." << endl;

            vector<Mat> augs = augment(crop, n_augs_this_crop);

            for (size_t i = 0; i < augs.size(); ++i) {
                string aug_path = dst_folder + "/aug_" + to_string(aug_index++) + ".jpg";
                cv::imwrite(aug_path, augs[i]);

                vector<float> aug_emb = rec.extract_embedding(augs[i]);
                if (aug_emb.empty())
                    continue;

                identity aug_ident;
                aug_ident.id        = person_id;
                aug_ident.name      = person_name;
                aug_ident.embedding = aug_emb;

                database.insert(aug_ident);
                valid++;
                cout << "[ENROLL] Augmented sample " << valid
                     << " saved (" << aug_path << ")" << endl;
            }
        }
    } // exit loop

    if (valid == 0) {
        cerr << "[ENROLL] No extracted embedding" << endl;
        return;
    }

    database.write();
    cout << "[ENROLL] Completed: id=" << person_id
         << " name=" << person_name
         << " | crop in: " << dst_folder
         << " | " << valid << " samples saved" << endl;
}

static void cmd_rec(RecState& rs, db& database, face_recognizer& rec)
{
    lock_guard<mutex> lk(rs.mtx);

    if (rs.active) {
        rs.active = false;
        rs.results.clear();
        cout << "[REC] Recognition deactivated" << endl;
    } else {
        if (database.all().empty()) {
            cout << "[REC] Empty database — run 'enroll <id> <name>'" << endl;
            return;
        }

        ClassifierMode mode = rec.select_and_train(database.all());

        if (!rec.classifier_ready()) {
            cout << "[REC] Cannot start classifier "
                 << "(at least 2 distinct identities required in DB)" << endl;
            return;
        }

        rs.active = true;
        rs.skip   = 0;

        string mode_name = (mode == ClassifierMode::NEAREST_CENTROID)
                           ? "NCC (Nearest Centroid)"
                           : "Linear SVM";
        cout << "[REC] Recognition activated | classifier: " << mode_name
             << " | identities in DB: " << database.all().size() << endl;
    }
}

static void print_help() {
    cout << "\n[CMD] Commands:" << endl;
    cout << "  enroll <id> <name> [OPTIONS]     — enroll new person/identity" << endl;
    cout << "  OPTIONS \n \t -i|--input <folder_path>" << endl;
    cout << "  rec                              — toggle recognition" << endl;
    cout << "  show                             — toggle streaming HTTP (client connection)" << endl;
    cout << "  list                             — show list of identities within DB" << endl;
    cout << "  remove <id>                      — remove person/identity from DB" << endl;
    cout << "  quit                             — close program" << endl;
    cout << "  help                             — show this message" << endl;
    cout << endl;
}

// inputThread
// read commands from stdin without locking main loop.
static void inputThread(SharedData& sd,
                        CascadeClassifier& face_cascade,
                        face_recognizer& rec,
                        db& database,
                        RecState& rs,
                        StreamState& stream_state)
{
    
    print_help();

    string line;
    bool print_prompt = true;
    while (sd.running.load()) {

        if (print_prompt) {
            cout << "> ";
            cout.flush();
            print_prompt = false; // instant reset
        }

        // configure select to configure input (FD 0)
        fd_set rfds;
        struct timeval tv;
        FD_ZERO(&rfds);
        FD_SET(0, &rfds);

        // 200ms for control
        tv.tv_sec = 0;
        tv.tv_usec = 200000; 

        int retval = select(1, &rfds, NULL, NULL, &tv);

        if (retval == -1) {
            // select error (e.g., signal interruption)
            break; 
        }
        else if (retval == 0) {
            // timeout: no input from stdin 
            // cicle restart and check if sd.running is false
            continue; 
        }

        if (!getline(cin, line)) {
            sd.running = false;
            break;
        }

        print_prompt = true;

        if (line.empty()) continue;

        istringstream ss(line);
        string cmd;
        ss >> cmd;

        if (cmd == "enroll") {
            int    id;
            string name;
            if (rs.active) {
                cout << "[CMD] rec is active, deactivate it by typing 'rec' again to use the command" << endl;
                continue;
            }
            if (!(ss >> id) || !(ss >> name)) {
                cout << "[CMD] Usage: enroll <id> <name>" << endl;
                continue;
            }
            // id already exists
            if (database.exists(id)) {
                cout << "[CMD] id=" << id << " already on DB ("
                     << database.find(id)->name
                     << "). Overwrite? [y/N] ";
                cout.flush();
                string confirm;
                getline(cin, confirm);
                if (confirm == "y" || confirm == "Y") {
                    database.remove(id);
                    database.write();
                    rm_rf(string(IMAGES_FOLDER) + "/" + to_string(id));
                } else {
                    cout << "[CMD] Cancelled" << endl;
                    continue;
                }
            }
            string flag, images_path;
            if (ss >> flag) {
                // there is something after <name>
                if ((flag == "--input" || flag == "-i") && (ss >> images_path)) {
                    cmd_enroll_folder(id, name, images_path, rec, database);
                } else {
                    cout << "[CMD] Unkown command: '" << flag
                         << "'. Usage: enroll <id> <name> [--input <path>]" << endl;
                }
            } else {
                // enrollment from camera
                cmd_enroll(id, name, sd, rec, database);
            }

        } else if (cmd == "rec") {
            cmd_rec(rs, database, rec);

        } else if (cmd == "show") {
            if (rs.streaming.load()) {
                rs.streaming.store(false);
                stream_state.cv.notify_all(); // wake up thread in order to let them exit from waiting loop
                cout << "[SHOW] Streaming deactivated" << endl;
            } else {
                rs.streaming.store(true);
                cout << "[SHOW] Streaming activated. Open on browser: http://<pynq_ip>:8080/video" << endl;
            }
        } else if (cmd == "list") {
            database.load();   // reload DB for updated entries
            const auto& all = database.all();
            if (all.empty()) {
                cout << "[CMD] Database empty" << endl;
            } else {
                int current_id;
                bool first = true;
                for (const auto& ident : all) {
                    if (first == true) {
                        current_id = ident.id;
                    }
                    if (current_id != ident.id || first) {
                        cout << "  id=" << ident.id << "  name=" << ident.name << endl;
                        current_id = ident.id;
                    }
                    if (first == true) {
                        first = false;
                    }
                }
            }

        } else if (cmd == "remove") {
            int rm_id;
            if (rs.active) {
                cout << "[CMD] rec is active, deactivate it by typing 'rec' again to use the command" << endl;
                continue;
            }
            if (!(ss >> rm_id)) {
                cout << "[CMD] Usage: remove <id>" << endl;
                continue;
            }
            if (!database.exists(rm_id)) {
                cout << "[CMD] id=" << rm_id << " not found in DB" << endl;
            } else {
                cout << "[CMD] Remove id=" << rm_id
                     << " (" << database.find(rm_id)->name << ")? [y/N] ";
                cout.flush();
                string confirm;
                getline(cin, confirm);
                if (confirm == "y" || confirm == "Y") {
                    database.remove(rm_id);
                    database.write();
                    rm_rf(string(IMAGES_FOLDER) + "/" + to_string(rm_id));
                    cout << "[CMD] id=" << rm_id << " removed" << endl;
                } else {
                    cout << "[CMD] Cancelled" << endl;
                }
            }

        } else if (cmd == "quit") {
            cout << "[CMD] Exiting ..." << endl;
            sd.running = false;
            break;

        } else if (cmd == "help") {
            print_help();

        } else {
            cout << "[CMD] Unkown command: '" << cmd
                 << "'. Type 'help'." << endl;
        }
    }
}

static void printOverlay(const vector<Rect>& faces,
                        const RecState& rs,
                        double fps){
    if (rs.active) {
        for (size_t i = 0; i < rs.results.size(); ++i) {
            const FaceResult& fr = rs.results[i];
            int print_id = fr.id;
            if (print_id >= 0) {
                cout << "[REC] recognized NAME=" << fr.name << " ID=" << print_id << "(FPS=" << fps << ")" <<endl;
            }
        }
    }
}

static void drawOverlay(Mat& frame,
                        const vector<Rect>& faces,
                        const RecState& rs,
                        double fps)
{
    for (const auto& f : faces)
        rectangle(frame, f, Scalar(255, 0, 0), 2);

    // FaceResult's label with anchored bbox at inference time
    if (rs.active) {
        for (size_t i = 0; i < rs.results.size(); ++i) {
            const FaceResult& fr = rs.results[i];
            Scalar color = (fr.id >= 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
            string label = (fr.id >= 0)
                           ? fr.name + " (id=" + to_string(fr.id) + ")"
                           : "unknown";
            putText(frame, label,
                    Point(fr.bbox.x, fr.bbox.y - 8),
                    FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
    }
    
    string rec_label = rs.active ? "REC: ON" : "REC: OFF";
    Scalar rec_color = rs.active ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
    putText(frame, rec_label, Point(frame.cols - 130, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2);

}


int main(int argc, const char** argv)
{
    // Classifier
    CascadeClassifier face_cascade;
    if (!face_cascade.load(faceClassifier)) {
        cout << "[FAIL] Cannot load the classifier" << endl;
        return -1;
    }
    cout << "[OK] Classifier loaded" << endl;

    // Camera
    VideoCapture cap("/dev/video0");
    if (!cap.isOpened()) { cout << "Cannot open camera" << endl; return -1; }
    cap.set(CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT);

    SharedData sd;
    RecState rs;

    // Http server
    StreamState stream_state;
    httplib::Server srv;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};

    srv.Get("/video", [&](const httplib::Request&, httplib::Response& res) {
    res.set_header("Cache-Control", "no-cache, private, max-age=0, no-store, must-revalidate");
    res.set_header("Connection", "keep-alive");
    
    // configure MJPEG streaming
    res.set_chunked_content_provider(
            "multipart/x-mixed-replace; boundary=frame",
            [&](size_t offset, httplib::DataSink &sink) {
                std::unique_lock<std::mutex> lk(stream_state.mtx);
                
                // wait unless there isn't a new frame or show is disabled
                stream_state.cv.wait(lk, [&]{
                    return stream_state.updated || !sd.running.load() || !rs.streaming.load(); 
                });
            
                // shutdown
                if (!sd.running.load() || !rs.streaming.load()) {
                    sink.done();
                    return false;
                }
            
                // local copy in order to release mutex
                std::vector<uchar> local_buf = stream_state.jpeg_buf;
                stream_state.updated = false;
                lk.unlock(); 
                
                // standard MJPEG packet building
                std::string header = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " 
                                + std::to_string(local_buf.size()) + "\r\n\r\n";
                
                if (!sink.write(header.data(), header.size())) return false;
                if (!sink.write(reinterpret_cast<const char*>(local_buf.data()), local_buf.size())) return false;
                if (!sink.write("\r\n", 2)) return false;
            
                return true;
            }
        );
    });

    // New thread where a new server will start
    std::thread tServer([&]() {
        srv.listen("0.0.0.0", 8080);
    });

    // DPU recognizer
    face_recognizer rec;
    if (!rec.is_valid()) return -1;

    // Database
    db database;
    if (!database.load()) {
        cerr << "[FAIL] Cannot load database" << endl;
        return -1;
    }

    // Thread detection
    thread tCapture(captureThread, ref(cap), ref(sd), face_cascade);
    thread tDetect(detectionThread, ref(sd), face_cascade);

    // Input thread
    thread tInput(inputThread, ref(sd), ref(face_cascade), ref(rec), ref(database), ref(rs), ref(stream_state));

    // Main loop (looping streaming)
    int    frameCount = 0;
    double fps_stream = 0.0;
    auto   lastReset  = chrono::steady_clock::now();

    while (sd.running.load()) {

        Mat          frame_send;
        vector<Rect> faces_snap;

        {
            lock_guard<mutex> lk(sd.mtx_capture);
            if (sd.frame_full.empty()) {
                this_thread::sleep_for(chrono::milliseconds(1));
                continue;
            }
            frame_send = sd.frame_full.clone();
        }
        {
            lock_guard<mutex> lk(sd.mtx_faces);
            faces_snap = sd.faces;
        }

        // every 5 frame -> recognition
        {
            lock_guard<mutex> lk(rs.mtx);

            if (rs.active && faces_snap.empty())
                rs.results.clear();

            if (rs.active && !faces_snap.empty() && rec.classifier_ready()) {
                if (++rs.skip % 5 == 0) {
                    rs.results.clear();
                    for (size_t i = 0; i < faces_snap.size(); ++i) {
                        Rect safe = faces_snap[i]
                                  & Rect(0, 0, frame_send.cols, frame_send.rows);
                        if (safe.area() == 0) continue;
                        Mat crop = frame_send(safe).clone();
                        vector<float> emb = rec.extract_embedding(crop);
                        FaceResult fr;
                        fr.bbox = faces_snap[i];  // ancora il bbox al momento dell'inferenza
                        if (!emb.empty()) {
                            pair<int,string> p = rec.identify(emb);
                            fr.id   = p.first;
                            fr.name = p.second;
                        }
                        rs.results.push_back(fr);
                    }
                }
            } else if (!rs.active) {
                rs.results.clear();
            }

            if (rs.streaming.load()) {
                drawOverlay(frame_send, faces_snap, rs, fps_stream);
            } else {
                printOverlay(faces_snap, rs, fps_stream);
            }

        }

        // FPS
        frameCount++;
        auto now = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(now - lastReset).count();
        if (elapsed >= 1.0) {
            fps_stream = frameCount / elapsed;
            frameCount = 0;
            lastReset  = now;
        }

        if (rs.streaming.load()) {
            // writing framerate
            cv::putText(frame_send, "fps: " + to_string(fps_stream),
                        cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN,
                        1.0, cv::Scalar(0, 255, 0), 2);
            
            {
                std::lock_guard<std::mutex> lk(stream_state.mtx);
                cv::imencode(".jpg", frame_send, stream_state.jpeg_buf, params); // write on server buffer
                stream_state.updated = true;
            }
            stream_state.cv.notify_all(); // notify all threads about cpp-httplib packet sending
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

    }

    sd.running = false;
    rs.streaming.store(false);
    stream_state.cv.notify_all(); // release all residual cond_var instances
    
    srv.stop(); // instantly stop server

    tCapture.join();
    tDetect.join();
    tInput.join();
    tServer.join();
    return 0;

}