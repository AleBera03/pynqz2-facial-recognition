#include "det.h"

// Thread A: Capture
void captureThread(VideoCapture& cap, SharedData& sd, CascadeClassifier face_cascade)
{
    Mat tmp;
    while (sd.running.load()) {
        cap.read(tmp);
        if (tmp.empty()) { sd.running = false; break; }

        Mat small;
        resize(tmp, small, Size(DETECT_WIDTH, DETECT_HEIGHT), 0, 0, INTER_LINEAR);

        {
            lock_guard<mutex> lk(sd.mtx_capture);
            sd.frame_full  = tmp.clone();
            sd.frame_small = small;
        }
    }
}

// Thread B: Detection
void detectionThread(SharedData& sd, CascadeClassifier face_cascade)
{
    Mat gray_small;

    while (sd.running.load()) {

        Mat small_copy;
        {
            lock_guard<mutex> lk(sd.mtx_capture);
            if (sd.frame_small.empty()) { this_thread::sleep_for(chrono::milliseconds(1)); continue; }
            small_copy = sd.frame_small.clone();
        }

        // BGR -> GRAY on small frame (320x240)
        cvtColor(small_copy, gray_small, COLOR_BGR2GRAY);
        // equalization for better accuracy in hard environment
        #if OPTIMIZED == 0
            equalizeHist(gray_small, gray_small);
        #endif

        vector<Rect> faces_local;
        face_cascade.detectMultiScale(
            gray_small,
            faces_local,
            SCALE_FACTOR,
            MIN_NEIGHBORS,
            0,              
            MIN_FACE_SIZE
        );

        // scale from small to full sizes
        float sx = (float)CAPTURE_WIDTH / DETECT_WIDTH;   // adapt for camera res
        float sy = (float)CAPTURE_HEIGHT / DETECT_HEIGHT;  // adapt for camera res
        for (auto& r : faces_local) {
            r.x      = (int)(r.x      * sx);
            r.y      = (int)(r.y      * sy);
            r.width  = (int)(r.width  * sx);
            r.height = (int)(r.height * sy);
        }

        {
            lock_guard<mutex> lk(sd.mtx_faces);
            sd.faces = move(faces_local);
        }

    }
}


void drawFaces(Mat& frame, const vector<Rect>& faces)
{
    for (const auto& f : faces) {
        rectangle(frame, f, Scalar(255, 0, 0), 2, 8, 0);
    }
}