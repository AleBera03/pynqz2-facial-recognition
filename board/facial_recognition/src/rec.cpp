#include "rec.h"

face_recognizer::face_recognizer() : kernel_(nullptr), task_(nullptr), valid_(false)
{
    dpuOpen();
    kernel_ = dpuLoadKernel(KERNEL_CONV);
    if (!kernel_) {
        cerr << "[REC] dpuLoadKernel failed for: " << KERNEL_CONV << endl;
        dpuClose();
        return;
    }
    task_ = dpuCreateTask(kernel_, 0);
    if (!task_) {
        cerr << "[REC] dpuCreateTask failed" << endl;
        dpuDestroyKernel(kernel_);
        dpuClose();
        return;
    }
    valid_ = true;

    // Warm-up: run one blank pass to initialize the DPU pipeline
    cout << "[REC] Preprocessing warm-up ..." << endl;
    {
        int8_t* warmup_addr = dpuGetInputTensorAddress(task_, CONV_INPUT_NODE);
        if (warmup_addr) {
            Mat dummy(INPUT_H, INPUT_W, CV_8UC3, Scalar(0, 0, 0));
            float warmup_scale = dpuGetInputTensorScale(task_, CONV_INPUT_NODE);
            preprocess(dummy, warmup_addr, warmup_scale);
            dpuRunTask(task_);
        }
    }

    cout << "[REC] DPU task ready — kernel: " << KERNEL_CONV
         << " | embedding: " << EMBEDDING_DIM << "-D" << endl;
}

face_recognizer::~face_recognizer()
{
    if (task_)   dpuDestroyTask(task_);
    if (kernel_) dpuDestroyKernel(kernel_);
    dpuClose();
}


vector<float> face_recognizer::extract_embedding(const Mat& face_crop_bgr)
{
    if (!valid_ || face_crop_bgr.empty()) return {};

    lock_guard<mutex> lk(dpu_mtx_);

    int8_t* input_addr  = dpuGetInputTensorAddress(task_, CONV_INPUT_NODE);
    float   input_scale = dpuGetInputTensorScale(task_, CONV_INPUT_NODE);

    if (!input_addr) {
        cerr << "[REC] dpuGetInputTensorAddress failed" << endl;
        return {};
    }

    preprocess(face_crop_bgr, input_addr, input_scale);
    dpuRunTask(task_);

    // Read the actual size of the output tensor
    int size = dpuGetOutputTensorSize(task_, CONV_OUTPUT_NODE);
    if (size != EMBEDDING_DIM) {
        cerr << "[REC] unexpected dpuGetOutputTensorSize: " << size << endl;
        return {};
    }

    vector<float> embedding(size);
    dpuGetOutputTensorInHWCFP32(task_, CONV_OUTPUT_NODE, embedding.data(), size);
    l2normalize(embedding);

    return embedding;
}


bool face_recognizer::train_svm(const vector<identity>& database)
{
    svm_trained_ = false;
    svm_class_ids_.clear();
    svm_class_names_.clear();

    if (database.size() < 2) {
        cerr << "[SVM] At least 2 total samples required (found: "
             << database.size() << ")" << endl;
        return false;
    }

    // Verify that all embeddings have the same dimension
    size_t dim = database[0].embedding.size();
    for (const auto& ident : database) {
        if (ident.embedding.size() != dim) {
            cerr << "[SVM] Inconsistent embedding dimensions in DB" << endl;
            return false;
        }
    }

    int n_samples = (int)database.size();

    // Build train data (cv::Mat float32, n_samples x dim)
    cv::Mat train_data(n_samples, (int)dim, CV_32F);
    cv::Mat labels(n_samples, 1, CV_32S);

    // Collect unique ids/names preserving insertion order
    vector<int>    unique_ids;
    vector<string> unique_names;
    for (int i = 0; i < n_samples; ++i) {
        bool found = false;
        for (int k = 0; k < (int)unique_ids.size(); ++k) {
            if (unique_ids[k] == database[i].id) { found = true; break; }
        }
        if (!found) {
            unique_ids.push_back(database[i].id);
            unique_names.push_back(database[i].name);
        }
    }

    if ((int)unique_ids.size() < 2) {
        cerr << "[SVM] At least 2 distinct identities required (found: "
             << unique_ids.size() << ")" << endl;
        return false;
    }

    svm_class_ids_   = unique_ids;
    svm_class_names_ = unique_names;

    for (int i = 0; i < n_samples; ++i) {
        const identity& ident = database[i];

        for (int j = 0; j < (int)dim; ++j)
            train_data.at<float>(i, j) = ident.embedding[j];

        int svm_label = -1;
        for (int k = 0; k < (int)unique_ids.size(); ++k)
            if (unique_ids[k] == ident.id) { svm_label = k; break; }
        labels.at<int>(i, 0) = svm_label;
    }

    // Configure SVM training
    svm_ = cv::ml::SVM::create();
    svm_->setType(cv::ml::SVM::C_SVC);
    svm_->setKernel(cv::ml::SVM::LINEAR);
    svm_->setC(10.0);
    svm_->setTermCriteria(
        cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
                         1000, 1e-6)
    );

    cv::Ptr<cv::ml::TrainData> td =
        cv::ml::TrainData::create(train_data, cv::ml::ROW_SAMPLE, labels);

    try {
        svm_->train(td);
    } catch (const cv::Exception& e) {
        cerr << "[SVM] Training failed: " << e.what() << endl;
        return false;
    }

    svm_trained_ = true;
    cout << "[SVM] Trained: " << unique_ids.size() << " identities, "
         << n_samples << " total samples"
         << " (dim=" << dim << ", C=10, kernel=LINEAR)" << endl;
    return true;
}

bool face_recognizer::build_ncc(const vector<identity>& database)
{
    ncc_trained_  = false;
    ncc_centroids_ = cv::Mat();
    ncc_class_ids_.clear();
    ncc_class_names_.clear();

    if (database.empty()) {
        cerr << "[NCC] Empty database" << endl;
        return false;
    }

    // Collect unique ids/names preserving insertion order
    vector<int>    unique_ids;
    vector<string> unique_names;
    for (const auto& ident : database) {
        bool found = false;
        for (int k = 0; k < (int)unique_ids.size(); ++k)
            if (unique_ids[k] == ident.id) { found = true; break; }
        if (!found) {
            unique_ids.push_back(ident.id);
            unique_names.push_back(ident.name);
        }
    }

    if ((int)unique_ids.size() < 2) {
        cerr << "[NCC] At least 2 distinct identities required (found: "
             << unique_ids.size() << ")" << endl;
        return false;
    }

    size_t dim     = database[0].embedding.size();
    int n_classes  = (int)unique_ids.size();

    // For each class: accumulate embeddings, divide by count, then L2-normalize
    cv::Mat centroid_matrix(n_classes, (int)dim, CV_32F, cv::Scalar(0));
    vector<int> counters(n_classes, 0);

    for (const auto& ident : database) {
        // Find the class index matching this id
        int class_idx = -1;
        for (int k = 0; k < n_classes; ++k)
            if (unique_ids[k] == ident.id) { class_idx = k; break; }
        if (class_idx < 0) continue;

        for (int j = 0; j < (int)dim; ++j)
            centroid_matrix.at<float>(class_idx, j) += ident.embedding[j];
        counters[class_idx]++;
    }

    // Divide each centroid row by its sample count, then L2-normalize
    // L2 normalization is essential for correct cosine similarity computation
    for (int i = 0; i < n_classes; ++i) {
        if (counters[i] > 0)
            centroid_matrix.row(i) /= (float)counters[i];
        cv::normalize(centroid_matrix.row(i), centroid_matrix.row(i),
                      1.0, 0.0, cv::NORM_L2);
    }

    ncc_centroids_   = centroid_matrix;
    ncc_class_ids_   = unique_ids;
    ncc_class_names_ = unique_names;
    ncc_trained_     = true;

    cout << "[NCC] Built: " << n_classes << " centroids"
         << " (dim=" << dim << ", cosine threshold=" << NCC_UNKNOWN_THRESHOLD << ")" << endl;
    return true;
}

ClassifierMode face_recognizer::select_and_train(const vector<identity>& database)
{
    if (SELECT_CLASSIFIER == 1) {
        cout << "[REC] Nearest Centroid (NCC)" << endl;
        active_mode_ = ClassifierMode::NEAREST_CENTROID;
        build_ncc(database);
    } else if (SELECT_CLASSIFIER == 0){
        cout << "[REC] SVM linear" << endl;
        active_mode_ = ClassifierMode::SVM;
        train_svm(database);
    }

    return active_mode_;
}

pair<int,string> face_recognizer::identify(const vector<float>& query_emb)
{
    if (query_emb.empty()) return {-1, "unknown"};

    if (active_mode_ == ClassifierMode::NEAREST_CENTROID)
        return identify_ncc_(query_emb);
    else
        return identify_svm_(query_emb);
}


pair<int,string> face_recognizer::identify_svm_(const vector<float>& query_emb)
{
    if (!svm_trained_) return {-1, "unknown"};

    // Convert embedding to a single-row cv::Mat
    cv::Mat sample(1, (int)query_emb.size(), CV_32F);
    for (int i = 0; i < (int)query_emb.size(); ++i)
        sample.at<float>(0, i) = query_emb[i];

    // Predict class index
    float pred_label = svm_->predict(sample);
    int   svm_idx    = (int)pred_label;

    if (svm_idx < 0 || svm_idx >= (int)svm_class_ids_.size())
        return {-1, "unknown"};

    // Distance from the separating hyperplane (RAW_OUTPUT)
    cv::Mat score_mat;
    svm_->predict(sample, score_mat, cv::ml::StatModel::RAW_OUTPUT);
    float confidence     = score_mat.at<float>(0, 0);
    float abs_confidence = fabsf(confidence);   // sign depends on OVO positive class

    if (abs_confidence <= SVM_UNKNOWN_THRESHOLD)
        return {-1, "unknown"};

    return {svm_class_ids_[svm_idx], svm_class_names_[svm_idx]};
}


pair<int,string> face_recognizer::identify_ncc_(const vector<float>& query_emb)
{
    if (!ncc_trained_ || ncc_centroids_.empty())
        return {-1, "unknown"};

    // Convert embedding to a single-row cv::Mat and L2-normalize
    cv::Mat sample(1, (int)query_emb.size(), CV_32F);
    for (int i = 0; i < (int)query_emb.size(); ++i)
        sample.at<float>(0, i) = query_emb[i];
    cv::normalize(sample, sample, 1.0, 0.0, cv::NORM_L2);

    // Compute cosine similarity against all centroids in one matrix multiply
    // This is valid because both the sample and all centroids are L2-normalized
    // Result: 1 x N_classes matrix
    cv::Mat similarities = sample * ncc_centroids_.t();

    // Find the centroid with the highest similarity
    double    max_sim = 0.0;
    cv::Point max_loc;
    cv::minMaxLoc(similarities, nullptr, &max_sim, nullptr, &max_loc);

    int class_idx = max_loc.x;

    // If similarity is below threshold -> unknown
    if (max_sim < NCC_UNKNOWN_THRESHOLD)
        return {-1, "unknown"};

    if (class_idx < 0 || class_idx >= (int)ncc_class_ids_.size())
        return {-1, "unknown"};

    return {ncc_class_ids_[class_idx], ncc_class_names_[class_idx]};
}


void l2normalize(vector<float>& v)
{
    float norm = 0.0f;
    for (float x : v) norm += x * x;
    norm = sqrtf(norm);
    if (norm < 1e-10f) return;
    for (float& x : v) x /= norm;
}

void preprocess(const Mat& face_bgr, int8_t* input_addr, float scale)
{
    // resize
    Mat resized;
    resize(face_bgr, resized, Size(INPUT_W, INPUT_H), 0, 0, INTER_LINEAR);

    // CLAHE
    Mat lab, clahe_out;
    cvtColor(resized, lab, COLOR_BGR2Lab);
    vector<Mat> lab_planes(3);
    split(lab, lab_planes);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(lab_planes[0], lab_planes[0]);
    merge(lab_planes, lab);
    cvtColor(lab, clahe_out, COLOR_Lab2BGR);

    // Quantize
    int idx = 0;
    for (int h = 0; h < INPUT_H; ++h) {
        const uchar* row = clahe_out.ptr<uchar>(h);
        for (int w = 0; w < INPUT_W; ++w) {
            float b = (float)row[w*3+0] - MEAN_B;
            float g = (float)row[w*3+1] - MEAN_G;
            float r = (float)row[w*3+2] - MEAN_R;
            input_addr[idx++] = (int8_t)(b * scale);
            input_addr[idx++] = (int8_t)(g * scale);
            input_addr[idx++] = (int8_t)(r * scale);
        }
    }
}
