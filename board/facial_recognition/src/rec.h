#ifndef RECOGNITION_H
#define RECOGNITION_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <numeric>
#include <cmath>
#include <mutex>
#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <dnndk/n2cube.h>
#include "db.h"

#define DATABASE_PATH "face_database.csv"

// DPU kernel
#define KERNEL_CONV       "face_resnet50"
#define CONV_INPUT_NODE   "conv1_7x7_s2"   // 224x224x3
#define CONV_OUTPUT_NODE  "pool5_7x7_s1"   // 1x1x2048
#define INPUT_W       224
#define INPUT_H       224
#define INPUT_C       3
#define EMBEDDING_DIM 2048
#define MEAN_B 0.0f
#define MEAN_G 0.0f
#define MEAN_R 0.0f

#define SELECT_CLASSIFIER 0 // NCC (0 for SVM)

// SVM confidence threshold: distance from the separating hyperplane
// Raise this value for higher precision (fewer false positives)
#define SVM_UNKNOWN_THRESHOLD 0.15f
#define NCC_UNKNOWN_THRESHOLD 0.65f // Cosine similarity threshold for the Nearest Centroid Classifier


using namespace std;
using namespace cv;

// Active classification mode
enum class ClassifierMode {
    SVM,
    NEAREST_CENTROID
};

class face_recognizer
{
private:
    DPUKernel*             kernel_;
    DPUTask*               task_;
    bool                   valid_;
    mutex                  dpu_mtx_;  // makes dpuRunTask thread-safe

    // --- SVM ---
    cv::Ptr<cv::ml::SVM>   svm_;
    bool                   svm_trained_ = false;
    vector<int>            svm_class_ids_;    // SVM index -> person id
    vector<string>         svm_class_names_;  // SVM index -> person name

    // --- Nearest Centroid Classifier ---
    bool                   ncc_trained_ = false;
    cv::Mat                ncc_centroids_;    // matrix [N_classes x EMBEDDING_DIM], float32, L2-normalized rows
    vector<int>            ncc_class_ids_;    // row index -> person id
    vector<string>         ncc_class_names_;  // row index -> person name

    // --- Active mode ---
    ClassifierMode         active_mode_ = ClassifierMode::SVM;

public:
    face_recognizer();
    ~face_recognizer();

    bool is_valid()     const { return valid_; }
    bool svm_ready()    const { return svm_trained_; }
    bool ncc_ready()    const { return ncc_trained_; }

    // Returns true if at least one classifier is ready
    bool classifier_ready() const { return svm_trained_ || ncc_trained_; }

    // Currently selected classifier mode
    ClassifierMode active_mode() const { return active_mode_; }

    // Extract a 2048-D embedding from the DPU
    vector<float> extract_embedding(const Mat& face_crop_bgr);

    bool train_svm(const vector<identity>& database);

    bool build_ncc(const vector<identity>& database);

    ClassifierMode select_and_train(const vector<identity>& database);

    // Classify query_emb using the active classifier (SVM or NCC)
    // Returns {id, name} or {-1, "unknown"} if the confidence is below threshold
    pair<int,string> identify(const vector<float>& query_emb);

private:
    // Internal SVM classification
    pair<int,string> identify_svm_(const vector<float>& query_emb);
    
    // Internal NCC classification (cosine similarity)
    pair<int,string> identify_ncc_(const vector<float>& query_emb);
};

void preprocess(const Mat& face_bgr, int8_t* input_addr, float scale);
void l2normalize(vector<float>& v);

#endif
