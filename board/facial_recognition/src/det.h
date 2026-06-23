#ifndef DETECTOR_H
#define DETECTOR_H

#include <chrono>
#include <algorithm>
#include <vector>
#include <atomic>
#include <queue>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <zconf.h>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <math.h>
#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <dnndk/n2cube.h>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define OPTIMIZED 0

#if OPTIMIZED
    #define faceClassifier "/home/root/facial_recognition/models/detector/lbpcascade_frontalface.xml"
#else
    #define faceClassifier "/home/root/facial_recognition/models/detector/haarcascade_frontalface_alt2.xml"
#endif

using namespace std;
using namespace cv;

// parameters
static const int   DETECT_WIDTH   = 320;   // work on 320px rather than full-res
static const int   DETECT_HEIGHT  = 240;
static const float SCALE_FACTOR   = 1.1f;  // 1.1 default
static const int   MIN_NEIGHBORS  = 5;     // 3 default, rise it for less false positive
static const Size  MIN_FACE_SIZE  = Size(60, 60); // avoid very small face

// capture resolution
#define CAPTURE_WIDTH 640
#define CAPTURE_HEIGHT 480

// shared data among thread
struct SharedData {
    Mat       frame_full;        // original frame from streaming or pure static image
    Mat       frame_small;       // reduced frame for detection
    vector<Rect> faces;          // last detected faces
    mutex     mtx_capture;       // protect frame_full / frame_small
    mutex     mtx_faces;         // protect faces
    atomic<bool> running{true};
    atomic<double> fps_detect{0.0};
};

void captureThread(VideoCapture& cap, SharedData& sd, CascadeClassifier face_cascade);
void detectionThread(SharedData& sd, CascadeClassifier face_cascade);
void drawFaces(Mat& frame, const vector<Rect>& faces);


#endif