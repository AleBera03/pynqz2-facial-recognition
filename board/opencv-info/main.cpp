#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << cv::getBuildInformation() << std::endl;
    return 0;
}