#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void harrisCornerDetection(const Mat& srcGray, Mat& dst) {
    Mat dst_norm;
    dst = Mat::zeros(srcGray.size(), CV_32FC1);

    cornerHarris(srcGray, dst, 2, 3, 0.04);

    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst);

    for (int i = 0; i < dst_norm.rows; i++) {
        for (int j = 0; j < dst_norm.cols; j++) {
            if ((int) dst_norm.at<float>(i, j) > 150) {
                circle(dst, Point(j, i), 3, Scalar(0), 1, 8, 0);
            }
        }
    }
}

void shiTomasiCornerDetection(const Mat& srcGray, Mat& dst) {
    vector<Point2f> corners;
    goodFeaturesToTrack(srcGray, corners, 100, 0.01, 10, noArray(), 3, false, 0.04);

    dst = srcGray.clone();
    cvtColor(dst, dst, COLOR_GRAY2BGR);

    for (size_t i = 0; i < corners.size(); i++) {
        circle(dst, corners[i], 3, Scalar(0, 255, 0), -1, 8);
    }
}

int main() {
    Mat srcColor = imread("house.png", IMREAD_COLOR);
    if (srcColor.empty()) {
        cerr << "Eroare: imaginea nu a fost găsită!" << endl;
        return -1;
    }

    Mat grayImg;
    cvtColor(srcColor, grayImg, COLOR_BGR2GRAY);

    Mat harrisResult, shiTomasiResult;
    harrisCornerDetection(grayImg, harrisResult);
    shiTomasiCornerDetection(grayImg, shiTomasiResult);

    imshow("Harris Corner Detection", harrisResult);
    imshow("Shi-Tomasi Corner Detection", shiTomasiResult);
    waitKey(0);

    return 0;
}
