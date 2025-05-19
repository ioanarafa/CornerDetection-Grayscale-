#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void harrisCorner(const Mat& srcGray, Mat& dst) {
    Mat gray;
    srcGray.convertTo(gray, CV_32F);

    Mat Ix, Iy;
    Sobel(gray, Ix, CV_32F, 1, 0, 3);
    Sobel(gray, Iy, CV_32F, 0, 1, 3);

    Mat Ix2 = Ix.mul(Ix);
    Mat Iy2 = Iy.mul(Iy);
    Mat Ixy = Ix.mul(Iy);

    Mat Sx2, Sy2, Sxy;
    GaussianBlur(Ix2, Sx2, Size(3, 3), 1.0);
    GaussianBlur(Iy2, Sy2, Size(3, 3), 1.0);
    GaussianBlur(Ixy, Sxy, Size(3, 3), 1.0);

    Mat response = Mat::zeros(srcGray.size(), CV_32FC1);
    float k = 0.04f;

    for (int i = 0; i < srcGray.rows; i++) {
        for (int j = 0; j < srcGray.cols; j++) {
            float a = Sx2.at<float>(i, j);
            float b = Sxy.at<float>(i, j);
            float c = Sy2.at<float>(i, j);
            float det = a * c - b * b;
            float trace = a + c;
            response.at<float>(i, j) = det - k * trace * trace;
        }
    }

    Mat responseNorm;
    normalize(response, responseNorm, 0, 255, NORM_MINMAX);

    Mat dstTemp = Mat::zeros(response.size(), CV_8UC1);
    for (int i = 1; i < response.rows - 1; i++) {
        for (int j = 1; j < response.cols - 1; j++) {
            float val = responseNorm.at<float>(i, j);
            if (val > 80) {
                bool isMax = true;
                for (int u = -1; u <= 1 && isMax; u++) {
                    for (int v = -1; v <= 1 && isMax; v++) {
                        if (responseNorm.at<float>(i + u, j + v) > val)
                            isMax = false;
                    }
                }
                if (isMax) dstTemp.at<uchar>(i, j) = 255;
            }
        }
    }

    convertScaleAbs(responseNorm, dst);
    for (int i = 0; i < dstTemp.rows; i++) {
        for (int j = 0; j < dstTemp.cols; j++) {
            if (dstTemp.at<uchar>(i, j) == 255) {
                circle(dst, Point(j, i), 3, Scalar(0), 1);
            }
        }
    }
}

void shiTomasi(const Mat& srcGray, Mat& dst) {
    Mat gray;
    srcGray.convertTo(gray, CV_32F);

    Mat Ix, Iy;
    Sobel(gray, Ix, CV_32F, 1, 0, 3);
    Sobel(gray, Iy, CV_32F, 0, 1, 3);

    Mat Ix2 = Ix.mul(Ix);
    Mat Iy2 = Iy.mul(Iy);
    Mat Ixy = Ix.mul(Iy);

    Mat Sx2, Sy2, Sxy;
    GaussianBlur(Ix2, Sx2, Size(3, 3), 1.0);
    GaussianBlur(Iy2, Sy2, Size(3, 3), 1.0);
    GaussianBlur(Ixy, Sxy, Size(3, 3), 1.0);

    Mat response = Mat::zeros(srcGray.size(), CV_32FC1);
    for (int i = 0; i < srcGray.rows; i++) {
        for (int j = 0; j < srcGray.cols; j++) {
            float a = Sx2.at<float>(i, j);
            float b = Sxy.at<float>(i, j);
            float c = Sy2.at<float>(i, j);
            float trace = a + c;
            float det = a * c - b * b;
            float sqrtTerm = sqrt(trace * trace - 4 * det);
            float lambda1 = 0.5f * (trace + sqrtTerm);
            float lambda2 = 0.5f * (trace - sqrtTerm);
            response.at<float>(i, j) = min(lambda1, lambda2);
        }
    }

    Mat responseNorm;
    normalize(response, responseNorm, 0, 255, NORM_MINMAX);

    Mat dstTemp = Mat::zeros(response.size(), CV_8UC1);
    for (int i = 1; i < response.rows - 1; i++) {
        for (int j = 1; j < response.cols - 1; j++) {
            float val = responseNorm.at<float>(i, j);
            if (val > 70) {
                bool isMax = true;
                for (int u = -1; u <= 1 && isMax; u++) {
                    for (int v = -1; v <= 1 && isMax; v++) {
                        if (responseNorm.at<float>(i + u, j + v) > val)
                            isMax = false;
                    }
                }
                if (isMax) dstTemp.at<uchar>(i, j) = 255;
            }
        }
    }

    cvtColor(srcGray, dst, COLOR_GRAY2BGR);
    for (int i = 0; i < dstTemp.rows; i++) {
        for (int j = 0; j < dstTemp.cols; j++) {
            if (dstTemp.at<uchar>(i, j) == 255) {
                circle(dst, Point(j, i), 3, Scalar(0, 255, 0), -1);
            }
        }
    }
}


Mat rotateImage(const Mat& src, double angle) {
    Point2f center(src.cols / 2.0F, src.rows / 2.0F);
    Mat rot = getRotationMatrix2D(center, angle, 1.0);
    Mat dst;
    warpAffine(src, dst, rot, src.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
    return dst;
}

int main() {
    Mat src = imread("house2.png");
    if (src.empty()) {
        cerr << "Imaginea nu a fost găsită!" << endl;
        return -1;
    }

    Mat gray, grayRotated;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    grayRotated = rotateImage(gray, 45);

    // Harris original si rotit
    Mat harrisOrig, harrisRotated;
    harrisCorner(gray, harrisOrig);
    harrisCorner(grayRotated, harrisRotated);

    // Shi-Tomasi original si rotit
    Mat shiOrig, shiRotated;
    shiTomasi(gray, shiOrig);
    shiTomasi(grayRotated, shiRotated);

    imshow("Harris - Original", harrisOrig);
    //imshow("Harris - Rotit 45°", harrisRotated);
    imshow("Shi-Tomasi - Original", shiOrig);
    //imshow("Shi-Tomasi - Rotit 45°", shiRotated);

    /*
    // Test Case : Cerc negru pe fundal alb
    Mat circleImg = Mat::ones(400, 400, CV_8UC1) * 255;
    circle(circleImg, Point(200, 200), 100, Scalar(0), FILLED);

    imshow("Imagine originala", circleImg);
    Mat harrisCircle, shiCircle;
    harrisCorner(circleImg, harrisCircle);
    shiTomasi(circleImg, shiCircle);

    imshow("Harris - Cerc negru", harrisCircle);
    imshow("Shi-Tomasi - Cerc negru", shiCircle);
    */

    waitKey(0);
    return 0;
}
