#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>

using namespace cv;
using namespace std;

bool isInside(int i, int j, Mat source) {
    if (i >= 0 && i < source.rows && j >= 0 && j < source.cols)
        return true;
    return false;
}

Mat bgr_2_grayscale(Mat source) {
    int rows = source.rows, cols = source.cols;
    Mat grayscale_image(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3b pixel = source.at<Vec3b>(i, j);
            float s = (float) pixel[0] + (float) pixel[1] + (float) pixel[2];
            float r = s / 3;
            grayscale_image.at<uchar>(i, j) = (uchar) r;
        }
    }
    return grayscale_image;
}

Mat to_grayscale(Mat img) {
    int rows = img.rows;
    int cols = img.cols;
    Mat gray(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3b color = img.at<Vec3b>(i, j);
            gray.at<uchar>(i, j) = (uchar) (0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]);
        }
    }
    return gray;
}

Mat box_blur(Mat img, int size) {
    int rows = img.rows, cols = img.cols;
    Mat blurred(rows, cols, CV_8UC1);
    int offset = size / 2;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int s = 0;
            int count = 0;
            for (int x = -offset; x <= offset; x++) {
                for (int y = -offset; y <= offset; y++) {
                    int nx = i + x;
                    int ny = j + y;
                    if (isInside(nx, ny, img)) {
                        s += img.at<uchar>(nx, ny);
                        count++;
                    }
                }
            }
            blurred.at<uchar>(i, j) = (uchar) (s / count);
        }
    }
    return blurred;
}

Mat sobel_edge(Mat img) {
    int rows = img.rows;
    int cols = img.cols;
    Mat edge = Mat::zeros(rows, cols, CV_8UC1);
    int gx[3][3] = {{-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1},
                    {0,  0,  0},
                    {1,  2,  1}};
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int si = 0;
            int sj = 0;
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    int pixel = img.at<uchar>(i + x, j + y);
                    si += gx[x + 1][y + 1] * pixel;
                    sj += gy[x + 1][y + 1] * pixel;
                }
            }
            int m = min(255, (int) (sqrt(si * si + sj * sj)));
            edge.at<uchar>(i, j) = (uchar) m;
        }
    }
    return edge;
}

Mat threshold_binary(Mat img, uchar t) {
    int rows = img.rows;
    int cols = img.cols;
    Mat binary(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (img.at<uchar>(i, j) <= t)
                binary.at<uchar>(i, j) = 0;
            else
                binary.at<uchar>(i, j) = 255;
        }
    }
    return binary;
}

struct BoundingBox {
    int xMin, yMin, xMax, yMax;
};

int area(int xMin, int xMax, int yMin, int yMax) {
    return (xMax - xMin + 1) * (yMax - yMin + 1);
}

float aspect(int xMin, int xMax, int yMin, int yMax) {
    return ((float) (xMax - xMin + 1)) / ((float) (yMax - yMin + 1));
}

BoundingBox detect_plate(Mat img) {
    int rows = img.rows;
    int cols = img.cols;
    Mat v = Mat::zeros(rows, cols, CV_8UC1);
    BoundingBox bestBox = {0, 0, 0, 0};
    int maxArea = 0;
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (img.at<uchar>(i, j) == 255 && v.at<uchar>(i, j) == 0) {
                int xMin = j, xMax = j, yMin = i, yMax = i;
                queue<pair<int, int>> q;
                q.push(pair<int, int>(i, j));
                v.at<uchar>(i, j) = 1;
                int count = 0;
                while (!q.empty()) {
                    pair<int, int> p;
                    p = q.front();
                    q.pop();
                    count++;
                    xMin = min(xMin, p.second);
                    xMax = max(xMax, p.second);
                    yMin = min(yMin, p.first);
                    yMax = max(yMax, p.first);

                    for (int d = 0; d < 4; d++) {
                        int nj = p.first + dy[d];
                        int ni = p.second + dx[d];
                        if (isInside(nj, ni, img) && img.at<uchar>(nj, ni) == 255 && v.at<uchar>(nj, ni) == 0) {
                            v.at<uchar>(nj, ni) = 1;
                            q.push(pair<int, int>(nj, ni));
                        }
                    }
                }
                int a = area(xMin, xMax, yMin, yMax);
                float asp = aspect(xMin, xMax, yMin, yMax);
                if (a > maxArea && asp > 2.0 && asp < 7.0 && (xMax - xMin) > 50 && (yMax - yMin) > 15) {
                    maxArea = a;
                    bestBox = {xMin, yMin, xMax, yMax};
                }
            }
        }
    }
    return bestBox;
}

float computeIoU(BoundingBox a, BoundingBox b) {
    int xA = max(a.xMin, b.xMin);
    int yA = max(a.yMin, b.yMin);
    int xB = min(a.xMax, b.xMax);
    int yB = min(a.yMax, b.yMax);

    int interWidth = max(0, xB - xA + 1);
    int interHeight = max(0, yB - yA + 1);
    int interArea = interWidth * interHeight;

    int boxAArea = area(a.xMin, a.xMax, a.yMin, a.yMax);
    int boxBArea = area(b.xMin, b.xMax, b.yMin, b.yMax);

    int unionArea = boxAArea + boxBArea - interArea;

    return (float) interArea / (float) unionArea;
}

int main() {
    Mat img = imread("C:/Others/An_III/piTest/project2.0/images/nr1.jpg", IMREAD_COLOR);

    Mat gray = to_grayscale(img);
    imshow("gray", gray);

    Mat blurred = box_blur(gray, 5);
    imshow("blurred", blurred);

    Mat edge = sobel_edge(blurred);
    imshow("edge", edge);

    Mat binary = threshold_binary(edge, 100);
    imshow("binary", binary);

    BoundingBox bestBox = detect_plate(binary);
    BoundingBox manualBox = {445, 450, 805, 540}; // nr1.jpg
//    BoundingBox manualBox = {75, 300, 775, 445}; // ch.jpg
//    BoundingBox manualBox = {250, 255, 480, 308}; // nr2.jpg

    Mat result = img.clone();
    rectangle(result, Point(manualBox.xMin, manualBox.yMin), Point(manualBox.xMax, manualBox.yMax), Scalar(0, 255, 0),
              3);
    if (area(bestBox.xMin, bestBox.xMax, bestBox.yMin, bestBox.yMax)) {
        rectangle(result, Point(bestBox.xMin, bestBox.yMin), Point(bestBox.xMax, bestBox.yMax), Scalar(0, 0, 255), 3);
    }
    float iou = computeIoU(manualBox, bestBox);
    cout << "Result: " << iou << '\n';
    imshow("plate", result);
    waitKey();
    return 0;
}