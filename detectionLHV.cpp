#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray, dst, mask1, mask2;
int thresh = 200;
int max_thresh = 255;

const char* source_window = "Source image";
const char* corners_window = "Corners detected";
string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}
void ThinSubiteration1(Mat& pSrc, Mat& pDst) {
    int rows = pSrc.rows;
    int cols = pSrc.cols;
    printf("nb rows = %d", rows); 

    pSrc.copyTo(pDst);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (pSrc.at<float>(i, j) == 1.0f) {
                /// get 8 neighbors
                /// calculate C(p)
                int neighbor0 = (int)pSrc.at<float>(i - 1, j - 1);
                int neighbor1 = (int)pSrc.at<float>(i - 1, j);
                int neighbor2 = (int)pSrc.at<float>(i - 1, j + 1);
                int neighbor3 = (int)pSrc.at<float>(i, j + 1);
                int neighbor4 = (int)pSrc.at<float>(i + 1, j + 1);
                int neighbor5 = (int)pSrc.at<float>(i + 1, j);
                int neighbor6 = (int)pSrc.at<float>(i + 1, j - 1);
                int neighbor7 = (int)pSrc.at<float>(i, j - 1);
                int C = int(~neighbor1 & (neighbor2 | neighbor3)) +
                    int(~neighbor3 & (neighbor4 | neighbor5)) +
                    int(~neighbor5 & (neighbor6 | neighbor7)) +
                    int(~neighbor7 & (neighbor0 | neighbor1));
                if (C == 1) {
                    /// calculate N
                    int N1 = int(neighbor0 | neighbor1) +
                        int(neighbor2 | neighbor3) +
                        int(neighbor4 | neighbor5) +
                        int(neighbor6 | neighbor7);
                    int N2 = int(neighbor1 | neighbor2) +
                        int(neighbor3 | neighbor4) +
                        int(neighbor5 | neighbor6) +
                        int(neighbor7 | neighbor0);
                    int N = min(N1, N2);
                    if ((N == 2) || (N == 3)) {
                        /// calculate criteria 3
                        int c3 = (neighbor1 | neighbor2 | ~neighbor4) & neighbor3;
                        if (c3 == 0) {
                            pDst.at<float>(i, j) = 0.0f;
                        }
                    }
                }
            }
        }
    }
}


void ThinSubiteration2(Mat& pSrc, Mat& pDst) {
    int rows = pSrc.rows;
    int cols = pSrc.cols;
    pSrc.copyTo(pDst);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (pSrc.at<float>(i, j) == 1.0f) {
                /// get 8 neighbors
                /// calculate C(p)
                int neighbor0 = (int)pSrc.at<float>(i - 1, j - 1);
                int neighbor1 = (int)pSrc.at<float>(i - 1, j);
                int neighbor2 = (int)pSrc.at<float>(i - 1, j + 1);
                int neighbor3 = (int)pSrc.at<float>(i, j + 1);
                int neighbor4 = (int)pSrc.at<float>(i + 1, j + 1);
                int neighbor5 = (int)pSrc.at<float>(i + 1, j);
                int neighbor6 = (int)pSrc.at<float>(i + 1, j - 1);
                int neighbor7 = (int)pSrc.at<float>(i, j - 1);
                int C = int(~neighbor1 & (neighbor2 | neighbor3)) +
                    int(~neighbor3 & (neighbor4 | neighbor5)) +
                    int(~neighbor5 & (neighbor6 | neighbor7)) +
                    int(~neighbor7 & (neighbor0 | neighbor1));
                if (C == 1) {
                    /// calculate N
                    int N1 = int(neighbor0 | neighbor1) +
                        int(neighbor2 | neighbor3) +
                        int(neighbor4 | neighbor5) +
                        int(neighbor6 | neighbor7);
                    int N2 = int(neighbor1 | neighbor2) +
                        int(neighbor3 | neighbor4) +
                        int(neighbor5 | neighbor6) +
                        int(neighbor7 | neighbor0);
                    int N = min(N1, N2);
                    if ((N == 2) || (N == 3)) {
                        int E = (neighbor5 | neighbor6 | ~neighbor0) & neighbor7;
                        if (E == 0) {
                            pDst.at<float>(i, j) = 0.0f;
                        }
                    }
                }
            }
        }
    }
}


double angleBetween(const Point& v1, const Point& v2)
{
    double delta_x, delta_y;
    double x1 = v1.x;
    double y1 = v1.y;
    double x2 = v2.x;
    double y2 = v2.y;
    delta_x = abs(x2 - x1);
    delta_y = abs(y2 - y1);

    double theta_radians = atan2(delta_y, delta_x);
    double angleDegres = atan2(delta_y, delta_x) * 180 / CV_PI;

    return angleDegres;
}

void getHorizontalLigne(double epsilon, vector<Vec4i> linesP, Mat cdstP) {
    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        Vec4i l = linesP[i];
        Point point1 = Point(l[0], l[1]);
        Point point2 = Point(l[2], l[3]);
        int x = point1.x;
        int y = point1.y;

        int x2 = point2.x;
        int y2 = point2.y;
        if ((y2 <= y + epsilon && y2 >= y - epsilon)) {
            line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
        }

    }

    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    // Wait and Exit
    waitKey();
}

void getVerticalLigne(double epsilon, vector<Vec4i> linesP, Mat cdstP) {
    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        //ajouter ici pour gérer angle
        Vec4i l = linesP[i];
        Point point1 = Point(l[0], l[1]);
        Point point2 = Point(l[2], l[3]);
        int x = point1.x;
        int y = point1.y;

        int x2 = point2.x;
        int y2 = point2.y;
        if ((x2 <= x + epsilon && x2 >= x - epsilon)) {
            line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
        }
    }

    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);

    // Wait and Exit
    waitKey();
}

// angle en radiant
void getAnglesLines(double epsilon, double angle, vector<Vec4i> linesP, Mat cdstP) {
    int cpt = 0;

    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        //ajouter ici pour gérer angle
        Vec4i l = linesP[i];

        Point point1 = Point(l[0], l[1]);
        Point point2 = Point(l[2], l[3]);

        if (angleBetween(point1, point2) <= (angle + epsilon) && angleBetween(point1, point2) >= (angle - epsilon)) {
            // check the density of the line
            LineIterator it(dst, point1, point2, 8);

            std::vector<cv::Vec3b> buf(it.count);
            std::vector<cv::Point> points(it.count);
            int countBlack = 0;

            for (int i = 0; i < it.count; i++, ++it){
                int gray = (int)dst.at<uchar>(it.pos());
                //les lignes sont blaches sur font noir
                if (gray == 255) {
                    countBlack++;
                }
            }
          
            // 90% of black pixel in the line
            double resultat = ((double)countBlack) / ((double)it.count);
            if (resultat > 0.9) {
                
                // version sans prolongement
                //line(cdstP, point1, point2, Scalar(0, 0, 255), 3, LINE_AA);

                // prolongement vertical
                if (angle == 90) {
                    

                    // ajout de la droite détéctée dans le masque
                    line(mask1, Point(l[0], 0), Point(l[2], src.rows), Scalar(255, 255, 255), 3, LINE_AA);

                    line(cdstP, Point(l[0], 0), Point(l[2], src.rows), Scalar(0, 0, 255), 3, LINE_AA);
                    cpt++;
                }

                // prolongement horizontal
                if (angle == 0) {
                    line(mask1, Point(l[0], 0), Point(l[2], src.rows), Scalar(255, 255, 255), 3, LINE_AA);

                    line(cdstP, Point(0, l[1]), Point(src.cols, l[3]), Scalar(0, 0, 255), 3, LINE_AA);
                    cpt++;

                }

            }
        }

    }
    printf("compteur %d", cpt);
    cv::imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    // Wait and Exit
    cv::waitKey();
}

// angle en radiant
void checkLinesMask(vector<Vec4i> linesMask1, Mat mask2) {
    int cpt = 0;
    printf("au debut de la detection de ligne dans le mask2 %d", cpt);
    // Draw the lines
    for (size_t i = 0; i < linesMask1.size(); i++)
    {
        //ajouter ici pour gérer angle
        Vec4i l = linesMask1[i];

        Point point1 = Point(l[0], l[1]);
        Point point2 = Point(l[2], l[3]);
        // version sans prolongement
        line(mask2, point1, point2, Scalar(0, 0, 255), 3, LINE_AA);
        cpt++;
        }

    
    printf("\n\nNOMBRE DE LIGNES DAND LE MASK2 %d\n", cpt);
    cv::imshow("VIRIFICATION MASK2", mask2);
    // Wait and Exit
    cv::waitKey();
}


int main(int argc, char** argv)
{
    printf("Entrer un angle en degre : \n");
    double angle;
    scanf_s("%lf", &angle);
  
     //Declare the output variables
    Mat cdst, cdstP;
    //const char* default_file = "ressources/eu-002/eu-002-1.jpg";
    const char* default_file = "ressources/eu-005/eu-005-03.jpg";
    const char* filename = argc >= 2 ? argv[1] : default_file;
    // Loads an image
    Mat bigSrc = imread(samples::findFile(filename), IMREAD_GRAYSCALE);

    cv::resize(bigSrc, src, cv::Size(), 0.35, 0.35);
    double epsilonX = src.cols * 0.05;
    double epsilonY = src.rows * 0.05;

    // Check if image is loaded fine
    if (src.empty()) {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", default_file);
        return -1;
    }

    // Edge detection
    Canny(src, dst, 50, 200, 3);

    // Copy edges to the images that will display the results in BGR
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    cdstP = cdst.clone();

    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(dst, linesP, 1, CV_PI / 180, 100, 20, 3); // runs the actual detection

    // notre premier masque, tout noir pour le moment
    mask1 = Mat::zeros(dst.size(), dst.type());

    // dilatation de notre masque
    int dilatationSize = 5;
    Mat kernel = Mat::ones(dilatationSize, dilatationSize, CV_8UC1);
    Mat mask1erode;
    // retourne les lignes suivant un angle séléctionné
    getAnglesLines(5, angle, linesP, cdstP);

    //dilatation
    dilate(mask1, mask1, kernel);
    imshow("mask1 apres dilatation", mask1);

    
    kernel = Mat::ones(8, 8, CV_8UC1);
    // doit-on garder le meme kernel ???
    erode(mask1, mask1, kernel);
    imshow("mask1 apres erosion", mask1);

    // creation du deuxieme mask
    mask2 = Mat::zeros(dst.size(), dst.type());
   
    Canny(mask1, mask1, 50, 200, 3);
    Mat mask1Copy, mask2Thin1, mask2Thin2;
    mask2Thin1 = mask1Copy.clone();
    mask2Thin2 = mask1Copy.clone();

    cvtColor(mask1, mask1Copy, COLOR_GRAY2BGR);
    Mat mask2 = mask1Copy.clone();

    string ty = type2str(mask1Copy.type());
    printf("Matrix: %s %dx%d \n", ty.c_str(), mask1Copy.cols, mask1Copy.rows);

    mask1Copy.convertTo(mask1Copy, CV_32F);

    /*ThinSubiteration1(mask1Copy, mask2Thin1);
    imshow("mask1 PREMIER THIN", mask2Thin1);*/

    ThinSubiteration2(mask1Copy, mask2Thin1);

    imshow("mask1 DEUXIEME THIN", mask2Thin1);

    vector<Vec4i> linesMask1; // lignes qui vont etre detectees dans le mask1
    HoughLinesP(mask1, linesMask1, 1, CV_PI / 180, 100, 20, 3); // runs the actual detection
    
    // on recherche les lignes dans le mask 1, puis on mets tout dans le masque 2
    checkLinesMask(linesMask1, mask2);

  
    // Show results

    imshow("mask2 a la fin", mask2);
  

    waitKey();

    return 0;
}