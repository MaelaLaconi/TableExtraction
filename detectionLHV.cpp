#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray, blackAndWhiteImage;
int thresh = 200;
int max_thresh = 255;

const char* source_window = "Source image";
const char* corners_window = "Corners detected";

/// Function header
void cornerHarris_demo(int, void*);

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

    return theta_radians;
}

void getHorizontalLigne(double epsilon, vector<Vec4i> linesP, Mat cdstP){
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
        if ((y2 <= y + epsilon && y2 >= y - epsilon)) {
            line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
        }
      
    }
 
    //imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
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

    //imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    // Wait and Exit
    waitKey();
}

// angle en radiant
void getAnglesLines(double epsilon, double angle, vector<Vec4i> linesP, Mat cdstP){
    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        //ajouter ici pour gérer angle
        Vec4i l = linesP[i];
        Point point1 = Point(l[0], l[1]);
        Point point2 = Point(l[2], l[3]);

        if (angleBetween(point1, point2) <= (angle + epsilon) && angleBetween(point1, point2) >= (angle - epsilon)) {
            // check the density of the line

            LineIterator it(src, point1, point2, 8);
            std::vector<cv::Vec3b> buf(it.count);
            std::vector<cv::Point> points(it.count);
            int countBlack = 0;
            for (int i = 0; i < it.count; i++, ++it)
            {
                buf[i] = *(const Vec3b*)*it;

                //Vec3b color = img.at<Vec3b>(it.pos());
                int blue = buf[i].val[0];// B
                int green = buf[i].val[1]; // G
                int red = buf[i].val[2]; // R

                //printf("rgb=(%d, %d, %d)\n", red, green, blue);
                // case black not white
                if (red == 255 && blue == 255 && green == 255) {
                    countBlack++;
                }
            }
            //printf("*************************\nnombre black = %d nombre total = %d \n", countBlack, it.count);

            // 70% of black pixel in the line
            double resultat = ((double)countBlack) / ((double)it.count);
            if (resultat > 0.8) {
                line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
            }
        }
        
    }

    //imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    cv::imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    // Wait and Exit
    cv::waitKey();
}


int main(int argc, char** argv)
{
    char* select = new char[50];
    /*printf("\n0. Selectionner lignes horizontales.\n1. Selectionner lignes verticales.\n2. Selectionner angle 180\n");
    cin >> select;*/

    // Declare the output variables
    Mat dst, cdst, cdstP;
    //const char* default_file = "ressources/eu-002/eu-002-1.jpg";
    const char* default_file = "ressources/eu-006/eu-006-3.jpg";
    const char* filename = argc >=2 ? argv[1] : default_file;
    // Loads an image
    Mat bigSrc = imread( samples::findFile( filename ), IMREAD_GRAYSCALE);

    cv::resize(bigSrc, src, cv::Size(), 0.35, 0.35);
    threshold(src, blackAndWhiteImage, 100, 255, THRESH_BINARY);

    double epsilonX = src.cols * 0.05;
    double epsilonY = src.rows * 0.05;
    /*Mat src1 = imread(samples::findFile(filename));
    cv::resize(src1, src1, cv::Size(), 0.35, 0.35);

    cvtColor(src1, src_gray, COLOR_BGR2GRAY);*/

    // Check if image is loaded fine
    if(src.empty()){
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
    
    HoughLinesP(dst, linesP, 1, CV_PI/180, 200, 20, 3 ); // runs the actual detection
   


    /*if (strcmp(select, "0") == 0) {
        getHorizontalLigne(epsilonX, linesP, cdstP);
    }
    if (strcmp(select, "1") == 0) {
        getVerticalLigne(epsilonY, linesP, cdstP);
    }*/

    //if (strcmp(select, "2") == 0) {
        // 0 rad pour horizontal, CV_PI/2
    getAnglesLines(0.8, 0 , linesP, cdstP);

    //}

    // Show results
    imshow("Source", src);
    waitKey();

    // harris
    /// Create a window and a trackbar
    /*namedWindow(source_window);
    createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
    imshow(source_window, src1);

    cornerHarris_demo(0, 0);
    // Wait and Exit
    waitKey();*/

    
    return 0;
}

/*void cornerHarris_demo(int, void*)
{
    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    /// Detecting corners
    //Mat dst = Mat::zeros(src1.size(), CV_32FC1);
    //cornerHarris(src_gray, dst, blockSize, apertureSize, k);

    /// Normalizing
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    /// Drawing a circle around corners
    for (int i = 0; i < dst_norm.rows; i++)
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if ((int)dst_norm.at<float>(i, j) > thresh)
            {
                circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
            }
        }
    }

    /// Showing the result
    namedWindow(corners_window);
    imshow(corners_window, dst_norm_scaled);
}*/