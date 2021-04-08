#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray, dst;
int thresh = 200;
int max_thresh = 255;
Mat cdstP;

const char* source_window = "Source image";
const char* corners_window = "Corners detected";

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
void getAnglesLines(double epsilon, double angle, vector<Vec4i> linesP, Mat cdstP,int densite) {
    // Draw the lines
    double dens = (double) densite / 10;
    for (size_t i = 0; i < linesP.size(); i++)
    {
        //ajouter ici pour gérer angle
        Vec4i l = linesP[i];
        Point point1 = Point(l[0], l[1]);
        Point point2 = Point(l[2], l[3]);
        if (angle == 91) {
            //Horizntal 
            line(cdstP, point1, point2, Scalar(0, 0, 255), 3, LINE_AA);         
        }
                
        if (angleBetween(point1, point2) <= (angle + epsilon) && angleBetween(point1, point2) >= (angle - epsilon)) {
            // check the density of the line
            LineIterator it(dst, point1, point2, 8);

            std::vector<cv::Vec3b> buf(it.count);
            std::vector<cv::Point> points(it.count);
            int countBlack = 0;
           
            for (int i = 0; i < it.count; i++, ++it) {
                int gray = (int)dst.at<uchar>(it.pos());
                //les lignes sont blaches sur font noir
                if (gray == 255) {
                    countBlack++;
                }
            }

            // 90% of black pixel in the line
            double resultat = ((double)countBlack) / ((double)it.count);

                if (resultat > dens) {
                    line(cdstP, point1, point2, Scalar(0, 0, 255), 3, LINE_AA);
                }
                  
        }

    }
   //cv::imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    // Wait and Exit
    //cv::waitKey();
}
int main(int argc, char** argv)
{
    Mat cdst;
    // Read original image 
    Mat Imgsrc = imread("ressources/eu-005/eu-005-03.jpg");
    cv::resize(Imgsrc, src, cv::Size(), 0.35, 0.35);

    //if fail to read the image
    if (!src.data)
    {
        cout << "Error loading the image" << endl;
        return -1;
    }

    // Create a window
    namedWindow("My Window", 1);

    //Create trackbar to change brightness
    int degre = 0;
    createTrackbar("Degré", "My Window", &degre, 91);

    //Create trackbar to change contrast
    int epaisseur = 5;
    createTrackbar("Épaisseur", "My Window", &epaisseur, 10);

    //Create trackbar to change contrast
    int densite = 9;
    createTrackbar("Densité", "My Window", &densite, 10);

    while (true)
    {
        Canny(src, dst, 50, 200, 3);
        // Copy edges to the images that will display the results in BGR
        cvtColor(dst, cdst, COLOR_GRAY2BGR);
        cdstP = cdst.clone();

        // Probabilistic Line Transform
        vector<Vec4i> linesP; // will hold the results of the detection
        HoughLinesP(dst, linesP, 1, CV_PI / 180, 100, 20, 3); // runs the actual detection

         // retourne les lignes suivant un angle séléctionné
        getAnglesLines(5, degre, linesP, cdstP, densite);
         imshow("My Window", cdstP);
      

        // Wait until user press some key for 50ms
        int iKey = waitKey(50);

        //if user press 'ESC' key
        if (iKey == 27)
        {
            break;
        }
    }

    return 0;
}
