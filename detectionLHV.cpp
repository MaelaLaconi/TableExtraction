#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

double angleBetween(const Point& v1, const Point& v2)
{
    double delta_x, delta_y;
    int x1 = v1.x;
    int y1 = v1.y;
    int x2 = v2.x;
    int y2 = v2.y;
    delta_x = x2 - x1;
    delta_y = y2 - y1;
    double theta_radians = atan2(delta_y, delta_x);

    
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
            line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
        }
        
    }

    //imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    // Wait and Exit
    waitKey();
}

int main(int argc, char** argv)
{
    char* select = new char[50];
    printf("\n0. Selectionner lignes horizontales.\n1. Selectionner lignes verticales.\n2. Selectionner angle 90\n");
    cin >> select;

    double epsilon = 50;
    // Declare the output variables
    Mat dst, cdst, cdstP;
    const char* default_file = "ressources/eu-001/eu-001-1.jpg";
    const char* filename = argc >=2 ? argv[1] : default_file;
    // Loads an image
    Mat bigSrc = imread( samples::findFile( filename ), IMREAD_GRAYSCALE);
    Mat src;
    cv::resize(bigSrc, src, cv::Size(), 0.35, 0.35);



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
    HoughLinesP(dst, linesP, 1, CV_PI/180, 50, 50, 10 ); // runs the actual detection
   
    if (strcmp(select, "0") == 0) {
        getHorizontalLigne(epsilon, linesP, cdstP);
    }
    if (strcmp(select, "1") == 0) {
        getVerticalLigne(epsilon, linesP, cdstP);
    }

    if (strcmp(select, "2") == 0) {
        // 0 rad pour horizontal, CV_PI/2
        getAnglesLines(0.8, 0, linesP, cdstP);
    }

    // Show results
    //imshow("Source", src);

    // Wait and Exit
    waitKey();

    
    return 0;
}