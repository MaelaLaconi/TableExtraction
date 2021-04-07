#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray, dst, mask1;
int thresh = 200;
int max_thresh = 255;

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
    Mat mask1dilate;
    // retourne les lignes suivant un angle séléctionné
    getAnglesLines(5, angle, linesP, cdstP);

    //dilatation
    dilate(mask1, mask1, kernel);

    /*int erosion_size = 5;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS,
        cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        cv::Point(erosion_size, erosion_size));    Mat mask1dilate;
    dilate(mask1, mask1dilate, kernel);*/
    // Show results
    imshow("mask1 avant dilatation", mask1);

    waitKey();

    return 0;
}