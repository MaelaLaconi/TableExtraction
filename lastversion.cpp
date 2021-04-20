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
Mat cdstP;
RNG rng(12345);

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
void getAnglesLines(double epsilon, double angle, vector<Vec4i> linesP, Mat cdstP, int densite, vector<Vec4i> linesH, vector<Vec4i> linesV){
    // Draw the lines
    int cpt = 0;
    double dens = (double)densite / 10;
    for (size_t i = 0; i < linesP.size(); i++)
    {
        //ajouter ici pour gérer angle
        Vec4i l = linesP[i];
        Point point1 = Point(l[0], l[1]);
        Point point2 = Point(l[2], l[3]);
        if (angle == 91) {
            //Horizntal 
            if ((angleBetween(point1, point2) <= (0 + epsilon) && angleBetween(point1, point2) >= (0 - epsilon)) || (angleBetween(point1, point2) <= (90 + epsilon) && angleBetween(point1, point2) >= (90 - epsilon))) {
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
                    linesV.push_back(l);

                }



            }


        }

        else if (angleBetween(point1, point2) <= (angle + epsilon) && angleBetween(point1, point2) >= (angle - epsilon)) {
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

                // version sans prolongement
                //line(cdstP, point1, point2, Scalar(0, 0, 255), 3, LINE_AA);

                // prolongement vertical
                if (angle == 90) {

                    // ajout de la droite détéctée dans le masque
                    line(mask1, Point(l[0], 0), Point(l[2], src.rows), Scalar(255, 255, 255), 3, LINE_AA);

                    line(cdstP, Point(l[0], 0), Point(l[2], src.rows), Scalar(0, 0, 255), 3, LINE_AA);
                    cpt++;
                    linesV.push_back(l);
                }

                // prolongement horizontal
                if (angle == 0) {
                    line(mask1, Point(l[0], 0), Point(l[2], src.rows), Scalar(255, 255, 255), 3, LINE_AA);

                    line(cdstP, Point(0, l[1]), Point(src.cols, l[3]), Scalar(0, 0, 255), 3, LINE_AA);
                    cpt++;
                    linesH.push_back(l);

                }

            }

        }

    }
    //cv::imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
     // Wait and Exit
     //cv::waitKey();
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


    printf("NOMBRE DE LIGNES DAND LE MASK2 %d\n", cpt);
    cv::imshow("getANGLE2", mask2);
    // Wait and Exit
    cv::waitKey();
}
void getCoin(double eps, double angle, vector<Vec4i> linesP,  Mat cdstP, int densite){
    vector<Vec4i> linesH;
    vector<Vec4i> linesV;
    Mat copy=cdstP.clone();
    std::vector<cv::Point2f> corners;
    getAnglesLines(eps,angle,linesP,cdstP,densite,linesH, linesV);
    for (size_t i=0; i< linesH.size();i++){
        Vec4i lho=linesH[i];
        double aH= (lho[3]-lho[1])/ (double) (lho[2]-lho[0]);
        double bH=lho[1]-aH*lho[0];
        for (size_t j = 0; j < linesV.size();j ++) {
            Vec4i lvect=linesV[i];
            double aV= (lvect[3]-lvect[1])/ (double) (lvect[2]-lvect[0]);
            double bV=lvect[1]-aH*lvect[0];
            double x= (bV-bH)/(aH-aV);
            double y= aH * x + bH;
            circle( copy,Point2f(x,y) ,4, Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), -1, 8, 0 );

        }
    }
    imshow("Corner",copy);

}

int main(int argc, char** argv)
{
    Mat cdst;
    // Read original image 
    Mat Imgsrc = imread("ressources/eu-005/eu-005-03.jpg");
    cv::resize(Imgsrc, src, cv::Size(), 0.35, 0.35);
    double epsilonX = src.cols * 0.05;
    double epsilonY = src.rows * 0.05;

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

    Canny(src, dst, 50, 200, 3);
    // Copy edges to the images that will display the results in BGR
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    cdstP = cdst.clone();

    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(dst, linesP, 1, CV_PI / 180, 100, 20, 3); // runs the actual detection

    // notre premier masque, tout noir pour le moment
    mask1 = Mat::zeros(dst.size(), dst.type());



    while (true)
    {
        Canny(src, dst, 50, 200, 3);
        // Copy edges to the images that will display the results in BGR
        cvtColor(dst, cdst, COLOR_GRAY2BGR);
        cdstP = cdst.clone();

        // Probabilistic Line Transform
        vector<Vec4i> linesP; // will hold the results of the detection
        HoughLinesP(dst, linesP, 1, CV_PI / 180, 100, 20, 3); // runs the actual detection     
         // notre premier masque, tout noir pour le moment
        mask1 = Mat::zeros(dst.size(), dst.type());
        // retourne les lignes suivant un angle séléctionné
         // retourne les lignes suivant un angle séléctionné
        //getAnglesLines(5, degre, linesP, cdstP, densite);
        getCoin(5, degre, linesP, cdstP, densite);
        imshow("My Window", cdstP);

       

        // dilatation de notre masque
        int dilatationSize = 5;
        Mat kernel = Mat::ones(dilatationSize, dilatationSize, CV_8UC1);
        Mat mask1erode;

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
        Mat mask1Copy;
        cvtColor(mask1, mask1Copy, COLOR_GRAY2BGR);
        Mat mask2 = mask1Copy.clone();

        vector<Vec4i> linesMask1; // lignes qui vont etre detectees dans le mask1
        HoughLinesP(mask1, linesMask1, 1, CV_PI / 180, 100, 20, 3); // runs the actual detection

        // on recherche les lignes dans le mask 1, puis on mets tout dans le masque 2
        checkLinesMask(linesMask1, mask2);


        // Show results

        imshow("mask2 a la fin", mask2);


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