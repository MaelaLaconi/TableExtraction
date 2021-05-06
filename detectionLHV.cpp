#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <streambuf>

using namespace cv;
using namespace std;

/// Global variables
Mat  maskPE, src, src_gray, dst, mask1, mask2, cdstP, maskHorizontal, maskVertical, maskVeriteTerrain, test;
int thresh = 200;
int max_thresh = 255;
RNG rng(12345);
const char* source_window = "Source image";
const char* corners_window = "Corners detected";
char myArray[10000];
String cordonner[10000];


vector<Vec4i> linesHori;
vector<Vec4i> linesVerti;
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

void ThinSubiteration1(Mat& pSrc, Mat& pDst, vector <Vec4i>& linesV, vector <Vec4i>& linesH) {
    int rows = pSrc.rows;
    int cols = pSrc.cols;


    Vec4i l;
    int cpt = 0;
    pSrc.copyTo(pDst);
    string ty = type2str(pSrc.type());
    pDst = Mat::zeros(pSrc.size(), pSrc.type());

    cvtColor(pDst, pDst, COLOR_GRAY2BGR);

    // printf("Matrix thin: %s %dx%d \n", ty.c_str(), pSrc.cols, pSrc.rows);

    int epsilon = 5;


    for (int i = 0; i < rows - epsilon; i++) {

        // soit 0 (noir) soit 255 (blanc)
        int currentColor = (int)pSrc.at<uchar>(i, 0);


        if (currentColor == 255) {
            l[0] = 0;
            l[1] = i;
            l[2] = cols;
            l[3] = i;
            // ligne horizontale
            line(pDst, Point(0, i), Point(cols, i), Scalar(255, 255, 255), 1, LINE_AA);
            linesH.push_back(l);
            cpt++;
            i += 5;
        }
        //printf("color = %d\n", currentColor);
    }
    //printf("channel == %d \n", pSrc.channels());

    // ne marche pas
    for (int i = 0; i < cols; i++) {
        //printf("i = %d\n", i);
        // soit 0 (noir) soit 255 (blanc)

        int currentColor = (int)pSrc.at<uchar>(0, i);


        if (currentColor == 255) {
            //line(pDst, Point(i, 0), Point(i, rows), Scalar(0, 0, 255), 3, LINE_AA);
            line(pDst, Point(i, 0), Point(i, rows), Scalar(255, 255, 255), 1, LINE_AA);
            l[0] = i;
            l[1] = 0;
            l[2] = i;
            l[3] = rows;
            linesV.push_back(l);
            cpt++;
            //printf("point = (%d, %d)\n", i, 0);
            i += 5;
        }
        //printf("color = %d\n", currentColor);

    }

    //printf("detection de %d lignes", cpt);

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
            line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
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
            line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
        }
    }

    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);

    // Wait and Exit
    waitKey();
}

// angle en radiant
void getAnglesLines(double epsilon, double angle, vector<Vec4i> linesP, Mat cdstP, int densite) {
    maskHorizontal = Mat::zeros(dst.size(), dst.type());
    maskVertical = Mat::zeros(dst.size(), dst.type());

    // Draw the lines
    double dens = (double)densite / 10;
    int cpt = 0;

    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        //ajouter ici pour gérer angle
        Vec4i l = linesP[i];

        Point point1 = Point(l[0], l[1]);
        Point point2 = Point(l[2], l[3]);

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


        if (angle == 91) {

            //Horizntal 
            if ((angleBetween(point1, point2) <= (0 + epsilon) && angleBetween(point1, point2) >= (0 - epsilon))) {
                if (resultat > dens) {
                    // avec prolongement

                    line(mask1, Point(0, l[1]), Point(src.cols, l[3]), Scalar(255, 255, 255), 1, LINE_AA);
                    line(maskHorizontal, point1, point2, Scalar(255, 255, 255), 1, LINE_AA);

                    //line(cdstP, Point(0, l[1]), Point(src.cols, l[3]), Scalar(0, 0, 255), 1, LINE_AA);
                    linesHori.push_back(l);
                    line(cdstP, point1, point2, Scalar(0, 0, 255), 1, LINE_AA);
                    //line(mask1, point1, point2, Scalar(255, 255, 255), 3, LINE_AA);

                }

            }
            else if ((angleBetween(point1, point2) <= (90 + epsilon) && angleBetween(point1, point2) >= (90 - epsilon))) {


                if (resultat > dens) {
                    line(cdstP, point1, point2, Scalar(0, 0, 255), 1, LINE_AA);
                    //line(mask1, point1, point2, Scalar(255, 255, 255), 3, LINE_AA);
                    //line(cdstP, Point(l[0], 0), Point(l[2], src.rows), Scalar(0, 0, 255), 1, LINE_AA);
                    line(mask1, Point(l[0], 0), Point(l[2], src.rows), Scalar(255, 255, 255), 1, LINE_AA);
                    line(maskVertical, point1, point2, Scalar(255, 255, 255), 1, LINE_AA);

                    linesVerti.push_back(l);


                }

            }
        }

        else  if (angleBetween(point1, point2) <= (angle + epsilon) && angleBetween(point1, point2) >= (angle - epsilon)) {

            if (resultat > dens) {

                // prolongement vertical
                if (angle >= 90 - epsilon && angle <= 90 + epsilon) {


                    // ajout de la droite détéctée dans le masque
                    line(mask1, Point(l[0], 0), Point(l[2], src.rows), Scalar(255, 255, 255), 1, LINE_AA);

                    line(cdstP, Point(l[0], 0), Point(l[2], src.rows), Scalar(0, 0, 255), 1, LINE_AA);
                    cpt++;
                }


                // prolongement horizontal
                else if (angle >= 0 - epsilon && angle <= 0 + epsilon) {

                    line(mask1, Point(0, l[1]), Point(src.cols, l[3]), Scalar(255, 255, 255), 1, LINE_AA);
                    line(cdstP, Point(0, l[1]), Point(src.cols, l[3]), Scalar(0, 0, 255), 1, LINE_AA);
                    cpt++;

                }
                else {
                    line(cdstP, point1, point2, Scalar(0, 0, 255), 1, LINE_AA);
                    line(mask1, point1, point2, Scalar(255, 255, 255), 1, LINE_AA);

                }


            }
        }

    }

    //printf("compteur %d", cpt);
    //cv::imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    // Wait and Exit
   // cv::waitKey();
}

// angle en radiant
void checkLinesMask(vector<Vec4i> linesMask1, Mat mask2) {
    int cpt = 0;
    // Draw the lines
    for (size_t i = 0; i < linesMask1.size(); i++)
    {
        //ajouter ici pour gérer angle
        Vec4i l = linesMask1[i];

        Point point1 = Point(l[0], l[1]);
        Point point2 = Point(l[2], l[3]);
        // version sans prolongement
        //line(mask2, point1, point2, Scalar(0, 0, 255), 1, LINE_AA);
        cpt++;
    }


    //printf("\n\nNOMBRE DE LIGNES DAND LE MASK2 %d\n", cpt);
    cv::imshow("VIRIFICATION MASK2", mask2);
    // Wait and Exit
    cv::waitKey();
}
vector<cv::Point2f> getCoin(Mat cdstP, vector <Vec4i> linesV, vector <Vec4i> linesH) {

    std::vector<cv::Point2f> corners;
    Mat copy = cdstP.clone();
    //getAnglesLines(epsilon, angle, linesP, cdstP, densite, linesH, linesV);
    int cpt = 0;

    for (size_t i = 0; i < linesH.size(); ++i) {
        Vec4i line = linesH[i];
        Point pointA = Point(line[0], line[1]);
        Point pointB = Point(line[2], line[3]);
        int aH = pointB.y - pointA.y;
        int bH = pointA.x - pointB.x;
        int cH = aH * (pointA.x) + bH * (pointA.y);
        int premintH = -1;
        int premintV = -1;
        for (size_t j = 0; j < linesV.size(); ++j) {
            Vec4i l = linesV[j];
            Point pointC = Point(l[0], l[1]);
            Point pointD = Point(l[2], l[3]);
            int aV = pointD.y - pointC.y;
            int bV = pointC.x - pointD.x;
            int cV = aV * (pointC.x) + bV * (pointC.y);
            int det = aH * bV - aV * bH;
            if (det != 0.) {
                int interx = (bV * cH - bH * cV) / det;
                int intery = (aH * cV - aV * cH) / det;
                Point p = Point(interx, intery);
                corners.push_back(p);
                cpt++;
            }
        }
    }

    for (size_t i = 0; i < corners.size(); i++) {
        circle(copy, corners[i], 4, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
    }
    //printf("NOMBREE DE CORNER = %d\n", cpt);
    imshow("Corner (in red) ", copy);
    return corners;
}

void partieEnglebante(vector<Point2f> corners, vector<Point2f> sommets, Mat src) {
    maskPE = Mat::zeros(src.size(), src.type());
    for (int i = 0; i < corners.size(); ++i) {
        Point2f p = corners[i];
        if (i < corners.size() - 1) {
            for (int j = i; j < corners.size(); ++j) {
                Point2f point2F = corners[j];
                if (p.y == point2F.y) {
                    line(maskPE, p, point2F, Scalar(255, 255, 255), 3, LINE_AA);
                }
                if ((p.x == point2F.x)) {
                    line(maskPE, p, point2F, Scalar(255, 255, 255), 3, LINE_AA);
                }

            }
        }
        if (p.x > sommets[0].x && p.x < sommets[1].x) {
            Point2f point = Point2f(p.x, sommets[0].y);
            Point2f point2 = Point2f(p.x, sommets[1].y);
            line(maskPE, p, point, Scalar(255, 0, 0), 3, LINE_AA);
            line(maskPE, p, point2, Scalar(255, 0, 0), 3, LINE_AA);
        }
        if (p.y > sommets[1].y && p.y < sommets[3].y) {
            Point2f point = Point2f(sommets[1].x, p.y);
            Point2f point2 = Point2f(sommets[3].x, p.y);
            line(maskPE, p, point, Scalar(255, 0, 0), 3, LINE_AA);
            line(maskPE, p, point2, Scalar(255, 0, 0), 3, LINE_AA);
        }

    }

    cv::imshow("Partie englobante", maskPE);
}
void vote(vector<cv::Point2f> corners, vector <Vec4i> linesV, vector <Vec4i> linesH, Mat cdstP, vector<Vec4i> linesP) {
    //printf("DEBUT \n");
    int epsilon = 5;
    string ty = type2str(maskVertical.type());


    // printf("\nMatrix maskVertical: %s %dx%d \n", ty.c_str(), maskVertical.cols, maskVertical.rows);
     // regroupe tous les corner sur une même ligne verticale
    vector<cv::Point2f> goodCorner;
    vector<cv::Point2f> pointV;
    vector<cv::Point2f> pointH;
    for (int i = 0; i < corners.size(); i++) {
        Point p = corners[i];

        int row = p.y;
        int col = p.x;

        int gray = (int)maskVertical.at<uchar>(row, col);
        // printf("%d avec channel = %d\n", gray, maskVertical.channels());

        if (gray != 0) {
            gray = (int)maskHorizontal.at<uchar>(row, col);

            if (gray != 0) {
                goodCorner.push_back(p);

            }
        }


    }

    //printf("corner = %d %d\n", p.x, p.y);
    Mat coinTest = cdstP.clone();
    vector<Point2f> quatresCoins;
    Point2f firstPoint;
    firstPoint = goodCorner[0];
    Point2f dernierPoint = goodCorner[goodCorner.size() - 1];
    quatresCoins.push_back(firstPoint);
    quatresCoins.push_back(dernierPoint);
    for (int i = 0; i < goodCorner.size(); i++) {
        Point2f point2F = goodCorner[i];
        if (point2F.x == firstPoint.x && point2F.y == dernierPoint.y) {
            quatresCoins.push_back(point2F);
        }
        if (point2F.x == dernierPoint.x && point2F.y == firstPoint.y) {
            quatresCoins.push_back(point2F);
        }
        //circle(coinTest, goodCorner[i], 4, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
    }


    for (int k = 0; k < quatresCoins.size(); ++k) {
        Point2f p = quatresCoins[k];
        
        
        for (int v = 0; v < linesV.size(); ++v) {
            if (linesV[v][0] == p.x) {
                for (int m = linesV[v][1]; m < linesV[v][3]; ++m) {
                    Point2f pbisV = Point2f(p.x, m);
                    int row = pbisV.y;
                    int col = pbisV.x;
                    int grayV = (int)maskVertical.at<uchar>(row, col);
                    if (grayV != 0) {
                        pointV.push_back(pbisV);

                    }
                }
            }
        }


        for (int l = 0; l < linesH.size(); l++) {
            if (linesH[l][1] == p.y) {
                for (int n = linesH[l][0]; n < linesH[l][2]; ++n) {
                    Point2f pbis = Point2f(n, p.y);
                    int row = pbis.y;
                    int col = pbis.x;

                    int gray = (int)maskHorizontal.at<uchar>(row, col);
                    printf("et la\n");

                    if (gray != 0) {

                        pointH.push_back(pbis);

                    }
                }

            }
        }


    }

    int x = 0;
    Point2f pointXPlusG;
    for (int i = 0; i < pointH.size(); ++i) {
        if (pointH[i].x > x) {
            x = pointH[i].x;
            pointXPlusG = pointH[i];
        }
    }
    Point2f pointXPlusB = pointXPlusG;
    for (int i = 0; i < pointH.size(); ++i) {
        if (pointH[i].x < pointXPlusB.x) {
            pointXPlusB = pointH[i];
        }
    }
    int y = 0;
    Point2f pointYPlusG;
    for (int i = 0; i < pointV.size(); ++i) {
        if (pointV[i].y > y) {
            y = pointV[i].y;
            pointYPlusG = pointV[i];
        }
    }
    Point2f pointYPlusB = pointYPlusG;
    for (int i = 0; i < pointV.size(); ++i) {
        if (pointV[i].y < pointYPlusB.y) {
            pointYPlusB = pointV[i];
        }
    }
    vector<Point2f> sommets;
    Point2f coinHG, coinHD, coinBG, coinBD;
    coinHG = Point2f(pointXPlusB.x, pointYPlusB.y);
    sommets.push_back(coinHG);
    goodCorner.push_back(coinHG);
    coinHD = Point2f(pointXPlusG.x, pointYPlusB.y);
    goodCorner.push_back(coinHD);
    sommets.push_back(coinHD);
    coinBD = Point2f(pointXPlusG.x, pointYPlusG.y);
    coinBG = Point2f(pointXPlusB.x, pointYPlusG.y);
    goodCorner.push_back(coinBD);
    goodCorner.push_back(coinBG);
    sommets.push_back(coinBD);
    sommets.push_back(coinBG);
    for (int i = 0; i < goodCorner.size(); ++i) {
        circle(coinTest, goodCorner[i], 4, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
    }
    partieEnglebante(goodCorner, sommets, cdstP);
    /* for (int i = 0; i < quatresCoins.size(); ++i) {
         circle(coinTest, quatresCoins[i], 4, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
     }*/
     //printf("NOMBREE DE CORNER = %d\n", cpt);
    imshow("GOOD CORNERS ", coinTest);

}

void veriteTerrainMask() {
    // Mat src = imread("ressources/eu-004/eu-004-3.jpg");
    maskVeriteTerrain = Mat::zeros(src.size(), src.type());
    String pixel;
    int num_characters = 0;
    int i = 0;
    int j = 0;
    std::string cmpt;


    ifstream myfile("ressources/testpdf4.txt");

    if (myfile.is_open())
    {
        while (!myfile.eof())
        {
            myfile >> myArray[i];
            i++;
            num_characters++;
        }
        //Partie du read pour recuperer les numeros du fichier dans un Array de string 
        for (i = 0; i < num_characters; i++) {
            if (myArray[i] != '-' && myArray[i] != ',' && myArray[i] != '.') {
                cmpt.push_back(myArray[i]);

            }
            else {
                cordonner[j] = cmpt;
                j++;
                cmpt = "";

            }
        }

        for (i = 0; i < j; i++) {
            if (i + 3 < j) {
                double x1 = stoi(cordonner[i]);
                double y1 = stoi(cordonner[i + 1]);

                int newX1 = (int)x1 * 0.35;
                int newY1 = (int)y1 * 0.35;

                int x2 = stoi(cordonner[i + 2]);
                int y2 = stoi(cordonner[i + 3]);

                int newX2 = (int)x2 * 0.35;
                int newY2 = (int)y2 * 0.35;
                Point point1 = Point(stoi(cordonner[i]), stoi(cordonner[i + 1]));
                Point point2 = Point(stoi(cordonner[i + 2]), stoi(cordonner[i + 3]));
                //line(maskVeriteTerrain, point1, point2, Scalar(255, 255, 255), 1, LINE_AA);
                line(maskVeriteTerrain, Point(newX1, newY1), Point(newX2, newY2), Scalar(255, 255, 255), 1, LINE_AA);

                i = i + 3;

            }
        }
        // a revoir ac le resize
       // cv::resize(maskVeriteTerrain, test, cv::Size(), 0.35, 0.35);
        cv::imshow("Verite Terrain", maskVeriteTerrain);
        int diff = 0;
        int point = 0;
        int pointPE = 0;
        int vraiPos = 0;
        for (int k = 0; k < maskPE.rows; ++k) {
            for (int l = 0; l < maskPE.cols; ++l) {
                int g = maskPE.at<uchar>(k, l);
                if (g != 0) {
                    pointPE++;
                }
            }
        }

        printf("channel = %d", maskVeriteTerrain.channels());
        for (int k = 0; k < maskVeriteTerrain.rows; ++k) {
            for (int l = 0; l < maskVeriteTerrain.cols; ++l) {
                int vT = maskVeriteTerrain.at<uchar>(k, l);
                if (vT != 0) {
                    point++;
                }
                int pE = maskPE.at<uchar>(k, l);
                if (vT != pE && vT != 0 && pE != 0) {
                    diff++;
                }
                if (vT == pE && vT != 0) {
                    vraiPos++;
                }

            }

        }
        Mat dest;
        addWeighted(maskPE, 0.3, maskVeriteTerrain, 0.7, 0.0, dest);
        printf("nb de points blancs dans le masque verite terrain %d\n", point);
        printf("nb de points dans blancs le masque partie Englebante %d\n", pointPE);
        printf("nb points diff %d\n", diff);
        printf("nb de points vraiPositifs  %d\n", vraiPos);
        // Wait and Exit
        imshow("test ", dest);
        cv::waitKey();
    }
}
int main(int argc, char** argv)
{
    Mat cdst;
    // Read original image 
    Mat Imgsrc = imread("ressources/eu-004/eu-004-3.jpg");
    cv::resize(Imgsrc, src, cv::Size(), 0.35, 0.35);

    //if fail to read the image
    if (!src.data)
    {
        cout << "Error loading the image" << endl;
        return -1;
    }

    int epsilon = 5;

    // Create a window
    namedWindow("My Window", 1);

    //Create trackbar to change brightness
    int degre = 91;
    createTrackbar("Degré", "My Window", &degre, 91);

    //Create trackbar to change contrast
    int epaisseur = 5;
    createTrackbar("Épaisseur", "My Window", &epaisseur, 10);

    //Create trackbar to change contrast
    int densite = 9;
    createTrackbar("Densité", "My Window", &densite, 10);


    while (true)
    {

        // Edge detection
        Canny(src, dst, 50, 200, 3);

        // Copy edges to the images that will display the results in BGR
        cvtColor(dst, cdst, COLOR_GRAY2BGR);
        cdstP = cdst.clone();

        // Probabilistic Line Transform
        vector<Vec4i> linesP; // will hold the results of the detection
        HoughLinesP(dst, linesP, 1, CV_PI / 180, 100, 20, 3); // runs the actual detectionver

        // notre premier masque, tout noir pour le moment
        mask1 = Mat::zeros(dst.size(), dst.type());


        // dilatation de notre masque
        int dilatationSize = 5;
        Mat kernel = Mat::ones(dilatationSize, dilatationSize, CV_8UC1);
        Mat mask1erode;

        getAnglesLines(epsilon, degre, linesP, cdstP, densite);

        // retourne les lignes suivant un angle séléctionné
        //getCoin(5, degre, linesP, cdstP, densite, vertical, horizontal);
        imshow("My Window", cdstP);

        //imshow("mask1 de depart", mask1);

        //dilatation
        dilate(mask1, mask1, kernel);
        imshow("mask1 apres dilatation", mask1);
        dilatationSize = 3;
        kernel = Mat::ones(dilatationSize, dilatationSize, CV_8UC1);
        dilate(maskVertical, maskVertical, kernel);
        dilate(maskHorizontal, maskHorizontal, kernel);

        kernel = Mat::ones(5, 5, CV_8UC1);
        // doit-on garder le meme kernel ???
        erode(mask1, mask1, kernel);
        imshow("mask1 apres erosion", mask1);


        // creation du deuxieme mask
        mask2 = Mat::zeros(dst.size(), dst.type());

        Canny(mask1, mask1, 50, 200, 3);
        Mat mask1Copy, mask2;
        mask2 = mask1Copy.clone();

        cvtColor(mask1, mask1Copy, COLOR_GRAY2BGR);
        //Mat mask2 = mask1Copy.clone();

        string ty = type2str(mask1Copy.type());
        //printf("Matrix: %s %dx%d \n", ty.c_str(), mask1Copy.cols, mask1Copy.rows);

        //mask1Copy.convertTo(mask1Copy, CV_32F);
        cvtColor(mask1Copy, mask1Copy, COLOR_BGR2GRAY, 1);

        vector <Vec4i> vertical;
        vector <Vec4i> horizontal;
        ThinSubiteration1(mask1Copy, mask2, vertical, horizontal);
        //thin(mask1Copy, mask2Thin1, 5);
        imshow("Mask2", mask2);
        //printf("\nligne hori = %d et ligne verti = %d\n", horizontal.size(), vertical.size());
        vector<cv::Point2f> coins = getCoin(mask2, vertical, horizontal);


        vote(coins, vertical, horizontal, cdstP, linesP);


        vector<Vec4i> linesMask1; // lignes qui vont etre detectees dans le mask1
        cvtColor(mask2, mask2, COLOR_BGR2GRAY, 1);
        imshow("hori", maskHorizontal);
        imshow("verti", maskVertical);
        //verite terrain
        veriteTerrainMask();


        // on recherche les lignes dans le mask 1, puis on mets tout dans le masque 2
        //checkLinesMask(linesMask1, mask2);


        // Show results

        //imshow("mask2 a la fin", mask2);

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