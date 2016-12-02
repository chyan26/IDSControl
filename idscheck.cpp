
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <getopt.h>
#include <wchar.h>
#include <wctype.h>
#include <iostream>
#include <string>


#include "opencv2\opencv.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"


using namespace cv;
using namespace std;


Mat analysisCenter(Mat imageData){
	RNG rng(12345);


    /* Convert to the grey scale image to color image for displaying */
    cv::Mat img;
    cv::cvtColor(imageData, img, CV_GRAY2BGR);


    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    Canny( imageData, canny_output, 200, 200*1.5, 3 );
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    /// Get the moments
     vector<Moments> mu(contours.size() );
     for( int i = 0; i < contours.size(); i++ )
        { mu[i] = moments( contours[i], false ); }

     ///  Get the mass centers:
     vector<Point2f> mc( contours.size() );
     for( int i = 0; i < contours.size(); i++ )
        { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

     /// Draw contours
     Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

     if (contours.size() > 0){
         drawContours( img, contours, 0, color, 2, 8, hierarchy, 0, Point() );
         circle( img, mc[0], 4, color, -1, 8, 0 );
         std::cout << mc[0].x <<' '<< mc[0].y << std::endl;
     }


    /// Show in a window
    //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	//imshow( "Contours", img );
    //cv::waitKey(0);

    //cv::imwrite(filename, img);
    
    return img;

}
Mat analysisAngle(Mat imageData){
	RNG rng(12345);

	/* Establish an array for storing the image data */
    //cv::Mat imageData(1024,1280, CV_8UC1);
    //for( int i = 0; i < 1280; i++ ){
    //	for( int j = 0; j < 1024; j++ ){
    //		imageData.at<uchar>(j,i)=pMem[j*1280+i];
    //	}
    //}

    /* Convert to the grey scale image to color image for displaying */
    cv::Mat img;
    cv::cvtColor(imageData, img, CV_GRAY2BGR);

    /* Convert to binary */

    cv::Mat thres;
    cv::threshold(imageData, thres, 0.7*255, 255, cv::THRESH_BINARY);

    /* Find contours */

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(thres, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);


    /// Find the rotated rectangles and ellipses for each contour
    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> minEllipse( contours.size() );

    for( int i = 0; i < contours.size(); i++ )
       { minRect[i] = minAreaRect( Mat(contours[i]) );
         if( contours[i].size() > 200 )
           { minEllipse[i] = fitEllipse( Mat(contours[i]) ); }
       }

    /// Draw contours + rotated rects + ellipses
    Mat drawing = Mat::zeros( thres.size(), CV_8UC3);
    for( int i = 0; i< contours.size(); i++ )
       {
         Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

         //printf("size %f\n",minAreaRect( Mat(contours[i]) ));
         // contour
         //drawContours( gray, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
         // ellipse
         ellipse( img, minEllipse[i], color, 2, 8 );
         circle(img,  minEllipse[i].center, 3, Scalar(0, 255, 255), -1,8);

         if( contours[i].size() > 200 ) std::cout << minEllipse[i].center.x<< ' '<<  minEllipse[i].center.y << std::endl;

       }


    /// Show in a window
    //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    //imshow( "Contours", img );
    //cv::waitKey(0);

    //cv::imwrite(filename, img);
    return img;

}


Mat analysisFocus(Mat imageData){
	RNG rng(12345);

    /* Convert to the grey scale image to color image for displaying */
    cv::Mat img;
    cv::cvtColor(imageData, img, CV_GRAY2BGR);
    //cvCvtColor(imageData, img, CV_GRAY2BGR);

    /* Convert to binary threshold image*/
    cv::Mat thres;
    cv::threshold(imageData, thres, 100, 255, cv::THRESH_BINARY);

    /* Find contours based on theshold image*/
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(thres, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);


    /// Find the rotated rectangles and ellipses for each contour
    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> minEllipse( contours.size() );

    /* Going through every contour for ellipse detecting */
    for( int i = 0; i < contours.size(); i++ )
       { minRect[i] = minAreaRect( Mat(contours[i]) );
         if( contours[i].size() > 5 )
           { minEllipse[i] = fitEllipse( Mat(contours[i]) ); }
       }

    /// Draw contours + rotated rects + ellipses
    Mat drawing = Mat::zeros( thres.size(), CV_8UC3);
    for( int i = 0; i< contours.size(); i++ )
       {
         Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
         ellipse( img, minEllipse[i], color, 2, 8 );
       }

    Point2f start_points;
    start_points=minEllipse[0].center;

    Scalar color = Scalar( rng.uniform(2, 255), rng.uniform(2,255), rng.uniform(2,255) );
    line(img,minEllipse[0].center, minEllipse[1].center, color, 1, 8 );
    line(img,minEllipse[1].center, minEllipse[2].center, color, 1, 8 );
    line(img,minEllipse[2].center, minEllipse[3].center, color, 1, 8 );
    line(img,minEllipse[3].center, minEllipse[0].center, color, 1, 8 );

    Point diff1 = minEllipse[0].center - minEllipse[1].center;
    Point diff2 = minEllipse[1].center - minEllipse[2].center;
    Point diff3 = minEllipse[2].center - minEllipse[3].center;
    Point diff4 = minEllipse[0].center - minEllipse[3].center;
    
    float res1= cv::sqrt(diff1.x*diff1.x + diff1.y*diff1.y);
    float res2= cv::sqrt(diff2.x*diff2.x + diff2.y*diff2.y);
    float res3= cv::sqrt(diff3.x*diff3.x + diff3.y*diff3.y);
    float res4= cv::sqrt(diff4.x*diff4.x + diff4.y*diff4.y);

	printf("dist %f %f %f %f %f\n",res1,res2,res3,res4,(res1+res2+res3+res4)/4);

	
    /// Show in a window
    //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    //imshow( "Contours", img );
    //cv::waitKey(0);
	
	//resImg=img;
	//imwrite(filename, img);

	return img;
}

printUsageSyntax(char *prgname) {
	fprintf(stderr,
	        "IDS exposure control.\n"
	        "Usage: %s <INPUT> <OUTPUT> [options...]\n"
	        "	-h, --help      display help message\n"
	        "	-f, --file      name of input image file.\n"
	        "	-p, --output    name of image file to be saved.\n"
	        "	-d, --display   display measured result.\n"
	        "	-c, --focus     analysis focus.\n"
	        "	-o, --center    analysis center.\n"
	        "	-a, --angle     analysis angle.\n"
	        "	-v, --verbose   turn on verbose.\n"
	        , prgname);

}


int main(int argc, char *argv[]) {
	int  opt,loops=1,i;
	int  verbose=0;
	int  focus=0,center=0,angle=0;
	int  display=0;
	double  etime;

	char  *file=NULL,*output=NULL;
	string  string;


	/** Check the total number of the arguments */
	struct option longopts[] = {
		{"file" ,1, NULL, 'f'},
		{"output" ,1, NULL, 'p'},
		{"focus" ,0, NULL, 'c'},
		{"center" ,0, NULL, 'o'},
		{"angle" ,0, NULL, 'a'},
		{"display" ,0, NULL, 'd'},
		{"verbose",0, NULL, 'v'},
		{"help", 0, NULL, 'h'},
		{0,0,0,0}
	};

	while((opt = getopt_long(argc, argv, "f:p:coavhd",
	                         longopts, NULL))  != -1) {
		switch(opt) {
			case 'f':
				file = optarg;
				break;
			case 'p':
				output = optarg;
				break;
			case 'c':
				focus = 1;
				break;
			case 'o':
				center = 1;
				break;
			case 'a':
				angle = 1;
				break;
			case 'd':
				display = 1;
			case 'v':
				verbose = 1;
				break;
			case 'h':
				printUsageSyntax(argv[0]);
				exit(EXIT_FAILURE);
				break;
			case '?':
				printUsageSyntax(argv[0]);
				exit(EXIT_FAILURE);
				break;
		}
	}

	/** Print the usage syntax if there is no input */
	if (argc < 2 ) {
		printUsageSyntax(argv[0]);
		return EXIT_FAILURE;
	}

	if (file == NULL) {
		fprintf(stderr, "Warning: (%s:%s:%d) there is no file name specified."
		        "\n", __FILE__, __func__, __LINE__);
		return EXIT_FAILURE;
	}
	Mat image;
	Mat resImg;
 	image = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
	
	if (focus==1) resImg=analysisFocus(image);
	if (angle==1) resImg=analysisAngle(image);
	if (center==1) resImg=analysisCenter(image);
	
	string="output-";
	string.append(file); 
   	imwrite(string, resImg);
   
    //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    //cvShowImage( "Display window", image );                   // Show our image inside it.
    
	if (display == 1){
		imshow( "Display window", resImg );                   // Show our image inside it.
 		cvWaitKey(0);
	}
	 		cvWaitKey(0);

	
	
	return EXIT_SUCCESS;
}
