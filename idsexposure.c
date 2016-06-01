#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <getopt.h>
#include <getopt.h>
#include <wchar.h>
#include <wctype.h>

#include <ueye.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;


int analysisCenter(char* pMem, char* filename){
	RNG rng(12345);

	/* Establish an array for storing the image data */
    cv::Mat imageData(1024,1280, CV_8UC1);

    for( int i = 0; i < 1280; i++ ){
    	for( int j = 0; j < 1024; j++ ){
    		imageData.at<uchar>(j,i)=pMem[j*1280+i];
    	}
    }


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

    cv::imwrite(filename, img);
    return 0;

}
int analysisAngle(char* pMem, char* filename){
	RNG rng(12345);

	/* Establish an array for storing the image data */
    cv::Mat imageData(1024,1280, CV_8UC1);
    for( int i = 0; i < 1280; i++ ){
    	for( int j = 0; j < 1024; j++ ){
    		imageData.at<uchar>(j,i)=pMem[j*1280+i];
    	}
    }

    /* Convert to the grey scale image to color image for displaying */
    cv::Mat img;
    cv::cvtColor(imageData, img, CV_GRAY2BGR);

    /* Convert to binary */

    cv::Mat thres;
    cv::threshold(imageData, thres, 0.5*255, 255, cv::THRESH_BINARY);

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

    cv::imwrite(filename, img);
    return 0;

}

int analysisFocus(char* pMem, char* filename){
	RNG rng(12345);


	/* Establish an array for storing the image data */
    cv::Mat imageData(1024,1280, CV_8UC1);
    for( int i = 0; i < 1280; i++ ){
    	for( int j = 0; j < 1024; j++ ){
    		imageData.at<uchar>(j,i)=pMem[j*1280+i];
    	}
    }

    /* Convert to the grey scale image to color image for displaying */
    cv::Mat img;
    cv::cvtColor(imageData, img, CV_GRAY2BGR);

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

    Point diff = minEllipse[0].center - minEllipse[1].center;
    float res= cv::sqrt(diff.x*diff.x + diff.y*diff.y);
	printf("dist %f\n",res);



    /// Show in a window
    //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    //imshow( "Contours", img );
    //cv::waitKey(0);

    cv::imwrite(filename, img);

	return 0;
}

/* Print out the proper program usage syntax */
static void
printUsageSyntax(char *prgname) {
   fprintf(stderr,
	   "IDS exposure control.\n"
	   "Usage: %s <INPUT> <OUTPUT> [options...]\n"
		"	-h, --help      display help message\n"
		"	-f, --file      name of image file to be saved.\n"
		"	-c, --focus     analysis focus.\n"
		"	-d, --device    selecting device ID.\n"
		"	-e, --exptime   exposure time in the unit of ms.\n"
		"	-i, --interval  interval of seconds between exposures.\n"
		"	-l, --loops     number of exposure loops.\n"
		"	-v, --verbose   turn on verbose.\n"
		, prgname);

}


int main(int argc, char *argv[]) {
	int  opt,loops=1,i;
	int  verbose=0;
	int  device=1;
	int  interval=0;
	int  focus=0;
	int  nRet;
	double  etime;

    char  *file=NULL;
    char  string[256];

    HIDS hCam = 2;

	/** Check the total number of the arguments */
	struct option longopts[] = {
         {"loops" ,0, NULL, 'l'},
	     {"file" ,1, NULL, 'f'},
	     {"focus" ,0, NULL, 'c'},
	     {"device" ,1, NULL, 'd'},
	     {"exptime" ,1, NULL, 'e'},
	     {"invterval" ,1, NULL, 'i'},
		 {"verbose",0, NULL, 'v'},
		 {"help", 0, NULL, 'h'},
		 {0,0,0,0}};

	while((opt = getopt_long(argc, argv, "cd:i:f:e:l:vh",
	   longopts, NULL))  != -1){
	      switch(opt) {
	         case 'l':
	               loops = atoi(optarg);
	               break;
	         case 'i':
	               interval = atoi(optarg);
	               break;
	         case 'd':
	               device = atoi(optarg);
	               break;
	         case 'f':
	               file = optarg;
	               break;
	         case 'c':
	               focus = 1;
	               break;
	         case 'e':
	               etime = strtold(optarg,NULL);
	               break;
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

	if (file == NULL){
		fprintf(stderr, "Warning: (%s:%s:%d) there is no file name specified, use \"exposureXX.png\"."
		"\n", __FILE__, __func__, __LINE__);
		file="exposure";
	}

	if (loops >= 1){
		if (verbose) printf("take %i exposures.\n",loops);
	}

	if (device > 3 || device < 1){
		fprintf(stderr, "Error: (%s:%s:%d) Device ID should not be less than "
		"3 .\n", __FILE__, __func__, __LINE__);
		is_ExitCamera(hCam);
		return EXIT_FAILURE;
	}


	hCam=device;

	nRet = is_InitCamera (&hCam, NULL);
	if (verbose) printf("Status Init %d\n",nRet);

	//Pixel-Clock Setting, the range of this camera is 7-35 MHz
	unsigned int nPixelClockDefault=35;

	nRet = is_PixelClock(hCam, IS_PIXELCLOCK_CMD_SET,
	                        (void*)&nPixelClockDefault,
	                        sizeof(nPixelClockDefault));

	if (verbose) printf("Status is_PixelClock = %i MHz\n",nPixelClockDefault);


	/* Setting the exposure time */
	unsigned int nCaps = 0;
	nRet = is_Exposure(hCam, IS_EXPOSURE_CMD_GET_CAPS, (void*)&nCaps, sizeof(nCaps));
	if (verbose) printf("Status exposure function %d Support Mode=%i\n",nRet,nCaps);

	/* Work on exposure time range*/
	double expinfo[3];
	nRet = is_Exposure(hCam, IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE,
			(void*)expinfo,sizeof(expinfo));
	if (verbose) printf("Getting the exposure time range. \n"
			"Status = %d, Max= %f Min=%f Inc=%f\n",nRet,expinfo[0],expinfo[1],expinfo[2]);

	if (etime < expinfo[0] || etime > expinfo[1]){
		fprintf(stderr, "Error: (%s:%s:%d) Exposure time is out of "
		"range (%8.3f - %8.3f).\n", __FILE__, __func__, __LINE__,expinfo[0],expinfo[1]);
		is_ExitCamera(hCam);
		return EXIT_FAILURE;
	}

	nRet = is_Exposure(hCam, IS_EXPOSURE_CMD_SET_EXPOSURE,
			(void*)&etime,sizeof(etime));
	if (verbose) printf("Setting exposure time. \n"
				"Status = %d, Value= %f \n",nRet,etime);


	int colorMode = IS_CM_MONO8;
	nRet = is_SetColorMode(hCam,colorMode);
	if (verbose) printf("Status SetColorMode %d\n",nRet);

	unsigned int formatID = 25;
	nRet = is_ImageFormat(hCam, IMGFRMT_CMD_SET_FORMAT, &formatID, 4);
	if (verbose) printf("Status ImageFormat %d\n",nRet);

	char* pMem = NULL;
	int memID = 0;
	nRet = is_AllocImageMem(hCam, 1280, 1024, 8, &pMem, &memID);
	if (verbose) printf("Status AllocImage %d\n",nRet);

	nRet = is_SetImageMem(hCam, pMem, memID);
	if (verbose) printf("Status SetImageMem %d\n",nRet);

	int displayMode = IS_SET_DM_DIB;
	nRet = is_SetDisplayMode (hCam, displayMode);
	if (verbose) printf("Status displayMode %d\n",nRet);

	i=1;
	while(loops) {
		nRet = is_FreezeVideo(hCam, IS_WAIT);
		if (verbose) printf("taking image with function is_FreezeVideo %d\n",nRet);

		sprintf(string,"%s%04i%s",file,i,".png");
		if (verbose) fprintf(stdout,"image save as: %s\n",string);

		wchar_t NameBuffer[256];
		mbstowcs(NameBuffer, string, 256);



		//EXAM the pixel data
		int nPitch;
		is_GetImageMemPitch (hCam, &nPitch);

		//printf("value=%i\n",nPitch);

//		int i,j;
//		for (i=0;i < 1280; i++) {
//			for (j=0;j<1024;j++){
//				printf("%i  ",pMem[i+j*nPitch]);
//			}
//			sleep(1);
//			printf("\n");
//
//		}
		//printf("value=%i\n",pMem[277849]);
		//printf("\n");

		//
		//nRet = is_ImageFile(hCam, IS_IMAGE_FILE_CMD_SAVE, (void*) &ImageFileParams, sizeof(ImageFileParams));
		//if (verbose) printf("Status is_ImageFile %d\n",nRet);

		if (interval > 0){
			sleep(interval);
			if (verbose) printf("Wait %i seconds before next exposure.\n",interval);
		}

		/* Passing the pointer of the image memory to analysis function */
		if (device == 1){
			analysisCenter(pMem,string);
		}else if (device == 2 && focus == 1){
			analysisFocus(pMem,string);
		} else if (device == 3) {
			analysisAngle(pMem,string);
		} else {
			IMAGE_FILE_PARAMS ImageFileParams;
			ImageFileParams.pwchFileName = NameBuffer;
			ImageFileParams.pnImageID = NULL;
			ImageFileParams.ppcImageMem = NULL;
			ImageFileParams.nQuality = 0;
			ImageFileParams.nFileType = IS_IMG_PNG;
			nRet = is_ImageFile(hCam, IS_IMAGE_FILE_CMD_SAVE, (void*) &ImageFileParams, sizeof(ImageFileParams));
			if (verbose) printf("Status is_ImageFile %d\n",nRet);
		}

		loops--;
		i++;
	}

	nRet = is_FreeImageMem(hCam, pMem, memID);
	if (verbose) printf("Status FreeImage %d\n",nRet);

	is_ExitCamera(hCam);
  
	return EXIT_SUCCESS;
}
