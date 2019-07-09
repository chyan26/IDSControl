
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <getopt.h>
#include <wchar.h>
#include <wctype.h>
#include <iostream>
#include <string>
#include <iterator>
#include <algorithm>
#include <winsock.h>

#include "mpfit.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace cv;
using namespace std;

#define CENTER_SIZE_X 128
#define CENTER_SIZE_Y 128

struct vars_struct {
   double *flux;     //DATA
   double *ferr;     //ESTIMATE OF ERROR
};


static int
gaussfunc2d(int m, int n, double *p, double *dy, double **dvec, void *vars)
{

   int i,j;    //COUNTERS, SIZE OF IMAGE
   int n1;     //SIZE OF IMAGE, ASSUMING A SQUARE
   struct vars_struct *v = (struct vars_struct *) vars;  //STRUCTURE TO PASS PRIVATE DATA
   double *flux;  //DATA VALUES
   double xc, yc;        //THE VALUE OF THE CENTRE POINTS OF THE IMAGE

   //SET THE LOCAL VARIABLES TO THE STRUCTURE.

   flux=v->flux;

   //ASSUMING A SQUARE IMAGE, GET THE DIMENSIONS OF EACH SIDE

   n1=(int)sqrt(m);


   //CYCLE THROUGH THE VALUES. THE DATA/RESIDUALS ARE ONE D,
   //MAP THE COORDINATES TO THAT ASSUMING A SQUARE

   for (i=0;i<n1;i++)
   {
      for (j=0;j<n1;j++)
      {
	 //CENTER VALUES
	 xc=i-p[0];
	 yc=j-p[1];

	 //EQUATION ASSUMING INDEPENDENT FWHM IN X AND Y DIRECTIONS

	 dy[i*n1+j] = ((flux[i*n1+j] - p[4]*exp(-0.5*(xc*xc/(p[2]*p[2]*0.180337)+yc*yc/(p[3]*p[3]*0.180337)))-p[5]));
      }
   }
   return 0;
}




// Function for calculating median
double findMedian(double a[], int n)
{
    // First we sort the array
    sort(a, a+n);

    // check for even case
    if (n % 2 != 0)
       return (double)a[n/2];

    return (double)(a[(n-1)/2] + a[n/2])/2.0;
}



/*
 * MPFIS method for centroid calculation on the image
 */
int *calculateCentroidMPFIT(u_char *image, int columns, int rows, float xest, float yest,
      float *xc, float *yc, int threshold) {

   struct vars_struct v; //PRIVATE STRUCTURE WITH DATA/FUNCTION INFORMATION
   mp_result result;     //STRUCTURE WITH RESULTS
   mp_par pars[6];	//VARIABLE THAT HOLDS INFORMATION ABOUT FIXING PARAMETERS - DATA TYPE IN MPFIT

   double median;

   double *ferr=NULL;

   // Array for median calculation

   double perror[6];	//ERRORS IN RETURNED PARAMETERS

   int i,j,k=0;
   //int npoints;

   /*
    *   First step, estimate the center of the point using Center of Mass
    */
   //float xest = 0;
   //float yest = 0;
   //calculateCentroid(image, columns, rows,&xest, &yest);
   //fprintf(stderr,"x=%f y=%f ",xest,yest);

   // /*
   //  *  Cut out the region near the point
   //  */
   int fpix[2];                       //FIRST PIXELS OF SUBREGION [X,Y]
   int lpix[2];                       //LAST PIXELS OF SUBREGION [X,Y]
   int subx, suby,np;
   double *subimage;
   //std::array<double,1280*1024> subimage = image
   fpix[0]=xest-CENTER_SIZE_X/4;
   fpix[1]=yest-CENTER_SIZE_Y/4;
   lpix[0]=xest+CENTER_SIZE_X/4-1;
   lpix[1]=yest+CENTER_SIZE_Y/4-1;

   // if (xest-CENTER_SIZE_X/4 < 0) fpix[0]=0;
   // if (yest-CENTER_SIZE_Y/4 < 0) fpix[1]=0;
   // if (xest-CENTER_SIZE_X/4 > 128) lpix[0]=128;
   // if (yest-CENTER_SIZE_Y/4 > 128) lpix[1]=128;


   // //GET THE DIMENSIONS OF THE SUBREGION
   subx=lpix[0]-fpix[0]+1;
   suby=lpix[1]-fpix[1]+1;

   // Copy the image region
   subimage = (double *)malloc(subx*suby*sizeof(double));
   ferr = (double *)malloc(subx*suby*sizeof(double));
   // //fprintf(stderr,"subx=%i %i \n",subx,suby);

   for (i=fpix[0];i<fpix[0]+subx;i++){
      for (j=fpix[1];j<fpix[1]+suby;j++){
	      subimage[k]=(double)image[j*columns+i];
          if (subimage[k] < (double)threshold){subimage[k] = 0;}
	      ferr[k]=1.0;
	      k++;
      }
   }
   //for (i=0;i<subx*suby;i++){
   //   subimage[i]=(double)image[i];
   //   ferr[i] = 1.0;
      //if (arr[i] < 0) {arr[i] =  0;}
   //}


   //npoints=columns*rows;
   np=subx*suby;
   median=findMedian(subimage,np);
   //fprintf(stderr," median=%f ",median);
   //if (median <= 0){median = 150;}
   // //ferr = malloc(npoints*sizeof(double));
   // //arr = malloc(npoints*sizeof(double));
   // for (i=0;i<npoints;i++){
   //    //ferr[i]=1.0;
   //    arr[i]=(double)image[i];
   // }
   //median=150;

   double p[] = {xest-fpix[0],yest-fpix[1],2.5,2.5, 200.0,median};

   memset(&result,0,sizeof(result));
   result.xerror = perror;
   memset(pars,0,sizeof(pars));
   // //fprintf(stderr,"init=%f %f\n",xest,yest);

   v.ferr = ferr;
   v.flux = subimage;

   //pars[1].fixed = 0;
   pars[2].fixed = 1;
   pars[3].fixed = 1;
   //pars[4].fixed = 1;
   //pars[5].fixed = 1;
   pars[5].fixed = 1;



   // // TODO: Error on the return value of this function need to be handled
   mpfit(gaussfunc2d, np, 6, p, pars, 0, (void *) &v, &result);


   free(subimage);
   free(ferr);

   if (fpix[1]+p[1] < 0){
      *yc=yest;
   } else {
      *yc=fpix[1]+p[1];
   }

   if (fpix[0]+p[0] < 0){
      *xc=xest;
   } else {
      *xc=fpix[0]+p[0];
   }
   return 0;

}

Mat analysisCenter(Mat imageData, int threshold){
	RNG rng(12345);
   int i = 0;
   float xc, yc;
   /* Convert to the grey scale image to color image for displaying */
   cv::Mat img;
   cv::cvtColor(imageData, img, COLOR_GRAY2BGR);

   double maxVal, minVal;
   float  thresVal;
   int    thresValint;

   if (threshold > 100) {
    	fprintf(stderr, "Warning: (%s:%s:%d) threshold should be 0 to 100."
        	"\n", __FILE__, __func__, __LINE__);
        exit(EXIT_FAILURE);
   }

   /* Calculate the maximum of the image */
   cv::minMaxLoc(imageData, &minVal, &maxVal, 0, 0);

   thresVal = maxVal*threshold/100.0;
   thresValint = (int) thresVal;
   //WriteFitsImage("test.fits", imageData.rows, imageData.cols, imageData.data);

   //std::cout << findMedian(image_ushort,imageData.total())<< std::endl;
   //std::cout << thresVal <<' '<< maxVal << std::endl;
   Mat canny_output;
   vector<vector<Point> > contours;
   vector<Vec4i> hierarchy;

   Canny( imageData, canny_output, thresValint/2, thresValint*2, 3 );
   findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
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
      drawContours( img, contours, 1, color, 2, 8, hierarchy, 0, Point() );

      circle( img, mc[0], 4, color, -1, 8, 0 );
      //std::cout << mc[0].x <<' '<< mc[0].y << std::endl;
   }

   calculateCentroidMPFIT(imageData.data, img.cols, img.rows,  mc[0].x, mc[0].y,&xc, &yc, thresValint);
   std::cout << xc << ' ' << yc << std::endl;
   Point2f center = Point2f( xc , yc );
   circle( img, center, 4, color, -1, 8, 0 );


    /// Show in a window
    //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	//imshow( "Contours", canny_output );
    //cv::waitKey(0);

    //cv::imwrite(filename, img);

    return img;

}
Mat analysisAngle(Mat imageData, int threshold){
	RNG rng(12345);

	/* Establish an array for storing the image data */
    //cv::Mat imageData(1024,1280, CV_8UC1);
    //for( int i = 0; i < 1280; i++ ){
    //	for( int j = 0; j < 1024; j++ ){
    //		imageData.at<uchar>(j,i)=pMem[j*1280+i];
    //	}
    //}
    double maxVal, minVal;
    float  thresVal;
    int    thresValint;

    if (threshold > 100) {
    	fprintf(stderr, "Warning: (%s:%s:%d) threshold should be 0 to 100."
        	"\n", __FILE__, __func__, __LINE__);
        exit(EXIT_FAILURE);
    }

    /* Convert to the grey scale image to color image for displaying */
    cv::Mat img;
    cv::cvtColor(imageData, img, COLOR_GRAY2BGR);

    /* Calculate the maximum of the image */
    cv::minMaxLoc(img, &minVal, &maxVal, 0, 0);

    thresVal = maxVal*threshold/100.0;
    thresValint = (int) thresVal;

    /* Convert to binary */
    cv::Mat thres;
    cv::threshold(imageData, thres, thresValint, 255, cv::THRESH_BINARY);

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


Mat analysisFocus(Mat imageData, int threshold){
	RNG rng(12345);
    int i;

    /* Convert to the grey scale image to color image for displaying */
    cv::Mat img;
    cv::cvtColor(imageData, img, COLOR_GRAY2BGR);
    //cvCvtColor(imageData, img, CV_GRAY2BGR);

    /* Convert to binary threshold image*/
    cv::Mat thres;
    cv::threshold(imageData, thres, threshold, 255, cv::THRESH_BINARY);

    /* Find contours based on theshold image*/
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(thres, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);


    /// Find the rotated rectangles and ellipses for each contour
    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> minEllipse( contours.size() );


    /* Going through every contour for ellipse detecting */
    for( i = 0; i < contours.size(); i++ )
       { minRect[i] = minAreaRect( Mat(contours[i]) );
         if( contours[i].size() > 10 )
           { minEllipse[i] = fitEllipse( Mat(contours[i]) ); }
       }

    /// Draw contours + rotated rects + ellipses
    Mat drawing = Mat::zeros( thres.size(), CV_8UC3);
    for( i = 0; i< contours.size(); i++ )
       {
         Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
         ellipse( img, minEllipse[i], color, 2, 8 );
       }

    //printf("total ellipses = %i\n",i);

    vector<Point2f> sec(4);

    for (i =0; i < contours.size(); i++ )
        {
            if (minEllipse[i].center.x != 0 && minEllipse[i].center.y != 0){
                //printf("x= %f y= %f %i %i \n",minEllipse[i].center.x,minEllipse[i].center.y,img.cols,img.rows);
                if (minEllipse[i].center.x < img.cols/2 && minEllipse[i].center.y < img.rows/2) sec[0]=minEllipse[i].center;
                if (minEllipse[i].center.x > img.cols/2 && minEllipse[i].center.y < img.rows/2) sec[1]=minEllipse[i].center;
                if (minEllipse[i].center.x > img.cols/2 && minEllipse[i].center.y > img.rows/2) sec[2]=minEllipse[i].center;
                if (minEllipse[i].center.x < img.cols/2 && minEllipse[i].center.y > img.rows/2) sec[3]=minEllipse[i].center;

            }    //sec1=minEllipse[i].center

        }

    Scalar color = Scalar( rng.uniform(2, 255), rng.uniform(2,255), rng.uniform(2,255) );
    line(img,sec[0], sec[1], color, 1, 8 );
    line(img,sec[1], sec[2], color, 1, 8 );
    line(img,sec[2], sec[3], color, 1, 8 );
    line(img,sec[3], sec[0], color, 1, 8 );

    Point diff1 = sec[0] - sec[1];
    Point diff2 = sec[1] - sec[2];
    Point diff3 = sec[2] - sec[3];
    Point diff4 = sec[3] - sec[0];

    float res1= cv::sqrt(diff1.x*diff1.x + diff1.y*diff1.y);
    float res2= cv::sqrt(diff2.x*diff2.x + diff2.y*diff2.y);
    float res3= cv::sqrt(diff3.x*diff3.x + diff3.y*diff3.y);
    float res4= cv::sqrt(diff4.x*diff4.x + diff4.y*diff4.y);

    printf("dist %f %f %f %f %f\n",res1,res2,res3,res4,(res1+res2+res3+res4)/4);


    /// Show in a window
    //namedWindow( "Contours", WINDOW_AUTOSIZE );
    //imshow( "Contours", img );
    //cv::waitKey(0);

	//resImg=img;
	//imwrite(filename, img);

	return img;
}

int printUsageSyntax(char *prgname) {
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
            "	-t, --threshold  setting the threshold.\n"
	        "	-v, --verbose   turn on verbose.\n"
	        , prgname);

}


int main(int argc, char *argv[]) {
	int  opt,loops=1,i;
	int  verbose=0;
	int  focus=0,center=0,angle=0;
	int  display=0;
        int  thres=0;

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
		{"threshold",1, NULL, 't'},
		{"verbose",0, NULL, 'v'},
                {"help", 0, NULL, 'h'},
		{0,0,0,0}
	};

	while((opt = getopt_long(argc, argv, "f:p:coavhdt:",
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
                                break;
			case 't':
				thres = atoi(optarg);
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

	if (file == NULL) {
		fprintf(stderr, "Warning: (%s:%s:%d) there is no file name specified."
		        "\n", __FILE__, __func__, __LINE__);
		return EXIT_FAILURE;
	}
	Mat image;
	Mat resImg;
 	image = imread(file, IMREAD_GRAYSCALE);

        if (thres == 0 ) thres = 70;
	if (focus==1) resImg=analysisFocus(image,thres);
	if (angle==1) resImg=analysisAngle(image,thres);
	if (center==1) resImg=analysisCenter(image,thres);

        //if (output == NULL){
            string="output-";
            string.append(file);
        //}
   	imwrite(string, resImg);

    //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    //cvShowImage( "Display window", image );                   // Show our image inside it.

	if (display == 1){
		imshow( "Display window", resImg );
		// Show our image inside it.
		cv::waitKey(0);
	}
	cv::waitKey(0);



	return EXIT_SUCCESS;
}
