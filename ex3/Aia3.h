//============================================================================
// Name        : Aia3.h
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : header file for second AIA assignment
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Aia3{

	public:
		// constructor
		Aia3(void){};
		// destructor
		~Aia3(void){};
		
		// processing routine
		// --> some parameters have to be set in this function
		void run(string, string);
		// testing routine
		void test(string, float, float);

	private:
		// --> these functions need to be edited
		void makeFFTObjectMask(vector<Mat>const & templ, double scale, double angle, Mat& fftMask);
		vector<Mat> makeObjectTemplate(Mat const& templateImage, double sigma, double templateThresh);
		vector< vector<Mat> > generalHough(Mat const& gradImage, vector<Mat> const& templ, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange);
		void plotHough(vector< vector<Mat> > const& houghSpace);
		// given functions
		void process(Mat const&, Mat const&, Mat const&);
		Mat makeTestImage(Mat const& temp, double angle, double scale, double* scaleRange);
		Mat rotateAndScale(Mat const& temp, double angle, double scale);
		Mat calcDirectionalGrad(Mat const& image, double sigma);
		void showImage(Mat const& img, string win, double dur);
		void circShift(Mat const& in, Mat& out, int dx, int dy);
		void findHoughMaxima(vector< vector<Mat> > const& houghSpace, double objThresh, vector<Scalar>& objList);
		void plotHoughDetectionResult(Mat const& testImage, vector<Mat> const& templ, vector<Scalar> const& objList, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange);
		
};
