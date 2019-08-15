//============================================================================
// Name        : Aia2.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

#include "Aia2.h"

/**
 * Calculates the contour line of all objects in an image.
 * @param img the input image
 * @param objList vector of contours, each represented by a two-channel matrix
 * @param thresh threshold used to binarize the image
 * @param k number of applications of the erosion operator
 */
void Aia2::getContourLine(const Mat& img, vector<Mat>& objList, int thresh, int k){
    // TODO !!!
    //showImage(img, "original", -1);

    // Threshold input image to get a binary image where leaves are white and background is black.
    cv::Mat img_thresholded;
    cv::threshold(img, img_thresholded, thresh, 255,cv::ThresholdTypes::THRESH_BINARY_INV);
    // showImage(img_thresholded, "thresholded", -1);

    // Erode thresholded image to delete small objects and connections between leafs.
    cv::Mat img_eroded;
    cv::erode(img_thresholded, img_eroded, cv::Mat::ones(3, 3, CV_8UC1), cv::Point(-1, -1), k);
    // showImage(img_eroded, "eroded", -1);

    // Extract contours from eroded image.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img_eroded, contours,
                     cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_NONE);

    /*
    cv::Mat img_contours = cv::Mat::zeros(img.size(), img.type());
    for(int i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(255, 255, 255);
        cv::drawContours(img_contours, contours, i, color, 2, 8, hierarchy, 0, Point());
    }
    showImage(img_contours, "contours", -1);
    */

    // Convert contours into output format.

    for(vector<cv::Point> contour : contours) {
        cv::Mat contour_mat = cv::Mat(contour).clone();
        objList.push_back(contour_mat);
    }
}

/**
 * Calculates the unnormalized fourier descriptor from a list of points.
 * @param contour 1xN 2-channel matrix,    cv::Mat img; containing N points (x in first, y in second channel)
 * @return fourier descriptor (not normalized)
 */
Mat Aia2::makeFD(const Mat& contour){
    // TODO !!!
    // Convert contour points from integers to floating point numbers.
    cv::Mat contour_float;
    contour.convertTo(contour_float, CV_32FC2);

    // Compute unnormalized fourier descriptor via discrete fourier transform.
    cv::Mat unnormalized_fourier_descriptor;
    cv::dft(contour_float, unnormalized_fourier_descriptor);

    return unnormalized_fourier_descriptor;
}

/**
 * Normalizes a given fourier descriptor.
 * @param fd the given fourier descriptor
 * @param n number of used frequencies (should be even)
 * @return the normalized fourier descriptor
 */
Mat Aia2::normFD(const Mat& fd, int n){
    // TODO !!!
    plotFD(fd, "fd not normalized", 0);

    // Make fourier descriptor invariant to translation.
    // TODO !!!
    cv::Mat translation = fd.clone();
    translation.at<Vec2f>(0, 0) = Vec2f(0, 0);
    plotFD(translation, "fd translation invariant", 0);

    // Make fourier descriptor invariant to scale.
    // TODO !!!
    cv::Mat translation_scale = translation.clone();
    if(cv::norm(translation.at<Vec2f>(0, 1)) != 0) {
        double scale_factor = cv::norm(translation.at<Vec2f>(0, 1));
        translation_scale = translation / scale_factor;
    } else {
        std::cout << "Warning: F(1) = 0" << std::endl;
        for(int i = 2; i < translation.size().height; i++) {
            if(cv::norm(translation.at<Vec2f>(0, i)) != 0) {
                double scale_factor = cv::norm(translation.at<Vec2f>(0, i));
                translation_scale = translation / scale_factor;
                break;
            }
        }
        translation_scale.at<Vec2f>(0, 1) = Vec2f(1, 0);
    }
    plotFD(translation_scale, "fd translation and scale invariant", 0);

    // Make fourier descriptor invariant to rotation.
    // TODO !!!
    cv::Mat translation_scale_rotation = translation_scale.clone();
    std::vector<cv::Mat> channels(2);
    cv::split(translation_scale, channels);
    cv::magnitude(channels[0], channels[1], translation_scale_rotation);
    plotFD(translation_scale_rotation, "fd translation, scale, and rotation invariant", 0);

    // Reduce fourier descriptor sensitivity for details.
    // TODO !!!
    cv::Mat translation_scale_rotation_details;
    Mat tmp1 = translation_scale_rotation.rowRange(0, n / 2);
    Mat tmp2 = translation_scale_rotation.rowRange(translation_scale_rotation.size().height - n/2,
                                                   translation_scale_rotation.size().height);
    cv::vconcat(tmp1, tmp2, translation_scale_rotation_details);
    plotFD(translation_scale_rotation_details, "fd translation, scale, and rotation invariant, smaller sensitivity", 0);

    return translation_scale_rotation_details;
}

/**
 * Plots fourier descriptor.
 * @param fd the fourier descriptor to be displayed
 * @param win the window name
 * @param dur number of ms or until key is pressed
 */
void Aia2::plotFD(const Mat& fd, string win, double dur){
    // TODO !!!
    cv::Mat complex_fd;
    if(fd.channels() == 1) {
        cv::Mat zero_mat = cv::Mat::zeros(fd.size(), CV_32FC1);
        std::vector<cv::Mat> channels = std::vector<cv::Mat>{fd.clone(), zero_mat};
        cv::merge(channels, complex_fd);
    } else {
        complex_fd = fd.clone();
    }

    if(cv::abs(cv::norm(complex_fd.at<Vec2f>(0, 1)) - 1) < 0.001) {
        complex_fd = complex_fd * 1000;
    }

    cv::Mat contour;

    cv::dft(complex_fd, contour, cv::DftFlags::DFT_SCALE | cv::DftFlags::DFT_INVERSE);

    int left_border = 400;
    int top_border = 400;
    cv::Mat img = cv::Mat::zeros(394 + top_border, 800 + left_border, CV_8UC1);
    cv::line(img, cv::Point(left_border, top_border), cv::Point(800 + left_border, top_border), cv::Scalar(255));
    cv::line(img, cv::Point(left_border, top_border), cv::Point(left_border, 394 + top_border), cv::Scalar(255));

    for(int i = 0; i < contour.size().height; i++) {
        cv::Point2f contour_point = contour.at<cv::Point2f>(0, i) + cv::Point2f(left_border, top_border);
        img.at<uchar>(int(contour_point.y), int(contour_point.x)) = 255;
    }

    cv::imwrite(win + ".png", img);
    showImage(img, win, dur);
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function loads input image, calls processing functions, and saves result
// in particular extracts FDs and compares them to templates
/*
img			path to query image
template1	path to template image of class 1
template2	path to template image of class 2
*/
void Aia2::run(string img, string template1, string template2){

	// process image data base
	// load image as gray-scale, paths in argv[2] and argv[3]
	Mat exC1 = imread( template1, 0);
	Mat exC2  = imread( template2, 0);
	if ( (!exC1.data) || (!exC2.data) ){
	    cout << "ERROR: Cannot load class examples in\n" << template1 << "\n" << template2 << endl;
	    cout << "Press enter to continue..." << endl;
	    cin.get();
	    exit(-1);
	}

	// parameters
	// these two will be adjusted below for each image individually
	int binThreshold;				      // threshold for image binarization
	int numOfErosions;			   	// number of applications of the erosion operator
	// these two values work fine, but it might be interesting for you to play around with them
	int steps = 32;					   // number of dimensions of the FD
	double detThreshold = 0.035;		// threshold for detection

	// get contour line from images
	vector<Mat> contourLines1;
	vector<Mat> contourLines2;
	// TODO !! --> Adjust threshold and number of erosion operations
	binThreshold = 127;
	numOfErosions = 2;

	getContourLine(exC1, contourLines1, binThreshold, numOfErosions);
	int mSize = 0, mc1 = 0, mc2 = 0, i = 0;
	for(vector<Mat>::iterator c = contourLines1.begin(); c != contourLines1.end(); c++,i++){
		if (mSize<c->rows){
			mSize = c->rows;
			mc1 = i;
		}
	}
	getContourLine(exC2, contourLines2, binThreshold, numOfErosions);
	for(vector<Mat>::iterator c = contourLines2.begin(); c != contourLines2.end(); c++, i++){
		if (mSize<c->rows){
			mSize = c->rows;
			mc2 = i;
		}
	}
	// calculate fourier descriptor
	Mat fd1 = makeFD(contourLines1.at(mc1));
	Mat fd2 = makeFD(contourLines2.at(mc2));

	// normalize  fourier descriptor
	Mat fd1_norm = normFD(fd1, steps);
	Mat fd2_norm = normFD(fd2, steps);

	// process query image
	// load image as gray-scale, path in argv[1]
	Mat query = imread( img, 0);
	if (!query.data){
	    cout << "ERROR: Cannot load query image in\n" << img << endl;
	    cout << "Press enter to continue..." << endl;
	    cin.get();
	    exit(-1);
	}
	
	// get contour lines from image
	vector<Mat> contourLines;
	// TODO !!! --> Adjust threshold and number of erosion operations
	binThreshold = 127;
	numOfErosions = 4;

	getContourLine(query, contourLines, binThreshold, numOfErosions);
	
	cout << "Found " << contourLines.size() << " object candidates" << endl;

	// just to visualize classification result
	Mat result(query.rows, query.cols, CV_8UC3);
	vector<Mat> tmp;
	tmp.push_back(query);
	tmp.push_back(query);
	tmp.push_back(query);
	merge(tmp, result);

	// loop through all contours found
	i = 1;
	for(vector<Mat>::iterator c = contourLines.begin(); c != contourLines.end(); c++, i++){

	    cout << "Checking object candidate no " << i << " :\t";
	  
		// color current object in yellow
	  	Vec3b col(0,255,255);
	    for(int p=0; p < c->rows; p++){
			result.at<Vec3b>(c->at<Vec2i>(p)[1], c->at<Vec2i>(p)[0]) = col;
	    }
	    showImage(result, "result", 0);
	    
	    // if fourier descriptor has too few components (too small contour), then skip it (and color it in blue)
	    if (c->rows < steps){
			cout << "Too less boundary points (" << c->rows << " instead of " << steps << ")" << endl;
			col = Vec3b(255,0,0);
	    }else{
			// calculate fourier descriptor
			Mat fd = makeFD(*c);
			// normalize fourier descriptor
			Mat fd_norm = normFD(fd, steps);
			// compare fourier descriptors
			double err1 = norm(fd_norm, fd1_norm)/steps;
			double err2 = norm(fd_norm, fd2_norm)/steps;
			// if similarity is too small, then reject (and color in cyan)
			if (min(err1, err2) > detThreshold){
				cout << "No class instance ( " << min(err1, err2) << " )" << endl;
				col = Vec3b(255,255,0);
			}else{
				// otherwise: assign color according to class
				if (err1 > err2){
					col = Vec3b(0,0,255);
					cout << "Class 2 ( " << err2 << " )" << endl;
				}else{
					col = Vec3b(0,255,0);
					cout << "Class 1 ( " << err1 << " )" << endl;
				}
			}
		}
		// draw detection result
	    for(int p=0; p < c->rows; p++){
			result.at<Vec3b>(c->at<Vec2i>(p)[1], c->at<Vec2i>(p)[0]) = col;
	    }
	    // for intermediate results, use the following line
	    showImage(result, "result", 0);

	}
	// save result
	imwrite("result.png", result);
	// show final result
	showImage(result, "result", 0);
}

/**
 * Shows an image.
 * @param img the image to be displayed
 * @param win the window name
 * @param dur number of ms or until key is pressed
 */
void Aia2::showImage(const Mat& img, string win, double dur){
  
    // use copy for normalization
    cv::Mat tempDisplay = img.clone();
    if (img.channels() == 1) cv::normalize(img, tempDisplay, 0, 255, CV_MINMAX);
    // create window and display image
    cv::namedWindow(win.c_str(), cv::WINDOW_AUTOSIZE);
    cv::imshow(win.c_str(), tempDisplay);
    // wait
    if (dur >= 0) {
        cv::waitKey(dur);
    } else {
        cv::waitKey(-1);
    }
    
}

// function loads input image and calls processing function
// output is tested on "correctness" 
void Aia2::test(void){
	
	test_getContourLine();
	test_makeFD();
	test_normFD();
	
}

void Aia2::test_getContourLine(void){

   // creates a black square on a white background
	Mat img(100, 100, CV_8UC1, Scalar(255));
	Mat roi(img, Rect(40,40,20,20));
	roi.setTo(0);
   // test correctness for #erosions=1
   // creates correct outline
	Mat cline(68,1,CV_32SC2);
	int k=0;
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41,i);
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i,58);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i,41);
   // computes outline
	vector<Mat> objList;
	getContourLine(img, objList, 128, 1);
   // compares correct and computed outlines
   // there should be only one object
	if ( objList.size() > 1 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cout << "--> found too many contours (#erosions=1)" << endl;
		cin.get();
	}
	if ( max(objList.at(0).rows, objList.at(0).cols) != 68 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cout << "--> contour has wrong number of pixels (#erosions=1)" << endl;
		cin.get();
	}
	if ( sum(cline != objList.at(0)).val[0] != 0 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cout << "--> computed wrong contour (#erosions=1)" << endl;
		cin.get();
	}
   // test correctness for #erosions=3
   // re-init
   objList.resize(0);
   // creates correct outline
   cline = Mat(52,1,CV_32SC2);
   k=0;
	for(int i=43; i<56; i++) cline.at<Vec2i>(k++) = Vec2i(43,i);
	for(int i=43; i<56; i++) cline.at<Vec2i>(k++) = Vec2i(i,56);
	for(int i=56; i>43; i--) cline.at<Vec2i>(k++) = Vec2i(56, i);
	for(int i=56; i>43; i--) cline.at<Vec2i>(k++) = Vec2i(i,43);
   // computes outline
	getContourLine(img, objList, 128, 3);
   // compares correct and computed outlines
	if ( objList.size() > 1 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cout << "--> found too many contours (#erosions=3)" << endl;
		cin.get();
	}
	if ( max(objList.at(0).rows, objList.at(0).cols) != 52 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cout << "--> contour has wrong number of pixels (#erosions=3)" << endl;
		cin.get();
	}
	if ( sum(cline != objList.at(0)).val[0] != 0 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cout << "--> computed wrong contour (#erosions=3)" << endl;
		cin.get();
	}
}

void Aia2::test_makeFD(void){

   // create example outline
	Mat cline(68,1,CV_32SC2);
	int k=0;
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41,i);
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i,58);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i,41);
	
   // test for correctness
	Mat fd = makeFD(cline);
	if (fd.rows != cline.rows){
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "--> The number of frequencies does not match the number of contour points!" << endl;
		cin.get();
	}
	if (fd.channels() != 2){
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "--> The fourier descriptor is supposed to be a two-channel, 1D matrix!" << endl;
		cin.get();
	}
   if (fd.type() != CV_32FC2){
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "--> Frequency amplitudes are not computed with floating point precision!" << endl;
		cin.get();
	}
}

void Aia2::test_normFD(void){

	double eps = pow(10,-3);

   // create example outline
	Mat cline(68,1,CV_32SC2);
	int k=0;
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41,i);
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i,58);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i,41);
	
	Mat fd = makeFD(cline);
	Mat nfd = normFD(fd, 32);
   // test for correctness
	if (nfd.channels() != 1){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "--> The normalized fourier descriptor is supposed to be a one-channel, 1D matrix" << endl;
		cin.get();
	}
   if (nfd.type() != CV_32FC1){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "--> The normalized fourier descriptor is supposed to be in floating point precision" << endl;
		cin.get();
	}
	if (abs(nfd.at<float>(0)) > eps){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "--> The F(0)-component of the normalized fourier descriptor F is supposed to be 0" << endl;
		cin.get();
	}
	if ((abs(nfd.at<float>(1)-1.) > eps) && (abs(nfd.at<float>(nfd.rows-1)-1.) > eps)){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "--> The F(1)-component of the normalized fourier descriptor F is supposed to be 1" << endl;
		cout << "--> But what if the unnormalized F(1)=0?" << endl;
		cin.get();
	}
	if (nfd.rows != 32){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "--> The number of components does not match the specified number of components" << endl;
		cin.get();
	}
}
