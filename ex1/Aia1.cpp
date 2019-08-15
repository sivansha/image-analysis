//============================================================================
// Name        : Aia1.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

/*
 * Automatic Image Analysis ex1
 * ============================
 * Group U
 * =======
 *
 * we have decided to perform DFT (Discrete Fourier Transform) on the iconic Lena image.
 * seems as nice idea, also Fourier Transform was big part of the lecture, and it will be nice to try it out in code.
 *
 * the coding done based on:
 * https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
 */

#include "Aia1.h"

// function performs some kind of (simple) image processing
/*
img	input image
return	output image
*/
Mat Aia1::doSomethingThatMyTutorIsGonnaLike(Mat const& img){

	// convert image to grayscale
	Mat img_gray;
	cvtColor(img, img_gray, CV_BGR2GRAY);

	// DFT is most efficient on images of certain sizes (sizes that are multiplication of 2, 3, 5)
	// opencv provides a function for optimizing run time by adjusting the image size.
	// as can be seen by the output, since the image dimensions are already a multiplication of 2
	// (512 = 2^9) the dimensions will not be change.
	cout << "dimensions of original image: " << img_gray.rows << " " << img_gray.cols << endl;
  
    int img_gray_rows_adjusted = getOptimalDFTSize( img_gray.rows );
    int img_gray_cols_adjusted = getOptimalDFTSize( img_gray.cols );

    cout << "dimensions after adjustment: " << img_gray_rows_adjusted << " " << img_gray_cols_adjusted << endl;

    // create an image capable of holding the results of the fourier transform:
    // matrix of type float, with additional channel for the complex part
    Mat img_float[] = {Mat_<float>(img_gray), Mat::zeros(img_gray.size(), CV_32F)};
    Mat img_complex;
    merge(img_float, 2, img_complex);

    // perform dft
    dft(img_complex, img_complex);

    // get results as opencv presentable image, including transformation to logaritmic scale and normalizing.
    split(img_complex, img_float);
    magnitude(img_float[0], img_float[1], img_float[0]);

    Mat img_dft_result = img_float[0];

    img_dft_result += Scalar::all(1);
    log(img_dft_result, img_dft_result);

   normalize(img_dft_result, img_dft_result, 0, 1, CV_MINMAX);

    // shift the image in order to get the origin at the center
    // (getting similar image to the one the lecturer presented in class)
    int col_halved = img_dft_result.cols/2;
    int row_halved = img_dft_result.rows/2;

    // get each quarter and rearrange them accordingly
    Mat q0(img_dft_result, Rect(0, 0, col_halved, row_halved));
    Mat q1(img_dft_result, Rect(col_halved, 0, col_halved, row_halved));
    Mat q2(img_dft_result, Rect(0, row_halved, col_halved, row_halved));
    Mat q3(img_dft_result, Rect(col_halved, col_halved, row_halved, row_halved));

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // finally, to cop with cv imwrite function:
    img_dft_result.convertTo(img_dft_result, CV_8UC3, 255.0);

	return img_dft_result;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function loads the input image, calls processing/test functions, and saves result
/*
fname	path to input image
*/
void Aia1::run(string fname){

	// window names
	string win1 = string ("Original image");
	string win2 = string ("Result");
  
	// some images
	Mat inputImage, outputImage;
  
	// load image
	cout << "load image" << endl;
	inputImage = imread( fname );
	cout << "done" << endl;
	
	// check if image can be loaded
	if (!inputImage.data){
	    cerr << "ERROR: Cannot read file " << fname << endl;
	    cout << "Press Enter to continue..." << endl;
	    cin.get();
	    exit(-1);
	}

	// show input image
	namedWindow( win1.c_str() );
	imshow( win1.c_str(), inputImage );
	
	// do something (reasonable!)
	outputImage = doSomethingThatMyTutorIsGonnaLike( inputImage );
	
	// show result
	namedWindow( win2.c_str() );
	imshow( win2.c_str(), outputImage );
	
	// save result
	imwrite("result.jpg", outputImage);
	
	// wait a bit
	waitKey(0);

}

// function loads input image and calls processing function
// output is tested on "correctness" 
/*
fname	path to input image
*/
void Aia1::test(string fname){

	// some image variables
	Mat inputImage, outputImage;
  
	// load image
	inputImage = imread( fname );

	// check if image can be loaded
	if (!inputImage.data){
	    cout << "ERROR: Cannot read file " << fname << endl;
	    cout << "Press Enter to continue..." << endl;
	    cin.get();
	    exit(-1);
	}

	// create output
	outputImage = doSomethingThatMyTutorIsGonnaLike( inputImage );
	// test output
	test_doSomethingThatMyTutorIsGonnaLike(inputImage, outputImage);
	
}

// function loads input image and calls processing function
// output is tested on "correctness" 
/*
inputImage	input image as used by doSomethingThatMyTutorIsGonnaLike()
outputImage	output image as created by doSomethingThatMyTutorIsGonnaLike()
*/
void Aia1::test_doSomethingThatMyTutorIsGonnaLike(Mat const& inputImage, Mat const& outputImage){

   Mat input = inputImage.clone();
	// ensure that input and output have equal number of channels
	if ( (input.channels() == 3) and (outputImage.channels() == 1) )
		cvtColor(input, input, CV_BGR2GRAY);

	// split (multi-channel) image into planes
	vector<Mat> inputPlanes, outputPlanes;
	split( input, inputPlanes );
	split( outputImage, outputPlanes );

	// number of planes (1=grayscale, 3=color)
	int numOfPlanes = inputPlanes.size();

	// calculate and compare image histograms for each plane
	Mat inputHist, outputHist;
	// number of bins
	int histSize = 100;
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	double sim = 0;
	for(int p = 0; p < numOfPlanes; p++){
		// calculate histogram
		calcHist( &inputPlanes[p], 1, 0, Mat(), inputHist, 1, &histSize, &histRange, uniform, accumulate );
		calcHist( &outputPlanes[p], 1, 0, Mat(), outputHist, 1, &histSize, &histRange, uniform, accumulate );
		// normalize
		inputHist = inputHist / sum(inputHist).val[0];
		outputHist = outputHist / sum(outputHist).val[0];
		// similarity as histogram intersection
		sim += compareHist(inputHist, outputHist, CV_COMP_INTERSECT);
	}
	sim /= numOfPlanes;

	// check whether images are to similar after transformation
	if (sim >= 0.8)
			cout << "The input and output image seem to be rather similar (similarity = " << sim << " ). Are you sure your tutor is gonna like your work?" << endl;

}
