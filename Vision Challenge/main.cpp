#include<opencv2/opencv.hpp>
#include<iostream>
#include<conio.h>     

using namespace cv;
using namespace std;


int main()
{
	cv::Mat diceOriginal;
	cv::Mat diceHSV;
	cv::Mat diceBlur;
	cv::Mat diceCanny;
	cv::Mat diceOutput;
	cv::Mat diceThreshold;
	cv::Mat diceThreshLow;
	cv::Mat diceThreshHigh;
	std::vector<cv::KeyPoint> keypoints;
	std::ostringstream str;
	std::ostringstream str1;
	
	//Load Original Image
	diceOriginal = cv::imread("dice6.png");
	
	//Image Pre-processing
	cv::cvtColor(diceOriginal, diceHSV, CV_BGR2HSV);
	cv::inRange(diceHSV, cv::Scalar(0,0,0,0), cv::Scalar(0,255,30,0), diceThreshLow);
	cv::inRange(diceHSV, cv::Scalar(0,0,200,0), cv::Scalar(200,255,255,0), diceThreshHigh);
	cv::add(diceThreshLow, diceThreshHigh, diceThreshold);
	cv::GaussianBlur(diceThreshold, diceBlur, cv::Size(5,5),0);
	cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::dilate(diceBlur, diceBlur, structuringElement);
	cv::erode(diceBlur, diceBlur, structuringElement);
	cv::Canny(diceBlur, diceCanny, 175, 620);
	imshow("Canny", diceCanny);
	imshow("Threshold", diceThreshold);
	
	//Detecting Dots Using Blob Detection
	cv::SimpleBlobDetector::Params params;
	params.filterByCircularity = false;
	params.filterByConvexity = true;
	params.minConvexity = 0.37;
	params.filterByArea = true;
	params.minArea = 200.0f;
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	//Detect Blobb
	detector->detect(diceBlur, keypoints);

	//Draw Keypoints
	drawKeypoints(diceOriginal, keypoints, diceOutput, Scalar(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//Text on Image
	int nBlobs = keypoints.size();
	if (nBlobs >= 0)
	{
		str << "Sum= " << nBlobs;
		cv::putText(diceOutput,str.str(), cv::Point2f(10, 25), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,255,0));

	}
	
	
	//Square Detection
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(diceCanny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> >hull(contours.size());
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());
	vector<Point2f>contourNumber;
	int diceCount = 0;
	int indCount[10] = {};
	
	for (size_t i = 0; i < contours.size(); i++)
	{
		convexHull(Mat(contours[i]), hull[i], false);
		
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle(contours_poly[i], center[i], radius[i]);
		double a = contourArea(hull[i]);
		if (a > 1000)
		{
			drawContours(diceOutput, contours, (int)i, Scalar(0,255, 0), 1, 8, vector<Vec4i>(), 0, Point());
			diceCount += 1;
			contourNumber.push_back(center[i]);
			int counter = 0;

			for (int j = 0; j < nBlobs; j++)
			{
				int x = center[i].x;
				int y = center[i].y;
				int x1 = keypoints[j].pt.x;
				int y1 = keypoints[j].pt.y;

				if (fabs(x - x1) < 100 && fabs(y - y1) < 100)
				{
					counter += 1;
				}
			}
			indCount[diceCount - 1] = counter;
		}
	}
	
	//Print Number of Dots on Each Die
	for (int i =0;i<diceCount;i++)
	{
		str1 << indCount[i];
		cv::putText(diceOutput, str1.str(), cv::Point2f(contourNumber[i] + Point2f(100, 0)), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,255,0));
		str1.str(std::string());
		str1.clear();
	}
	
	cv::imshow("Original Image", diceOriginal);
	cv::imshow("Final Image", diceOutput);
	cv::imwrite("output_dice6.jpg", diceOutput);

	cv::waitKey();

	return 0;
}