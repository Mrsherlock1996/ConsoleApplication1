#include <iostream>
#include <opencv2/opencv.hpp>
/*利用该程序对opencvAPI进行学习*/
using namespace std;
using namespace cv;
int main()
{
	//练习imread参数IMREAD_GRAYSCALE
	Mat srcGray = imread("C:\\Users\\xpp19\\Desktop\\test.jpg", IMREAD_GRAYSCALE);
	Mat src= imread("C:\\Users\\xpp19\\Desktop\\test.jpg");
	string srcWindowName = "src_image";
	namedWindow(srcWindowName, WINDOW_AUTOSIZE);
	imshow(srcWindowName, src);
	//练习src.rows & src.cols
	cout << "src width = " << src.cols << endl;
	cout << "src height = " << src.rows << endl;
	Mat testImage0;
	//练习src.copyTo();
	srcGray.copyTo(testImage0);
	//练习cvtColor();
	cvtColor(testImage0, testImage0, COLOR_GRAY2BGR);
	string testWindowName = "testImage0";
	namedWindow(testWindowName, WINDOW_AUTOSIZE);
	imshow(testWindowName, testImage0);
	//练习filter2D()和Sobel算子
	Mat sobelX= (Mat_<int>(3, 3) << -1, 0, -1, -2, 0, -2, -1, 0, 1);
	Mat sobelY = (Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	Mat filterImage;
	filter2D(src, filterImage, src.depth(), sobelX,Point(-1,-1));   //经测试, 当只进行sobelX时,显示黑色图片
	filter2D(src, filterImage, src.depth(), sobelY, Point(-1, -1));
	imshow("filterImage", filterImage);
	//练习获取像素指针和像素值
	Mat ptrImage;
	src.copyTo(ptrImage);
	//用像素指针输出每个像素值,太久了不看了
	/*int channels = ptrImage.channels();
	int width = ptrImage.cols* channels;
	int height = ptrImage.rows;
	for (int row = 0; row < height-1; row++) {
		const uchar* rowPtr = ptrImage.ptr<uchar>(row);
		for (int col = 0; col < width-1; col++) {
			cout <<"("<<row<<","<<col<<")"<<"pixel = "<< rowPtr[col] << endl;
		}
	}*/
	//直接输出某个像素值
	Mat pixelImage;
	srcGray.copyTo(pixelImage);
	//错误示范:
	//uchar pixelValue = pixelImage.at<uchar>(1, 1);
	//单通道图片像素值
	int pixelValue = pixelImage.at<uchar>(32, 2);
	//多通道图片像素值
	src.copyTo(pixelImage);
	int bPixel = pixelImage.at<Vec3b>(1,1)[0];
	int gPixel = pixelImage.at<Vec3b>(1, 1)[1];
	int rPixel = pixelImage.at<Vec3b>(1, 1)[2];
	cout << "(1,1)bPixel =" << bPixel << endl;
	cout << "(1,1)gPixel =" << gPixel << endl;
	cout << "(1,1)rPixel =" << rPixel << endl;
	waitKey(0);
	return 0;
}