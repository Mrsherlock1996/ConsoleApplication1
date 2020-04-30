#include <iostream>
#include <opencv2/opencv.hpp>
/*利用该程序对opencvAPI进行学习*/
using namespace std;
using namespace cv;

void  getGaussionMatrix(int size, double sigma, Mat& Gauss);

int main()
{
	//练习imread参数IMREAD_GRAYSCALE
	Mat srcGray = imread("C:\\Users\\xpp19\\Desktop\\test.jpg", IMREAD_GRAYSCALE);
	Mat src= imread("C:\\Users\\xpp19\\Desktop\\test.jpg");
	string srcWindowName = "src_image";
	namedWindow(srcWindowName, WINDOW_AUTOSIZE);
	imshow(srcWindowName, src);
	waitKey(10);
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
	waitKey(10);

	Mat GaussianImage;
	Mat output;
	Mat GaussMatrix;
	int size = 3;
	src.copyTo(GaussianImage);
	src.copyTo(output);
	int channel = GaussianImage.channels();
	int rows = GaussianImage.rows;
	int cols = GaussianImage.cols;
	getGaussionMatrix(3, 1, GaussMatrix);
	for (int row = 1; row < rows - 1; row++) {
		const uchar* current = GaussianImage.ptr<uchar>(row);
		const uchar* previous = GaussianImage.ptr<uchar>(row-1);
		const uchar* next = GaussianImage.ptr<uchar>(row+1);
		uchar* outputPixels = output.ptr<uchar>(row);
		for (int col = channel; col < channel * (cols - 1); col++) {
		/*Mat矩阵点乘操作 A.dot(B) */
			Mat temp = (Mat_<float>(size, size) <<
				previous[col - 1], previous[col], previous[col + 1],
				current[col - 1], current[col], current[col + 1],
				next[col - 1], next[col], next[col + 1]);
			outputPixels[col] = GaussMatrix.dot(temp);

		}

	}
	namedWindow("outputGaussImage", WINDOW_AUTOSIZE);
	imshow("outputGaussImage", output);
	Mat gaussionBlurImage;
	src.copyTo(gaussionBlurImage);
	Mat outputGaussBlur;
	GaussianBlur(gaussionBlurImage, outputGaussBlur, Size(3, 3), 1.0);
	namedWindow("GaussionBlur", WINDOW_AUTOSIZE);
	imshow("GaussionBlur", outputGaussBlur);
	waitKey(0);
	return 0;

	
}
void  getGaussionMatrix(int size, double sigma, Mat &Gauss) { 
	//只用了一个sigma, 公式是sigmax和sigmay
	Gauss.create(Size(size, size), CV_32FC1);
	float sum = 0.0;
	float center = size / 2;
	/*  要想得到一个高斯滤波器的模板，
	可以对高斯函数进行离散化，
	得到的高斯函数值作为模板的系数。
	例如：要产生一个3×3的高斯滤波器模板，
	以模板的中心位置为坐标原点进行取样。
	center只是一个用来平衡周围像素的值罢了*/
	const float PI = 4 * tan(1.0);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			//写出该点Gaussion权值, 
			Gauss.at<float>(i, j) = (1 / (2 * PI * sigma * sigma)) * 
				exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma));
			sum += Gauss.at<float>(i, j);
		}
	}
	//归一化
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			Gauss.at<float>(i, j) /= sum;
		}
	}
}