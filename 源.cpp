#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include <cstring>
#include <algorithm>
#include  <map>
#include <string>

using namespace cv;
using namespace std;

#define ChessBoardWidth 7
#define ChessBoardHeight 7
#define WIDTH 720
#define HEIGHT 480
#define ImageNum 3
#define beishu 1  
#define range 40

string str_head = "上汽数据\\4\\";
string str[10] = { "11", "22", "00", "1", "2", "8" };

int img_height, img_width;
int checked_img_num = 0;

double sq_sz = 41.27;        //真实棋盘格方块宽度
double f = 161.85;           //焦距
double pixel_size = 0.006;   //LUT系数
double ellip = 1.125;        //椭圆系数
vector<double> table;        //LUT表
double step;                 //LUT步长
int LUT_num;                 //LUT数量

Size s = Size(WIDTH,HEIGHT);//Size(1440,960);

vector<Point2f> corners;               //棋盘格角点集
vector<vector<Point3d>>  obj_points;   //真实物体点集(Z=0)
vector<vector<Point2d> > img_points;   //成像平面点集
vector<Point3d> obj_temp;              //用于存放一张图的点集



/**********************************************
*  找棋盘格
*  若找到,则将点放入成像平面点集img_points
**********************************************/
void ready_go()
{
	cout << "Image_list: ";
	for (int i = 0; i < ImageNum; i++)
	{
		Mat img1 = imread(str_head + str[i] + ".jpg");
		cout << str[i] + ".jpg" << " ";
		img_width = img1.cols;
		img_height = img1.rows;

		bool found = findChessboardCorners(img1, Size(ChessBoardWidth,ChessBoardHeight), corners);

		Mat gray;
		cvtColor(img1, gray, CV_BGR2GRAY);
		//cornerSubPix(gray, corners, Size(10, 10), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
		drawChessboardCorners(img1, Size(ChessBoardWidth, ChessBoardHeight), corners, found);

		///*
		namedWindow(str[i], 0);
		imshow(str[i], img1);
		waitKey(10);
		//*/

		//如果棋盘格寻找成功，建立成像平面点集
		if (found)
		{
			vector<Point2d> img_temp;
			for (int j = 0; j < ChessBoardHeight*ChessBoardWidth; j++)
			{
				Point2d temp = corners[j];
				img_temp.push_back(temp);
			}
			img_points.push_back(img_temp);
			checked_img_num++;
		}
	}

	//构造物体三维坐标点集
	for (int i = 0; i < ChessBoardHeight; i++) {
		for (int j = 0; j < ChessBoardWidth; j++) {
			obj_temp.push_back(Point3d(double(j * sq_sz), double(i * sq_sz), 0));
		}
	}
	for (int i = 0; i < checked_img_num; i++) obj_points.push_back(obj_temp);
}


/**********************************************
*  xml输出K,D
**********************************************/
void xml_out(Matx33d K,Vec4d D)
{
	//创建XML文件写出  
	FileStorage fs("intrinsics.xml", FileStorage::WRITE);
	Mat mat = Mat(K);
	fs << "intrinsics" << mat;
	mat = Mat(D);
	fs << "coefficient" << mat;
	fs.release();
}


/**********************************************
*LUT读入
**********************************************/
void init_LUT()
{
	int LUT_num = 0;
	FILE *in;
	in = fopen("curve101_LUT.txt", "r");
	fscanf(in, "%lf", &step);
	double angle;
	while (fscanf(in, "%lf", &angle) != NULL&&LUT_num<18000)
	{
		table.push_back(angle);
		LUT_num++;
	}
	fclose(in);
}

/**********************************************
*根据矫正图像坐标(i,j)计算对应的鱼眼图像坐标(resultx,resulty)
**********************************************/
void correspondence(double i, double j, int xcenter, int ycenter, double* resultx, double* resulty)
{
	//计算所需的变量
	double theta;
	double ru;
	double x, y;
	double rd;

	//christian
	//i = i * ellip;
	//ycenter = ycenter * ellip;
	ru = sqrt(pow(i - ycenter * beishu, 2) + pow((j - xcenter * beishu)/ellip, 2));
	if (ru != 0)
	{
		//对顶角相等,求出实际世界中的入射角
		theta = atan(ru / f);
		//求出鱼眼镜头内的角度
		//----LUT----
		if (1)
		{
			double angle = theta * 180 / CV_PI;
			int r = angle / step;
			rd = (table[r] + (table[r + 1] - table[r])*((angle - r*step) / step)) / pixel_size;
		}
		else
		{
			//公式
			rd = 2 * f * sin(theta / 2);
		}
		//求出鱼眼镜头内相对光轴的坐标
		//christian
		x = rd * abs(j - xcenter * beishu) / ru;
		y = rd * abs(i - ycenter * beishu) / ru;
	}
	else
	{
		x = y = 0;
	}

	//christian
	if (i <= ycenter * beishu && j <= xcenter * beishu) { x = -x; y = -y; }
	else if (i <= ycenter * beishu && j > xcenter * beishu) { y = -y; }
	else if (i > ycenter * beishu && j <= xcenter * beishu) { x = -x; }

	//变换为绝对坐标
	*resultx = x*ellip + xcenter;
	*resulty = (y + ycenter);
	return;
}



int main(){


	ready_go();

	cv::Matx33d K,K2;
	cv::Vec4d D;
	int flag = 0;
	flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flag |= cv::fisheye::CALIB_CHECK_COND;
	flag |= cv::fisheye::CALIB_FIX_SKEW;
	//cout << "flag: "<<flag << endl;
	//标定求出内参矩阵K和畸变系数矩阵D
	double calibrate_error=fisheye::calibrate(obj_points, img_points, Size(img_width, img_height), K, D, noArray(), noArray(), flag, TermCriteria(3, 20, 1e-6));
	getOptimalNewCameraMatrix(K, D, s, 1.0, s);
	//得到矫正相机矩阵
	fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, Size(720, 480), cv::noArray(), K2, 0.8, s ,1.0);
	
	
	cout << endl << "K:" << K << endl << "D:" << D << endl << "calibrate_error:  " << calibrate_error << endl;

	xml_out(K, D);

	//鱼眼矫正后图像
	for (int i = 0; i <ImageNum; i++)
	{
		Mat output;// = Mat(Size(img_height, img_width), CV_8UC3);
		Mat img1 = imread(str_head + str[i] + ".jpg");
		fisheye::undistortImage(img1, output, K, D, K2, s);
		//namedWindow("img"+str[i], 0);
		//imshow("img"+str[i], output);
		//waitKey();
	}

	//=============================================================================================
	double ChessScore[range][range],ChessScore2[range][range];
	Mat u_img[range][range]; 
	memset(ChessScore, 0, sizeof(ChessScore));
	memset(ChessScore2, 0, sizeof(ChessScore2));
	Mat img = imread(str_head + str[2] + ".jpg");
	init_LUT();
  	Mat P = Mat(K);
	double* T = (double*)(P.data);
	double center_x = *(T + 2), center_y = *(T + 5);
	for (int i = 0; i < range; i++)
		for (int j = 0; j < range; j++)
		{
		     double center_xx = center_x + i - range / 2;
		     double center_yy = center_y + j - range / 2;
			 int u_width = beishu *WIDTH, u_height = beishu*HEIGHT;
			 u_img[i][j] = Mat(Size(u_width, u_height), CV_8UC3);
			 for (int k1 = 0; k1 < u_width; k1++)
				 for (int k2 = 0; k2 < u_height; k2++)
				 {
				        double resultx, resulty;
						correspondence(k2, k1, center_xx, center_yy, &resultx, &resulty);
						int intx = resultx; 
						int inty = resulty;

						//超出鱼眼图片范围，防止越界，边缘自动补全
						if (intx >= WIDTH)
							intx = WIDTH - 1;
						else if (intx < 0)
							intx = 0;
						if (inty >= HEIGHT)
							inty = HEIGHT - 1;
						else if (inty < 0)
							inty = 0;
						//if (k1 == 479 && k2 == 479) system("pause");
						u_img[i][j].at<Vec3b>(k2, k1) = img.at<Vec3b>(inty,intx);
				 }
		}


	for (int i = 0; i < range; i++)
		for (int j = 0; j < range; j++)
		{
		    bool found = findChessboardCorners(u_img[i][j], Size(ChessBoardWidth, ChessBoardHeight), corners);
		    //drawChessboardCorners(u_img[i][j], Size(ChessBoardWidth, ChessBoardHeight), corners, found);
			if (!found)
			{
				ChessScore[i][j] = 99.9999;
				ChessScore2 [i][j] = 99.9999;
				continue;
			}

			//横线打分
			for (int k = 0; k < ChessBoardHeight; k++)
			{
				double dis = 0;
				double sum_x = 0, sum_y = 0, avg_x = 0, avg_y, A = 0, B = 0, C = 0;
				for (int h = 0; h < ChessBoardWidth; h++)
					sum_x += corners[k*ChessBoardWidth + h].x, sum_y += corners[k*ChessBoardWidth + h].y;

				avg_x = sum_x / ChessBoardWidth;
				avg_y = sum_y / ChessBoardWidth;

				for (int h = 0; h < ChessBoardWidth; h++)
					C += (corners[k*ChessBoardWidth + h].x - avg_x)*(corners[k*ChessBoardWidth + h].x - avg_x);

				for (int h = 0; h < ChessBoardWidth; h++)
					A += (corners[k*ChessBoardWidth + h].x - avg_x)*(corners[k*ChessBoardWidth + h].y - avg_y);

				A /= C;
				B = avg_y - A*avg_x;
				for (int h = 0; h < ChessBoardWidth; h++)
					dis += fabs(A*corners[k*ChessBoardWidth + h].x - corners[k*ChessBoardWidth + h].y + B) / sqrt(A*A + 1);
				ChessScore[i][j] += dis ;
				//line(u_img[i][j], Point(0, B), Point(WIDTH*beishu, WIDTH*beishu * A + B), Scalar{ 255, 255, 0 }, 1, CV_AA);
				//for (int z = 0; z < corners.size(); z++)
				//	circle(u_img[i][j], corners[z], 3, Scalar{ 0 , 255, 0 }, 1.5);
				//cout << "Line " << k << "   Score:  " << dis << endl;
				//cout << "A:  " << A << "  B:  " << B << endl;
			}


			//竖线
			double AA[ChessBoardWidth], BB[ChessBoardWidth], CC[ChessBoardWidth];  //储存每条直线的A,B,C系数  y=A*x+B
			for (int k = 0; k < ChessBoardWidth; k++)
			{
				double dis = 0;
				double sum_x = 0, sum_y = 0, avg_x = 0, avg_y, A = 0, B = 0, C = 0;
				for (int h = 0; h < ChessBoardHeight; h++)
					sum_x += corners[h*ChessBoardWidth + k].x, sum_y += corners[h*ChessBoardWidth + k].y;

				avg_x = sum_x / ChessBoardHeight;
				avg_y = sum_y / ChessBoardHeight;

				for (int h = 0; h < ChessBoardHeight; h++)
					C += (corners[h*ChessBoardWidth + k].x - avg_x)*(corners[h*ChessBoardWidth + k].x - avg_x);

				for (int h = 0; h < ChessBoardHeight; h++)
					A += (corners[h*ChessBoardWidth + k].x - avg_x)*(corners[h*ChessBoardWidth + k].y - avg_y);

				A /= C;
				B = avg_y - A*avg_x;
				for (int h = 0; h < ChessBoardHeight; h++)
					dis += fabs(A*corners[h*ChessBoardWidth + k].x - corners[h*ChessBoardWidth + k].y + B) / sqrt(A*A + 1);
				ChessScore[i][j] += dis;

				AA[k] = A; BB[k] = B; CC[k] = C;
				//line(u_img[i][j], Point(-B / A, 0), Point((HEIGHT*beishu - B) / A, HEIGHT*beishu), Scalar{ 255, 255, 0 }, 1, CV_AA);
				//cout << "Line " << k << "   Score:  " << dis << endl;
				//cout << "A:  " << A << "  B:  " << B << endl;
			}


			//打分方式二，按交点密集程度打分
			int P_num = 0;  //交点个数
			Point2f PP[ChessBoardHeight*ChessBoardWidth];
			for (int k = 1; k <= ChessBoardWidth; k++)
				for (int h = k + 1; h <= ChessBoardHeight; h++)
				{
				   P_num++;
				   PP[P_num] = Point((BB[h] - BB[k]) / (AA[k] - AA[h]), AA[k] * (BB[h] - BB[k]) / (AA[k] - AA[h]) + BB[k]);
				   //circle(u_img[i][j], PP[P_num], 3, Scalar{ 255, 0, 255 }, 1);
				}

			//求出中心点
			double avg_x = 0, avg_y = 0;
			for (int k = 1; k <= P_num; k++)
			{
				avg_x += PP[k].x;
				avg_y += PP[k].y;
			}
			avg_x /= P_num; avg_y /= P_num;
			//circle(u_img[i][j], Point(avg_x, avg_y), 3, Scalar{ 0, 255, 255 }, 1);
			//求聚散系数
			double dis = 0;
			for (int k = 1; k <= P_num; k++)
			{
				dis += sqrt((avg_x - PP[k].x)*(avg_x - PP[k].x) + (avg_y - PP[k].y)*(avg_y - PP[k].y));
			}

			//选择打分方式二的打分覆盖分数
			ChessScore2[i][j] = dis / P_num;


		    //namedWindow("t11", 0);
		    //imshow("t11", u_img[i][j]);
		    //waitKey();
		}

	//找出分数最小的即为最优点
	freopen("score.txt", "w", stdout);
	double MinScore = 1000000;
	int marki = 0, markj = 0;
	for (int i = 0; i < range; i++)
		for (int j = 0; j < range; j++)
		{
		//cout << "-----------x: " << i - range / 2 << "    y: " << j - range / 2 << "-------------" << endl;
		if (ChessScore2[i][j] < MinScore)
		{
			MinScore = ChessScore2[i][j];
			marki = i;
			markj = j;
		}
		printf("%.4lf\t", ChessScore2[i][j]);
		if (j == range - 1) printf("\n");
		}
	cout << "Best Center:  x: " << center_x + marki - range / 2 << "     y: " << center_y + markj - range / 2 << endl;
	fclose(stdout);
	//cout << endl << endl;
	//cout << "Best Center:  x: " << center_x + marki - range / 2 << "     y: " << center_y+markj - range / 2 << endl;
	

	//system("pause");
	return 0;
}
