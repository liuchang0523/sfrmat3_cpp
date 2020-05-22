#include <opencv2/opencv.hpp>

using namespace cv;

#define min(a,b) (((a) < (b)) ? (a) : (b))

Mat matRotateCounterClockWise90(Mat src)
{
	if (src.empty())
	{
		std::cout << "RorateMat src is empty!";
	}
	// 矩阵转置
	transpose(src, src);
	//0: 沿X轴翻转； >0: 沿Y轴翻转； <0: 沿X轴和Y轴翻转
	flip(src, src, 0);// 翻转模式，flipCode == 0垂直翻转（沿X轴翻转），flipCode>0水平翻转（沿Y轴翻转），flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
	return src;
}


cv::Mat rotatev2(cv::Mat img)
{
	int mm = 1;
	int nn = 3;
	cv::Mat v1 = img.row(nn - 1);
	cv::Mat v2 = img.row(img.rows - nn - 1);
	double testv = cv::abs(cv::mean(v1).val[0] - cv::mean(v2).val[0]);

	cv::Mat h1 = img.col(nn - 1);
	cv::Mat h2 = img.col(img.cols - nn - 1);
	double testh = cv::abs(cv::mean(h1).val[0] - cv::mean(h2).val[0]);

	int rflag = 0;
	cv::Mat output = img.clone();
	if (testv > testh)
	{
		rflag = 1;
		//旋转90度
		output = matRotateCounterClockWise90(img);
	}

	return output;
}

//generates a general asymmetric Hamming-type window array
//If mid = (n+1)/2 then the usual symmetric Hamming window is returned
cv::Mat ahamming(int n, double mid)
{
	cv::Mat hammingData = cv::Mat::zeros(n, 1, CV_64FC1);
	double wid1 = mid - 1;
	double wid2 = n - mid;
	double wid = cv::max(wid1, wid2);
	double pie = 3.141592653589793;
	auto *ptr = hammingData.ptr<double>();
	for (int i = 0; i < n; i++)
	{
		double arg = i - mid + 1;
		ptr[i] = cosf(pie*arg / wid);
	}
	hammingData = hammingData * 0.46 + 0.54;

	return hammingData;
}


// Computes first derivative via FIR(1xn) filter
//  Edge effects are suppressed and vector size is preserved
//  Filter is applied in the npix direction only
//   a = (nlin, npix) data array
//   fil = array of filter coefficients, eg[-0.5 0.5]
cv::Mat deriv1(const cv::Mat &input, cv::Mat fil)
{
	int nlin = input.rows;
	int npix = input.cols;
	int nn = fil.cols;
	cv::Mat calc_mat;
	input.convertTo(calc_mat, fil.type());
	cv::Mat b = cv::Mat::zeros(nlin, npix, calc_mat.type());
	for (int i = 0; i < nlin; i++)
	{
		if (3 == nn)
		{
			cv::Mat row_mat = calc_mat.row(i);
			cv::Mat temp;
			cv::filter2D(row_mat, temp, fil.type(), fil, cv::Point(-1, -1), 0, cv::BORDER_ISOLATED);
			temp = -temp;//与matlab结果相差一个负号
			temp.at<double>(0, 0) = temp.at<double>(0, 1);
			cv::Mat temp_zero = cv::Mat::zeros(1, 1, temp.type());
			cv::Mat temp_1;
			cv::hconcat(temp_zero, temp, temp_1);
			cv::Rect roi_rect(0, 0, npix, 1);
			temp = temp_1(roi_rect).clone();
			cv::Mat row_b = b.row(i);
			temp.copyTo(row_b);
		}
		else if (2 == nn)
		{
			cv::Mat row_mat = calc_mat.row(i);
			cv::Mat temp;
			cv::filter2D(row_mat, temp, fil.type(), fil);
			temp = -temp;
			temp.at<double>(0, 0) = temp.at<double>(0, 1);
			cv::Mat row_b = b.row(i);
			temp.copyTo(row_b);
		}
	}
	return b;
}


double centroid(const cv::Mat &input)
{
	int n = input.rows;
	cv::Mat n_mat = cv::Mat::zeros(1, n, input.type());
	auto *ptr = n_mat.ptr<double>();
	for (int i = 0; i < n; i++)
	{
		ptr[i] = i + 1;
	}

	double sumx = cv::sum(input).val[0];
	if (sumx < 1e-4)
	{
		return 0;
	}
	else
	{
		double loc = cv::sum(n_mat*input).val[0] / sumx;
		return loc;//-0.5 shift for FIR phase
	}
}

Mat polyfit(std::vector<cv::Point2d>& in_point, int n)
{
	int size = in_point.size();
	//所求未知数个数
	int x_num = n + 1;
	//构造矩阵U和Y
	Mat mat_u(size, x_num, CV_64F);
	Mat mat_y(size, 1, CV_64F);

	for (int i = 0; i < mat_u.rows; ++i)
		for (int j = 0; j < mat_u.cols; ++j)
		{
			mat_u.at<double>(i, j) = pow(in_point[i].x, j);
		}

	for (int i = 0; i < mat_y.rows; ++i)
	{
		mat_y.at<double>(i, 0) = in_point[i].y;
	}

	//矩阵运算，获得系数矩阵K
	Mat mat_k(x_num, 1, CV_64F);
	mat_k = (mat_u.t()*mat_u).inv()*mat_u.t()*mat_y;
	//std::cout << mat_k << std::endl;
	return mat_k;
}

cv::Mat fir2fix(int n, int m)
{
	cv::Mat correct = cv::Mat::ones(n, 1, CV_64FC1);

	m = m - 1;
	int scale = 1;
	auto ptr = correct.ptr<double>();
	for (int i = 1; i < n; i++)
	{
		ptr[i] = abs(CV_PI*(i + 1)*m / (2 * (n + 1))) / sinf(CV_PI*(i + 1)*m / (2 * (n + 1)));
		ptr[i] = 1 + scale * (ptr[i] - 1);
		if (ptr[i] > 10) //limiting the correction to the range[1, 10]
		{
			ptr[i] = 10;
		}
	}

	return correct;
}

cv::Mat project(const cv::Mat &bb, double loc, double slope, int fac = 4)
{
	int nlin = bb.rows;
	int npix = bb.cols;
	int big = 0;
	int nn = npix * fac;

	// smoothing window
	cv::Mat win = ahamming(nn, fac*loc);

	slope = 1 / slope;

	int offset = round(fac*(0 - (nlin - 1) / slope));
	int del = abs(offset);
	if (offset > 0)
	{
		offset = 0;
	}

	cv::Mat barray = cv::Mat::zeros(2, nn + del + 100, CV_64FC1);
	auto *ptr_barray_1 = barray.ptr<double>(0);
	auto *ptr_barray_2 = barray.ptr<double>(1);

	for (int n = 0; n < npix; n++)
	{
		for (int m = 0; m < nlin; m++)
		{
			int x = n;
			int y = m;
			int ling = ceil((x - y / slope)*fac) + 1 - offset;
			ling = ling - 1;//与matalb的坐标差别
			ptr_barray_1[ling] = ptr_barray_1[ling] + 1;
			ptr_barray_2[ling] = ptr_barray_2[ling] + bb.at<uchar>(m, n);
		}
	}

	int start = 1 + round(0.5*del);
	int nz = 0;
	int status = 0;
	for (int i = start - 1; i < start + nn; i++)
	{
		if (0 == ptr_barray_1[i])
		{
			nz++;
			status = 0;
			if (1 == i)
			{
				ptr_barray_1[i] = ptr_barray_1[i + 1];
			}
			else
			{
				ptr_barray_1[i] = (ptr_barray_1[i - 1] + ptr_barray_2[i + 1]) / 2.0;
			}
		}
	}

	cv::Mat point = cv::Mat::zeros(nn, 1, CV_64FC1);
	auto *ptr_point = point.ptr<double>();
	for (int i = 0; i < nn; i++)
	{
		ptr_point[i] = ptr_barray_2[i + start - 1] / ptr_barray_1[i + start - 1];
	}

	return point;
}

cv::Mat cent(const cv::Mat &a, int center)
{
	int n = a.rows;
	cv::Mat b = cv::Mat::zeros(n, 1, a.type());
	int mid = round((n + 1) / 2.0);
	int del = round(center - mid);
	auto ptr_a = a.ptr<double>();
	auto ptr_b = b.ptr<double>();
	if (del > 0)
	{
		for (int i = 0; i < n - del; i++)
		{
			ptr_b[i] = ptr_a[i + del];
		}
	}
	else
	{
		for (int i = -del; i < n; i++)
		{
			ptr_b[i] = ptr_a[i + del];
		}
	}

	return b;
}

cv::Mat complex_abs(const cv::Mat &input)
{
	std::vector<cv::Mat> channels;
	cv::split(input, channels);
	auto ptr_1 = channels[0].ptr<double>();
	auto ptr_2 = channels[1].ptr<double>();
	cv::Mat output = cv::Mat::zeros(input.rows, 1, CV_64FC1);
	auto ptr_output = output.ptr<double>();
	for (int i = 0; i < input.rows; i++)
	{
		double a = ptr_1[i];
		double b = ptr_2[i];
		ptr_output[i] = sqrtf(a*a + b * b);
	}
	return output;
}

void findfreq(const cv::Mat &dat, double val, int imax,
	double &frequval, double &sfrval)
{
	int n = dat.rows;
	int nc = 1;

	std::vector<cv::Mat> channels;
	cv::split(dat, channels);
	auto ptr_1 = channels[0].ptr<double>();
	auto ptr_2 = channels[1].ptr<double>();
	double maxf = ptr_1[imax - 1];

	cv::Mat fil = cv::Mat::ones(3, 1, CV_64FC1);
	fil = fil / 3.0;

	cv::Mat test = channels[1] - val;
	auto ptr_test = test.ptr<double>();
	std::vector<int> x;
	for (int i = 0; i < dat.rows; i++)
	{
		if (ptr_test[i] < 0)
		{
			x.push_back(i);
		}
	}
	double sval = 0;
	double s = 0;
	if (x.empty() || 0 == x[0])
	{
		double s = maxf;
		sfrval = ptr_2[imax];
	}
	else
	{
		int x_pos = x[0];
		sval = ptr_2[x_pos - 1];
		s = ptr_1[x_pos - 1];
		double y = sval;
		double y2 = ptr_2[x_pos];
		double slope = (y2 - y) / ptr_1[1];
		double dely = ptr_test[x_pos - 1];
		s = s - dely / slope;
		sval = sval - dely;
	}
	if (s > maxf)
	{
		s = maxf;
		sval = ptr_2[imax - 1];
	}
	frequval = s;
	sfrval = sval;
}


void sampeff(const cv::Mat &dat, const cv::Mat &val,
	double del, cv::Mat &eff, cv::Mat &freqval, cv::Mat &sfrval)
{
	int pflag = 0;
	int fflag = 0;
	double mmin;
	int mindex[2];
	cv::minMaxIdx(val, &mmin, nullptr, &mindex[0]);
	if (mmin > 0.1)
	{
		std::cout << "Warning: sampling efficiency is based on SFR = " <<
			mmin << std::endl;
	}
	double delf = dat.at<double>(1, 0) + 1e-6;
	double hs = 0.5 / del;
	std::vector<int> x;
	std::vector<cv::Mat> channels;
	cv::split(dat, channels);
	auto ptr_1 = channels[0].ptr<double>();
	auto ptr_2 = channels[1].ptr<double>();
	for (int i = 0; i < dat.rows; i++)
	{
		if (ptr_1[i] > 1.1*hs)
		{
			x.push_back(i);
		}
	}
	int immax;
	int imax;
	cv::Mat dat_new;
	if (x.empty())
	{
		imax = dat.rows;
		immax = imax;
	}
	else
	{
		std::vector<int> xx;
		for (int i = 0; i < dat.rows; i++)
		{
			if (ptr_1[i] > hs - delf)
			{
				xx.push_back(i);
			}
		}
		imax = x[1];
		immax = xx[1];
		cv::Rect dat_roi(0, 0, 1, imax);
		dat_new = dat(dat_roi);
	}

	int n = dat.rows;
	int nc = 1;
	int nval = val.cols;
	eff = cv::Mat::zeros(1, nc, CV_64FC1);
	freqval = cv::Mat::zeros(nval, nc, CV_64FC1);
	sfrval = cv::Mat::zeros(nval, nc, CV_64FC1);

	for (int i = 0; i < nval; i++)
	{
		double freqval_temp;
		double sfr_val_temp;
		findfreq(dat_new, val.at<double>(0, i), imax, freqval_temp, sfr_val_temp);
		freqval.at<double>(i, 0) = freqval_temp;
		sfrval.at<double>(i, 0) = sfr_val_temp;
	}

	//Efficiency computed only for lowest value of SFR requested
	eff.at<double>(0, 0) = min(round(100 * freqval.at<double>(mindex[1], 0) / ptr_1[immax - 1]), 100);

	// 	if (pflag != 0)
	// 	{
	// 		//TODO
	// 	}
}

double Mtf50Compute(const cv::Mat &img)
{
	cv::Mat img_roi = rotatev2(img);

	int img_height = img_roi.rows;//nlin
	int img_width = img_roi.cols;//npix

	cv::Rect roiA(0, 0, 5, img_height);
	//tleft  = sum(sum(a(:,      1:5,  1),2));
	cv::Rect roiB(img_width - 6, 0, 6, img_height);
	//tright = sum(sum(a(:, npix-5:npix,1),2));
	//这里tleft是5列，tright是6列，估计是个bug

	cv::Mat A = img_roi(roiA);
	cv::Mat B = img_roi(roiB);
	double tleft = cv::sum(A).val[0];
	double tright = cv::sum(B).val[0];
	cv::Mat fil1 = (cv::Mat_<double>(1, 2) << 0.5, -0.5);
	cv::Mat fil2 = (cv::Mat_<double>(1, 3) << 0.5, 0, -0.5);
	if (tleft > tright)
	{
		fil1 = (cv::Mat_<double>(1, 2) << -0.5, 0.5);
		fil2 = (cv::Mat_<double>(1, 3) << -0.5, 0, 0.5);
	}

	double test = cv::abs((tleft - tright) / (tleft + tright));
	if (test < 0.2)
	{
		std::cout << "** WARNING: Edge contrast is less that 20%, this can" <<
			"lead to high error in the SFR measurement.\n";
	}

	cv::Mat filme = cv::Mat::zeros(1, 3, CV_64FC1);
	double slout = 0;

	//汉明窗
	cv::Mat win1 = ahamming(img_width, (img_width + 1) / 2.0);
	//求一阶导数
	cv::Mat c = deriv1(img_roi, fil1);
	//计算质心
	cv::Mat loc = cv::Mat::zeros(1, img_height, CV_64FC1);
	auto *ptr_loc = loc.ptr<double>();
	for (int i = 0; i < img_height; i++)
	{
		cv::Mat temp = c.row(i).t();
		temp = temp.mul(win1);
		double centroid_temp = centroid(temp) - 0.5;
		ptr_loc[i] = centroid_temp;
	}

	//曲线拟合
	std::vector<cv::Point2d> fit_points;
	for (int i = 0; i < img_height; i++)
	{
		fit_points.emplace_back(i, ptr_loc[i]);
	}
	cv::Mat fitme = polyfit(fit_points, 1);//这里和Matlab位置不一样
	cv::Mat place = cv::Mat::zeros(img_height, 1, CV_64FC1);
	auto *ptr_place = place.ptr<double>();
	double fitme_a = fitme.at<double>(1, 0);
	double fitme_b = fitme.at<double>(0, 0);
	for (int i = 0; i < img_height; i++)
	{
		ptr_place[i] = fitme_a * (i + 1) + fitme_b;
		cv::Mat win2 = ahamming(img_width, ptr_place[i]);

		cv::Mat temp = c.row(i).t();
		temp = temp.mul(win2);
		double centroid_temp = centroid(temp);
		ptr_loc[i] = centroid_temp;
	}
	fit_points.clear();
	for (int i = 0; i < img_height; i++)
	{
		fit_points.emplace_back(i, ptr_loc[i]);
	}
	fitme = polyfit(fit_points, 1);//这里和Matlab位置不一样

	//Limit number of lines to integer
	//对应oldflag=0
	int nlin1 = img_height * cv::abs(fitme.at<double>(1, 0));
	nlin1 = round(nlin1 / cv::abs(fitme.at<double>(1, 0)));
	cv::Rect interger_roi(0, 0, img_width, nlin1);
	cv::Mat img_roi_new = img_roi(interger_roi);

	double vslope = cv::abs(fitme.at<double>(1, 0));
	double slope_deg = 180 * atan(cv::abs(vslope)) / CV_PI;
	if (slope_deg < 3.5)
	{
		std::cout << "High slope warning : " << slope_deg << " degrees\n";
	}

	double del2 = 0;
	//Correct sampling inverval for sampling parallel to edge
	//对应oldflag=0
	double del = 1;
	int nbin = 4;
	double delfac = cosf(atan(vslope));
	del = del * delfac;
	del2 = del / nbin;

	int nn = img_width * nbin;
	cv::Mat mtf = cv::Mat::zeros(nn, 1, CV_64FC1);
	int nn2 = nn / 2 + 1;

	//Derivative correction
	cv::Mat dcorr = fir2fix(nn2, 3);

	cv::Mat freq = cv::Mat::zeros(nn, 1, CV_64FC1);
	auto ptr_freq = freq.ptr<double>();
	for (int i = 0; i < nn; i++)
	{
		ptr_freq[i] = nbin * i / (del*nn);
	}

	int freqlim = 1;
	if (1 == nbin)
	{
		freqlim = 2;
	}

	int nn2out = round(nn2*freqlim / 2.0);

	double nfreq = nn / (2.0*del*nn); // half - sampling frequency
	cv::Mat win = ahamming(nbin*img_width, (nbin*img_width + 1) / 2.0);

	//Large SFR loop for each color record
	cv::Mat esf = cv::Mat::zeros(nn, 1, CV_64FC1);
	// project and bin data in 4x sampled array
	cv::Mat point = project(img_roi_new, loc.at<double>(0, 0), fitme.at<double>(1, 0), nbin);

	esf = point;

	//compute first derivative via FIR(1x3) filter fil
	c = deriv1(point.t(), fil2);
	c = c.t();

	cv::Mat psf = c;
	double mid = centroid(c);

	cv::Mat temp = cent(c, round(mid));
	c = temp;
	// apply window(symmetric Hamming)
	c = win.mul(c);

	//Transform, scale and correct for FIR filter response
	//temp = abs(fft(c, nn));
	cv::dft(c, temp, cv::DFT_COMPLEX_OUTPUT);
	temp = complex_abs(temp);

	cv::Rect roi_temp(0, 0, 1, nn2);
	cv::Mat temp1 = temp(roi_temp).clone();
	temp1 = temp1 / temp1.at<double>(0, 0);
	cv::Mat mtf_roi = mtf(roi_temp);
	temp1.copyTo(mtf_roi);
	//对应oldflag=0
	mtf_roi = mtf_roi.mul(dcorr);

	std::vector<cv::Point2d> dat;
	auto ptr_mtf = mtf.ptr<double>();
	for (int i = 0; i < nn2out; i++)
	{
		dat.emplace_back(ptr_freq[i], ptr_mtf[i]);
	}
	cv::Mat dat_mat(dat);//仅用于显示

	//Sampling efficiency
	//cv::Mat val = (cv::Mat_<double>(1, 2) << 0.1, 0.5);
	cv::Mat val = (cv::Mat_<double>(1, 1) << 0.5);//只关注MTF50

	cv::Mat eff, freqval, sfrval;
	sampeff(dat_mat, val, del, eff, freqval, sfrval);

	// Plot SFRs on same axes
	//后面就是绘图和保存结果了
	// 

	std::vector<cv::Mat> channels;
	cv::split(dat_mat, channels);

	cv::Mat dat_output;
	cv::hconcat(channels[0], channels[1], dat_output);
	auto ptr_1 = channels[0].ptr<double>();
	auto ptr_2 = channels[1].ptr<double>();
	//打印输出
	for (int i = 0; i < dat_output.rows; i++)
	{
		std::cout << ptr_1[i] << "," << ptr_2[i] << std::endl;
	}

	return freqval.at<double>(0, 0);
}

int main(int argc, char** argv)
{
	Mat img = imread("bb.bmp", -1);
	//Mat img = imread(argv[1], -1);
	//cv::Rect roi_rect(74, 30, 249 - 75 + 1, 84 - 31 + 1);
	double mtf50 = Mtf50Compute(img);

	return 0;
}
