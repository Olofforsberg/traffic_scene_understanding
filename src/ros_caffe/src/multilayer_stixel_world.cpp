#include "multilayer_stixel_world.h"
#include "matrix.h"
#include "cost_function.h"



#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

bool isZero (int i)
{
    return i == 0;
}

MultiLayerStixelWrold::MultiLayerStixelWrold(const Parameters& param) : param_(param)
{
}

std::vector<Stixel> MultiLayerStixelWrold::compute(const cv::Mat& djup, cv::Mat& predictions, const image_geometry::PinholeCameraModel& cam_model_)
{                              //CV_32F
    //CV_Assert(disparity.type() == CV_32F);

	const int stixelWidth = param_.stixelWidth;
    const int w = 1024/stixelWidth; //predictions.width() / stixelWidth;
    const int h = 512; //predictions.height();
	const int fnmax = static_cast<int>(param_.dmax);

    // compute median of classified prediction image
    Matrixf columns_class(w, h);

    std::vector<cv::Mat> classes(predictions.cols, cv::Mat(w, h, CV_32FC1));

    for (int nr_of_class = 0; nr_of_class < predictions.cols; nr_of_class++)
    {
        cv::Mat current_class(w, h, CV_32FC1);
        for (int v = 0; v < h; v++)
        {
            for (int u = 0; u < w; u++)
            {
                    for (int du = 0; du < stixelWidth; du++)
                    {
                        cv::Mat row = predictions.row(nr_of_class);
                        current_class.at<float>(u,v) += - row.at<float>(v*w*stixelWidth + u*stixelWidth + du);

                    }
            }
        }
        current_class = current_class / stixelWidth;
        classes.at(nr_of_class) = current_class.clone();
    }

//    Matrixf columns_depth(w, h);
//    cv::Mat depth;
//    Matrix<cv::Point> index;
//    for (int v = 0; v < h; v++)
//	{
//		for (int u = 0; u < w; u++)
//		{
//			// compute horizontal median
//			std::vector<float> buf(stixelWidth);
//            std::vector<float> buf_class(stixelWidth);

//			for (int du = 0; du < stixelWidth; du++)
//            {
//                buf[du] = djup.at<float>(v, u * stixelWidth + du);

//            }
//            std::vector<float>::iterator newIter = std::remove_if( buf.begin() , buf.end() , isZero);
//            buf.resize( newIter -  buf.begin());
//            if (buf.empty() == 0) {
//                std::sort(std::begin(buf), std::end(buf));
//                const float m = buf[stixelWidth / 2];
//                columns_depth(u, h - 1 - v) = m;
//                index.push_back(cv::Point(u,v));
//            }
//            else {
//                const float m = 0;
//                columns_depth(u, h - 1 - v) = m;
//                }
//            // In columns we now have reverse order of data so that v = 0 points to the bottom
//		}
//	}

	// get camera parameters
	const CameraParameters& camera = param_.camera;
	const float sinTilt = sinf(camera.tilt);
	const float cosTilt = cosf(camera.tilt);

    // compute expected ground depth
    Matrixf ground_depth(w, h);
	std::vector<float> groundDisparity(h);
    double fx = this->param_.camera.fu;
    double fy = this->param_.camera.fv;
    cv::Mat ground_distance = cv::Mat::zeros(h, w*stixelWidth, CV_32F);

    cv::Mat diff = cv::Mat::zeros(h, w*stixelWidth, CV_32F);
    for (int u = 0; u < w*stixelWidth; u++)
    {
        for (int v = 0; v < h; v++)
        {
                cv::Point3d vector = cam_model_.projectPixelTo3dRay(cv::Point2d(u,v));
                if (columns_depth(u, h - 1 - v) > 0)
                {
                    ground_depth(u, h - 1 - v) =  - this->param_.d/(this->param_.a*vector.x+this->param_.b*vector.y+this->param_.c*vector.z);
                }
                else
                {
                    ground_depth(u, h - 1 - v) = 0;
                }
        }
    }

    const float vhor = this->param_.camera.image_heigth/2.0; //- (fy*this->param_.c) / this->param_.b;
    std::cout << "vhor2: " << vhor << std::endl;

//    for (int v = 0; v < h; v++)
//        groundDisparity[h - 1 - v] = std::max((camera.baseline / camera.height) * (camera.fu * sinTilt + (v - camera.v0) * cosTilt), 0.f);
//	const float vhor = h - 1 - (camera.v0 * cosTilt - camera.fu * sinTilt) / cosTilt;
				
	// create data cost function of each segment
    NegativeLogDataTermGrd dataTermG(param_.dmax, param_.dmin, param_.sigmaG, param_.pOutG, param_.pInvG, ground_depth);
    NegativeLogDataTermObj dataTermO(param_.dmax, param_.dmin, param_.sigmaO, param_.pOutO, param_.pInvO);
	NegativeLogDataTermSky dataTermS(param_.dmax, param_.dmin, param_.sigmaS, param_.pOutS, param_.pInvS);
    std::cout << "data term klar" << std::endl;

	// create prior cost function of each segment
	const int G = NegativeLogPriorTerm::G;
	const int O = NegativeLogPriorTerm::O;
	const int S = NegativeLogPriorTerm::S;
	NegativeLogPriorTerm priorTerm(h, vhor, param_.dmax, param_.dmin, camera.baseline, camera.fu, param_.deltaz,
		param_.eps, param_.pOrd, param_.pGrav, param_.pBlg, groundDisparity);
	
	// data cost LUT
	Matrixf costsG(w, h), costsO(w, h, fnmax), costsS(w, h), sum(w, h);
	Matrixi valid(w, h);

	// cost table
	Matrixf costTable(w, h, 3), dispTable(w, h, 3);
	Matrix<cv::Point> indexTable(w, h, 3);
	
	// process each column
	int u;
#pragma omp parallel for
	for (u = 0; u < w; u++)
	{
		//////////////////////////////////////////////////////////////////////////////
		// pre-computate LUT
		//////////////////////////////////////////////////////////////////////////////
		float tmpSumG = 0.f;
		float tmpSumS = 0.f;
		std::vector<float> tmpSumO(fnmax, 0.f);

		float tmpSum = 0.f;
		int tmpValid = 0;

		for (int v = 0; v < h; v++)
		{
			// measured disparity
            const float d = columns_depth(u, v);

			// pre-computation for ground costs
            if (d != 0) {
                tmpSumG += dataTermG(d, u, v);
            }
			costsG(u, v) = tmpSumG;

			// pre-computation for sky costs
            if (d != 0) {
                tmpSumS += dataTermS(d);
            }
			costsS(u, v) = tmpSumS;
			
			// pre-computation for object costs
			for (int fn = 0; fn < fnmax; fn++)
            {   if (d != 0) {
				tmpSumO[fn] += dataTermO(d, fn);
                }
				costsO(u, v, fn) = tmpSumO[fn];
			}

			// pre-computation for mean disparity of stixel
			if (d >= 0.f)
			{
				tmpSum += d;
				tmpValid++;
			}
			sum(u, v) = tmpSum;
			valid(u, v) = tmpValid;
		}

		//////////////////////////////////////////////////////////////////////////////
		// compute cost tables
		//////////////////////////////////////////////////////////////////////////////
		for (int vT = 0; vT < h; vT++)
		{
			float minCostG, minCostO, minCostS;
			float minDispG, minDispO, minDispS;
			cv::Point minPosG(G, 0), minPosO(O, 0), minPosS(S, 0);

			// process vB = 0
			{
				// compute mean disparity within the range of vB to vT
				const float d1 = sum(u, vT) / std::max(valid(u, vT), 1);
				const int fn = cvRound(d1);

				// initialize minimum costs
				minCostG = costsG(u, vT) + priorTerm.getG0(vT);
				minCostO = costsO(u, vT, fn) + priorTerm.getO0(vT);
				minCostS = costsS(u, vT) + priorTerm.getS0(vT);
				minDispG = minDispO = minDispS = d1;
			}
			
			for (int vB = 1; vB <= vT; vB++)
			{
				// compute mean disparity within the range of vB to vT
				const float d1 = (sum(u, vT) - sum(u, vB - 1)) / std::max(valid(u, vT) - valid(u, vB - 1), 1);
				const int fn = cvRound(d1);

				// compute data terms costs
				const float dataCostG = vT < vhor ? costsG(u, vT) - costsG(u, vB - 1) : N_LOG_0_0;
				const float dataCostO = costsO(u, vT, fn) - costsO(u, vB - 1, fn);
				const float dataCostS = vT < vhor ? N_LOG_0_0 : costsS(u, vT) - costsS(u, vB - 1);

				// compute priors costs and update costs
				const float d2 = dispTable(u, vB - 1, 1);

#define UPDATE_COST(C1, C2) \
				const float cost##C1##C2 = dataCost##C1 + priorTerm.get##C1##C2(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, C2); \
				if (cost##C1##C2 < minCost##C1) \
				{ \
					minCost##C1 = cost##C1##C2; \
					minDisp##C1 = d1; \
					minPos##C1 = cv::Point(C2, vB - 1); \
				} \

				UPDATE_COST(G, G);
				UPDATE_COST(G, O);
				UPDATE_COST(G, S);
				UPDATE_COST(O, G);
				UPDATE_COST(O, O);
				UPDATE_COST(O, S);
				UPDATE_COST(S, G);
				UPDATE_COST(S, O);
				UPDATE_COST(S, S);
			}

			costTable(u, vT, G) = minCostG;
			costTable(u, vT, O) = minCostO;
			costTable(u, vT, S) = minCostS;

			dispTable(u, vT, G) = minDispG;
			dispTable(u, vT, O) = minDispO;
			dispTable(u, vT, S) = minDispS;

			indexTable(u, vT, G) = minPosG;
			indexTable(u, vT, O) = minPosO;
			indexTable(u, vT, S) = minPosS;
		}
	}

	//////////////////////////////////////////////////////////////////////////////
	// backtracking step
	//////////////////////////////////////////////////////////////////////////////
	std::vector<Stixel> stixels;
	for (int u = 0; u < w; u++)
	{
		float minCost = std::numeric_limits<float>::max();
		cv::Point minPos;
		for (int c = 0; c < 3; c++)
		{
			const float cost = costTable(u, h - 1, c);
			if (cost < minCost)
			{
				minCost = cost;
				minPos = cv::Point(c, h - 1);
			}
		}

		while (minPos.y > 0)
		{
			const cv::Point p1 = minPos;
			const cv::Point p2 = indexTable(u, p1.y, p1.x);
			if (p1.x == O) // object
			{
				Stixel stixel;
				stixel.u = stixelWidth * u + stixelWidth / 2;
				stixel.vT = h - 1 - p1.y;
				stixel.vB = h - 1 - (p2.y + 1);
				stixel.width = stixelWidth;
				stixel.disp = dispTable(u, p1.y, p1.x);
				stixels.push_back(stixel);
			}
			minPos = p2;
		}
	}

	return stixels;
}
