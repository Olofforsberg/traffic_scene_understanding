#ifndef __MULTILAYER_STIXEL_WORLD_H__
#define __MULTILAYER_STIXEL_WORLD_H__

#include <opencv2/opencv.hpp>
#include <pcl/range_image/range_image_planar.h>
#include <image_geometry/pinhole_camera_model.h>
#include <caffe/caffe.hpp>





struct Stixel
{
	int u;
	int vT;
	int vB;
	int width;
	float disp;
};

class MultiLayerStixelWrold
{
public:

	struct CameraParameters
	{
		float fu;
		float fv;
		float u0;
		float v0;
		float baseline;
		float height;
		float tilt;
        float image_width;
        float image_heigth;
        float pixel_length_x;
        float pixel_length_y;


		// default settings
		CameraParameters()
		{
			fu = 1.f;
			fv = 1.f;
			u0 = 0.f;
			v0 = 0.f;
			baseline = 0.2f;
			height = 1.f;
			tilt = 0.f;
		}
	};


	struct Parameters
	{
		// stixel width
		int stixelWidth;

        // depth range
		float dmin;
		float dmax;

		// disparity measurement uncertainty
		float sigmaG;
		float sigmaO;
		float sigmaS;

		// outlier rate
		float pOutG;
		float pOutO;
		float pOutS;

		// probability of invalid disparity
		float pInvG;
		float pInvO;
		float pInvS;

		// probability for regularization
		float pOrd;
		float pGrav;
		float pBlg;

		float deltaz;
		float eps;

        // a*x+b*y+c*z+d = 0
        // this is the equation params to describe a flat ground surface
        float a;
        float b;
        float c;
        float d;

		// camera parameters
		CameraParameters camera;

		// default settings
		Parameters()
		{
			// stixel width
			stixelWidth = 7;

			// disparity range
			dmin = 0;
            dmax = 100;

			// disparity measurement uncertainty
            sigmaG = 0.01f;//1.5f;
            sigmaO = 0.01f;//1.5f;
            sigmaS = 0.01f;//1.2f;

			// outlier rate
            pOutG = 0.05; //0.15f;
            pOutO = 0.05; //0.15f;
            pOutS = 0.05; //0.4f;

			// probability of invalid disparity
            pInvG = 1.f/3.f;//0.34f;
            pInvO = 1.f/3.f;//0.3f;
            pInvS = 1.f/3.f;//0.36f;

			// probability for regularization
			pOrd = 0.1f;
			pGrav = 0.1f;
			pBlg = 0.001f;

			deltaz = 3.f;
			eps = 1.f;

            a = 0.0f;
            b = 0.0f;
            c = 0.0f;
            d = 0.0f;

			// camera parameters
			camera = CameraParameters();
		}
	};

	MultiLayerStixelWrold(const Parameters& param);

    std::vector<Stixel> compute(const cv::Mat &djup, cv::Mat &predictions, const image_geometry::PinholeCameraModel &cam_model_);

private:

	Parameters param_;
};

#endif // !__STIXEL_WORLD_H__
