#ifndef __MULTILAYER_STIXEL_WORLD_H__
#define __MULTILAYER_STIXEL_WORLD_H__

#include <opencv2/opencv.hpp>
#include <pcl/range_image/range_image_planar.h>
#include <image_geometry/pinhole_camera_model.h>
#include <caffe/caffe.hpp>
#include <sensor_msgs/PointCloud2.h>

struct Stixel
{
	int u;
	int vT;
	int vB;
	int width;
    int sem_class;
    cv::Mat depth;
};

class Olofs_stixlar
{
public:

	struct CameraParameters
	{
		float u0;
		float v0;
        int image_width;
        int image_heigth;

		// default settings
		CameraParameters()
		{
			u0 = 0.f;
			v0 = 0.f;
            image_width = 1024;
            image_heigth = 512;
		}
	};

	struct Parameters
	{

		// stixel width
		int stixelWidth;

        // camera model for stuff on camera
        image_geometry::PinholeCameraModel cam_model_;

        // a*x+b*y+c*z+d = 0
        // this is the equation params to describe a flat ground surface
        cv::Mat trans_plane;

        float a;
        float b;
        float c;
        float d;

        // paramters of how many pixel step we go in direction of heigth within the dynamical programming
        int optimisation_step;

		// camera parameters
		CameraParameters camera;

        // direction of color look up table, keeps a cool color for every class
        std::string LUT_file;

        //scale of how the deep network changes size of image
        float scale_x;
        float scale_y;

		// default settings
		Parameters()
		{
			// stixel width
            stixelWidth = 8;

            a = 0.0f;
            b = 0.0f;
            c = 0.0f;
            d = 0.0f;

			// camera parameters
			camera = CameraParameters();
		}
	};

    Olofs_stixlar(const Parameters& param);

    std::vector<Stixel> compute(std::vector<cv::Point3d> &djup, std::vector<uint16_t> &ring_nr, cv::Mat &predictions);

    void compute_pointcloud(const std::vector<Stixel> &stixels, sensor_msgs::PointCloud2 &pc, cv::Mat &draw);

    void stixels_in_image(cv::Mat &draw, const std::vector<Stixel> &stixels);

    void change_params(Parameters &new_params);

    void fix_cam_model(const sensor_msgs::CameraInfoConstPtr& cam_info);


private:

	Parameters param_;
};

#endif // !__STIXEL_WORLD_H__
