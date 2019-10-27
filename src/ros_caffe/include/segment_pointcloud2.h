#include <string>

#include <sensor_msgs/PointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>

#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/model_outlier_removal.h>
#include <pcl/range_image/range_image_planar.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>



class Segment_pointcloud
{
public:
    Segment_pointcloud(const double& scale1, const double& scale2, const double& threshold, const double& segradius, const sensor_msgs::PointCloud2 msg_pc, double plane_x, double plane_y, double plane_z, double plane_d, double focal_x, double focal_y);

    ///The smallest scale to use in the DoN filter.
    double scale1;

    ///The largest scale to use in the DoN filter.
    double scale2;

    ///The minimum DoN magnitude to threshold by
    double threshold;

    ///segment scene into clusters with given distance tolerance using euclidean clustering
    double segradius;

    /// RangeImage to use in stixel optimization

    //boost::shared_ptr<pcl::RangeImagePlanar> range_image_ptr(new pcl::RangeImagePlanar);

    pcl::RangeImagePlanar range_image;
    pcl::RangeImagePlanar range_image_plane;

    //pcl::RangeImagePlanar rangeImage;
    ///pointcloud to segment
    //pcl::PointCloud<pcl::PointXYZ>::Ptr *cloud;
    //pcl::PointCloud<PointXYZ>::Ptr cloud(new pcl::PointCloud<PointXYZ>);



//private:


};

