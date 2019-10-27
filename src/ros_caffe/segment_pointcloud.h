#include <string>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/features/don.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>


class Segment_pointcloud
{
public:
    Segment_pointcloud(const double& scale1, const double& scale2, const double& threshold, const double& segradius, const sensor_msgs::PointCloud2 msg_pc, double plane_x, double plane_y, double plane_z, double plane_d);

    ///The smallest scale to use in the DoN filter.
    double scale1;

    ///The largest scale to use in the DoN filter.
    double scale2;

    ///The minimum DoN magnitude to threshold by
    double threshold;

    ///segment scene into clusters with given distance tolerance using euclidean clustering
    double segradius;

    ///pointcloud to segment
    pcl::PointCloud<pcl::PointXYZ>::Ptr *cloud;
    //pcl::PointCloud<PointXYZ>::Ptr cloud(new pcl::PointCloud<PointXYZ>);



//private:


};

