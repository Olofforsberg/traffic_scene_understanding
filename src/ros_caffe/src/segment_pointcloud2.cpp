
/**
 * This code is based on Yani Ioannou example Difference of Normals Example for PCL Segmentation Tutorials.
 * @file don_segmentation.cpp
 * With several modification to be used as a class to segment pointclouds into vertical/obstacle and horisonal/ground
 */
#include <segment_pointcloud2.h>
#include <pcl/filters/extract_indices.h>



using namespace pcl;
using namespace std;

void setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f(0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f(0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}


Segment_pointcloud::Segment_pointcloud(const double& scale1, const double& scale2, const double& threshold, const double& segradius, const sensor_msgs::PointCloud2 msg_pc, double normal_x, double normal_y, double normal_z, double normal_d, double focal_x, double focal_y)
{
    this->scale1 = scale1;            // small scale
    this->scale2 = scale2;            // large scale
    this->threshold = threshold;      // threshold for DoN magnitude
    this->segradius = segradius;      // threshold for radius segmentation
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_object(new pcl::PointCloud<pcl::PointXYZ>);



    // Convert cloud from sensor_msgs/poincloud2 to pcl type
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(msg_pc,pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2,*cloud);

    pcl::ModelCoefficients coefficients;
      coefficients.values.resize(4);
      coefficients.values[0] = normal_x;
      coefficients.values[1] = normal_y;
      coefficients.values[2] = normal_z;
      coefficients.values[3] = normal_d;  //1.265

      pcl::ModelOutlierRemoval<pcl::PointXYZ> plane_filter;
        plane_filter.setModelCoefficients (coefficients);
        plane_filter.setThreshold (0.3);
        plane_filter.setModelType (pcl::SACMODEL_PLANE);
        plane_filter.setInputCloud (cloud);
        plane_filter.filter (*cloud_plane);
        //pcl::IndicesConstPtr inliers;
        pcl::IndicesConstPtr outliers;
        //inliers = plane_filter.getIndices();
        outliers = plane_filter.getRemovedIndices();

        //get a pointcloud of the outliers
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        // Extract the outliers
            extract.setInputCloud (cloud);
            extract.setIndices (outliers);
            extract.setNegative (false);
            extract.filter (*cloud_object);

        // We now want to create a range image from the above point cloud, with a 1deg angular resolution
        Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
        pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
        float noiseLevel=1000;
        float minRange = 0.0f;
        boost::shared_ptr<pcl::RangeImagePlanar> range_image_ptr(new pcl::RangeImagePlanar);
        pcl::RangeImagePlanar& rangeImage = *range_image_ptr;
        //pcl::RangeImage rangeImage;
        rangeImage.createFromPointCloudWithFixedSize(*cloud_object, 1360.0f, 1024.0f, 1360.0f/2.0f, 1024.0f/2.0f, focal_x, focal_y,sensorPose, coordinate_frame, noiseLevel, minRange);
        this->range_image = rangeImage;
        rangeImage.createFromPointCloudWithFixedSize(*cloud_plane, 1360.0f, 1024.0f, 1360.0f/2.0f, 1024.0f/2.0f, focal_x, focal_y,sensorPose, coordinate_frame, noiseLevel, minRange);
        this->range_image_plane = rangeImage;

        std::cout << rangeImage << "\n";
        // --------------------------------------------
          // -----Open 3D viewer and add point cloud-----
          // --------------------------------------------
          pcl::visualization::PCLVisualizer viewer ("3D Viewer");
          viewer.setBackgroundColor (1, 1, 1);
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler (range_image_ptr, 0, 0, 0);
          viewer.addPointCloud (range_image_ptr, range_image_color_handler, "range image");
          viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "range image");
          //viewer.addCoordinateSystem (1.0f, "global");
          //PointCloudColorHandlerCustom<PointType> point_cloud_color_handler (point_cloud_ptr, 150, 150, 150);
          //viewer.addPointCloud (point_cloud_ptr, point_cloud_color_handler, "original point cloud");
          viewer.initCameraParameters ();
          setViewerPose(viewer, rangeImage.getTransformationToWorldSystem ());

          // --------------------------
          // -----Show range image-----
          // --------------------------
          pcl::visualization::RangeImageVisualizer range_image_widget ("Range image");
          range_image_widget.showRangeImage (*range_image_ptr);

          //--------------------
          // -----Main loop-----
          //--------------------
          while (!viewer.wasStopped ())
          {
            range_image_widget.spinOnce ();
            viewer.spinOnce ();
            pcl_sleep (0.01);
          }
}

