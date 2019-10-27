
/**
 * This code is based on Yani Ioannou example Difference of Normals Example for PCL Segmentation Tutorials.
 * @file don_segmentation.cpp
 * With several modification to be used as a class to segment pointclouds into vertical/obstacle and horisonal/ground
 */
#include <segment_pointcloud.h>
#include <pcl/visualization/pcl_visualizer.h>


using namespace pcl;
using namespace std;


Segment_pointcloud::Segment_pointcloud(const double& scale1, const double& scale2, const double& threshold, const double& segradius, const sensor_msgs::PointCloud2ConstPtr& msg_pc)
{
    this->scale1 = scale1;            // small scale
    this->scale2 = scale2;            // large scale
    this->threshold = threshold;      // threshold for DoN magnitude
    this->segradius = segradius;      // threshold for radius segmentation
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Convert cloud from sensor_msgs/poincloud2 to pcl type
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*msg_pc,pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2,*cloud);

  // Create a search tree, use KDTreee for non-organized data.
  pcl::search::Search<PointXYZ>::Ptr tree;
  if (cloud->isOrganized ())
  {
    tree.reset (new pcl::search::OrganizedNeighbor<PointXYZ> ());
  }
  else
  {
    tree.reset (new pcl::search::KdTree<PointXYZ> (false));
  }

  // Set the input pointcloud for the search tree
  tree->setInputCloud (cloud);

  if (scale1 >= scale2)
  {
    cerr << "Error: Large scale must be > small scale!" << endl;
    exit (EXIT_FAILURE);
  }

  // Compute normals using both small and large scales at each point
  pcl::NormalEstimationOMP<PointXYZ, PointNormal> ne;
  ne.setInputCloud (cloud);
  ne.setSearchMethod (tree);

  /**
   * NOTE: setting viewpoint is very important, so that we can ensure
   * normals are all pointed in the same direction!
   */
  ne.setViewPoint (std::numeric_limits<float>::max (), std::numeric_limits<float>::max (), std::numeric_limits<float>::max ());

  // calculate normals with the small scale
  cout << "Calculating normals for scale..." << scale1 << endl;
  pcl::PointCloud<PointNormal>::Ptr normals_small_scale (new pcl::PointCloud<PointNormal>);

  ne.setRadiusSearch (scale1);
  ne.compute (*normals_small_scale);

  // calculate normals with the large scale
  cout << "Calculating normals for scale..." << scale2 << endl;
  pcl::PointCloud<PointNormal>::Ptr normals_large_scale (new pcl::PointCloud<PointNormal>);

  ne.setRadiusSearch (scale2);
  ne.compute (*normals_large_scale);

  // Create output cloud for DoN results
  PointCloud<PointNormal>::Ptr doncloud (new pcl::PointCloud<PointNormal>);
  copyPointCloud<PointXYZ, PointNormal>(*cloud, *doncloud);

  cout << "Calculating DoN... " << endl;
  // Create DoN operator
  pcl::DifferenceOfNormalsEstimation<PointXYZ, PointNormal, PointNormal> don;
  don.setInputCloud (cloud);
  don.setNormalScaleLarge (normals_large_scale);
  don.setNormalScaleSmall (normals_small_scale);

  if (!don.initCompute ())
  {
    std::cerr << "Error: Could not intialize DoN feature operator" << std::endl;
    exit (EXIT_FAILURE);
  }

  // Compute DoN
  don.computeFeature (*doncloud);

  // Save DoN features
//  pcl::PCDWriter writer;
//  writer.write<pcl::PointNormal> ("don.pcd", *doncloud, false);

  // Filter by magnitude
  cout << "Filtering out DoN mag <= " << threshold << "..." << endl;

  // Build the condition for filtering
  pcl::ConditionOr<PointNormal>::Ptr range_cond (
    new pcl::ConditionOr<PointNormal> ()
    );
  range_cond->addComparison (pcl::FieldComparison<PointNormal>::ConstPtr (
                               new pcl::FieldComparison<PointNormal> ("curvature", pcl::ComparisonOps::GT, threshold))
                             );
  // Build the filter
  pcl::ConditionalRemoval<PointNormal> condrem (range_cond);
  condrem.setInputCloud (doncloud);

  pcl::PointCloud<PointNormal>::Ptr doncloud_filtered (new pcl::PointCloud<PointNormal>);

  // Apply filter
  condrem.filter (*doncloud_filtered);

  doncloud = doncloud_filtered;

  // Save filtered output
  std::cout << "Filtered Pointcloud: " << doncloud->points.size () << " data points." << std::endl;

  //writer.write<pcl::PointNormal> ("don_filtered.pcd", *doncloud, false);

  // Filter by magnitude
  cout << "Clustering using EuclideanClusterExtraction with tolerance <= " << segradius << "..." << endl;

  pcl::search::KdTree<PointNormal>::Ptr segtree (new pcl::search::KdTree<PointNormal>);
  segtree->setInputCloud (doncloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointNormal> ec;

  ec.setClusterTolerance (segradius);
  ec.setMinClusterSize (2);
  ec.setMaxClusterSize (2);
  ec.setSearchMethod (segtree);
  ec.setInputCloud (doncloud);
  ec.extract (cluster_indices);

  boost::shared_ptr< ::pcl::visualization::PCLVisualizer > viewer(new ::pcl::visualization::PCLVisualizer("3D Viewer"));

  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud< PointNormal>(doncloud, "sample cloud");
  viewer->setPointCloudRenderingProperties( ::pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem(1);
  //viewer->initCameraParameters();
  while (!viewer->wasStopped())
  {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }

  int j = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it, j++)
  {
    pcl::PointCloud<PointNormal>::Ptr cloud_cluster_don (new pcl::PointCloud<PointNormal>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    {
      cloud_cluster_don->points.push_back (doncloud->points[*pit]);
    }

    cloud_cluster_don->width = int (cloud_cluster_don->points.size ());
    cloud_cluster_don->height = 1;
    cloud_cluster_don->is_dense = true;

    //Save cluster
//    cout << "PointCloud representing the Cluster: " << cloud_cluster_don->points.size () << " data points." << std::endl;
//    stringstream ss;
//    ss << "don_cluster_" << j << ".pcd";
//    writer.write<pcl::PointNormal> (ss.str (), *cloud_cluster_don, false);

        boost::shared_ptr< ::pcl::visualization::PCLVisualizer > viewer(new ::pcl::visualization::PCLVisualizer("3D Viewer"));

        viewer->setBackgroundColor(0, 0, 0);
        viewer->addPointCloud< PointNormal >(cloud_cluster_don, "sample cloud");
        viewer->setPointCloudRenderingProperties( ::pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
        viewer->addCoordinateSystem(1);
        //viewer->initCameraParameters();
        while (!viewer->wasStopped())
        {
          viewer->spinOnce(100);
          boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }


  }
}

