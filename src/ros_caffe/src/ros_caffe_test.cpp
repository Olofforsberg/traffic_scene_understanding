#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <cv_bridge/cv_bridge.h>

#include "test_segmentation.h"
#include "Olofs_stixlar.h"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <camera_info_manager/camera_info_manager.h>

#include <tf/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

#include <algorithm>
#include "matrix.h"

#include <segment_pointcloud2.h>

 #include <pcl/visualization/cloud_viewer.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>


#include <visualization_msgs/Marker.h>

//some global variables and shit
// if some smart ass thinks OH NOW global varialbles are bad bla bla...
// all these global variables are given values within tha main function and are addressed in the callback
// exept one -> param_ which is given new values within the callback.

//subscription and pub topics
std::string CAMERA_FRAME_TOPIC;
std::string CAMERA_INFO_TOPIC;
std::string VELODYNE_TOPIC;
std::string PUBLISH_TOPIC;

//FCN-network for classification
Classifier* classifier;
std::string model_path;
std::string weights_path;
std::string image_path;
std::string LUT_file;

//publish cool shit
image_transport::Publisher pub, pub2;
ros::Publisher pub_pc, pub_pc2, marker_pub;

using namespace ros;
using namespace message_filters;

//the big callback everything important is done within
void Callback(const sensor_msgs::ImageConstPtr& msg_img, const sensor_msgs::CameraInfoConstPtr& msg_info,
              const sensor_msgs::PointCloud2ConstPtr& msg_pc) {
    ROS_INFO_STREAM("GOOOO CALLBACK");
    //push image through FCN-network -> classify image
    cv::Mat img, predictions, pred_channels, pred_img, img_resized;
    //caffe::Blob<float> pred_channels(1,19,512,1024);
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg_img, "bgr8");
        img = cv_ptr->image;
        classifier->Predict(img,LUT_file, predictions, pred_channels);
        pred_img = classifier->Visualization(predictions, LUT_file);
        cv::resize(img, img_resized, pred_img.size());

        sensor_msgs::ImagePtr msg_img_classed = cv_bridge::CvImage(std_msgs::Header(), "bgr8", pred_img).toImageMsg();
        sensor_msgs::ImagePtr msg_img_not_c = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();

        pub.publish(msg_img_classed);
        pub2.publish(msg_img_not_c);
        ROS_INFO_STREAM("published new classified image");
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg_img->encoding.c_str());
    }

//    //transform LIDAR - pointcloud to camera frame with tf
    tf::StampedTransform trans;
    geometry_msgs::TransformStamped transform;
    sensor_msgs::PointCloud2 cloud_camframe;
    tf::TransformListener listener;
   // try{
    listener.waitForTransform(msg_img->header.frame_id, msg_pc->header.frame_id, msg_img->header.stamp,ros::Duration(2));
      //                      target                  source
    listener.lookupTransform(msg_img->header.frame_id,msg_pc->header.frame_id,ros::Time(0),trans);
   // }

    //tf2 transforms sensor_msgs/pointcloud2 need to repack the tf1-transformation to the type tf2 uses to transform, stupid indeed!
    transform.child_frame_id = trans.child_frame_id_;
    transform.transform.rotation.x = trans.getRotation().getX();
    transform.transform.rotation.y = trans.getRotation().getY();
    transform.transform.rotation.z = trans.getRotation().getZ();
    transform.transform.rotation.w = trans.getRotation().getW();
    transform.transform.translation.x = trans.getOrigin().getX();
    transform.transform.translation.y = trans.getOrigin().getY();
    transform.transform.translation.z = trans.getOrigin().getZ();
    transform.header.frame_id = trans.frame_id_;
    transform.header.stamp = trans.stamp_;
    tf2::doTransform(*msg_pc,cloud_camframe,transform);

    ROS_INFO_STREAM("transform pointcloud done");

    //projecera LIDAR - pointcloud into image
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_camframe, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_camframe, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_camframe, "z");
    sensor_msgs::PointCloud2Iterator<uint16_t> iter_r(cloud_camframe, "ring");

    image_geometry::PinholeCameraModel cam_model_;
    cam_model_.fromCameraInfo(msg_info);
    cv::Rect frame(cam_model_.Tx(),cam_model_.Ty(), msg_img.get()->width, msg_img.get()->height);
    std::vector<cv::Point3d> xyz_cords;
    std::vector<cv::Point3d> uvz_cords;
    std::vector<uint16_t> ring_info;
  //  static const int RADIUS = 1;

    for(; iter_x !=iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_r) {
       if(*iter_z>0) {
       cv::Point3d pt_cv(*iter_x, *iter_y, *iter_z);
       cv::Point2d uv;
       uv = cam_model_.project3dToPixel(pt_cv);
       if(uv.inside(frame)) {
           cv::Point3d image_range_point(uv.x*classifier->scale_x, uv.y*classifier->scale_y, pt_cv.z);
           uvz_cords.push_back(image_range_point);
           xyz_cords.push_back(pt_cv);
           ring_info.push_back(*iter_r);
           }
       }
    }

    //transformera normal of a plane to cameras coordinate system
    //tf::TransformListener listener;
    double normal_d = 0.6;  //-1.6;
    tf::StampedTransform plane_trans;
    ROS_INFO_STREAM(" msg_img->header.frame_id ");
    ROS_INFO_STREAM(msg_img->header.frame_id);

    listener.lookupTransform(msg_img->header.frame_id, "base_link", ros::Time(1), plane_trans);
    cv::Mat plane_normal = cv::Mat::zeros(4, 1, CV_32FC1);
    plane_normal.at<float>(2,0) = 1.0f;
    plane_normal.at<float>(3,0) = normal_d;
    cv::Mat transform_matrix(4, 4, CV_32FC1);
    cv::Mat transl(1, 3, CV_32FC1);
    cv::Mat rot_cv(3, 3, CV_32FC1);
    tf::Vector3 translation = plane_trans.getOrigin();
    tf::Quaternion rot = plane_trans.getRotation();
    tf::Matrix3x3 mat(rot);
    transl.at<float>(0,0) = translation.getX();
    transl.at<float>(0,1) = translation.getY();
    transl.at<float>(0,2) = translation.getZ();
    rot_cv.at<float>(0,0) = mat.getRow(0).getX();
    rot_cv.at<float>(0,1) = mat.getRow(0).getY();
    rot_cv.at<float>(0,2) = mat.getRow(0).getZ();
    rot_cv.at<float>(1,0) = mat.getRow(1).getX();
    rot_cv.at<float>(1,1) = mat.getRow(1).getY();
    rot_cv.at<float>(1,2) = mat.getRow(1).getZ();
    rot_cv.at<float>(2,0) = mat.getRow(2).getX();
    rot_cv.at<float>(2,1) = mat.getRow(2).getY();
    rot_cv.at<float>(2,2) = mat.getRow(2).getZ();
    cv::Mat M(1, 3, CV_32FC1);
    M = -transl*rot_cv;
    transform_matrix.at<float>(0,0) = mat.getRow(0).getX();
    transform_matrix.at<float>(0,1) = mat.getRow(0).getY();
    transform_matrix.at<float>(0,2) = mat.getRow(0).getZ();
    transform_matrix.at<float>(1,0) = mat.getRow(1).getX();
    transform_matrix.at<float>(1,1) = mat.getRow(1).getY();
    transform_matrix.at<float>(1,2) = mat.getRow(1).getZ();
    transform_matrix.at<float>(2,0) = mat.getRow(2).getX();
    transform_matrix.at<float>(2,1) = mat.getRow(2).getY();
    transform_matrix.at<float>(2,2) = mat.getRow(2).getZ();
    transform_matrix.at<float>(0,3) = 0.0f;
    transform_matrix.at<float>(1,3) = 0.0f;
    transform_matrix.at<float>(2,3) = 0.0f;
    transform_matrix.at<float>(3,3) = 1.0f;
    transform_matrix.at<float>(3,0) = M.at<float>(0,0);
    transform_matrix.at<float>(3,1) = M.at<float>(0,1);
    transform_matrix.at<float>(3,2) = M.at<float>(0,2);
    cv::Mat trans_plane;
    trans_plane = transform_matrix*plane_normal;

    //stixel stuff
    // Stixel class
    Olofs_stixlar::Parameters param_;
    param_.camera.u0 = 0.0f;
    param_.camera.v0 = 0.0f;
    param_.optimisation_step = 1;
    param_.stixelWidth = 8;
    param_.LUT_file = LUT_file;
    param_.camera.image_heigth = msg_img->height*classifier->scale_y;
    param_.camera.image_width = msg_img->width*classifier->scale_x;
    param_.a = trans_plane.at<float>(0);
    param_.b = trans_plane.at<float>(1);
    param_.c = trans_plane.at<float>(2);
    param_.d = trans_plane.at<float>(3);
    trans_plane.copyTo(param_.trans_plane);
    param_.scale_x = classifier->scale_x;
    param_.scale_y = classifier->scale_y;
    param_.cam_model_ = cam_model_;
    //stixel_stuff->change_params(*param_);
    ROS_INFO_STREAM( " param done " );

    Olofs_stixlar stixel_stuff(param_);
    ROS_INFO_STREAM( " stixel class done " );

    const std::vector<Stixel> stixels = stixel_stuff.compute(uvz_cords, ring_info, pred_channels);
    ROS_INFO_STREAM("stixlar klara");
    //const auto t2 = std::chrono::system_clock::now();
    //const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    //std::cout << "stixel computation time: " << duration << "[msec]" << std::endl;

    // draw stixels
    cv::Mat draw = img_resized;
    stixel_stuff.stixels_in_image(draw,stixels);
    cv::imshow(" stixels " , draw);

    //compute a colored point cloud from image
    sensor_msgs::PointCloud2 new_pointcloud;
    new_pointcloud.header.frame_id = msg_img->header.frame_id;

    stixel_stuff.compute_pointcloud(stixels, new_pointcloud, draw);

    pub_pc.publish(new_pointcloud);
    pub_pc2.publish(cloud_camframe);

    cv::waitKey(0);


}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ros_caffe_test");
    ros::NodeHandle n;
    image_transport::ImageTransport it(n);

    n.getParam("/ros_caffe/camera_frame_topic", CAMERA_FRAME_TOPIC);
    n.getParam("/ros_caffe/camera_info_topic", CAMERA_INFO_TOPIC);
    n.getParam("/ros_caffe/velodyne_topic", VELODYNE_TOPIC);
    n.getParam("/ros_caffe/publish_topic", PUBLISH_TOPIC);
    ROS_INFO_STREAM(CAMERA_FRAME_TOPIC);
    ROS_INFO_STREAM(CAMERA_INFO_TOPIC);
    ROS_INFO_STREAM(VELODYNE_TOPIC);
    ROS_INFO_STREAM(PUBLISH_TOPIC);

    image_transport::SubscriberFilter image_sub(it,CAMERA_FRAME_TOPIC, 0);
    message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub(n, CAMERA_INFO_TOPIC, 0);
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(n, VELODYNE_TOPIC, 0);

    typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::PointCloud2> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, info_sub, cloud_sub);
    sync.registerCallback(boost::bind(&Callback, _1, _2, _3));

    pub  = it.advertise(PUBLISH_TOPIC,1);
    pub2  = it.advertise(PUBLISH_TOPIC + "2",1);
    pub_pc = n.advertise<sensor_msgs::PointCloud2> ("points2", 1);
    pub_pc2 = n.advertise<sensor_msgs::PointCloud2> ("points3s", 1);

    const std::string ROOT_SAMPLE = ros::package::getPath("ros_caffe");
    model_path = ROOT_SAMPLE +    "/model/enet_deploy_final_unupgrade.prototxt";
    weights_path = ROOT_SAMPLE +  "/model/cityscapes_weights.caffemodel";
    image_path = ROOT_SAMPLE + "/enet.png";
    LUT_file = ROOT_SAMPLE + "/model/cityscapes19.png";

    classifier = new Classifier(model_path,weights_path);

    ROS_INFO_STREAM( " classifier done " );

    ros::spin();
    delete classifier;
    ros::shutdown();
    return 0;
}
