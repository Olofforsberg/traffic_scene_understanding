#include "Olofs_stixlar.h"
#include "matrix.h"
#include "cost_function.h"
#include <algorithm>
#include <ros/package.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <pcl_conversions/pcl_conversions.h>


#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace cv;

//functions for displaying stixels
static void drawStixel(cv::Mat& img, const Stixel& stixel, cv::Scalar color)
{
    const int radius = std::max(stixel.width / 2, 1);
    const cv::Point tl(stixel.u - radius, stixel.vT);
    const cv::Point br(stixel.u + radius, stixel.vB);
    cv::rectangle(img, cv::Rect(tl, br), color, -1);
    cv::rectangle(img, cv::Rect(tl, br), cv::Scalar(255, 255, 255), 1);
}

static cv::Scalar classToColor(const int& semantic_class, const cv::Mat& label_colours)
{
    cv::Mat output;
    cv::Mat input(1,1,CV_32FC1,semantic_class);
    input.convertTo(input,CV_8U);
    cv::cvtColor(input.clone(), input, CV_GRAY2BGR);
    cv::LUT(input, label_colours, output);
    return cv::Scalar(output.at<cv::Vec3b>(0)[0],output.at<cv::Vec3b>(0)[1],output.at<cv::Vec3b>(0)[2]);
}



float Cost_c(int& v, int& C1, int& C2)
{
    if(C1 == C2)
        return 0.02f;
    else
        return 0.00f;
}

// denna kod kan användas för att optimera stixlar över djup och klassifiering
//std::array<double,3> depth_model_cost(std::vector<double>& v, std::vector<double>& y, std::array<std::vector<int>, 3>& classes, int& u, int& vT, int& vB, const image_geometry::PinholeCameraModel& cam_model_, cv::Mat& plan)
//{
//    double a = plan.at<float>(0);
//    double b = plan.at<float>(1);
//    double c = plan.at<float>(2);
//    double d = plan.at<float>(3);
//        /* classes(0) contains i-liggande model */
//    if(v.size()==0) {
//        std::array<double,3> cost = {4,4,1};
//        return cost;

//    }
//    else if (v.size()==1) {
//        cv::Point3d p = cam_model_.projectPixelTo3dRay(cv::Point2d(u/0.752941,y.at(0)/0.5));
//        float lambda = -d/(a*p.x+b*p.y+c*p.z);
//        float diff = v.at(0) - lambda;
//        std::array<double,3> cost;
//        if (diff >= 1.0) {
//            cost = {4,2,20};

//        } else {
//            cost = {2,4,20};
//        }
//        return cost;
//    } else {
//        //cout << " liggande model " << endl;
//        /* sky model */
//        std::vector<double> diff_ground;

//       for (int i2 = 0; i2 < y.size(); i2++)
//       {
//           cv::Point3d p = cam_model_.projectPixelTo3dRay(cv::Point2d(u/0.752941,y.at(i2)/0.5));
//           float lambda = -d/(a*p.x+b*p.y+c*p.z);
//           diff_ground.push_back(v.at(i2) - lambda);
//           //cout << " lamda " << lambda << " djup " << v.at(i2) << " diff " << diff_ground.at(i2) << endl;
//       }
//       double sum_g = std::accumulate(diff_ground.begin(), diff_ground.end(), 0.0);
//       double mean_g = sum_g / v.size();

//       std::vector<double> diff_g(diff_ground.size());
//       std::transform(diff_ground.begin(), diff_ground.end(), diff_g.begin(),
//                      std::bind2nd(std::minus<double>(), mean_g));
//       double sq_sum_g = std::inner_product(diff_g.begin(), diff_g.end(), diff_g.begin(), 0.0);
//       double stdev_g = std::sqrt(sq_sum_g / diff_ground.size());
//       //cout << " sky model " << endl;
//       //cout << " liggande model " << mean_g << " " << stdev_g << " " << stdev_g*(vT-vB) << " vB "<< vB << " vT " << vT << endl;

//        /* classes(2) contains i-stående model */
//        double sum_o = std::accumulate(v.begin(), v.end(), 0.0);
//        double mean_o = sum_o / v.size();

//        std::vector<double> diff(v.size());
//        std::transform(v.begin(), v.end(), diff.begin(),
//                       std::bind2nd(std::minus<double>(), mean_o));
//        double sq_sum_o = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
//        double stdev_o = std::sqrt(sq_sum_o / v.size());
//        //cout << " stående model " << mean_o << " " << stdev_o << " " << stdev_o*(vT-vB) << " vB "<< vB << " vT " << vT << endl;

//        double stdev_s = 3.0;
//        std::array<double,3> cost = {stdev_g,stdev_o,stdev_s};
//        return cost;
//    }
//}

cv::Mat stixel_depth(int& u, int& sem_class, const int& vT, const int& vB, std::array<double, 16>& Lidar_column, std::array<double, 16>& y_coords, cv::Mat &plan, const image_geometry::PinholeCameraModel &cam_model_)
{
    if (Lidar_column.empty()){
        return cv::Mat::zeros(vT-vB,1,CV_32FC1);
    }
    else
    {
        std::vector<double> stixel_lidar;
        std::vector<double> y_points;
        auto start = std::lower_bound(y_coords.begin(), y_coords.end(), 512 - vT);
        auto end = std::upper_bound(y_coords.begin(), y_coords.end(), 512 - vB);
        for (auto it = start; it != end; it++) {
                y_points.push_back(*it);
                stixel_lidar.push_back(Lidar_column.at(std::distance(y_coords.begin(), it)));
        }
        if (stixel_lidar.empty()){
            return cv::Mat::zeros(vT-vB,1,CV_32FC1);
        }
        float d;
        cv::Point3d abc(plan.at<float>(0),plan.at<float>(1),plan.at<float>(2));

        cv::Mat depth(vT-vB,1,CV_32FC1);
        if (sem_class == 13)
        {
            return cv::Mat::zeros(vT-vB,1,CV_32FC1);

            // sky and we should not have any depth! if we are rigth
        } // else if måste ändras till de nummer som är liggande kategorier i classifiering, ifall ett byte av klassifierings nätverk, 0 = väg, 1 = trottoar
        else if ((sem_class == 0)||(sem_class == 1)||( sem_class == 9))
        {
            // liggande model
            std::vector<double> y_diff;
            std::vector<double> z_diff;

            for (int i = 0; i < stixel_lidar.size(); i++)
            {
                if (i > 0)
                {
                    z_diff.push_back(stixel_lidar.at(i) - stixel_lidar.at(i-1));
                    y_diff.push_back((y_points.at(i) - y_points.at(i-1)));
                }
            }

            if (stixel_lidar.size() > 1)
            {
                int nr = 0;
                depth = cv::Mat::zeros(vT-vB,1,CV_32FC1);
                for (int v = 0; v < vT-vB; v++)
                {
                    if (512-vT + v < (int) (y_points.at(0)) )
                    {
                        depth.at<float>(v) = stixel_lidar.at(0) - z_diff.at(0)/y_diff.at(0)*(vT - 512 +(int) (y_points.at(0))-v);
                        //cout << " 1v " << v << " vT +v " << 512-vT + v <<  " z/y*v " << (vT - 512 +(int) (y_points.at(0))-v) << " z " << depth.at<float>(v) << endl;
                    }
                    else if (nr == stixel_lidar.size() - 1)
                    {
                        depth.at<float>(v) = stixel_lidar.at(nr) + z_diff.at(nr-1)/y_diff.at(nr-1)*(v-(vT - 512 +(int) (y_points.at(nr))) );
                        //cout << " 4v " << v << " vT +v " << 512-vT + v <<  " z/y*v " << (v-(vT - 512 +(int) (y_points.at(nr))) ) << " z " << depth.at<float>(v) << endl;
                    }
                    else if ( ((int) y_points.at(nr) <= 512-vT + v)&&(512-vT + v < (int) y_points.at(nr+1)) )
                    {
                        depth.at<float>(v) = stixel_lidar.at(nr) + z_diff.at(nr)/y_diff.at(nr)*(v -(vT-512 + (int) y_points.at(nr)));
                        //cout << " 2v " << v << " vT +v " << 512-vT + v <<  " z/y*v " << (v -(vT-512 + (int) y_points.at(nr))) << " z " << depth.at<float>(v) << endl;
                    }
                    else
                    {
                       // cout << " y_points.at(nr) " << (int) y_points.at(nr) << " <= 512-vT + v = " <<  512-vT + v << " <  y_points.at(nr+1) = " << (int) y_points.at(nr+1) << endl;
                        nr += 1;
                            depth.at<float>(v) = stixel_lidar.at(nr);
                    }
                }
            }
            else
            {
                cv::Point3d plane_point = stixel_lidar.at(0)*cam_model_.projectPixelTo3dRay(cv::Point2d(u/0.752941,(y_points.at(0)*2.0f)));
                double d =  plane_point.ddot(abc);
//                cout << " vT " << vT << " vB " << vB << endl;
                 for (int v = 0; v < (int) y_points.at(0) - 1 - 512 + vT; v++)
                 {
                     cv::Point3d u_vec = cam_model_.projectPixelTo3dRay(cv::Point2d(u/0.752941, (v+512-vT)/0.5));
                     //cout << " u_vec " << u_vec << endl;

                     double new_z =   d/(abc.ddot(u_vec));
                     depth.at<float>(v) = new_z;
//                     cout << " liggande model, 1 punkt, z " << new_z << " v img " << (v+512-vT)/0.5 << " depth v " << v << endl;
                 }
                 depth.at<float>( (int) y_points.at(0)-512+vT) = stixel_lidar.at(0);
                 //cout << " liggande model, 1 punkt, z " << stixel_lidar.at(0) << " v img " << (int) y_cord.at(0)*0.5-512+vT << " depth v " << y_cord.at(0)*0.5 << " unik " << endl;
                // cout << " v = y_cord.at(0)+1, vT " << 512 - y_cord.at(0)*0.5+1 << " vT " << vT << endl;
                 for (int v = (int) y_points.at(0)+1 - 512 + vT; v < vT-vB; v++)
                 {
                     cv::Point3d u_vec = cam_model_.projectPixelTo3dRay(cv::Point2d(u/0.752941,(v+512-vT)/0.5));
                     double new_z =  d/(abc.ddot(u_vec));
                     if (new_z > 200)
                        depth.at<float>(v) = 200;
                     else if (new_z < -200 )
                         depth.at<float>(v) = -200;
                     else
                         depth.at<float>(v) = new_z;
                 }


            }

        }
        else
        {
            // stående model
            if(stixel_lidar.size() == 1)
            {
                depth = stixel_lidar.at(0);
            }
            else
            {
                std::sort(stixel_lidar.begin(), stixel_lidar.end());
                 if(stixel_lidar.size() % 2 == 0)
                         d = (stixel_lidar.at(stixel_lidar.size()/2 - 1) + stixel_lidar.at(stixel_lidar.size()/2)) / 2;
                 else
                         d = stixel_lidar.at(stixel_lidar.size()/2);
                 depth = d;
//                 cout << " depth " << depth << endl;


            }

        }
        return depth;
    }
}

Olofs_stixlar::Olofs_stixlar(const Parameters& param) : param_(param)
{
}

std::vector<Stixel> Olofs_stixlar::compute(std::vector<cv::Point3d>& djup, std::vector<uint16_t>& ring_nr, cv::Mat& predictions)
{
	const int stixelWidth = param_.stixelWidth;
    const int w = param_.camera.image_width/stixelWidth; // predictions.width() / stixelWidth;
    const int w1 = param_.camera.image_width;
    const int h = param_.camera.image_heigth; // predictions.height();
    const int h_step = param_.optimisation_step;//4;
    cv::Mat plan = param_.trans_plane;
    const image_geometry::PinholeCameraModel cam_model_ = param_.cam_model_;

    std::array<float,19> nr_stixelclasses = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}; //predictions.cols;
    cv::Mat predictions2(predictions.rows,nr_stixelclasses.size(),CV_32FC1);
    for (int i = 0; i < nr_stixelclasses.size();i++)
    {
        predictions.col(nr_stixelclasses.at(i)).copyTo(predictions2.col(i));
    }

    // compute median of classified prediction image
    cv::Mat pred2(w*h, nr_stixelclasses.size(), CV_64FC1);
    for (int u = 0; u < w*h; u++) {
        cv::reduce(predictions2.rowRange(u*stixelWidth,u*stixelWidth+stixelWidth-1),pred2.row(u),0,CV_REDUCE_SUM,CV_64FC1);
    }

    cv::Mat colum_sum(pred2.rows, pred2.cols, CV_64FC1);
    cv::Mat row_min(pred2.rows, pred2.cols, CV_64FC1);
    cv::Mat row_min_pos(pred2.rows, pred2.cols, CV_64FC1);
    cv::Mat over_zero(pred2.rows, pred2.cols, CV_64FC1);

    cv::reduce(pred2, row_min.col(0), 1, CV_REDUCE_MIN,CV_64FC1);
    for (int i = 1; i < pred2.cols; i++) {
        row_min.col(0).copyTo(row_min.col(i));
    }

    row_min_pos = cv::abs(row_min);
    cv::add(pred2,row_min_pos,over_zero);
    cv::reduce(over_zero, colum_sum.col(0),1,CV_REDUCE_SUM,CV_64FC1);
    for (int i = 1; i < colum_sum.cols; i++) {
        colum_sum.col(0).copyTo(colum_sum.col(i));
    }
    cv::Mat pred3(pred2.rows,pred2.cols,CV_64FC1);
    cv::divide(over_zero,colum_sum,pred3);
    std::cout << "pred3 " << endl;

    cv::Mat sumofheigth(w*h, pred3.cols, CV_64FC1);
    for (int u = 0; u < w; u++) {
        for (int i = 0; i < h; i++) {
                if (i == 0) {
                    pred3.row(u).copyTo(sumofheigth.row(h*u));
                    }
                else {
                    cv::Mat hm(2,pred3.cols,CV_64FC1);
                    sumofheigth.row(i-1+u*h).copyTo(hm.row(0));
                    pred3.row(w*i+u).copyTo(hm.row(1));
                    cv::reduce(hm, sumofheigth.row(i+u*h),0,CV_REDUCE_SUM,CV_64FC1);
                    }
        }
    }

    std::vector<cv::Point3d>::iterator it = djup.begin();
    std::vector<uint16_t>::iterator it2 = ring_nr.begin();

    //första listan innehåller 128 listor för vardera stixel column, för vardera column finns en lista för medelvärdet av lidarns djup, 16 värden för lidarns 16 scan linjer.
    std::array<std::array<std::vector<double>, 16>, 128> Lidar_colum;
    std::array<std::array<std::vector<double>, 16>, 128> ring_y_coord;
    std::array<std::array<double, 16>, 128> Lidar_colum_medel;
    std::array<std::array<double, 16>, 128> ring_y_coord_medel;

    for(; it != djup.end(); ++it, ++it2)
    {
        Lidar_colum.at((int) it->x/stixelWidth).at(15-*it2).push_back(it->z);
        ring_y_coord.at((int) it->x/stixelWidth).at(15-*it2).push_back(it->y);
    }
    for(int j = 0; j < 128; j++)
    {
        for(int i = 0; i < 16; i++)
        {
            if (Lidar_colum.at(j).at(i).size() == 0){
                if (j==0) {
                    Lidar_colum_medel.at(j).at(i) = Lidar_colum_medel.at(j+1).at(i);
                }
                else{
                    Lidar_colum_medel.at(j).at(i) = Lidar_colum_medel.at(j-1).at(i);
                }
            }
            else{
                Lidar_colum_medel.at(j).at(i) = std::accumulate( Lidar_colum.at(j).at(i).begin(), Lidar_colum.at(j).at(i).end(), 0.0)/Lidar_colum.at(j).at(i).size();
            }
            if (ring_y_coord.at(j).at(i).size() == 0){
                if (j==0) {
                    ring_y_coord_medel.at(j).at(i) = ring_y_coord_medel.at(j+1).at(i);
                }
                else{
                    ring_y_coord_medel.at(j).at(i) = ring_y_coord_medel.at(j-1).at(i);
                }
            }
            else{
                ring_y_coord_medel.at(j).at(i) = std::accumulate( ring_y_coord.at(j).at(i).begin(), ring_y_coord.at(j).at(i).end(), 0.0)/ring_y_coord.at(j).at(i).size();
            }
        }
    }

	// cost table
    Matrixf costTable(w1, h/h_step, nr_stixelclasses.size());
    Matrix<cv::Point> indexTable(w1, h/h_step, nr_stixelclasses.size());

    cout << " stixel optimization started " << endl;

	// process each column
    int u;
#pragma omp parallel for
    for (u = 0; u < w; u++)
    {
		//////////////////////////////////////////////////////////////////////////////
		// compute cost tables
        //////////////////////////////////////////////////////////////////////////////
        for (int vT = 0; vT < h; vT+= h_step)
        {
            // initialize minimum costs
            std::array<double, nr_stixelclasses.size()> minCost;
            std::array<cv::Point, nr_stixelclasses.size()> minPose;
            for (int i = 0; i < nr_stixelclasses.size(); i++)
            {
                minCost.at(i) = 0.0f;
                minPose.at(i) = cv::Point(i,0);
            }

            for (int vB = 1; vB <= vT; vB+=h_step)
            {
//				// compute data terms costs
                std::array<double, nr_stixelclasses.size()> dataCost;

                for (int i = 0; i < nr_stixelclasses.size(); i++)
                {
                    dataCost.at(i) = - ( - sumofheigth.at<double>(u*h + h-vT, i) + sumofheigth.at<double>(u*h + h-vB + 1, i)); //+ depth_cost.at(depth_model);
                }

                cv::Mat cost(nr_stixelclasses.size(),nr_stixelclasses.size(),CV_64FC1);
#define UPDATE_COST(C1, C2) \
                cost.at<double>(C1,C2) = dataCost.at(C1)  + Cost_c(vB, C1, C2)+ costTable(u, vB/h_step - 1, C2); \
                if (cost.at<double>(C1,C2) < minCost.at(C1)) \
				{ \
                    minCost.at(C1) = cost.at<double>(C1,C2); \
                    minPose.at(C1) = cv::Point(C2, vB - 1); \
				} \

                for (int i = 0; i < nr_stixelclasses.size(); i++)
                {
                    for (int j = 0; j < nr_stixelclasses.size(); j++)
                    {
                        UPDATE_COST(i, j);
                    }
                }

            }
            for (int i = 0; i < nr_stixelclasses.size(); i++)
            {
                costTable(u, vT/h_step, i) = minCost.at(i);
                indexTable(u, vT/h_step, i) = minPose.at(i);
            }

        }
    }
    std::cout << " backtracking " << std::endl;

	//////////////////////////////////////////////////////////////////////////////
	// backtracking step
	//////////////////////////////////////////////////////////////////////////////
    std::vector<Stixel> stixels;
    for (int u = 0; u < w; u++)
    {
        float minCost = std::numeric_limits<float>::max();
        cv::Point minPos;
        for (int c = 0; c < nr_stixelclasses.size(); c++)
        {
            const float cost = costTable(u, h/h_step - 1, c);
            if (cost < minCost)
            {
                minCost = cost;
                minPos = cv::Point(c, h - 1);
            }
        }

        while (minPos.y > 0)
        {
            const cv::Point p1 = minPos;
            const cv::Point p2 = indexTable(u, p1.y/h_step, p1.x);
            Stixel stixel;
            stixel.u = stixelWidth * u + stixelWidth / 2;
            stixel.vT = h  - p1.y;
            stixel.vB = h  - (p2.y + 1);
            stixel.width = stixelWidth;
            stixel.sem_class = p1.x;
            stixel.depth = stixel_depth(u,stixel.sem_class, p1.y , p2.y + 1 ,Lidar_colum_medel.at(u), ring_y_coord_medel.at(u), plan, cam_model_);
            stixels.push_back(stixel);
            minPos = p2;
        }
    }
    return stixels;
}

//function to compute a colored pointcloud from stixels
void Olofs_stixlar::compute_pointcloud(const std::vector<Stixel> &stixels, sensor_msgs::PointCloud2 &pc, cv::Mat &draw)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_pcl_pc (new pcl::PointCloud<pcl::PointXYZRGB>);
    new_pcl_pc->header.frame_id = pc.header.frame_id;
    cv::Mat label_colours = cv::imread(param_.LUT_file,1);
    for (const auto& stixel : stixels)
    {
//        cv::Mat stixelImg = cv::Mat::zeros(draw.size(), draw.type());
        cv::Scalar color = classToColor( stixel.sem_class, label_colours );
//        drawStixel(stixelImg, stixel, color);
//        std::cout << " depth " << stixel.depth << std::endl;
//        cv::imshow(" stixelImg bfa" , stixelImg);
//        cv::waitKey(0);
        for (int v = stixel.vT; v < stixel.vB; v ++)
        {
            cv::Point3d vec = stixel.depth.at<float>(v-stixel.vT)*param_.cam_model_.projectPixelTo3dRay(cv::Point2d(stixel.u/param_.scale_x, v/param_.scale_y));
            if (vec != cv::Point3d(0,0,0) )
            {
                double delta = param_.cam_model_.getDeltaX(1, stixel.depth.at<float>(v-stixel.vT));
                for (int dx = - param_.stixelWidth/2; dx < param_.stixelWidth/2; dx++)
                {
                    pcl::PointXYZRGB point;
                    point.b = color(0), point.g = color(1), point.r = color(2);
                    point.x = vec.x+dx*delta, point.y = vec.y, point.z = vec.z;
                    if(std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
                    {
                        new_pcl_pc->push_back(point);
                    }
                    else
                    {
                        std::cout << " point in infinity! " << std::endl;
                    }
                }
            }
        }
    }
    pcl::toROSMsg(*new_pcl_pc, pc);
}

void Olofs_stixlar::stixels_in_image(cv::Mat &draw, const std::vector<Stixel> &stixels)
{
    cv::Mat stixelImg = cv::Mat::zeros(draw.size(), draw.type());
    cv::Mat label_colours = cv::imread(param_.LUT_file,1);
    cv::cvtColor( label_colours, label_colours, CV_RGB2BGR );
    for (const auto& stixel : stixels)
    {
//        std::cout << " stixel class " << stixel.sem_class << std::endl;

        cv::Scalar color = classToColor(stixel.sem_class,label_colours);
        drawStixel(stixelImg, stixel, color);
//        cv::imshow(" stixelImg bfa" , stixelImg);
//        cv::waitKey(0);
    }
    draw = draw + 0.5 * stixelImg;
}

void Olofs_stixlar::change_params(Parameters &new_params)
{
    this->param_ = new_params;
}

void Olofs_stixlar::fix_cam_model(const sensor_msgs::CameraInfoConstPtr& cam_info)
{
    this->param_.cam_model_.fromCameraInfo(cam_info);
}


