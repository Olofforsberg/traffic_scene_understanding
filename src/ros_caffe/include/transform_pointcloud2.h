#include <pcl_conversions/pcl_conversions.h>



void
   transformPointCloud2 (const Eigen::Matrix4f &transform, const sensor_msgs::PointCloud2 &in, sensor_msgs::PointCloud2 &out)
   {
     // Get X-Y-Z indices
     int x_idx = pcl::getFieldIndex (in, "x");
     int y_idx = pcl::getFieldIndex (in, "y");
     int z_idx = pcl::getFieldIndex (in, "z");

     if (x_idx == -1 || y_idx == -1 || z_idx == -1)
     {
       ROS_ERROR ("Input dataset has no X-Y-Z coordinates! Cannot convert to Eigen format.");
       return;
     }

     if (in.fields[x_idx].datatype != sensor_msgs::PointField::FLOAT32 ||
         in.fields[y_idx].datatype != sensor_msgs::PointField::FLOAT32 ||
         in.fields[z_idx].datatype != sensor_msgs::PointField::FLOAT32)
     {
       ROS_ERROR ("X-Y-Z coordinates not floats. Currently only floats are supported.");
       return;
     }

     // Check if distance is available
     int dist_idx = pcl::getFieldIndex (in, "distance");

     // Copy the other data
     if (&in != &out)
     {
       out.header = in.header;
       out.height = in.height;
       out.width  = in.width;
       out.fields = in.fields;
       out.is_bigendian = in.is_bigendian;
       out.point_step   = in.point_step;
       out.row_step     = in.row_step;
       out.is_dense     = in.is_dense;
       out.data.resize (in.data.size ());
       // Copy everything as it's faster than copying individual elements
       memcpy (&out.data[0], &in.data[0], in.data.size ());
     }

     Eigen::Array4i xyz_offset (in.fields[x_idx].offset, in.fields[y_idx].offset, in.fields[z_idx].offset, 0);

     for (size_t i = 0; i < in.width * in.height; ++i)
     {
       Eigen::Vector4f pt (*(float*)&in.data[xyz_offset[0]], *(float*)&in.data[xyz_offset[1]], *(float*)&in.data[xyz_offset[2]], 1);
       Eigen::Vector4f pt_out;

       bool max_range_point = false;
       int distance_ptr_offset = i*in.point_step + in.fields[dist_idx].offset;
       float* distance_ptr = (dist_idx < 0 ? NULL : (float*)(&in.data[distance_ptr_offset]));
       if (!std::isfinite (pt[0]) || !std::isfinite (pt[1]) || !std::isfinite (pt[2]))
       {
         if (distance_ptr==NULL || !std::isfinite(*distance_ptr))  // Invalid point
         {
           pt_out = pt;
         }
         else  // max range point
         {
           pt[0] = *distance_ptr;  // Replace x with the x value saved in distance
           pt_out = transform * pt;
           max_range_point = true;
           //std::cout << pt[0]<<","<<pt[1]<<","<<pt[2]<<" => "<<pt_out[0]<<","<<pt_out[1]<<","<<pt_out[2]<<"\n";
         }
       }
       else
       {
         pt_out = transform * pt;
       }

       if (max_range_point)
       {
         // Save x value in distance again
         *(float*)(&out.data[distance_ptr_offset]) = pt_out[0];
         pt_out[0] = std::numeric_limits<float>::quiet_NaN();
       }

       memcpy (&out.data[xyz_offset[0]], &pt_out[0], sizeof (float));
       memcpy (&out.data[xyz_offset[1]], &pt_out[1], sizeof (float));
      memcpy (&out.data[xyz_offset[2]], &pt_out[2], sizeof (float));


       xyz_offset += in.point_step;
     }

     // Check if the viewpoint information is present
     int vp_idx = pcl::getFieldIndex (in, "vp_x");
     if (vp_idx != -1)
     {
       // Transform the viewpoint info too
       for (size_t i = 0; i < out.width * out.height; ++i)
       {
         float *pstep = (float*)&out.data[i * out.point_step + out.fields[vp_idx].offset];
         // Assume vp_x, vp_y, vp_z are consecutive
         Eigen::Vector4f vp_in (pstep[0], pstep[1], pstep[2], 1);
         Eigen::Vector4f vp_out = transform * vp_in;

         pstep[0] = vp_out[0];
         pstep[1] = vp_out[1];
         pstep[2] = vp_out[2];
       }
     }
   }
