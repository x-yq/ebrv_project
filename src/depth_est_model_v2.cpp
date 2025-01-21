#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <opencv2/highgui.hpp> 
#include <fast_dynamic/motion_compensate_node.h>
#include <dvs_msgs/Event.h>
#include <numeric>
#include <random>

using namespace cv;
using namespace std;

namespace motion_compensate{
    
void MotionCompensate::initialize_v2(const std::vector<dvs_msgs::Event>& events_subset){

 if(random_initial){
    
    for (int i = 0; i < this->Z.rows; ++i) {
      for (int j = 0; j < this->Z.cols; ++j) {
        if(this->event_count_.at<double>(i, j) < 1.){
          continue;
        }
        else{
          this->Z.at<double>(i, j) = 2.;
        }
      
      }
    }
    if(slice_number == 0){
      this->linear_vel_cam = cv::Vec3f(0.0, 0.0, 0.0);
      this->angular_vel_cam = cv::Vec3f(0.0, 0.0, 0.0);
    }

 }
 else{

  this->Z = getGTDepthMap_v2(events_subset.front().ts.toSec());
  if(slice_number == 0){
    if(this->bag_ind == 0) this->Z /= 1000.;
    this->linear_vel_cam = Vec3f(0.0,0.0,0.0);
    this->angular_vel_cam = Vec3f(0.0,0.0,0.0);
  }
 }


}


cv::Mat MotionCompensate::computeImageOfWarpedEvents_v2(const std::vector<dvs_msgs::Event>& events_subset) {
  
  this->mc_time_map_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);
  this->mc_event_count_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);

  const double t_ref = events_subset.front().ts.toSec();
  int valid = 0;
  for (const dvs_msgs::Event& ev : events_subset)
  {
    double xx = ev.x;
    double yy = ev.y;
    double dt = ev.ts.toSec() - t_ref; 
    double w_x, w_y;

    double depth = this->Z.at<double>(yy, xx);
    
    cv::Matx23f A = A_v2(xx,yy);
    cv::Matx23f B = B_v2(xx,yy);
    cv::Vec2f v;
    if (depth < 0.) {
      v = cv::Vec2f(0.,0.);
      // continue;
    }else{

      v = (1.0f / depth) * A * this->linear_vel_cam + B * this->angular_vel_cam;
    }
    
    w_x = xx + v[0] * dt;
    w_y = yy + v[1] * dt;

    if (0. <= w_x && w_x < img_width && 0. <= w_y && w_y < img_height)
    {
      valid ++;
      this->mc_time_map_.at<double>(w_y, w_x) += dt;
      this->mc_event_count_.at<double>(w_y, w_x) += 1.;
    }

  }

  cv::Mat invalid_mask = this->mc_event_count_ < 1.;
  this->mc_time_map_.setTo(0.0, invalid_mask);
  this->mc_event_count_.setTo(0.000001, invalid_mask);
  this->mc_time_map_ = this->mc_time_map_.mul(1.0 / this->mc_event_count_);
  this->mc_time_map_.setTo(0.0, invalid_mask);
  return this->mc_event_count_;

}

cv::Matx23f MotionCompensate::A_v2(const int x, const int y){

    cv::Mat cMatrix;
  
    switch(this->bag_ind){
      case 0:
        cMatrix = (cv::Mat_<double>(3, 3) << 
          536.3332593298378, 0, 320.90009280822994, 
          0, 536.31797700847164, 234.04853514480661, 
          0, 0, 1);
        break;
      case 1:
        cMatrix = (cv::Mat_<double>(3, 3) << 
          335.4194629584808, 0.0, 129.9246633794451, 
          0.0, 335.3529356120773, 99.18643034473205, 
          0.0, 0.0, 1.0);
          break;
      default:
        cMatrix = (cv::Mat_<double>(3, 3) << 
          171.37776185565394, 0.0, 120.0, 
          0.0, 171.37776185565394, 90.0, 
          0.0, 0.0, 1.0);
        break;
    }

    double fx = cMatrix.at<double>(0, 0);
    double fy = cMatrix.at<double>(1, 1);
    double cxx = cMatrix.at<double>(0, 2);
    double cyy = cMatrix.at<double>(1, 2);
    cv::Matx23f A =cv::Matx23f(fx, 0., -(x-cxx), 0., fy, -(y-cyy));
    return A;
}

cv::Matx23f MotionCompensate::B_v2(const int x, const int y){

    cv::Mat cMatrix;
  
  switch(this->bag_ind){
    case 0:
      cMatrix = (cv::Mat_<double>(3, 3) << 
        536.3332593298378, 0, 320.90009280822994, 
        0, 536.31797700847164, 234.04853514480661, 
        0, 0, 1);
      break;
    case 1:
      cMatrix = (cv::Mat_<double>(3, 3) << 
        335.4194629584808, 0.0, 129.9246633794451, 
        0.0, 335.3529356120773, 99.18643034473205, 
        0.0, 0.0, 1.0);
        break;
    default:
      cMatrix = (cv::Mat_<double>(3, 3) << 
        171.37776185565394, 0.0, 120.0, 
        0.0, 171.37776185565394, 90.0, 
        0.0, 0.0, 1.0);
      break;
  }

  double fx = cMatrix.at<double>(0, 0);
  double fy = cMatrix.at<double>(1, 1);
  double cxx = cMatrix.at<double>(0, 2);
  double cyy = cMatrix.at<double>(1, 2);

  cv::Matx23f B = cv::Matx23f( 
    (x - cxx)*(y-cyy)/fx, -(fx*fx + (x-cxx)*(x-cxx))/fx, y-cyy, 
    (fy*fy + (y-cyy)*(y-cyy))/fy, -(x-cxx)*(y-cyy)/fx, -(x-cxx));
  return B;
}

cv::Mat MotionCompensate::getGTDepthMap_v2(const double time){

   size_t index = 0;
    for (size_t i = 0; i < depth_map_timestamps.size(); ++i)
    {
        if (depth_map_timestamps[i] > time)
        {
            index = i;
            break;
        }
    }

    cv::Mat depth_map = depth_maps_[index];
    return depth_map;

}

void MotionCompensate::computeGrad_v2(const cv::Mat& image, const double t_ref, const cv::Mat& Z) {

    this->grad_linear_vel = cv::Vec3f(0.0f, 0.0f, 0.0f);
    this->grad_angular_vel = cv::Vec3f(0.0f, 0.0f, 0.0f);
    this->grad_Z = cv::Mat::zeros(this->Z.size(), this->Z.type());
    cv::Mat grad_Z_count = cv::Mat::zeros(this->Z.size(), this->Z.type());

    cv::Mat grad_x, grad_y, grad_xx, grad_yy;
    cv::Sobel(image, grad_x, CV_64FC1, 1, 0, 3);
    cv::Sobel(image, grad_y, CV_64FC1, 0, 1, 3); 
    cv::Sobel(grad_x, grad_xx, CV_64FC1, 1, 0, 3); 
    cv::Sobel(grad_y, grad_yy, CV_64FC1, 0, 1, 3);

    int valid_pixel = 0;
    int invalid = 0;

    int patch_size = 20;
    for (int y = patch_size / 2; y < img_height - patch_size / 2; y += patch_size) {
        for (int x = patch_size / 2; x < img_width - patch_size / 2; x += patch_size) {
          if (std::abs(grad_x.at<double>(y, x)) <= 1e-7 || std::abs(grad_xx.at<double>(y, x)) <= 1e-7
            || std::abs(grad_y.at<double>(y, x)) <= 1e-7 || std::abs(grad_yy.at<double>(y, x)) <= 1e-7) {
                continue;
            }

            if(this->mc_time_map_.at<double>(y, x) < 1e-7){
                invalid ++;
                continue;
            }
                       cv::Matx23f A = A_v2(x,y);
            cv::Matx23f B = B_v2(x,y);

            double dt = this->mc_time_map_.at<double>(y, x) - t_ref;
            
            double squre_r = std::sqrt(grad_x.at<double>(y, x) *  grad_x.at<double>(y, x) + grad_y.at<double>(y, x) * grad_y.at<double>(y, x));
            // double squre_r = 1.;

            double obj_vx = -dt * (grad_x.at<double>(y, x) / squre_r) * grad_xx.at<double>(y, x);
            double obj_vy = -dt * (grad_y.at<double>(y, x) / squre_r) * grad_yy.at<double>(y, x);
            cv::Matx12f grad_I(obj_vx, obj_vy);
            double depth = this->Z.at<double>(y, x);
            
            if(depth < 0.){
              continue;
            }

            this->grad_Z.at<double>(y, x) += (grad_I * (-1.0 / (depth * depth)) * A * this->linear_vel_cam)[0];
            
            grad_Z_count.at<double>(y, x) += 1.;
        }
    }

    this->grad_Z /= (grad_Z_count + 1e-8);

    for (int y = 1; y < grad_x.rows - 1; ++y) {
        for (int x = 1; x < grad_x.cols - 1; ++x) {

            if (std::abs(grad_x.at<double>(y, x)) <= 1e-7 || std::abs(grad_xx.at<double>(y, x)) <= 1e-7
            || std::abs(grad_y.at<double>(y, x)) <= 1e-7 || std::abs(grad_yy.at<double>(y, x)) <= 1e-7) {
                continue;
            }

            if(this->mc_time_map_.at<double>(y, x) < 1e-7){
                invalid ++;
                continue;
            }

            cv::Matx23f A = A_v2(x,y);
            cv::Matx23f B = B_v2(x,y);

            double dt = this->mc_time_map_.at<double>(y, x) - t_ref;
            
            double squre_r = std::sqrt(grad_x.at<double>(y, x) *  grad_x.at<double>(y, x) + grad_y.at<double>(y, x) * grad_y.at<double>(y, x));
            // double squre_r = 1.;

            double obj_vx = -dt * (grad_x.at<double>(y, x) / squre_r) * grad_xx.at<double>(y, x);
            double obj_vy = -dt * (grad_y.at<double>(y, x) / squre_r) * grad_yy.at<double>(y, x);

            // ROS_WARN("obj vx and vy %f, %f", obj_vx, obj_vy);

            cv::Matx12f grad_I(obj_vx, obj_vy);
            double depth = this->Z.at<double>(y, x);
            
            if(depth < 0.){
              continue;
            }
            this->grad_linear_vel += cv::Vec3f((grad_I * (1.0 / depth) * A).val);
            this->grad_angular_vel += cv::Vec3f((grad_I * B).val);
            
            valid_pixel++;
        }
    }
    // std::cout << "valid pixel!! for grad " << valid_pixel << std::endl;
    // std::cout << "invalid pixel!! for grad " << invalid << std::endl;

    if(valid_pixel > 10){
      this->grad_linear_vel /= static_cast<double>(valid_pixel);
      this->grad_angular_vel /= static_cast<double>(valid_pixel);
      // this->grad_Z /= (grad_Z_count + 1e-8);
    }else{
      this->grad_linear_vel = cv::Vec3f(0.0f, 0.0f, 0.0f);;
      this->grad_angular_vel = cv::Vec3f(0.0f, 0.0f, 0.0f);
      // this->grad_Z = cv::Mat::zeros(Z.size(), Z.type());
    }

}

cv::Mat MotionCompensate::bilinearInterpolate(const cv::Mat& depth_map, double patch_size){

    cv::Mat interpolated_depth_map = cv::Mat::zeros(img_height, img_width, CV_64FC1);

    for (int y = patch_size / 2; y < img_height - patch_size / 2; y += patch_size) {
        for (int x = patch_size / 2; x < img_width - patch_size / 2; x += patch_size) {

            double depth = depth_map.at<double>(y, x);
            
            interpolated_depth_map.at<double>(y, x) = depth;

            for (int dy = -patch_size / 2; dy < patch_size / 2; ++dy) {
                for (int dx = -patch_size / 2; dx < patch_size / 2; ++dx) {
                    int patch_x = x + dx;
                    int patch_y = y + dy;

                    if (patch_x >= 0 && patch_x < img_width && patch_y >= 0 && patch_y < img_height) {
                        double depth_top_left = depth_map.at<double>(patch_y - 1, patch_x - 1);
                        double depth_top_right = depth_map.at<double>(patch_y - 1, patch_x + 1);
                        double depth_bottom_left = depth_map.at<double>(patch_y + 1, patch_x - 1);
                        double depth_bottom_right = depth_map.at<double>(patch_y + 1, patch_x + 1);

                        double depth_interpolated = (depth_top_left + depth_top_right + depth_bottom_left + depth_bottom_right) / 4.0;
                        
                        interpolated_depth_map.at<double>(patch_y, patch_x) = depth_interpolated;
                    }
                }
            }
        }
    }

    return interpolated_depth_map;
}

double MotionCompensate::computeContrast_v2(const cv::Mat& image)
{

//// std deviation
//   cv::Scalar mean, stddev;
//   cv::meanStdDev(image, mean, stddev);
//   return stddev[0] * stddev[0];

  // norm
  return cv::norm(image,cv::NORM_L2SQR) / static_cast<double>(image.rows*image.cols);

//// magnitude
  // cv::Mat grad_x, grad_y;
  // cv::Sobel(image, grad_x, CV_64FC1, 1, 0, 3);
  // cv::Sobel(image, grad_y, CV_64FC1, 0, 1, 3);

  // cv::Mat magnitude;
  // cv::magnitude(grad_x, grad_y, magnitude);

  // double mean_value;
  // int valid_pixel_count = 0;

  // for (int i = 0; i < magnitude.rows; i++) {
  //     for (int j = 0; j < magnitude.cols; j++) {
  //         if (magnitude.at<double>(i, j) > 1e-7) {  
  //             mean_value += magnitude.at<double>(i, j);
  //             valid_pixel_count++;
  //         }
  //     }
  // }

  // if (valid_pixel_count > 0) {
  //     mean_value /= valid_pixel_count; 
  // }

  // return mean_value;

}

void MotionCompensate::updateModel_v2() {
      // this->grad_linear_vel = cv::Vec3f(0.0001f, 0.0001f, 0.00001f);
      this->linear_vel_cam -= 0.0001 * this->grad_linear_vel;
      // this->angular_vel_cam += 0.0001 * this->grad_angular_vel;
      this->Z -= 0.0001 * this->grad_Z;
      // this->Z = bilinearInterpolate(this->Z, 20);
      
      // cv::Mat temp;
      // temp = this->Z + 0.001 * this->grad_Z; 
      // cv::exp(temp, this->Z);

}

void MotionCompensate::printInfo_v2(const double& contrast, const double& density){

  std::cout << "current linear vel x: " << linear_vel_cam[0] 
          << " y: " << linear_vel_cam[1] 
          << " z: " << linear_vel_cam[2] << std::endl;
  std::cout << "current angular vel x: " << angular_vel_cam[0] 
          << " y: " << angular_vel_cam[1] 
          << " z: " << angular_vel_cam[2] << std::endl;

  std::cout << "linear vel dx: " << grad_linear_vel[0] 
          << " dy: " << grad_linear_vel[1] 
          << " dz: " << grad_linear_vel[2] << std::endl;
  std::cout << "angular vel dx: " << grad_angular_vel[0] 
          << " dy: " << grad_angular_vel[1] 
          << " dz: " << grad_angular_vel[2] << std::endl;
  std::cout << "current contrast: " << contrast << std::endl;
  std::cout << "current density: " << density << std::endl;

}
}