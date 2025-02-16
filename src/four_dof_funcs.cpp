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
#include <ceres/ceres.h>
#include <random>
#include "cnpy.h"  // Include cnpy header for .npy saving

using namespace cv;
using namespace std;
using namespace Eigen;

namespace motion_compensate
{

void MotionCompensate::findInitialFlow(const std::vector<dvs_msgs::Event>& events_subset)
{
  // Hierarchical search to find a good initial flow
  double hx_temp = 0.0, hy_temp = 0.0, hz_temp = 0.0, hth_temp = 0.0;
  std::vector<double> xy_steps = {300., 200., 100., 50., 25.};
  std::vector<double> zth_steps = {0.1, 0.06, 0.02, 0.01, 0.005};


  for (size_t i = 1; i < xy_steps.size(); ++i)
  {
    const double prev_step_xy = xy_steps[i - 1];
    const double step_xy = xy_steps[i];
    const double prev_step_zth = zth_steps[i - 1];
    const double step_zth = zth_steps[i];

    // Define search range for each parameter
    std::array<double, 8> param_range = {
        hx_temp - prev_step_xy, hx_temp + prev_step_xy,
        hy_temp - prev_step_xy, hy_temp + prev_step_xy,
        hz_temp - prev_step_zth, hz_temp + prev_step_zth,
        hth_temp - prev_step_zth, hth_temp + prev_step_zth};

    // Find the best parameters in the given range
    std::array<double, 4> best_params = findBestFlowInRangeBruteForce(events_subset, param_range, step_xy, step_zth);

    // Update current parameters
    hx_temp = best_params[0];
    hy_temp = best_params[1];
    hz_temp = best_params[2];
    hth_temp = best_params[3];
  }

  this->hx = hx_temp;
  this->hy = hy_temp;
  this->hz = hz_temp;
  this->htheta = hth_temp;
}

std::array<double, 4> MotionCompensate::findBestFlowInRangeBruteForce(const std::vector<dvs_msgs::Event>& events_subset, const std::array<double, 8>& param_range, double step_xy, double step_zth)
{
  // Brute-force search over a grid of possible global flows for 4 parameters
  double hx_min = param_range[0], hx_max = param_range[1];
  double hy_min = param_range[2], hy_max = param_range[3];
  double hz_min = param_range[4], hz_max = param_range[5];
  double hth_min = param_range[6], hth_max = param_range[7];

  double minimum_cost = std::numeric_limits<double>::max();
  double opt_hx = 0.0, opt_hy = 0.0, opt_hz = 0.0, opt_hth = 0.0;

  for (double hx_ = hx_min; hx_ <= hx_max; hx_ += step_xy)
  {
    for (double hy_ = hy_min; hy_ <= hy_max; hy_ += step_xy)
    {

      for (double hth_ = hth_min; hth_ <= hth_max; hth_ += step_zth)
      {
        if(enable_depth){
          double cost = contrast_f_numerical(events_subset, hx_, hy_, 0., hth_);
          if (cost < minimum_cost)
          {
            minimum_cost = cost;
            opt_hx = hx_;
            opt_hy = hy_;
            opt_hz = 0.;
            opt_hth = hth_;
          }
          continue;
        }
        
        for (double hz_ = hz_min; hz_ <= hz_max; hz_ += step_zth)
        {
          double cost = contrast_f_numerical(events_subset, hx_, hy_, hz_, hth_);

          if (cost < minimum_cost)
          {
            minimum_cost = cost;
            opt_hx = hx_;
            opt_hy = hy_;
            opt_hz = hz_;
            opt_hth = hth_;

          }
        }
      }
    }
  }


return {opt_hx, opt_hy, opt_hz, opt_hth};
}

double MotionCompensate::contrast_f_numerical(const std::vector<dvs_msgs::Event>& events_subset, const double hx_, const double hy_, const double hz_, const double hth_)
{
  // Compute cost for the given parameters
  cv::Mat image_warped = computeImageOfWarpedEvents(events_subset, hx_, hy_, hz_, hth_); 

  double contrast = computeContrast(image_warped); 

  return -contrast; 
}

cv::Mat MotionCompensate::computeImageOfWarpedEvents(const std::vector<dvs_msgs::Event>& events_subset, double hx_, double hy_, double hz_, double hth_)
{

  this->mc_time_map_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);
  this->mc_event_count_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);

  const double t_ref = events_subset.front().ts.toSec();

  for (const dvs_msgs::Event& ev : events_subset)
  {

    double w_x, w_y;
    double xx = ev.x;
    double yy = ev.y;
    double dt = ev.ts.toSec() - t_ref; 

    double cosTheta = std::cos(hth_);
    double sinTheta = std::sin(hth_);

    double rotX = cosTheta * xx - sinTheta * yy;
    double rotY = sinTheta * xx + cosTheta * yy;

    w_x = xx + dt * (hx_ + (hz_ + 1) * rotX - xx);
    w_y = yy + dt * (hy_ + (hz_ + 1) * rotY - yy);

    if (0. <= w_x && w_x < img_width && 0. <= w_y && w_y < img_height)
    {
      this->mc_time_map_.at<double>(w_y, w_x) += dt;
      this->mc_event_count_.at<double>(w_y, w_x) += 1.;
    }
  }

  cv::Mat invalid_mask = this->mc_event_count_ < 1.0;
  this->mc_time_map_.setTo(0.0, invalid_mask);
  this->mc_event_count_.setTo(0.000001, invalid_mask);
  this->mc_time_map_ = this->mc_time_map_.mul(1.0 / this->mc_event_count_);
  this->mc_time_map_.setTo(0.0, invalid_mask);

  if(optimize_image_type == EventCount) return this->mc_event_count_;
  else return this->mc_time_map_;

}

double MotionCompensate::computeContrast(const cv::Mat& image)
{

  double contrast = 0.0;;
  switch (contrast_ind){
    case NORM:
      contrast = cv::norm(image,cv::NORM_L2SQR) / static_cast<double>(image.rows*image.cols);
      break;

    case VAR: {
      cv::Scalar mean, stddev;
      cv::meanStdDev(image, mean, stddev);
      contrast = stddev[0] * stddev[0];
      break;
    }

    case MAG:{
      cv::Mat grad_x, grad_y;
      cv::Sobel(image, grad_x, CV_64FC1, 1, 0, 3);
      cv::Sobel(image, grad_y, CV_64FC1, 0, 1, 3);

      cv::Mat magnitude;
      cv::magnitude(grad_x, grad_y, magnitude);

      cv::Scalar mean_mag = cv::mean(magnitude, magnitude > 1e-7);
      contrast = mean_mag[0];
      break;

    }

    default:
      contrast = 0.;
      break;
  }
    return contrast;
}

double MotionCompensate::computeError(const double& l_hx,const double& l_hy,const double& l_hz,const double& l_hth){
  return std::sqrt(
    std::pow(this->hx - l_hx, 2) +
    std::pow(this->hy - l_hy, 2) +
    std::pow(this->hz - l_hz, 2) +
    std::pow(this->htheta - l_hth, 2)
  );
}

void MotionCompensate::updateModel(){

  this->hx += this->lr_x * this->dx;
  this->hy += this->lr_y * this->dy; 
  if(!enable_depth) this->hz += this->lr_div * this->dz;
  this->htheta += this->lr_rot * this->dth;

}

void MotionCompensate::diffTimeImage(const cv::Mat& image){

    cv::Mat grad_x, grad_y;
    cv::Sobel(image, grad_x, CV_64FC1, 1, 0, 3);
    cv::Sobel(image, grad_y, CV_64FC1, 0, 1, 3);

    const double EPS = 1e-5;

    int cnt = 0;
    double dx_, dy_, div, rot;

    // TODO: acc!

    for (int y = 0; y < grad_x.rows; y+=1) {
        for (int x = 0; x < grad_x.cols; x+=1) {

          if(image.at<double>(y,x) > EPS) continue;

            double rx = x - img_width/2.;
            double ry = y - img_height/2.;
            
            dx_ += grad_x.at<double>(y, x);
            dy_ += grad_y.at<double>(y, x);
            rot += rx * grad_y.at<double>(y, x) - ry * grad_x.at<double>(y, x);
            div += rx * grad_x.at<double>(y, x) + ry * grad_y.at<double>(y, x);
            cnt ++;

        }
    }

      if (cnt > 10) { 

        dx_ /= cnt;
        dy_ /= cnt;
        rot /= cnt;
        div /= cnt;

        if(!use_adam){
          this->dth = rot;
          this->dz = div;
          this->dx = dx_;
          this->dy = dy_;
        }
        else{
          ap_rot.update(rot, &dth);
          ap_div.update(div, &dz);
          ap_x.update(dx_, &dx);
          ap_y.update(dy_, &dy);
        }
        
      } else {
          this->dx = this->dy = this->dz = this->dth = 0.0;
      }

}

double MotionCompensate::getDensity(const cv::Mat& image, double threshold){
  cv::Mat normalized;

  if(threshold < 1.){
    cv::normalize(image, normalized, 1., 255., cv::NORM_MINMAX);
    threshold = 0.1;
  }else{
    image.copyTo(normalized);
  }
  double total_count = cv::sum(normalized)[0];
  cv::Mat mask = (normalized >= threshold);
  double count = cv::countNonZero(mask);
  double density = total_count / count;
  return density;
}

void MotionCompensate::printInfo(const double& error, const double& contrast, const double& density){
  std::cout << "current model " << hx << " " << hy << " " << hz << " " << htheta << std::endl;
  std::cout << "dx: " << dx << " dy: " << dy << " dz: " << dz << " dth: " << dth << std::endl;
  std::cout << "lr_x: " << lr_x << " lr_y: " << lr_y << " lr_div: " << lr_div << " lr_rot: " << lr_rot << std::endl;
  std::cout << "current contrast: " << contrast << std::endl;
  std::cout << "current error: " << error << std::endl;
  std::cout << "current density: " << density << std::endl;

}

void MotionCompensate::detectMovingObjects(const cv::Mat& avg_time_map, 
                                           const cv::Mat& mc_time_map, 
                                           const double& dt,
                                           cv::Mat& background_mask, 
                                           cv::Mat& foreground_mask) {

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    std::vector<double> non_zero_values;
    double sum;

    // TODO: acc!
    sum = 0.0;

    // Using OpenCV matrix operations to extract non-zero elements
    cv::Mat non_zero_mask = mc_time_map > 0.0;  // Create a mask for non-zero values
    cv::Mat non_zero_values_mat;
    mc_time_map.copyTo(non_zero_values_mat, non_zero_mask);  // Copy only non-zero elements

    // Iterate through the non-zero values
    for (int i = 0; i < non_zero_values_mat.total(); ++i) {
        double val = non_zero_values_mat.at<double>(i);
        if (val > 0.) {
            non_zero_values.push_back(val);
            sum += val;
        }
    }
    // for (int i = 0; i < mc_time_map.rows; ++i) {
    //     for (int j = 0; j < mc_time_map.cols; ++j) {
    //         double val = mc_time_map.at<double>(i, j);
    //         if (val > 0.) {
    //             non_zero_values.push_back(val);
    //             sum += val;
    //         }
    //     }
    // }
    double avg = sum/float(non_zero_values.size());

    std::sort(non_zero_values.begin(), non_zero_values.end());

    int size = non_zero_values.size();
    double med;

    med = non_zero_values[int(size / 2)];
    std::cout << duration << std::endl;

    // med = size % 2 == 0 ? (non_zero_values[2*size / 3 - 1] + non_zero_values[2*size / 3]) / 2.0 : non_zero_values[size / 2];

    rho = (this->mc_time_map_ - med) / dt;

    background_mask = rho <= 0.;

    foreground_mask = rho > lambda;

    if(filter_small_compo)
       filterComponents(foreground_mask, foreground_mask, 30, 2.5);

    cv::Mat labels;
    int num_objects = cv::connectedComponents(foreground_mask, labels);

    ROS_INFO("Detected %d independently moving objects.", num_objects - 1);

}


void MotionCompensate::filterComponents(const cv::Mat& binary_image, cv::Mat& filtered_image, int min_area, float max_aspect_ratio) {
    CV_Assert(binary_image.type() == CV_8UC1);

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary_image, labels, stats, centroids);

    filtered_image = cv::Mat::zeros(binary_image.size(), CV_8UC1);

    int largest_area = 0;
    int largest_label = -1;

    for (int label = 1; label < num_labels; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

        float aspect_ratio = static_cast<float>(std::max(width, height)) / std::min(width, height);

        if (area >= min_area && aspect_ratio <= max_aspect_ratio) {
            if (area > largest_area) {
                largest_area = area;
                largest_label = label;
            }
        }
    }

    if (largest_label != -1) {
        filtered_image.setTo(255, labels == largest_label);
    }
}

}