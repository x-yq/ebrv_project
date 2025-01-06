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
        // if(enable_depth){
        //   double cost = contrast_f_numerical(events_subset, hx_, hy_, 0., hth_);
        //   if (cost < minimum_cost)
        //   {
        //     minimum_cost = cost;
        //     opt_hx = hx_;
        //     opt_hy = hy_;
        //     opt_hz = 0.;
        //     opt_hth = hth_;
        //   }
        //   continue;
        // }
        
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

            // ROS_WARN("opt x:%f, y:%f, z:%f, th: %f", opt_hx, opt_hy, opt_hz, opt_hth);
            // ROS_WARN("min cost: %f", minimum_cost);

            // cv::Mat image_warped = computeImageOfWarpedEvents(events_subset, opt_hx, opt_hy, opt_hz, opt_hth); 
            // cv::normalize(image_warped, image_warped, 0., 1., cv::NORM_MINMAX);
            // image_warped.convertTo(image_warped, CV_64FC1);
            // cv::imshow("MC Initial", image_warped);
            // cv::waitKey(0);
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

double MotionCompensate::getDiversion(double time, double x, double y)
{
    // Ensure timestamps and depth maps have the same size
    if (depth_maps_.size() != depth_map_timestamps.size())
    {
        std::cerr << "Error: Size of timestamps and data must be the same!" << std::endl;
        return -1;
    }

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

    if (x < 1 || x >= depth_map.cols - 1 || y < 1 || y >= depth_map.rows - 1)
    {
        return 0.;
    }

    // Correct the depth value based on intrinsic parameters
    double fx = K_depth.at<double>(0, 0);
    double fy = K_depth.at<double>(1, 1);
    double cxx = K_depth.at<double>(0, 2);
    double cyy = K_depth.at<double>(1, 2);

    double total_divergence = 0.0;
    int valid_points = 0;

    // Iterate over the 3x3 neighborhood
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue; // Skip the center point

            int nx = x + dx;
            int ny = y + dy;

            double z_center = depth_map.at<float>(y, x) / 1000.0;
            double z_neighbor = depth_map.at<float>(ny, nx) / 1000.0;

            // Skip invalid or zero depth values
            if (z_center <= 0.0 || z_neighbor <= 0.0)
                continue;

            // Calculate 3D coordinates for center and neighbor
            double X_center = (x - cxx) * z_center / fx;
            double Y_center = (y - cyy) * z_center / fy;

            double X_neighbor = (nx - cxx) * z_neighbor / fx;
            double Y_neighbor = (ny - cyy) * z_neighbor / fy;

            // Calculate divergence for this pair
            double div_x = X_neighbor - X_center;
            double div_y = Y_neighbor - Y_center;

            total_divergence += div_x + div_y;
            valid_points++;
        }
    }

    if (valid_points == 0)
        return 0.0;

    return total_divergence / valid_points;
}

double MotionCompensate::calculateAverageDiv(const std::vector<dvs_msgs::Event>& events_subset) {
    double totalDivergence = 0.0;
    int validCount = 0;

    for (const auto& ev : events_subset) {
        double divergence = getDiversion(ev.ts.toSec(), ev.x, ev.y);
        if (divergence != 0.0) {
            totalDivergence += divergence;
            validCount++;
        }

        std::cout<< totalDivergence << std::endl;
    }

    if (validCount == 0) {
        return 0.0;
    }

    return totalDivergence / validCount;
}

cv::Mat MotionCompensate::computeImageOfWarpedEvents(const std::vector<dvs_msgs::Event>& events_subset, double hx_, double hy_, double hz_, double hth_)
{

  this->mc_time_map_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);
  this->mc_event_count_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);

  this->mc_event_count_pos_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);
  this->mc_event_count_neg_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);

  this->mc_time_map_pos_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);
  this->mc_time_map_neg_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);

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

    w_x = xx + dt * hx_;
    w_y = xx + dt * hx_;

    if(enable_depth){
      hz_ = getDiversion(ev.ts.toSec(), w_x, w_y);
      this->hz = hz_;
    }

    w_x = xx + dt * (hx_ + (hz_ + 1) * rotX - xx);
    w_y = yy + dt * (hy_ + (hz_ + 1) * rotY - yy);

    if (0. <= w_x && w_x < img_width && 0. <= w_y && w_y < img_height)
    {
      this->mc_time_map_.at<double>(w_y, w_x) += dt;
      this->mc_event_count_.at<double>(w_y, w_x) += 1.;
      if(ev.polarity){
        this->mc_event_count_pos_.at<double>(w_y, w_x) += 1.;
        this->mc_time_map_pos_.at<double>(w_y, w_x) += dt;
      }
      else{
        this->mc_event_count_neg_.at<double>(w_y, w_x) += 1.;
        this->mc_time_map_neg_.at<double>(w_y, w_x) += dt;
      }
    }
  }

  cv::Mat invalid_mask = this->mc_event_count_ < 1.0;
  this->mc_time_map_.setTo(0.0, invalid_mask);
  this->mc_event_count_.setTo(0.000001, invalid_mask);
  this->mc_time_map_ = this->mc_time_map_.mul(1.0 / this->mc_event_count_);
  this->mc_time_map_.setTo(0.0, invalid_mask);

  return this->mc_event_count_;

}

double MotionCompensate::computeContrast(const cv::Mat& image)
{

  // cv::Scalar mean, stddev;
  // cv::meanStdDev(image, mean, stddev);
  // double contrast = stddev[0] * stddev[0];
  // return contrast; 

  // Compute mean square value of the image
  double contrast = cv::norm(image,cv::NORM_L2SQR) / static_cast<double>(image.rows*image.cols);
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
          dx = dy = dz = dth = 0.0;
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
       filterComponents(foreground_mask, foreground_mask, 15, 3.0);

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

void MotionCompensate::plotHist(const cv::Mat& avg_image, const cv::Mat& mc_image){
  
  cv::Mat avg_img, mc_img;
  cv::Mat avg_time_map_normalized, mc_time_map_normalized;
  
  cv::normalize(mc_image, mc_time_map_normalized, 0., 1., cv::NORM_MINMAX);
  cv::normalize(avg_image, avg_time_map_normalized, 0., 1., cv::NORM_MINMAX);
  mc_time_map_normalized.convertTo(mc_img, CV_32F);
  avg_time_map_normalized.convertTo(avg_img, CV_32F);

  int histSize = 100; 
  float range[] = {0., 1.};
  const float* histRange = {range};


  bool uniform = true, accumulate = false;
  cv::Mat hist_mc, hist_avg;

  cv::Mat mask_mc = mc_img > 0.0;
  cv::Mat mask_avg = avg_img > 0.0;

  cv::calcHist(&mc_img, 1, 0, mask_mc, hist_mc, 1, &histSize, &histRange, uniform, accumulate);
  cv::calcHist(&avg_img, 1, 0, mask_avg, hist_avg, 1, &histSize, &histRange, uniform, accumulate);

  int maxVal = 0;
  for (int i = 0; i < histSize; i++) {
      maxVal = std::max(maxVal, (int)hist_mc.at<float>(i));
      maxVal = std::max(maxVal, (int)hist_avg.at<float>(i));
  }

  int histImageWidth = 512; 
  int histImageHeight = 400; 
  int padding = 50;
  int binWidth = cvRound((double)(histImageWidth - 2 * padding) / histSize);

  cv::Mat histImage(histImageHeight + padding * 2, histImageWidth + padding * 2, CV_8UC3, cv::Scalar(255, 255, 255));

  cv::normalize(hist_mc, hist_mc, 0, histImageHeight, cv::NORM_MINMAX, -1, cv::Mat());
  cv::normalize(hist_avg, hist_avg, 0, histImageHeight, cv::NORM_MINMAX, -1, cv::Mat());

  line(histImage, cv::Point(padding, histImageHeight + padding), cv::Point(histImageWidth - padding, histImageHeight + padding), Scalar(0, 0, 0), 1); // x axis
  line(histImage, cv::Point(padding, padding), cv::Point(padding, histImageHeight + padding), Scalar(0, 0, 0), 1); // y axis

  for (int i = 0; i < histSize; i++) {
      int x1 = padding + binWidth * i;
      int y1 = histImageHeight + padding - cvRound(hist_mc.at<float>(i));
      int y2 = histImageHeight + padding - cvRound(hist_avg.at<float>(i));
      int x2 = x1 + binWidth;

      // blue is the motion compensated one
      cv::Mat roi_mc = histImage(cv::Rect(x1, y1, binWidth, histImageHeight + padding - y1));
      cv::Mat blue(roi_mc.size(), roi_mc.type(), cv::Scalar(255, 0, 0)); // Blue color
      cv::addWeighted(roi_mc, 0.5, blue, 0.5, 0.0, roi_mc);

      cv::Mat roi_avg = histImage(cv::Rect(x1, y2, binWidth, histImageHeight + padding - y2));
      cv::Mat green(roi_avg.size(), roi_avg.type(), cv::Scalar(0, 255, 0)); // Green color
      cv::addWeighted(roi_avg, 0.5, green, 0.5, 0.0, roi_avg);


  }


  for (int i = 0; i <= 5; i++) {
      int x = padding + i * (histImageWidth - 2 * padding) / 5;
      line(histImage, cv::Point(x, histImageHeight + padding), cv::Point(x, histImageHeight + padding + 5), Scalar(0, 0, 0), 1);

      string label = format("%.1f", range[0] + (range[1] - range[0]) / 5 * i);
      putText(histImage, label, cv::Point(x - 10, histImageHeight + padding + 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
  }

  for (int i = 0; i <= 5; i++) {
      int y = histImageHeight + padding - i * (histImageHeight / 5);
      line(histImage, cv::Point(padding - 5, y), cv::Point(padding, y), Scalar(0, 0, 0), 1);

      string label = format("%d", maxVal / 5 * i);
      putText(histImage, label, cv::Point(padding - 40, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
  }

  // Show the images
  // cv::imshow("Avg Image", avg_time_map_normalized);
  // cv::imshow("MC Image", mc_time_map_normalized);
  cv::imshow("hist", histImage);
  cv::waitKey(0);
}


void MotionCompensate::saveMapsAsMultiChannels(){

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    std::tm tm_now;
    localtime_r(&time_t_now, &tm_now);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y%m%d_%H%M%S");

     if (this->mc_time_map_.size() != this->mc_event_count_pos_.size() || this->mc_event_count_neg_.size() != this->mc_time_map_.size()) {
        std::cerr << "Error: Input images must have the same size!" << std::endl;
    }

    // cv::Mat normalized1, normalized2, normalized3;
    // cv::normalize(this->mc_time_map_, normalized1, 0., 1., cv::NORM_MINMAX);
    // cv::normalize(this->mc_event_count_pos_, normalized2, 0., 1., cv::NORM_MINMAX);
    // cv::normalize(this->mc_event_count_neg_, normalized3, 0., 1., cv::NORM_MINMAX);

    // normalized1.convertTo(normalized1, CV_64FC1);
    // normalized2.convertTo(normalized2, CV_64FC1);
    // normalized3.convertTo(normalized3, CV_64FC1);

    // cv::Mat npy_merged;
    // std::vector<cv::Mat> channels_npy = {normalized1, normalized2, normalized3};
    // cv::merge(channels_npy, npy_merged);

    // npy_merged = npy_merged(cv::Rect(0, 0, this->img_height, this->img_height));

    // std::string npy_folder = "/home/x-yq/catkin_ws/src/fast_dynamic/images/what_is_background/npy";
    // std::string name_npy = npy_folder + "/" + oss.str() + ".npy";

    // std::vector<float> merged_vec(npy_merged.begin<float>(), npy_merged.end<float>());
    // try {
    //     cnpy::npy_save(name_npy, merged_vec.data(), {npy_merged.rows, npy_merged.cols, 3}, "w");
    //     std::cout << "Successfully saved the .npy file!" << std::endl;
    // } catch (const std::exception& e) {
    //     std::cerr << "Error: " << e.what() << std::endl;
    // }



    cv::Mat png_normalized1, png_normalized2, png_normalized3;
    cv::normalize(this->mc_event_count_pos_, png_normalized1, 0, 255, cv::NORM_MINMAX);
    cv::normalize(this->mc_event_count_neg_, png_normalized2, 0, 255, cv::NORM_MINMAX);
    cv::normalize(this->mc_time_map_, png_normalized3, 0, 255, cv::NORM_MINMAX);

    png_normalized1.convertTo(png_normalized1, CV_8UC1);
    png_normalized2.convertTo(png_normalized2, CV_8UC1);
    png_normalized3.convertTo(png_normalized3, CV_8UC1);


    cv::Mat png_merged;
    std::vector<cv::Mat> channels_png = {png_normalized1, png_normalized2, png_normalized3};
    cv::merge(channels_png, png_merged);

    png_merged = png_merged(cv::Rect(0, 0, this->img_height, this->img_height));

    std::string png_folder = "/home/x-yq/catkin_ws/src/fast_dynamic/images/what_is_background/png_multi_event_count";

    try {
        // std::filesystem::create_directories(npy_folder);
        std::filesystem::create_directories(png_folder);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    std::string name_png = png_folder + "/" + oss.str() + ".png";

    if (!cv::imwrite(name_png, png_merged)) {
        std::cerr << "Error: Could not save the PNG file!" << std::endl;
    }




    cv::normalize(this->mc_time_map_pos_, png_normalized1, 0, 255, cv::NORM_MINMAX);
    cv::normalize(this->mc_time_map_neg_, png_normalized2, 0, 255, cv::NORM_MINMAX);
    cv::normalize(this->mc_event_count_, png_normalized3, 0, 255, cv::NORM_MINMAX);

    png_normalized1.convertTo(png_normalized1, CV_8UC1);
    png_normalized2.convertTo(png_normalized2, CV_8UC1);
    png_normalized3.convertTo(png_normalized3, CV_8UC1);


    png_merged;
    channels_png = {png_normalized1, png_normalized2, png_normalized3};
    cv::merge(channels_png, png_merged);

    png_merged = png_merged(cv::Rect(0, 0, this->img_height, this->img_height));

    png_folder = "/home/x-yq/catkin_ws/src/fast_dynamic/images/what_is_background/png_multi_time_map";


    try {
        // std::filesystem::create_directories(npy_folder);
        std::filesystem::create_directories(png_folder);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    name_png = png_folder + "/" + oss.str() + ".png";

    if (!cv::imwrite(name_png, png_merged)) {
        std::cerr << "Error: Could not save the PNG file!" << std::endl;
    }
    

    std::cout << "Successfully saved images" << std::endl;
}

}