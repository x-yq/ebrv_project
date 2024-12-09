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


using namespace cv;
using namespace std;

namespace motion_compensate
{

std::vector<dvs_msgs::Event> MotionCompensate::computeImageOfWarpedEvents(const std::vector<dvs_msgs::Event>& events_subset)
{

  this->mc_time_map_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);
  this->mc_event_count_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);

  this->mc_event_count_pos_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);
  this->mc_event_count_neg_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);

  const double t_ref = events_subset.front().ts.toSec();
  std::vector<dvs_msgs::Event> warped_events;

  for (const dvs_msgs::Event& ev : events_subset)
  {
    double w_x, w_y;
    double xx = ev.x;
    double yy = ev.y;
    double dt = ev.ts.toSec() - t_ref; 

    double cosTheta = std::cos(htheta);
    double sinTheta = std::sin(htheta);

    double rotX = cosTheta * xx - sinTheta * yy;
    double rotY = sinTheta * xx + cosTheta * yy;

    w_x = xx + dt * (hx + (hz + 1) * rotX - xx);
    w_y = yy + dt * (hy + (hz + 1) * rotY - yy);

    if (0. <= w_x && w_x < img_width && 0. <= w_y && w_y < img_height)
    {
      this->mc_time_map_.at<double>(w_y, w_x) += dt;
      this->mc_event_count_.at<double>(w_y, w_x) += 1.;
      if(ev.polarity) this->mc_event_count_pos_.at<double>(w_y, w_x) += 1.;
      else this->mc_event_count_neg_.at<double>(w_y, w_x) += 1.;

      dvs_msgs::Event warped_ev;
      warped_ev.x = w_x;
      warped_ev.y = w_y;
      warped_ev.ts = ev.ts;
      warped_ev.polarity = ev.polarity;
      warped_events.push_back(warped_ev);
    }
  }

  for (int y = 0.; y < img_height; y+=1.) {
    for (int x = 0.; x < img_width; x+=1.) {
      double count = mc_event_count_.at<double>(y, x);
      if(count >= 1.){
        this->mc_time_map_.at<double>(y, x) /= this->mc_event_count_.at<double>(y, x);
      }
      else{
        this->mc_time_map_.at<double>(y, x) = 0.;
      }
      
    }

}

  return warped_events;

}

double MotionCompensate::computeContrast(const cv::Mat& image)
{

  cv::Scalar mean, stddev;
  cv::meanStdDev(image, mean, stddev);
  return stddev[0]; 

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
  this->hz += this->lr_div * this->dz;
  this->htheta += this->lr_rot * this->dth;

}


void MotionCompensate::diffTimeImage(const cv::Mat& image){

    cv::Mat grad_x, grad_y;
    cv::Sobel(image, grad_x, CV_64FC1, 1, 0, 3);
    cv::Sobel(image, grad_y, CV_64FC1, 0, 1, 3);

    const double EPS = 1e-7;

    int cnt = 0;
    double dx_, dy_, div, rot;

    for (int y = 0; y < grad_x.rows; y+=1) {
        for (int x = 0; x < grad_x.cols; x+=1) {

          if(image.at<double>(y,x) > EPS) continue;

            double rx = x - img_width;
            double ry = y - img_height;
            
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

double MotionCompensate::getDensity(const cv::Mat& event_count){
  double total_count = cv::sum(event_count)[0];
  cv::Mat mask = (event_count >= 1.);
  double count = cv::countNonZero(mask);
  double density = total_count / count;
  return density;
}

void MotionCompensate::printInfo(const double& error, const double& density){
  std::cout << "current model " << hx << " " << hy << " " << hz << " " << htheta << std::endl;
  std::cout << "dx: " << dx << " dy: " << dy << " dz: " << dz << " dth: " << dth << std::endl;
  std::cout << "lr_x: " << lr_x << " lr_y: " << lr_y << " lr_div: " << lr_div << " lr_rot: " << lr_rot << std::endl;
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
    for (int i = 0; i < mc_time_map.rows; ++i) {
        for (int j = 0; j < mc_time_map.cols; ++j) {
            double val = mc_time_map.at<double>(i, j);
            if (val > 0.) {
                non_zero_values.push_back(val);
                sum += val;
            }
        }
    }
    double avg = sum/float(non_zero_values.size());

    std::sort(non_zero_values.begin(), non_zero_values.end());

    int size = non_zero_values.size();
    double med;

    med = non_zero_values[int(2*size / 3)];
    std::cout << duration << std::endl;

    // med = size % 2 == 0 ? (non_zero_values[2*size / 3 - 1] + non_zero_values[2*size / 3]) / 2.0 : non_zero_values[size / 2];

    rho = (this->mc_time_map_ - med) / dt;

    background_mask = rho <= 0.;

    foreground_mask = rho > lambda;

    if(filter_small_compo)
      cv::morphologyEx(foreground_mask, foreground_mask, cv::MORPH_CLOSE, kernel);

    // filterComponents(foreground_mask, foreground_mask, 30, 3.0);

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
  mc_time_map_normalized.convertTo(mc_img, CV_32FC1);
  avg_time_map_normalized.convertTo(avg_img, CV_32FC1);

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

  cv::Mat mmm = rho > lambda;

  // int cnt = 0;

  //   for (int i = 0; i < this->foreground_mask.rows; ++i) {
  //       for (int j = 0; j < this->foreground_mask.cols; ++j) {
  //           if (mmm.at<int>(i, j) > 127 && this->mc_time_map_.at<double>(i, j) > 0.) { 
  //             cnt++;
  //               float timestamp = this->mc_time_map_.at<double>(i, j) / duration;
  //               // std::cout << "mc_time_map_ " << timestamp << std::endl;
  //               int bin_index = int(timestamp * histSize);
  //               // std::cout << "bin index" << bin_index << std::endl;
  //               int x = padding + binWidth * bin_index + binWidth / 2;
  //               int y = histImageHeight + padding - 5;
  //               cv::circle(histImage, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
  //           }
  //       }
  //   }

  //   std::cout<<"count "<<cnt<<std::endl;


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
     if (this->mc_time_map_.size() != this->mc_event_count_pos_.size() || this->mc_event_count_neg_.size() != this->mc_time_map_.size()) {
        std::cerr << "Error: Input images must have the same size!" << std::endl;
    }

    cv::Mat normalized1, normalized2, normalized3;
    cv::normalize(this->mc_time_map_, normalized1, 0, 255, cv::NORM_MINMAX);
    cv::normalize(this->mc_event_count_pos_, normalized2, 0, 255, cv::NORM_MINMAX);
    cv::normalize(this->mc_event_count_neg_, normalized3, 0, 255, cv::NORM_MINMAX);

    normalized1.convertTo(normalized1, CV_8UC1);
    normalized2.convertTo(normalized2, CV_8UC1);
    normalized3.convertTo(normalized3, CV_8UC1);

    cv::Mat merged;
    std::vector<cv::Mat> channels = {normalized1, normalized2, normalized3};
    cv::merge(channels, merged);

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    std::tm tm_now;
    localtime_r(&time_t_now, &tm_now);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y%m%d_%H%M%S");

    std::string folder = "/home/x-yq/catkin_ws/src/fast_dynamic/images/what_is_back_ground";

    // if (!std::filesystem::exists(folder)) {
    //     if (!std::filesystem::create_directory(folder)) {
    //         std::cerr << "Error: Could not create folder " << folder << std::endl;
    //     }
    // }

    try {
        std::filesystem::create_directories(folder);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    std::string name = folder + "/" + oss.str() + ".png";

    if (!cv::imwrite(name, merged)) {
        std::cerr << "Error: Could not save the PNG file!" << std::endl;
    }

    std::cout << "Successfully saved the image" << std::endl;
}

}