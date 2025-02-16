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

void MotionCompensate::get_intrinsic_params(){
    switch(this->bag_ind){
    case slider_depth:
      cameraMatrix = (cv::Mat_<double>(3, 3) << 
        335.4194629584808, 0.0, 129.9246633794451, 
        0.0, 335.3529356120773, 99.18643034473205, 
        0.0, 0.0, 1.0);
      distCoeffs = (cv::Mat_<double>(5, 1) << -0.1385927674081299, 0.09337366641919795, -0.0003355869875320301, 0.0001737201582276446, 0.0);
      R = (cv::Mat_<double>(3,3) <<
                    1.0, 0.0, 0.0, 
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0);
      P = (cv::Mat_<double>(3,4) <<
            328.3079223632812, 0.0, 129.7166999252022, 0.0, 
            0.0, 330.1483459472656, 98.78069942616821, 0.0, 
            0.0, 0.0, 1.0, 0.0);
        break;
    case slider_far:
      cameraMatrix = (cv::Mat_<double>(3, 3) <<
        335.4194629584808, 0.0, 129.9246633794451, 
        0.0, 335.3529356120773, 99.18643034473205, 
        0.0, 0.0, 1.0);
      distCoeffs = (cv::Mat_<double>(5,1) << -0.1385927674081299, 0.09337366641919795, -0.0003355869875320301, 0.0001737201582276446, 0.0);
      R = (cv::Mat_<double>(3,3) <<
                    1.0, 0.0, 0.0, 
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0);
      P = (cv::Mat_<double>(3,4)<<
        328.307922, 0.0, 129.7167, 0.0, 
        0.0, 330.148346, 98.780699, 0.0, 
        0.0, 0.0, 1.0, 0.0);
      break;

    case what_is_background:
      cameraMatrix = (cv::Mat_<double>(3, 3) << 
        199.0923665423112, 0.0, 132.1920713777002, 
        0.0, 198.8288204700886, 110.7126600112956, 
        0.0, 0.0, 1.0);
      distCoeffs = (cv::Mat_<double>(5, 1) << 
        -0.3684363117977873, 0.1509472435566583, 
        -0.0002961305343848646, -0.000759431726241032, 0.0);
      R = (cv::Mat_<double>(3,3) <<
                    1.0, 0.0, 0.0, 
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0);
      P =  (cv::Mat_<double>(3,4) <<
            168.6294097900391, 0.0, 135.348079770296, 0.0, 
            0.0, 178.5641784667969, 113.6189973794753, 0.0, 
            0.0, 0.0, 1.0, 0.0);
      break;
    
    case test_vins:
      cameraMatrix = (cv::Mat_<double>(3, 3) << 
        536.3332593298378, 0, 320.90009280822994, 
        0, 536.31797700847164, 234.04853514480661, 
        0, 0, 1);
      distCoeffs = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);
      R = (cv::Mat_<double>(3,3) <<
                    1.0, 0.0, 0.0, 
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0);
      P = (cv::Mat_<double>(3,4) <<
            5.3633325932983780e+02, 0.0, 3.2090009280822994e+02, 0.0, 
            0.0, 5.3631797700847164e+02, 2.3404853514480661e+02, 0.0, 
            0.0, 0.0, 1.0, 0.0);
      break;

    default:
      cameraMatrix = (cv::Mat_<double>(3, 3) << 
        171.37776185565394, 0.0, 120.0, 
        0.0, 171.37776185565394, 90.0, 
        0.0, 0.0, 1.0);

      distCoeffs = (cv::Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);

      R = (cv::Mat_<double>(3,3) <<
                    0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0);
      P = (cv::Mat_<double>(3,4) <<
            171.37776185565394, 0.0, 120.0, 
        0.0, 171.37776185565394, 90.0, 
        0.0, 0.0, 1.0);
      break;
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


}//namespace