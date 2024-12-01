#include <cv_bridge/cv_bridge.h>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <opencv2/highgui.hpp> //cv::imwrite
#include <fast_dynamic/motion_compensate_node.h>
#include <dvs_msgs/Event.h>
#include <numeric>

namespace motion_compensate
{

cv::Point2d MotionCompensate::warpEvent(
  const Model& model,
  const dvs_msgs::Event& event,
  const double& t_ref
){
        cv::Point2d warped_pt;

        double x = event.x;
        double y = event.y;
        double t = event.ts.toSec()  - t_ref; 

        double cosTheta = std::cos(model.htheta);
        double sinTheta = std::sin(model.htheta);

        double rotX = cosTheta * x - sinTheta * y;
        double rotY = sinTheta * x + cosTheta * y;

        double hx = model.hx;
        double hy = model.hy;
        double hz = model.hz;

        // move upward, hy should be negative; move downward, hy should be positive;
        // move leftward, hx should be negative; move rightward, hx should be positive;

        warped_pt.x = x - t * (hx + (hz + 1) * rotX - x);
        warped_pt.y = y - t * (hy + (hz + 1) * rotY - y);

        // std::cout << "warped x displacement : " << warped_pt.x - event.x << " warped y displacement: " << warped_pt.y - event.y << std::endl;
        return warped_pt;

}

void MotionCompensate::computeImageOfWarpedEvents(
  const Model& model,
  const std::vector<dvs_msgs::Event>& events_subset,
  const int width,
  const int height,
  cv::Mat* image_warped,
  cv::Mat* image_event_count
)
{

  *image_event_count = cv::Mat::zeros(height, width, CV_64FC1);
  *image_warped = cv::Mat::zeros(height, width, CV_64FC1);
  const double t_ref = events_subset.front().ts.toSec();

  // Loop through all events
  for (const dvs_msgs::Event& ev : events_subset)
  {
    cv::Point2d warped_pt;
    warped_pt = warpEvent(model, ev, t_ref);
    if (0 <= warped_pt.x && warped_pt.x < width && 0 <= warped_pt.y && warped_pt.y < height)
    {
      double time = ev.ts.toSec() - t0.toSec();
      (*image_warped).at<double>(warped_pt.y, warped_pt.x) += time;
      (*image_event_count).at<double>(warped_pt.y, warped_pt.x) += 1.;
    }
    // ROS_WARN("displacement for x %f", ev.x - warped_pt.x);
  }
  for (int y = 0.; y < height; y+=1) {
    for (int x = 0.; x < width; x+=1) {
      image_warped->at<double>(y, x) /= (image_event_count->at<double>(y, x) + 1e-7);
    }
}

}

double MotionCompensate::computeError(
  const Model& modelPrev,
  const Model& modelCurr
){
  return std::sqrt(
                std::pow(modelCurr.hx - modelPrev.hx, 2) +
                std::pow(modelCurr.hy - modelPrev.hy, 2) +
                std::pow(modelCurr.hz - modelPrev.hz, 2) +
                std::pow(modelCurr.htheta - modelPrev.htheta, 2)
            ); 
}

double MotionCompensate::computeError2(const cv::Mat& image, const cv::Mat& wp_image)
{
  // double contrast = cv::norm(image,cv::NORM_L2SQR) / static_cast<double>(image.rows*image.cols);
  // std::cout << contrast << std::endl;
  // return contrast;
  cv::Mat laplacian;
  cv::Laplacian(image, laplacian, CV_64FC1);
  double laplacian_var = cv::mean(cv::abs(laplacian))[0];
  cv::Mat wp_laplacian;
  cv::Laplacian(wp_image, wp_laplacian, CV_64FC1);
  double wp_laplacian_var = cv::mean(cv::abs(wp_laplacian))[0];
  return laplacian_var - wp_laplacian_var;

}

void MotionCompensate::updateModel(const Model& pre, 
const Grad& grad, Model* cur, const double& contrast){
  // cur->hx = pre.hx + lr_x * grad.dx;
  // cur->hy = pre.hy + lr_y * grad.dy;
  // cur->hz = pre.hz + lr_div *  grad.dz;
  // cur->htheta = pre.htheta + lr_rot * grad.dth;

  cur->hx = pre.hx + grad.dx;
  cur->hy = pre.hy + grad.dy;

  // if(contrast < -0.1){  
    // ROS_WARN("update the rotation and div! Contrast: %f", contrast);
    cur->hz = pre.hz + grad.dz;
    cur->htheta = pre.htheta + grad.dth;
  // }

}

void MotionCompensate::dZThetaUpdate(const cv::Mat &grad_x, const cv::Mat &grad_y,
                                     double &div_r, double &rot_r) {
    double rot = 0.0, div = 0.0, pixel_cnt = 0.0;
    const double EPS = 1e-7;

    for (int y = 0; y < grad_x.rows; ++y) {
      for (int x = 0; x < grad_x.cols; ++x) {
        double dx = grad_x.at<double>(y, x);
        double dy = grad_y.at<double>(y, x);

        if (std::abs(dx) < EPS || std::abs(dy) < EPS) continue;

        double rx = x - grad_x.cols / 2.0;
        double ry = y - grad_x.rows / 2.0;

        if (std::abs(rx) < 0.5 || std::abs(ry) < 0.5) continue;

        rot += rx * dy - ry * dx;
        div += rx * dx + ry * dy;
        pixel_cnt++;
      }
    }

    if (pixel_cnt > 10.) {
        rot /= pixel_cnt;
        div /= pixel_cnt;
    } else {
        rot = div = 0.0;
    }

    // if (std::abs(rot) < 0.5) {
      rot_r = lr_rot * ap_rot.update(rot);
    // } else {
    //   rot_r = 0.0;
    // }
    
    // if (std::abs(div) < 0.5) {
      div_r = lr_div * ap_div.update(div);
    // } else {
    //   div_r = 0.0;
    // }
}

void MotionCompensate::dXYUpdate(const cv::Mat &grad_x,const cv::Mat &grad_y,
                    double &dx_val_r, double &dy_val_r) {
    const double ORIENTATION_THRESHOLD = 30.0;  

    cv::Mat magnitude, orientation;
    cv::cartToPolar(grad_x, grad_y, magnitude, orientation, true);
    const double EPS = 1e-7;

    double dx_sum = 0.0, dy_sum = 0.0;
    int dx_count = 0, dy_count = 0, cnt = 0;
    double dx_, dy_;

    for (int y = 0; y < grad_x.rows; ++y) {
        for (int x = 0; x < grad_x.cols; ++x) {
            double theta = orientation.at<double>(y, x);

            dx_ += grad_x.at<double>(y, x);
            dy_ += grad_y.at<double>(y, x);
            cnt ++;

            if ((theta >= -ORIENTATION_THRESHOLD && theta < ORIENTATION_THRESHOLD) || 
                (theta > 180 - ORIENTATION_THRESHOLD && theta <= 180 + ORIENTATION_THRESHOLD)) {
                dx_sum += grad_x.at<double>(y, x);
                dx_count++;
            } 
            else if ((theta > 90 - ORIENTATION_THRESHOLD && theta < 90 + ORIENTATION_THRESHOLD) || 
                     (theta > 270 - ORIENTATION_THRESHOLD && theta < 270 + ORIENTATION_THRESHOLD)) {
                dy_sum += grad_y.at<double>(y, x);
                dy_count++;
            }
        }
    }

    if (cnt > 100) { 
        dx_sum /= dx_count;
        dy_sum /= dy_count;
        dx_ /= cnt;
        dy_ /=cnt;

        double ratio = std::abs(dx_sum/(dy_sum+1e-7));
        // std::cout << ratio << std::endl;

        double ada_x = ap_x.update(dx_);
        double ada_y = ap_y.update(dy_);
        // if(dx_ * ada_x < 0.){
        //   ada_x = -ada_x;
        // }
        // if(dy_ * ada_y < 0.){
        //   ada_y = -ada_y;
        // }

        if(ratio > 5.0){
          dx_val_r = ada_x * lr_x;
          dy_val_r = (1/ratio) * ada_y;
          dy_val_r = 0.0;
        }
        else if(ratio < 0.2){
          dx_val_r = ratio * ada_x;
          dy_val_r = lr_y * ada_y;
        }
        else{
          dx_val_r = lr_x * ada_x;
          dy_val_r = lr_y * ada_y;
        }

        // dx_val_r = lr_x * ap_x.update(dx_);
        // dy_val_r = lr_y * ap_y.update(dy_);
        
        // std::cout << "dx and dy" << std::endl;
        // std::cout << dx_sum << " " << dy_sum << std::endl;
    } else {
        dx_val_r = dy_val_r = 0.0;
    }
}


void MotionCompensate::diffTimeImage(const cv::Mat &time_image,
                   Grad* grad) {
    double dx_val_r = 0., dy_val_r = 0., dz_val_r = 0., dth_val_r = 0.0;

    // cv::GaussianBlur(time_image, time_image_blurred, cv::Size(0, 0), 1.0);

     cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
        199.0923665423112, 0.0, 132.1920713777002, 
        0.0, 198.8288204700886, 110.7126600112956, 
        0.0, 0.0, 1.0);

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 
        -0.3684363117977873, 0.1509472435566583, 
        -0.0002961305343848646, -0.000759431726241032, 0.0);


    cv::Mat undistortedImage;
    cv::undistort(time_image, undistortedImage, cameraMatrix, distCoeffs);

    cv::Mat time_image_blurred;
    cv::GaussianBlur(undistortedImage, time_image_blurred, cv::Size(0, 0), 1.0);


    cv::Mat grad_x, grad_y;
    // cv::Sobel(time_image, grad_x, CV_64FC1, 1, 0, 3);
    // cv::Sobel(time_image, grad_y, CV_64FC1, 0, 1, 3);
    cv::Mat kernel_x = (cv::Mat_<double>(3, 3) <<
                         -1, 0, 1,
                         -2, 0, 2,
                         -1, 0, 1); 

    cv::Mat kernel_y = (cv::Mat_<double>(3, 3) <<
                         -1, -2, -1,
                          0,  0,  0,
                          1,  2,  1); 

    cv::filter2D(time_image_blurred, grad_x, CV_64FC1, kernel_x);
    cv::filter2D(time_image_blurred, grad_y, CV_64FC1, kernel_y);

    dXYUpdate(grad_x, grad_y, dx_val_r,dy_val_r);
    dZThetaUpdate(grad_x, grad_y, dz_val_r, dth_val_r);
    grad->dx = dx_val_r;
    grad->dy = dy_val_r;
    grad->dz = dz_val_r;
    grad->dth = dth_val_r;

}

double MotionCompensate::getDensity(const cv::Mat& event_count){
  double total_count = cv::sum(event_count)[0];
  cv::Mat mask = (event_count > 1e-5);
  double count = cv::countNonZero(mask); //TODO: debug
  double density = total_count / count;
  return density;
}

void MotionCompensate::printInfo(const Grad& grad, const Model& m, const double& error, const double& density){
  std::cout << "current model " << m.hx << " " << m.hy << " " << m.hz << " " << m.htheta << std::endl;
  std::cout << "dx: " << grad.dx << " dy: " << grad.dy << " dz: " << grad.dz << " dth: " << grad.dth << std::endl;
  std::cout << "current error: " << error << std::endl;
  std::cout << "current density: " << density << std::endl;

}

}