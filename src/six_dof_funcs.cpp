#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <iostream>
#include <opencv2/highgui.hpp> 
#include <fast_dynamic/motion_compensate_node.h>
#include <dvs_msgs/Event.h>
#include <numeric>
#include <gsl/gsl_vector.h>
#include <utility>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <ctime>
#include <sys/stat.h>

using namespace cv;
using namespace std;

typedef struct {
  std::vector<dvs_msgs::Event> *poEvents_subset;
  cv::Size * img_size;
  cv::Mat * depth_map;
  cv::Mat * event_count;
  cv::Mat * event_depth_map;
  cv::Vec3f * linear_vel;
  cv::Vec3f * angular_vel;
  cv::Size * depth_patch_num;
  int* bag_ind;
  int* optimise_method;
  int* contrast_ind;
  double contrast_score = 0.0;

} AuxdataBestFlow;

cv::Mat get_camera_matrix(int bag_ind){
    
    cv::Mat cameraMatrix;
    switch(bag_ind){
        case 0:
            cameraMatrix = (cv::Mat_<double>(3, 3) << 
                335.4194629584808, 0.0, 129.9246633794451, 
                0.0, 335.3529356120773, 99.18643034473205, 
                0.0, 0.0, 1.0);
            break;
        
        case 1:
            cameraMatrix = (cv::Mat_<double>(3, 3) <<
                335.4194629584808, 0.0, 129.9246633794451, 
                0.0, 335.3529356120773, 99.18643034473205, 
                0.0, 0.0, 1.0);
            break;
        
        case 2:
            cameraMatrix = (cv::Mat_<double>(3, 3) << 
                199.0923665423112, 0.0, 132.1920713777002, 
                0.0, 198.8288204700886, 110.7126600112956, 
                0.0, 0.0, 1.0);
            break;
        
        case 3:
            cameraMatrix = (cv::Mat_<double>(3, 3) << 
                536.3332593298378, 0, 320.90009280822994, 
                0, 536.31797700847164, 234.04853514480661, 
                0, 0, 1);
            break;
        
        default:
            cameraMatrix = (cv::Mat_<double>(3, 3) << 
                171.37776185565394, 0.0, 120.0, 
                0.0, 171.37776185565394, 90.0, 
                0.0, 0.0, 1.0);
            break;
  }

  return cameraMatrix;

}

cv::Matx23f A_v2(const int x, const int y, const int bag_ind){
  cv::Mat cMatrix = get_camera_matrix(bag_ind);

  double fx = cMatrix.at<double>(0, 0);
  double fy = cMatrix.at<double>(1, 1);
  double cxx = cMatrix.at<double>(0, 2);
  double cyy = cMatrix.at<double>(1, 2);
  cv::Matx23f A =cv::Matx23f(fx, 0., -(x-cxx), 0., fy, -(y-cyy));
  return A;
}

cv::Matx23f B_v2(const int x, const int y, const int bag_ind){

  cv::Mat cMatrix = get_camera_matrix(bag_ind);

  double fx = cMatrix.at<double>(0, 0);
  double fy = cMatrix.at<double>(1, 1);
  double cxx = cMatrix.at<double>(0, 2);
  double cyy = cMatrix.at<double>(1, 2);

  cv::Matx23f B = cv::Matx23f( 
    (x - cxx)*(y-cyy)/fx, -(fx*fx + (x-cxx)*(x-cxx))/fx, y-cyy, 
    (fy*fy + (y-cyy)*(y-cyy))/fy, -(x-cxx)*(y-cyy)/fx, -(x-cxx));
  return B;
}


double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2) {

    const double C1 = 6.5025, C2 = 58.5225;

    cv::Mat img1_float, img2_float;
    img1.convertTo(img1_float, CV_32FC1);
    img2.convertTo(img2_float, CV_32FC1);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1_float, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_float, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    cv::GaussianBlur(img1_float.mul(img1_float), sigma1_sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_float.mul(img2_float), sigma2_sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img1_float.mul(img2_float), sigma12, cv::Size(11, 11), 1.5);

    sigma1_sq -= mu1_sq;
    sigma2_sq -= mu2_sq;
    sigma12 -= mu1_mu2;

    cv::Mat ssim_map = ((2 * mu1_mu2 + C1).mul(2 * sigma12 + C2)) /
                       ((mu1_sq + mu2_sq + C1).mul(sigma1_sq + sigma2_sq + C2));

    double ssim = cv::mean(ssim_map)[0];
    return ssim;
}

double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat s1;
    cv::absdiff(img1, img2, s1);
    s1.convertTo(s1, CV_32FC1);
    s1 = s1.mul(s1);

    double mse = cv::sum(s1)[0] / (double)(img1.total());
    if (mse == 0) return 0.;

    double max_pixel = 1.0;
    double psnr = 10.0 * std::log10((max_pixel * max_pixel) / mse);
    return psnr;
}


std::pair<double, double> computeC(const cv::Mat& image, const cv::Mat& inf_image, const int& contrast_ind, int patch_size = 60, int stride = 60) {
    double total_score = 0.0;
    int count = 0;
    
    for (int y = 0; y <= image.rows - patch_size; y += stride) {
        for (int x = 0; x <= image.cols - patch_size; x += stride) {
            cv::Rect roi(x, y, patch_size, patch_size);
            cv::Mat patch1 = image(roi);
            cv::Mat patch2 = inf_image(roi);

            double score;

            if(contrast_ind == 2){
              // magnitude
              cv::Mat grad_x, grad_y;
              cv::Sobel(patch1, grad_x, CV_64FC1, 1, 0, 3);
              cv::Sobel(patch1, grad_y, CV_64FC1, 0, 1, 3);

              cv::Mat magnitude;
              cv::magnitude(grad_x, grad_y, magnitude);

              int valid_pixel_count = 0;

              for (int i = 0; i < magnitude.rows; i++) {
                  for (int j = 0; j < magnitude.cols; j++) {
                      if (magnitude.at<double>(i, j) > 1e-7) {  
                          score += magnitude.at<double>(i, j);
                          valid_pixel_count++;
                      }
                  }
              }

              if (valid_pixel_count > 0) {
                  score /= valid_pixel_count; 
              }

            }else if(contrast_ind == 0){
              // norm
              score = cv::norm(patch1, cv::NORM_L2SQR) / static_cast<double>(patch1.rows * patch1.cols);
            }else{
              // variance
              cv::Scalar mean, stddev;
              cv::meanStdDev(patch1, mean, stddev);
              score = stddev[0] * stddev[0];
            }
            
            double ssim = calculateSSIM(patch1, patch2);
            total_score += score + ssim;
            count++;
        }
    }

    if(count > 0){
       total_score /= count;
    }else{
      total_score = 0.;
    }

    double contrast_score = 0.;
    switch (contrast_ind){
    case 0:
      contrast_score = cv::norm(image,cv::NORM_L2SQR) / static_cast<double>(image.rows*image.cols);
      break;

    case 1: {
      cv::Scalar mean, stddev;
      cv::meanStdDev(image, mean, stddev);
      contrast_score = stddev[0] * stddev[0];
      break;
    }

    case 2:{
      cv::Mat grad_x, grad_y;
      cv::Sobel(image, grad_x, CV_64FC1, 1, 0, 3);
      cv::Sobel(image, grad_y, CV_64FC1, 0, 1, 3);

      cv::Mat magnitude;
      cv::magnitude(grad_x, grad_y, magnitude);

      cv::Scalar mean_mag = cv::mean(magnitude, magnitude > 1e-7);
      contrast_score = mean_mag[0];
      break;

    }

    default:
      contrast_score = 0.;
      break;
  }

    // double psnr = calculatePSNR(image, inf_image);
    // total_score += 0.01*psnr;

    // double ssim = calculateSSIM(image, inf_image);
    // total_score += 5.*ssim;
    
    // std::cout << "psnr " << psnr << std::endl;
    // std::cout << "total_score " << total_score << std::endl;

    return std::make_pair(total_score, contrast_score);
}


cv::Mat computeImage(const cv::Size& size, const std::vector<dvs_msgs::Event>& events_subset, const cv::Mat& Z, const cv::Vec3f& linear_vel, const cv::Vec3f& angular_vel, const int bag_ind) {
  int img_width = size.width;
  int img_height = size.height;
  cv::Mat mc_time_map_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);
  cv::Mat mc_event_count_ = cv::Mat::zeros(img_height,img_width, CV_64FC1);

  const double t_ref = events_subset.front().ts.toSec();
  int valid = 0;
  for (const dvs_msgs::Event& ev : events_subset)
  {
    double xx = ev.x;
    double yy = ev.y;
    double dt = ev.ts.toSec() - t_ref; 
    double w_x, w_y;

    double depth = Z.at<double>(yy, xx);
    
    cv::Matx23f A = A_v2(xx, yy, bag_ind);
    cv::Matx23f B = B_v2(xx, yy, bag_ind);
    cv::Vec2f v;
    if (depth <= 0.) {
      v = cv::Vec2f(0.,0.);
      // continue;
    }else{

      v = (1.0f / depth) * A * linear_vel + B * angular_vel;
    }
    
    w_x = xx + v[0] * dt;
    w_y = yy + v[1] * dt;

    if (0. <= w_x && w_x < img_width && 0. <= w_y && w_y < img_height)
    {
      valid ++;
      mc_time_map_.at<double>(w_y, w_x) += dt;
      mc_event_count_.at<double>(w_y, w_x) += 1.;
    }

  }

  cv::Mat invalid_mask = mc_event_count_ < 1.;
  mc_time_map_.setTo(0.0, invalid_mask);
  mc_event_count_.setTo(0.000001, invalid_mask);
  mc_time_map_ = mc_time_map_.mul(1.0 / mc_event_count_);
  mc_time_map_.setTo(0.0, invalid_mask);
  return mc_event_count_;

}

cv::Mat computeAvgImage(const cv::Size& size, const std::vector<dvs_msgs::Event>& events_subset) {
  int img_width = size.width;
  int img_height = size.height;
  cv::Mat avg_time_map_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);
  cv::Mat avg_event_count_ = cv::Mat::zeros(img_height,img_width, CV_64FC1);

  const double t_ref = events_subset.front().ts.toSec();
  int valid = 0;
  for (const dvs_msgs::Event& ev : events_subset)
  {
    double xx = ev.x;
    double yy = ev.y;
    double dt = ev.ts.toSec() - t_ref; 

    avg_time_map_.at<double>(yy, xx) += dt;
    avg_event_count_.at<double>(yy, xx) += 1.;

  }

  cv::Mat invalid_mask = avg_event_count_ < 1.;
  avg_time_map_.setTo(0.0, invalid_mask);
  avg_event_count_.setTo(0.000001, invalid_mask);
  avg_time_map_ = avg_time_map_.mul(1.0 / avg_event_count_);
  avg_time_map_.setTo(0.0, invalid_mask);
  return avg_event_count_;

}

double bilinearInterpolate(double x, double y, double q11, double q12, double q21, double q22) {
    return q11 * (1. - x) * (1. - y) +
           q12 * (1. - x) * y +
           q21 * x * (1. - y) +
           q22 * x * y;

}

cv::Mat generateDepthMap(const std::vector<double>& depth_patches, const cv::Mat& event_count, const cv::Size& size, int x, int y) {
    
    int rows = size.width;
    int cols = size.height;
    
    cv::Mat depth_map(rows, cols, CV_64FC1, cv::Scalar(0.));
    
    int patch_width = cols / x;
    int patch_height = rows / y;


    for (int i = 0; i < y; ++i) {
        for (int j = 0; j < x; ++j) {
            int center_x = j * patch_width + patch_width / 2.;
            int center_y = i * patch_height + patch_height / 2.;
            depth_map.at<double>(center_y, center_x) = depth_patches[i * x + j];
        }
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {

            if(event_count.at<double>(r, c) < 1.){
              depth_map.at<double>(r, c) = 0.;
              continue;
            }

            int left_patch = std::max(0, (c / patch_width));
            int right_patch = std::min(x - 1, left_patch +1 );
            int top_patch = std::max(0, (r / patch_height));
            int bottom_patch = std::min(y - 1, top_patch + 1);

            int left_center_x = left_patch * patch_width + patch_width / 2;
            int right_center_x = right_patch * patch_width + patch_width / 2;
            int top_center_y = top_patch * patch_height + patch_height / 2;
            int bottom_center_y = bottom_patch * patch_height + patch_height / 2;

            double q11 = depth_map.at<double>(top_center_y, left_center_x);
            double q12 = depth_map.at<double>(bottom_center_y, left_center_x);
            double q21 = depth_map.at<double>(top_center_y, right_center_x);
            double q22 = depth_map.at<double>(bottom_center_y, right_center_x);

            double x_diff = right_center_x - left_center_x;
            double y_diff = bottom_center_y - top_center_y;

            double x_ratio = (x_diff == 0) ? 0.5 : (double)(c - left_center_x) / x_diff;
            double y_ratio = (y_diff == 0) ? 0.5 : (double)(r - top_center_y) / y_diff;

            // x_ratio = std::clamp(x_ratio, 0.0, 1.0);
            // y_ratio = std::clamp(y_ratio, 0.0, 1.0);

            depth_map.at<double>(r, c) = bilinearInterpolate(x_ratio, y_ratio, q11, q12, q21, q22);
        }
    }

    return depth_map;
}


cv::Mat image_warped;
double contrast_ff_numerical_vel (const gsl_vector *v, void *adata)
{
    AuxdataBestFlow *poAux_data = (AuxdataBestFlow *) adata;

  // Parameter vector (from GSL to OpenCV)
   cv::Vec3f linear_vel( gsl_vector_get(v,0), gsl_vector_get(v,1), gsl_vector_get(v,2) );
   cv::Vec3f angular_vel( gsl_vector_get(v,3), gsl_vector_get(v,4), gsl_vector_get(v,5) );
  
  // Compute cost
  // double total_score, contrast_score;
  cv::Mat last_img_warped;
  if(image_warped.empty()){
     last_img_warped = computeAvgImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset));
    
  }else{
      image_warped.copyTo(last_img_warped);
  }
  image_warped = computeImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset), *(poAux_data->depth_map), linear_vel, angular_vel, *(poAux_data->bag_ind));
  
  
  auto [total_score, contrast_score] = computeC(image_warped, last_img_warped, *(poAux_data->contrast_ind));
  poAux_data->contrast_score = contrast_score;

  return -total_score;
}

double contrast_ff_numerical_depth (const gsl_vector *v, void *adata)
{
    AuxdataBestFlow *poAux_data = (AuxdataBestFlow *) adata;

  cv::Size s = *(poAux_data->depth_patch_num);
  int patch_num_w = s.width;
  int patch_num_h = s.height;
  int patch_num = patch_num_w*patch_num_h;
  std::vector<double> depth_patches_temp(patch_num, 0.);
  for(int i = 0 ; i < patch_num ; i++){
    depth_patches_temp[i] = gsl_vector_get(v, i);
  }

  cv::Mat d_map = generateDepthMap(depth_patches_temp, *(poAux_data->event_count), *(poAux_data->img_size), patch_num_w, patch_num_h);
    
  
  // int count = 0;
  // cv::Mat d_img = *(poAux_data->event_depth_map);
  // cv::Mat d_map = cv::Mat::zeros(d_img.rows, d_img.cols, CV_64FC1);
  // for (int y = 0; y < d_img.rows; y ++) {
  //   for (int x = 0; x < d_img.cols; x ++) {
  //     if(d_img.at<double>(y,x)==0.){
  //       continue;
  //     }
  //     d_map.at<double>(y,x) = gsl_vector_get(v, count);
  //     count ++;
  //   }
  // }

  // Compute cost
  // double total_score, contrast_score;
  cv::Mat last_img_warped;
  if(image_warped.empty()){
     last_img_warped = computeAvgImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset));
    
  }else{
      image_warped.copyTo(last_img_warped);
  }
  image_warped = computeImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset), d_map, *(poAux_data->linear_vel),  *(poAux_data->angular_vel), *(poAux_data->bag_ind));
  
  
  auto [total_score, contrast_score] = computeC(image_warped, last_img_warped, *(poAux_data->contrast_ind));
  poAux_data->contrast_score = contrast_score;
  return -total_score;
}

double contrast_ff_numerical (const gsl_vector *v, void *adata)
{
    AuxdataBestFlow *poAux_data = (AuxdataBestFlow *) adata;

  cv::Size s = *(poAux_data->depth_patch_num);
  int patch_num_w = s.width;
  int patch_num_h = s.height;
  int patch_num = patch_num_w*patch_num_h;
  std::vector<double> depth_patches_temp(patch_num, 0.);
  for(int i = 0 ; i < patch_num ; i++){
    if(gsl_vector_get(v, i) < 0.) 
      depth_patches_temp[i] = 0.;
    else 
      depth_patches_temp[i] = gsl_vector_get(v, i);
  }

  cv::Mat d_map = generateDepthMap(depth_patches_temp, *(poAux_data->event_count),*(poAux_data->img_size), patch_num_w, patch_num_h);
    
  // Compute cost
  // double total_score, contrast_score;
  cv::Mat last_img_warped;
  if(image_warped.empty()){
     last_img_warped = computeAvgImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset));
    
  }else{
      image_warped.copyTo(last_img_warped);
  }

  cv::Vec3f linear_vel( gsl_vector_get(v,patch_num), gsl_vector_get(v,patch_num + 1), gsl_vector_get(v,patch_num + 2) );
  cv::Vec3f angular_vel( gsl_vector_get(v,patch_num + 3), gsl_vector_get(v,patch_num + 4), gsl_vector_get(v,patch_num + 5) );
  image_warped = computeImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset), d_map, linear_vel, angular_vel, *(poAux_data->bag_ind));  
  
  auto [total_score, contrast_score] = computeC(image_warped, last_img_warped, *(poAux_data->contrast_ind));
  poAux_data->contrast_score = contrast_score;
  return -total_score;
}

double vs_gsl_Gradient_ForwardDiff (
    const gsl_vector * x, /**< [in] Point at which the gradient is to be evaluated */
    void * data,          /**< [in] Optional parameters passed directly to the function func_f */
    double (*func_f)(const gsl_vector * x, void *data), /**< [in] User-supplied routine that returns the value of the function at x */
    gsl_vector * J,       /**< [out] Gradient vector (same length as x) */
    double dh = 1e-6      /**< [in] Increment in variable for numerical differentiation */
    )
{
  // Evaluate vector function at x
  double fx = func_f(x, data);

  // Clone the parameter vector x
  gsl_vector *xh = gsl_vector_alloc (J->size);
  gsl_vector_memcpy(xh, x);

  for (int j=0; j < J->size; j++)
  {
    gsl_vector_set(xh,j,gsl_vector_get(x,j)+dh); // Take a (forward) step in the current dimension
    double fh = func_f(xh, data); // Evaluate vector function at new x
    gsl_vector_set(J ,j,fh-fx); // Finite difference approximation (except for 1/dh factor)
    gsl_vector_set(xh,j,gsl_vector_get(x,j)); // restore original value of the current variable
  }
  gsl_vector_scale(J, 1.0/dh);

  gsl_vector_free(xh);

  return fx;
}


void contrast_fdf_numerical (const gsl_vector *v, void *adata, double *f, gsl_vector *df)
{
  // Finite difference approximation
  AuxdataBestFlow *poAux_data = (AuxdataBestFlow *) adata;

  if(*(poAux_data->optimise_method) == 0){
    *f = vs_gsl_Gradient_ForwardDiff (v, adata, contrast_ff_numerical_vel, df, 1e-5);
  }else if(*(poAux_data->optimise_method) == 1){
    *f = vs_gsl_Gradient_ForwardDiff (v, adata, contrast_ff_numerical_depth, df, 1e-3);
  }else{
    *f = vs_gsl_Gradient_ForwardDiff (v, adata, contrast_ff_numerical, df, 1e-3);
  }
}


void contrast_df_numerical (const gsl_vector *v, void *adata, gsl_vector *df)
{
  double cost;
  contrast_fdf_numerical (v, adata, &cost, df);
}



namespace motion_compensate
{

// cv::Mat MotionCompensate::getGTDepthMap_v2(const double time){
    
//   std::cout << "time of the first event in the slice " << time << std::endl;

//    size_t index = 1;
//     for (size_t i = 0; i < this->depth_map_timestamps.size(); ++i)
//     {
//         if (this->depth_map_timestamps[i] > time)
//         {
//             index = i;
//             break;
//         }
//     }

//     std::cout << "depth index " << index << ": " << this->depth_map_timestamps[index] << std::endl;
//     cv::Mat depth_map = depth_maps_[index];
//     return depth_map;

// }

cv::Mat MotionCompensate::getGTDepthMap_v2(int slice_number) {
    const int window_size = 8;
    const int total_depth_maps = 30;

    int depth_index = slice_number / window_size + 1;

    depth_index = std::min(depth_index, total_depth_maps - 1);
    
    std::cout << "Total number of depth maps " << this->depth_maps_.size() << std::endl;

    std::cout << "Slice " << slice_number << " -> Depth map " << depth_index << std::endl;
    return this->depth_maps_[depth_index];
}



void MotionCompensate::initialize_v2(const std::vector<dvs_msgs::Event>& events_subset){

 if(random_initial){
    
    double min_depth = 1.0; // meters
    double max_depth = 2.0; // meters
    for (int i = 0; i < this->Z.rows; ++i) {
      for (int j = 0; j < this->Z.cols; ++j) {
        if(this->event_count_.at<double>(i, j) < 1.){
          this->Z.at<double>(i, j) = 0.;
          // continue;
        }
        else{
          this->Z.at<double>(i, j) = 1.;
          // this->Z.at<double>(i, j) = min_depth + static_cast<double>(rand()) / RAND_MAX * (max_depth - min_depth);

        }
      
      }
    }
    
    if(slice_number == 0){
    
      this->linear_vel_cam = cv::Vec3f(0.0, 0., 0.);
      this->angular_vel_cam = cv::Vec3f(0.0, 0.0, 0.0);
    }

 }
 else{
    // this->Z = getGTDepthMap_v2((events_subset.back().ts.toSec()+ events_subset.front().ts.toSec())/2.);
    this->Z = getGTDepthMap_v2(slice_number);
    //TODO: test the bag_ind should be only in the slice 0?
    if(slice_number == 0){
      this->linear_vel_cam = Vec3f(0.0,0.0,0.0);
      this->angular_vel_cam = Vec3f(0.0,0.0,0.0);
    }
 }

}

cv::Matx23f MotionCompensate::A_v2(const int x, const int y){

    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cxx = cameraMatrix.at<double>(0, 2);
    double cyy = cameraMatrix.at<double>(1, 2);
    cv::Matx23f A =cv::Matx23f(fx, 0., -(x-cxx), 0., fy, -(y-cyy));
    return A;
}

cv::Matx23f MotionCompensate::B_v2(const int x, const int y){

  double fx = cameraMatrix.at<double>(0, 0);
  double fy = cameraMatrix.at<double>(1, 1);
  double cxx = cameraMatrix.at<double>(0, 2);
  double cyy = cameraMatrix.at<double>(1, 2);

  cv::Matx23f B = cv::Matx23f( 
    (x - cxx)*(y-cyy)/fx, -(fx*fx + (x-cxx)*(x-cxx))/fx, y-cyy, 
    (fy*fy + (y-cyy)*(y-cyy))/fy, -(x-cxx)*(y-cyy)/fx, -(x-cxx));
  return B;
}

cv::Mat MotionCompensate::computeImageOfWarpedEvents_v2(const std::vector<dvs_msgs::Event>& events_subset, const int ImageType) {
  
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
    if (depth <= 0.) {
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
  if(ImageType==EventCount) return this->mc_event_count_;
  else return this->mc_time_map_;

}

void MotionCompensate::printInfo_v2(const double& total_score, const double& contrast_score){

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
  std::cout << "current total score: " << total_score << std::endl;
  std::cout << "current contrast: " << contrast_score << std::endl;
  std::cout << "current iteration num: " << this->iter << std::endl;

}

void MotionCompensate::logInfo_v2(const int slice_number, 
                                   const std::string& minimizer_type,
                                   const double& total_score, 
                                   const double& contrast_score, 
                                   bool evaluate = false,
                                   bool remove = false)
{
    std::ostringstream oss;
    oss << "/home/x-yq/catkin_ws/src/fast_dynamic/files/gt_bag_" << this->bag_ind 
        << "_cType_" << this->contrast_ind << ".txt";
    const std::string log_filename = oss.str();

    if (remove) {
        struct stat buffer;
        if (stat(log_filename.c_str(), &buffer) == 0) {  // 文件存在
            std::remove(log_filename.c_str());  // 删除文件
            std::cout << "File exists, deleted: " << log_filename << std::endl;
        }
    }

    // 检查文件是否存在，如果不存在则创建
    std::ifstream infile(log_filename);
    if (!infile.is_open()) {
        std::ofstream outfile(log_filename);  // 创建文件
        if (outfile.is_open()) {
            std::cout << "File created: " << log_filename << std::endl;
        }
    }

    // ========== EVALUATE ==========
    if (evaluate) {
        std::ifstream infile(log_filename);
        std::string line;
        
        // For best slice tracking
        int best_event_count_slice = -1;
        double best_event_count_contrast = -std::numeric_limits<double>::infinity();
        std::string best_event_count_minimizer;
        std::vector<std::string> best_event_count_block;

        int best_time_map_slice = -1;
        double best_time_map_contrast = -std::numeric_limits<double>::infinity();
        std::string best_time_map_minimizer;
        std::vector<std::string> best_time_map_block;

        // For averages
        double event_count_contrast_sum = 0;
        int event_count_count = 0;
        double time_map_contrast_sum = 0;
        int time_map_count = 0;

        std::vector<std::string> current_block;
        int current_slice = -1;
        std::string current_minimizer;

        while (std::getline(infile, line)) {
            if (line.find("Slice") == 0) {
                if (!current_block.empty()) {
                    // 处理当前 block 的 contrast
                    for (const auto& l : current_block) {
                        if (l.find("current contrast:") != std::string::npos) {
                            std::istringstream iss(l);
                            std::string token;
                            double contrast_val;
                            while (iss >> token) {
                                if (std::istringstream(token) >> contrast_val) {
                                    // 处理 Event Count Minimizer
                                    if (current_minimizer == "Event Count Minimizer") {
                                        if (contrast_val > best_event_count_contrast) {
                                            best_event_count_contrast = contrast_val;
                                            best_event_count_slice = current_slice;
                                            best_event_count_minimizer = current_minimizer;
                                            best_event_count_block = current_block;
                                        }
                                        event_count_contrast_sum += contrast_val;
                                        event_count_count++;
                                    }
                                    // 处理 Time Map Minimizer
                                    else if (current_minimizer == "Time Map Minimizer") {
                                        if (contrast_val > best_time_map_contrast) {
                                            best_time_map_contrast = contrast_val;
                                            best_time_map_slice = current_slice;
                                            best_time_map_minimizer = current_minimizer;
                                            best_time_map_block = current_block;
                                        }
                                        time_map_contrast_sum += contrast_val;
                                        time_map_count++;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    current_block.clear();
                }

                std::istringstream iss(line);
                std::string dummy;
                iss >> dummy >> current_slice >> dummy >> current_minimizer;
            }
            current_block.push_back(line);
        }

        // 最后一块也处理
        if (!current_block.empty()) {
            for (const auto& l : current_block) {
                if (l.find("current contrast:") != std::string::npos) {
                    std::istringstream iss(l);
                    std::string token;
                    double contrast_val;
                    while (iss >> token) {
                        if (std::istringstream(token) >> contrast_val) {
                            if (current_minimizer == "Event Count Minimizer") {
                                if (contrast_val > best_event_count_contrast) {
                                    best_event_count_contrast = contrast_val;
                                    best_event_count_slice = current_slice;
                                    best_event_count_minimizer = current_minimizer;
                                    best_event_count_block = current_block;
                                }
                                event_count_contrast_sum += contrast_val;
                                event_count_count++;
                            }
                            else if (current_minimizer == "Time Map Minimizer") {
                                if (contrast_val > best_time_map_contrast) {
                                    best_time_map_contrast = contrast_val;
                                    best_time_map_slice = current_slice;
                                    best_time_map_minimizer = current_minimizer;
                                    best_time_map_block = current_block;
                                }
                                time_map_contrast_sum += contrast_val;
                                time_map_count++;
                            }
                            break;
                        }
                    }
                }
            }
        }

        // 计算平均值
        double event_count_average = event_count_count > 0 ? event_count_contrast_sum / event_count_count : 0.0;
        double time_map_average = time_map_count > 0 ? time_map_contrast_sum / time_map_count : 0.0;

        // 在日志文件结尾写入最佳结果和平均值
        std::ofstream outfile(log_filename, std::ios_base::app);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open log file!" << std::endl;
            return;
        }

        outfile << "========= Best Event Count Minimizer =========" << std::endl;
        outfile << "Best slice number: " << best_event_count_slice << std::endl;
        outfile << "Best contrast: " << best_event_count_contrast << std::endl;
        outfile << "Minimizer type: " << best_event_count_minimizer << std::endl;
        outfile << "Full log block:" << std::endl;
        for (const auto& l : best_event_count_block) {
            outfile << l << std::endl;
        }

        outfile << "========= Best Time Map Minimizer =========" << std::endl;
        outfile << "Best slice number: " << best_time_map_slice << std::endl;
        outfile << "Best contrast: " << best_time_map_contrast << std::endl;
        outfile << "Minimizer type: " << best_time_map_minimizer << std::endl;
        outfile << "Full log block:" << std::endl;
        for (const auto& l : best_time_map_block) {
            outfile << l << std::endl;
        }

        outfile << "========= Averages =========" << std::endl;
        outfile << "Average contrast (Event Count Minimizer): " << event_count_average << std::endl;
        outfile << "Average contrast (Time Map Minimizer): " << time_map_average << std::endl;

        outfile << "------------------------------" << std::endl;
        outfile.close();

        return;
    }

    // ========== 正常 LOG 写入 ==========
    std::ofstream outfile(log_filename, std::ios_base::app);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open log file!" << std::endl;
        return;
    }

    outfile << "Slice " << slice_number << " MinimizerType " << minimizer_type << std::endl;
    outfile << std::fixed << std::setprecision(6);
    outfile << "current linear vel x: " << linear_vel_cam[0] 
            << " y: " << linear_vel_cam[1] 
            << " z: " << linear_vel_cam[2] << std::endl;
    outfile << "current angular vel x: " << angular_vel_cam[0] 
            << " y: " << angular_vel_cam[1] 
            << " z: " << angular_vel_cam[2] << std::endl;
    outfile << "linear vel dx: " << grad_linear_vel[0] 
            << " dy: " << grad_linear_vel[1] 
            << " dz: " << grad_linear_vel[2] << std::endl;
    outfile << "angular vel dx: " << grad_angular_vel[0] 
            << " dy: " << grad_angular_vel[1] 
            << " dz: " << grad_angular_vel[2] << std::endl;
    outfile << "current total score: " << total_score << std::endl;
    outfile << "current contrast: " << contrast_score << std::endl;
    outfile << "current iteration num: " << this->iter << std::endl;
    outfile << "------------------------------" << std::endl;

    outfile.close();
}


// cv::Mat MotionCompensate::generateDepthMap(const std::vector<double>& depth_patches) {

//     this->Z = cv::Mat::zeros(this->img_height, this->img_width, CV_64FC1);
    
//     int patch_width = this->img_width / this->depth_x_bin_num;
//     int patch_height = this->img_height / this->depth_y_bin_num;


//     for (int i = 0; i < this->depth_y_bin_num; ++i) {
//         for (int j = 0; j < this->depth_x_bin_num; ++j) {
//             int center_x = j * patch_width + patch_width / 2;
//             int center_y = i * patch_height + patch_height / 2;
//             this->Z.at<double>(center_y, center_x) = depth_patches[i * this->depth_x_bin_num + j];
//         }
//     }

//     for (int r = 0; r < this->img_height; ++r) {
//         for (int c = 0; c < this->img_width; ++c) {

//           if(this->event_count_.at<double>(r, c) < 1.){
//             this->Z.at<double>(r, c) = 0.;
//             continue;
//           }

//           int left_patch = std::max(0, (c / patch_width));
//           int right_patch = std::min(int(this->depth_x_bin_num) - 1, left_patch +1 );
//           int top_patch = std::max(0, (r / patch_height));
//           int bottom_patch = std::min(int(this->depth_y_bin_num) - 1, top_patch + 1);

//           int left_center_x = left_patch * patch_width + patch_width / 2;
//           int right_center_x = right_patch * patch_width + patch_width / 2;
//           int top_center_y = top_patch * patch_height + patch_height / 2;
//           int bottom_center_y = bottom_patch * patch_height + patch_height / 2;

//           double q11 = this->Z.at<double>(top_center_y, left_center_x);
//           double q12 = this->Z.at<double>(bottom_center_y, left_center_x);
//           double q21 = this->Z.at<double>(top_center_y, right_center_x);
//           double q22 = this->Z.at<double>(bottom_center_y, right_center_x);


//           double x_diff = right_center_x - left_center_x;
//           double y_diff = bottom_center_y - top_center_y;

//           double x_ratio = (x_diff == 0) ? 0.5 : (double)(c - left_center_x) / x_diff;
//           double y_ratio = (y_diff == 0) ? 0.5 : (double)(r - top_center_y) / y_diff;

//           // x_ratio = std::clamp(x_ratio, 0.0, 1.0);
//           // y_ratio = std::clamp(y_ratio, 0.0, 1.0);

//           this->Z.at<double>(r, c) = bilinearInterpolate(x_ratio, y_ratio, q11, q12, q21, q22);
//         }
//     }

//     return this->Z;
// }


double MotionCompensate::maximizeContrast(const std::vector<dvs_msgs::Event>& events_subset, const int& method)
{
  //Solver/minimizer type (algorithm):
  const gsl_multimin_fdfminimizer_type *solver_type;
  solver_type = gsl_multimin_fdfminimizer_conjugate_fr;

  //Auxiliary data for the cost function
  AuxdataBestFlow oAuxdata;
  oAuxdata.poEvents_subset = const_cast<std::vector<dvs_msgs::Event>*>(&events_subset);
  oAuxdata.linear_vel = &this->linear_vel_cam;
  oAuxdata.angular_vel = &this->angular_vel_cam;
  oAuxdata.event_count = &this->event_count_;

  // this->Z = generateDepthMap(this->depth_patches);
  // this->event_depth_map_.copyTo(this->Z);

  oAuxdata.depth_map = &this->Z;
  oAuxdata.event_depth_map = &this->event_depth_map_;
  oAuxdata.img_size = new cv::Size(this->img_width, this->img_height);
  oAuxdata.depth_patch_num = new cv::Size(this->depth_x_bin_num, this->depth_y_bin_num);
  oAuxdata.bag_ind = new int(this->bag_ind);
  oAuxdata.contrast_ind = new int(this->contrast_ind);
  oAuxdata.optimise_method = new int(method);

  //Routines to compute the cost function and its derivatives
  gsl_multimin_function_fdf solver_info;

  int num_params;
  gsl_vector *vx;

  if(method == ByVel){

    num_params = 6; // Size of global flow
    solver_info.n = num_params; // Size of the parameter vector
    solver_info.f = contrast_ff_numerical_vel; // Cost function
    solver_info.df = contrast_df_numerical; // Gradient of cost function
    solver_info.fdf = contrast_fdf_numerical; // Cost and gradient functions
    solver_info.params = &oAuxdata; // Auxiliary data

    //Initial parameter vector
    vx = gsl_vector_alloc (num_params);

    // FILL IN ...
    // gsl_vector_set (vx, ...  
    gsl_vector_set(vx, 0, this->linear_vel_cam[0]);
    gsl_vector_set(vx, 1, this->linear_vel_cam[1]);
    gsl_vector_set(vx, 2, this->linear_vel_cam[2]);
    gsl_vector_set(vx, 3, this->angular_vel_cam[0]);
    gsl_vector_set(vx, 4, this->angular_vel_cam[1]);
    gsl_vector_set(vx, 5, this->angular_vel_cam[2]);

  }else if(method == ByDepth){

    num_params = this->depth_x_bin_num * this->depth_y_bin_num; // Size of global flow
    solver_info.n = num_params; // Size of the parameter vector
    solver_info.f = contrast_ff_numerical_depth; // Cost function
    solver_info.df = contrast_df_numerical; // Gradient of cost function
    solver_info.fdf = contrast_fdf_numerical; // Cost and gradient functions
    solver_info.params = &oAuxdata; // Auxiliary data

    //Initial parameter vector
    vx = gsl_vector_alloc (num_params);

    for(int i = 0 ; i < this->depth_patches.size() ; i++){
      gsl_vector_set(vx, i, this->depth_patches[i]);
    }

    // int nonzero_depth_value = cv::countNonZero(this->event_depth_map_);
    // num_params = nonzero_depth_value;
    // solver_info.n = num_params; // Size of the parameter vector
    // solver_info.f = contrast_ff_numerical_depth; // Cost function
    // solver_info.df = contrast_df_numerical; // Gradient of cost function
    // solver_info.fdf = contrast_fdf_numerical; // Cost and gradient functions
    // solver_info.params = &oAuxdata; // Auxiliary data

    // vx = gsl_vector_alloc (num_params);

    // int count = 0;
    // for (int y = 0; y < this->event_depth_map_.rows; ++y) {
    //   for (int x = 0; x < this->event_depth_map_.cols; ++x) {
    //     if(this->event_depth_map_.at<double>(y,x) > 0.){
    //       gsl_vector_set(vx, count, this->event_depth_map_.at<double>(y,x));
    //       count ++;
    //     }
    //   }
    // }
    
  }else{
    num_params = 6 + this->depth_x_bin_num * this->depth_y_bin_num; // Size of global flow
    solver_info.n = num_params; // Size of the parameter vector
    solver_info.f = contrast_ff_numerical; // Cost function
    solver_info.df = contrast_df_numerical; // Gradient of cost function
    solver_info.fdf = contrast_fdf_numerical; // Cost and gradient functions
    solver_info.params = &oAuxdata; // Auxiliary data

    //Initial parameter vector
    vx = gsl_vector_alloc (num_params);

    for(int i = 0 ; i < this->depth_patches.size() ; i++){
      gsl_vector_set(vx, i, this->depth_patches[i]);
    }

    gsl_vector_set(vx, this->depth_patches.size(), this->linear_vel_cam[0]);
    gsl_vector_set(vx, this->depth_patches.size() + 1, this->linear_vel_cam[1]);
    gsl_vector_set(vx, this->depth_patches.size() + 2, this->linear_vel_cam[2]);
    gsl_vector_set(vx, this->depth_patches.size() + 3, this->angular_vel_cam[0]);
    gsl_vector_set(vx, this->depth_patches.size() + 4, this->angular_vel_cam[1]);
    gsl_vector_set(vx, this->depth_patches.size() + 5, this->angular_vel_cam[2]);
  }

  //Initialize solver
  gsl_multimin_fdfminimizer *solver = gsl_multimin_fdfminimizer_alloc (solver_type, num_params);
  const double initial_step_size = 10;
  double tol = 0.01;

  gsl_multimin_fdfminimizer_set (solver, &solver_info, vx, initial_step_size, tol);

  const double initial_cost = solver->f;

  //ITERATE

  const int num_max_line_searches = this->maxIterations;
  int status;
  const double epsabs_grad = 1e-5, tolfun=1e-7;
  double cost_new = 1e9, cost_old = 1e9;
  size_t iter = 0;

  do
  {
    iter++;
    cost_old = cost_new;
    status = gsl_multimin_fdfminimizer_iterate (solver);

    if (status == GSL_SUCCESS)
    {
      //Test convergence due to stagnation in the value of the function
      cost_new = gsl_multimin_fdfminimizer_minimum(solver);
      if ( fabs( 1-cost_new/(cost_old+1e-7) ) < tolfun )
      {

        break;
      }
      else
        status = GSL_CONTINUE;
    }

    //Test convergence due to absolute norm of the gradient
    if (GSL_SUCCESS == gsl_multimin_test_gradient (solver->gradient, epsabs_grad))
    {
      break;
    }

    if (status != GSL_CONTINUE)
    {
      // The iteration was not successful (did not reduce the function value)
      break;
    }
  }
  while (status == GSL_CONTINUE && iter < num_max_line_searches);

  //Convert from GSL to OpenCV format
  gsl_vector *final_x = gsl_multimin_fdfminimizer_x(solver);

  // FILL IN ...  the return value of vel_ using  final_x

  if(method == ByVel){
    this->linear_vel_cam[0] = gsl_vector_get(final_x, 0);
    this->linear_vel_cam[1]= gsl_vector_get(final_x, 1);
    this->linear_vel_cam[2] = gsl_vector_get(final_x, 2);
    
    this->angular_vel_cam[0] = gsl_vector_get(final_x, 3);
    this->angular_vel_cam[1] = gsl_vector_get(final_x, 4);
    this->angular_vel_cam[2] = gsl_vector_get(final_x, 5);

  }else if(method == ByDepth){

    for(int i = 0 ; i < this->depth_patches.size(); i++){
      if(gsl_vector_get(final_x, i) < 0.) 
        this->depth_patches[i] = 0.;
      else 
        this->depth_patches[i] = gsl_vector_get(final_x, i);
    }

    //TODO: event depth map
    // int count = 0;
    // for (int y = 0; y < this->event_depth_map_.rows; ++y) {
    //   for (int x = 0; x < this->event_depth_map_.cols; ++x) {
    //     if(this->event_depth_map_.at<double>(y,x)==0.){
    //       continue;
    //     }
    //     this->event_depth_map_.at<double>(y,x) = gsl_vector_get(final_x, count);
    //     count ++;
    //   }
    // }


  }else{

    for(int i = 0 ; i < this->depth_patches.size(); i++){
      if(gsl_vector_get(final_x, i) < 0.) 
        this->depth_patches[i] = 0.;
      else 
        this->depth_patches[i] = gsl_vector_get(final_x, i);
    }

    this->linear_vel_cam[0] = gsl_vector_get(final_x, this->depth_patches.size());
    this->linear_vel_cam[1]= gsl_vector_get(final_x, this->depth_patches.size() + 1);
    this->linear_vel_cam[2] = gsl_vector_get(final_x, this->depth_patches.size() + 2);
    
    this->angular_vel_cam[0] = gsl_vector_get(final_x, this->depth_patches.size() + 3);
    this->angular_vel_cam[1] = gsl_vector_get(final_x, this->depth_patches.size() + 4);
    this->angular_vel_cam[2] = gsl_vector_get(final_x, this->depth_patches.size() + 5);

  }
  // this->Z = generateDepthMap(this->depth_patches);
  // this->event_depth_map_.copyTo(this->Z);
  
  const double final_cost = gsl_multimin_fdfminimizer_minimum(solver);

  //Release memory used during optimization
  gsl_multimin_fdfminimizer_free (solver);
  gsl_vector_free (vx);

  this->ContrastScore = oAuxdata.contrast_score;
  this->iter = (int)iter;
  return final_cost;
}

} //namespace