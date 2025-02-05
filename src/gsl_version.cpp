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

} AuxdataBestFlow;

cv::Matx23f A_v2(const int x, const int y, const int bag_ind){
  cv::Mat cMatrix;
  
  switch(bag_ind){
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

cv::Matx23f B_v2(const int x, const int y, const int bag_ind){

  cv::Mat cMatrix;
  
  switch(bag_ind){
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

double computeC(const cv::Mat& image, const cv::Mat& inf_image, const int& magnitude_score, int patch_size = 60, int stride = 60) {
    double total_score = 0.0;
    int count = 0;
    
    for (int y = 0; y <= image.rows - patch_size; y += stride) {
        for (int x = 0; x <= image.cols - patch_size; x += stride) {
            cv::Rect roi(x, y, patch_size, patch_size);
            cv::Mat patch1 = image(roi);
            cv::Mat patch2 = inf_image(roi);

            double score;

            if(magnitude_score == 0){
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

            }else if(magnitude_score == 1){
              // norm
              score = cv::norm(patch1, cv::NORM_L2SQR) / static_cast<double>(patch1.rows * patch1.cols);
            }else{
              // std deviation
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
    
    // double psnr = calculatePSNR(image, inf_image);
    // total_score += 0.01*psnr;

    // double ssim = calculateSSIM(image, inf_image);
    // total_score += 5.*ssim;
    
    // std::cout << "psnr " << psnr << std::endl;
    // std::cout << "total_score " << total_score << std::endl;

    return total_score;
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
  double contrast;
  cv::Mat last_img_warped;
  if(image_warped.empty()){
     last_img_warped = computeAvgImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset));
    
  }else{
      image_warped.copyTo(last_img_warped);
  }
  image_warped = computeImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset), *(poAux_data->depth_map), linear_vel, angular_vel, *(poAux_data->bag_ind));
  
  
  contrast = computeC(image_warped, last_img_warped, 2);
  return -contrast;
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
  double contrast;
  cv::Mat last_img_warped;
  if(image_warped.empty()){
     last_img_warped = computeAvgImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset));
    
  }else{
      image_warped.copyTo(last_img_warped);
  }
  image_warped = computeImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset), d_map, *(poAux_data->linear_vel),  *(poAux_data->angular_vel), *(poAux_data->bag_ind));
  
  
  contrast = computeC(image_warped, last_img_warped, 2);
  
  return -contrast;
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
  double contrast;
  cv::Mat last_img_warped;
  if(image_warped.empty()){
     last_img_warped = computeAvgImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset));
    
  }else{
      image_warped.copyTo(last_img_warped);
  }

  cv::Vec3f linear_vel( gsl_vector_get(v,patch_num), gsl_vector_get(v,patch_num + 1), gsl_vector_get(v,patch_num + 2) );
  cv::Vec3f angular_vel( gsl_vector_get(v,patch_num + 3), gsl_vector_get(v,patch_num + 4), gsl_vector_get(v,patch_num + 5) );
  image_warped = computeImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset), d_map, linear_vel, angular_vel, *(poAux_data->bag_ind));  
  
  contrast = computeC(image_warped, last_img_warped, 2);
  
  return -contrast;
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
  this->Z = generateDepthMap(this->depth_patches);
  // this->event_depth_map_.copyTo(this->Z);
  
  const double final_cost = gsl_multimin_fdfminimizer_minimum(solver);

  //Release memory used during optimization
  gsl_multimin_fdfminimizer_free (solver);
  gsl_vector_free (vx);

  return final_cost;
}

} //namespace