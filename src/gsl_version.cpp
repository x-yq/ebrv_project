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
  int* bag_ind;

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


double computeC(const cv::Mat& image)
{

// // std deviation
//   cv::Scalar mean, stddev;
//   cv::meanStdDev(image, mean, stddev);
//   return stddev[0] * stddev[0];

//   // norm
  return cv::norm(image,cv::NORM_L2SQR) / static_cast<double>(image.rows*image.cols);

// // magnitude
//   cv::Mat grad_x, grad_y;
//   cv::Sobel(image, grad_x, CV_64FC1, 1, 0, 3);
//   cv::Sobel(image, grad_y, CV_64FC1, 0, 1, 3);

//   cv::Mat magnitude;
//   cv::magnitude(grad_x, grad_y, magnitude);

//   double mean_value;
//   int valid_pixel_count = 0;

//   for (int i = 0; i < magnitude.rows; i++) {
//       for (int j = 0; j < magnitude.cols; j++) {
//           if (magnitude.at<double>(i, j) > 1e-7) {  
//               mean_value += magnitude.at<double>(i, j);
//               valid_pixel_count++;
//           }
//       }
//   }

//   if (valid_pixel_count > 0) {
//       mean_value /= valid_pixel_count; 
//   }

//   return mean_value;

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
    if (depth < 0.) {
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


double contrast_ff_numerical (const gsl_vector *v, void *adata)
{
    AuxdataBestFlow *poAux_data = (AuxdataBestFlow *) adata;

  // Parameter vector (from GSL to OpenCV)
   cv::Vec3f linear_vel( gsl_vector_get(v,0), gsl_vector_get(v,1), gsl_vector_get(v,2));
    cv::Vec3f angular_vel( gsl_vector_get(v,3), gsl_vector_get(v,4), gsl_vector_get(v,5) );

  // Compute cost
  double contrast;
  cv::Mat image_warped;
  image_warped = computeImage(*(poAux_data->img_size), *(poAux_data->poEvents_subset), *(poAux_data->depth_map), linear_vel, angular_vel, *(poAux_data->bag_ind));
  contrast = computeC(image_warped);
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
  *f = vs_gsl_Gradient_ForwardDiff (v, adata, contrast_ff_numerical, df, 1e-3);
}


void contrast_df_numerical (const gsl_vector *v, void *adata, gsl_vector *df)
{
  double cost;
  contrast_fdf_numerical (v, adata, &cost, df);
}


namespace motion_compensate
{

double MotionCompensate::maximizeContrast(const std::vector<dvs_msgs::Event>& events_subset)
{
  //Solver/minimizer type (algorithm):
  const gsl_multimin_fdfminimizer_type *solver_type;
  solver_type = gsl_multimin_fdfminimizer_conjugate_fr;

  //Auxiliary data for the cost function
  AuxdataBestFlow oAuxdata;
//   oAuxdata.poEvents_subset = &events_subset;
  oAuxdata.poEvents_subset = const_cast<std::vector<dvs_msgs::Event>*>(&events_subset);
  oAuxdata.depth_map = &this->Z;
  oAuxdata.img_size = new cv::Size(this->img_width, this->img_height);
  oAuxdata.bag_ind = new int(this->bag_ind);

  //Routines to compute the cost function and its derivatives
  gsl_multimin_function_fdf solver_info;

  const int num_params = 6; // Size of global flow
  solver_info.n = num_params; // Size of the parameter vector
  solver_info.f = contrast_ff_numerical; // Cost function
  solver_info.df = contrast_df_numerical; // Gradient of cost function
  solver_info.fdf = contrast_fdf_numerical; // Cost and gradient functions
  solver_info.params = &oAuxdata; // Auxiliary data

  //Initial parameter vector
  gsl_vector *vx = gsl_vector_alloc (num_params);

  // FILL IN ...
  // gsl_vector_set (vx, ...  
  gsl_vector_set(vx, 0, this->linear_vel_cam[0]);
  gsl_vector_set(vx, 1, this->linear_vel_cam[1]);
  gsl_vector_set(vx, 2, this->linear_vel_cam[2]);
  gsl_vector_set(vx, 3, this->angular_vel_cam[0]);
  gsl_vector_set(vx, 4, this->angular_vel_cam[1]);
  gsl_vector_set(vx, 5, this->angular_vel_cam[2]);


  //Initialize solver
  gsl_multimin_fdfminimizer *solver = gsl_multimin_fdfminimizer_alloc (solver_type, num_params);
  const double initial_step_size = 10;
  double tol = 0.01;

  gsl_multimin_fdfminimizer_set (solver, &solver_info, vx, initial_step_size, tol);

  const double initial_cost = solver->f;

  //ITERATE

  const int num_max_line_searches = this->maxIterations;
  int status;
  const double epsabs_grad = 1e-7, tolfun=1e-7;
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

  //SAVE RESULTS (best global flow velocity)

  //Convert from GSL to OpenCV format
  gsl_vector *final_x = gsl_multimin_fdfminimizer_x(solver);

  // FILL IN ...  the return value of vel_ using  final_x
  this->linear_vel_cam[0] = gsl_vector_get(final_x, 0);
  this->linear_vel_cam[1]= gsl_vector_get(final_x, 1);
  this->linear_vel_cam[2] = gsl_vector_get(final_x, 2);
  
  this->angular_vel_cam[0] = gsl_vector_get(final_x, 3);
  this->angular_vel_cam[1] = gsl_vector_get(final_x, 4);
  this->angular_vel_cam[2] = gsl_vector_get(final_x, 5);
  
  const double final_cost = gsl_multimin_fdfminimizer_minimum(solver);

  //Release memory used during optimization
  gsl_multimin_fdfminimizer_free (solver);
  gsl_vector_free (vx);

  return final_cost;
}

} //namespace