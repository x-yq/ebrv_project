#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <geometry_msgs/PointStamped.h>

#include <opencv2/core/core.hpp>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <deque>

#include <kindr/minimal/quat-transformation.h>
#include <image_geometry/pinhole_camera_model.h>
#include <fast_dynamic/image_util.h>

#include <filesystem>
#include <chrono>
#include <ctime>

#include <string>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <mutex>

#include <Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <random>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>

using namespace autodiff;
using namespace std;
using namespace Eigen;

namespace motion_compensate
{

using Transformation = kindr::minimal::QuatTransformation;

class MotionCompensate {
public:
  MotionCompensate(ros::NodeHandle & nh, ros::NodeHandle nh_private);
  virtual ~MotionCompensate();

cv::Vec3f linear_vel_cam;
cv::Vec3f angular_vel_cam;
cv::Mat Z;
cv::Vec3f grad_linear_vel;
cv::Vec3f grad_angular_vel;
cv::Mat grad_Z;


double hx, hy, hz, htheta;
double dx, dy, dz, dth;
int img_width, img_height;
int iter;
double cx, cy;

int bag_ind;

private:
  ros::NodeHandle nh_;   // Node handle used to subscribe to ROS topics
  ros::NodeHandle pnh_;  // Private node handle for reading parameters

  int packet_number;
  int slice_number;
  long total_event_count;
  long total_depth_map_count;

  // Callback functions
//   void callback(const dvs_msgs::EventArray::ConstPtr& msg);
  void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg);
  void depthCallback(const sensor_msgs::ImageConstPtr& depth_msg = nullptr);
  void checkAndProcess();
  void processMessages();

  int expected_events_msg_ = 0, expected_depth_msg_ = 0;
  int total_events_msg_size_ = 0, total_depth_msg_size_ = 0;

  // Subscribers
  ros::Subscriber event_sub_;
  ros::Subscriber depth_image_sub_;
  ros::Subscriber imu_sub_;

  // Publishers
  image_transport::Publisher event_count_pub_;
  image_transport::Publisher avg_time_map_pub_;
  image_transport::Publisher mc_event_count_pub_;
  image_transport::Publisher mc_time_map_pub_;
  image_transport::Publisher ground_mask_pub_;
  image_transport::Publisher depth_map_pub_;

  cv::Mat event_count_;
  cv::Mat avg_time_map_;
  cv::Mat mc_event_count_;
  cv::Mat mc_time_map_;
  cv::Mat depth_image_;
  cv::Mat ground_mask_;
  cv::Mat foreground_mask, background_mask;
  cv::Mat rho;
  cv::Mat mc_event_count_pos_;
  cv::Mat mc_event_count_neg_;
  cv::Mat mc_time_map_pos_;
  cv::Mat mc_time_map_neg_;

  double duration;

  ros::Time t0;

  void publishMap(const double t_ref);
  ros::Time time_packet_;

  std::deque<dvs_msgs::Event> events_;
  std::deque<cv::Mat> depth_maps_;
  std::deque<double> depth_map_timestamps;
  std::vector<dvs_msgs::Event> events_subset_temp;

  std::mutex buffer_mutex_;

  double acc_threshold_;
  double num_events_map_update_;
  double maxIterations;
  int idx_first_ev_map_;

void findInitialFlow(const std::vector<dvs_msgs::Event>& events_subset);
std::array<double, 4> findBestFlowInRangeBruteForce(const std::vector<dvs_msgs::Event>& events_subset, 
                                                                    const std::array<double, 8>& param_range, double step_xy, double step_zth);

double contrast_f_numerical(const std::vector<dvs_msgs::Event>& events_subset, const double hx_, const double hy_, const double hz_, const double hth_);

double getDiversion(double time, double x, double y);

cv::Mat computeImageOfWarpedEvents(const std::vector<dvs_msgs::Event>& events_subset, double hx_, double hy_, double hz_, double hth_);

double computeError(
  const double& l_hx,const double& l_hy,const double& l_hz,const double& l_hth
);

std::string bag_args;
double lr_x, lr_y, lr_div, lr_rot;
int depth_window_size;
int velocity_calculated_ = 0;
bool random_initial;
bool filter_small_compo, use_adam, enable_undistort, save_frames, plot_hist, better_initial, enable_depth, depth_image_received = false;
double initial_lr_x, initial_lr_y, initial_lr_div, initial_lr_rot;


struct AdamParam {
    double t, m, v;  // Transformation parameters

    // Default constructor
    AdamParam() : t(1.0), m(0.0), v(0.0) {}

    // Parameterized constructor
    AdamParam(double t_, double m_, double v_) 
        : t(t_), m(m_), v(v_) {}

    AdamParam& operator=(const AdamParam& other) {
        if (this != &other) { 
            t = other.t;
            v = other.v;
            m = other.m;
        }
        return *this; 
    }
};

struct AdamOptimizer {
    AdamParam param;
    double beta1, beta2, alpha;

    AdamOptimizer(double beta1 = 0.9, double beta2 = 0.999, double alpha = 1.0)
        : beta1(beta1), beta2(beta2), alpha(alpha) {}

    void init(double m_init = 0.0, double v_init = 0.0, int t_init = 1) {
        param.m = m_init;
        param.v = v_init;
        param.t = t_init;
    }

    void update(const double& grad, double* scale) {
        param.m = beta1 * param.m + (1 - beta1) * grad;
        param.v = beta2 * param.v + (1 - beta2) * grad * grad;

        double m_hat = param.m / (1 - std::pow(beta1, param.t));
        double v_hat = param.v / (1 - std::pow(beta2, param.t));

        param.t += 1;

        *scale = alpha * m_hat / (std::sqrt(v_hat) + 1e-7);  

    }
};


// const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
//         199.0923665423112, 0.0, 132.1920713777002, 
//         0.0, 198.8288204700886, 110.7126600112956, 
//         0.0, 0.0, 1.0);

// const cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << 
//         -0.3684363117977873, 0.1509472435566583, 
//         -0.0002961305343848646, -0.000759431726241032, 0.0);

// const cv::Mat R = (cv::Mat_<double>(3,3) <<
//                     1.0, 0.0, 0.0, 
//                     0.0, 1.0, 0.0,
//                     0.0, 0.0, 1.0);
// const cv::Mat P = (cv::Mat_<double>(3,4) <<
//             168.6294097900391, 0.0, 135.348079770296, 0.0, 
//             0.0, 178.5641784667969, 113.6189973794753, 0.0, 
//             0.0, 0.0, 1.0, 0.0);


// // //test_vins
const cv::Mat K_depth = (cv::Mat_<double>(3, 3) <<  
        5.3633325932983780e+02, 0, 3.2090009280822994e+02, 
        0, 5.3631797700847164e+02, 2.3404853514480661e+02, 
        0, 0, 1);


// const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
//         536.3332593298378, 0, 320.90009280822994, 
//         0, 536.31797700847164, 234.04853514480661, 
//         0, 0, 1);

// const cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);

// const cv::Mat R = (cv::Mat_<double>(3,3) <<
//                     1.0, 0.0, 0.0, 
//                     0.0, 1.0, 0.0,
//                     0.0, 0.0, 1.0);
// const cv::Mat P = (cv::Mat_<double>(3,4) <<
//             5.3633325932983780e+02, 0.0, 3.2090009280822994e+02, 0.0, 
//             0.0, 5.3631797700847164e+02, 2.3404853514480661e+02, 0.0, 
//             0.0, 0.0, 1.0, 0.0);


// slider_depth

const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
        335.4194629584808, 0.0, 129.9246633794451, 
        0.0, 335.3529356120773, 99.18643034473205, 
        0.0, 0.0, 1.0);

const cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << -0.1385927674081299, 0.09337366641919795, -0.0003355869875320301, 0.0001737201582276446, 0.0);

const cv::Mat R = (cv::Mat_<double>(3,3) <<
                    1.0, 0.0, 0.0, 
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0);
const cv::Mat P = (cv::Mat_<double>(3,4) <<
            328.3079223632812, 0.0, 129.7166999252022, 0.0, 
            0.0, 330.1483459472656, 98.78069942616821, 0.0, 
            0.0, 0.0, 1.0, 0.0);


// simulation3dplane

// const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
//         171.37776185565394, 0.0, 120.0, 
//         0.0, 171.37776185565394, 90.0, 
//         0.0, 0.0, 1.0);

// const cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);

// const cv::Mat R = (cv::Mat_<double>(3,3) <<
//                     0.0, 0.0, 0.0, 
//                     0.0, 0.0, 0.0,
//                     0.0, 0.0, 0.0);
// const cv::Mat P = (cv::Mat_<double>(3,4) <<
//             171.37776185565394, 0.0, 120.0, 
//         0.0, 171.37776185565394, 90.0, 
//         0.0, 0.0, 1.0);



AdamOptimizer ap_x,ap_y,ap_div,ap_rot;

double lambda;

double computeContrast(const cv::Mat& image);

void updateModel();

void diffTimeImage(const cv::Mat& image);

double getDensity(const cv::Mat& event_count, double threshold);

void printInfo(const double& error, const double& contrast, const double& density);

void detectMovingObjects(const cv::Mat& avg_time_map, 
                        const cv::Mat& mc_time_map, 
                        const double& dt,
                        cv::Mat& background_mask, 
                        cv::Mat& foreground_mask);

void filterComponents(const cv::Mat& binary_image, cv::Mat& filtered_image, int min_area, float max_aspect_ratio);
void plotHist(const cv::Mat& avg_image, const cv::Mat& mc_image);
cv::Mat denoiseTimeMap(const cv::Mat &time_map, int n, int window_size);
void saveMapsAsMultiChannels();

//TODO:signiture of v2

double angular_vel_x0 = 0.;
double angular_vel_y0 = 0.;
double angular_vel_z0 = 0.;

double linear_vel_x0 = 0.;
double linear_vel_y0 = 0.;
double linear_vel_z0 = 0.;

ros::Time prev_time_;

void imuDataCallback(const sensor_msgs::Imu::ConstPtr& imu_msg);
void processMessages_v2();
void initialize_v2(const std::vector<dvs_msgs::Event>& events_subset);
cv::Mat computeImageOfWarpedEvents_v2(const std::vector<dvs_msgs::Event>& events_subset);
cv::Matx23f A_v2(const int x, const int y);
cv::Matx23f B_v2(const int x, const int y);
cv::Mat getGTDepthMap_v2(const double time);
void computeGrad_v2(const cv::Mat& image, const double t_ref, const cv::Mat& Z);
double computeContrast_v2(const cv::Mat& image);
void updateModel_v2();
void printInfo_v2(const double& contrast, const double& density);
cv::Mat bilinearInterpolate(const cv::Mat& depth_map, double patch_size);

double maximizeContrast(const std::vector<dvs_msgs::Event>& events_subset);


// template <typename T>
// T computeLoss_v2_autodiff(const std::vector<dvs_msgs::Event>& events_subset,
//                           const Eigen::Matrix<T, 3, 1>& linear,
//                           const Eigen::Matrix<T, 3, 1>& angular,
//                           const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& depthMap);

// template <typename T>
// cv::Mat computeImage_v2_autodiff(const std::vector<dvs_msgs::Event>& events_subset,
//                                  const Eigen::Matrix<T, 3, 1>& linear,
//                                  const Eigen::Matrix<T, 3, 1>& angular,
//                                  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& depthMap);

// dual computeLoss_v2_autodiff(const MatrixXd& grad_x, const MatrixXd& grad_y);

// // Eigen::Matrix<dual, Eigen::Dynamic, Eigen::Dynamic> convertToAutodiffMatrix(const cv::Mat& mat);

// void computeGradients_v2_autodiff(const std::vector<dvs_msgs::Event>& events_subset, 
//                                         const cv::Mat& depthMap, 
//                                         const cv::Vec3f& linear_vel_cam, 
//                                         const cv::Vec3f& angular_vel_cam);

// template <typename T>
// Eigen::Matrix<T, 2, 3> A_v2_autodiff(T xx, T yy);

// template <typename T>
// Eigen::Matrix<T, 2, 3> B_v2_autodiff(T xx, T yy);

// dual computeCost4SinglePixel_autodiff(const dual& grad_x, const dual& grad_y);

};


} // namespace
