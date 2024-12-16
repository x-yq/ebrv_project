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


namespace motion_compensate
{

using Transformation = kindr::minimal::QuatTransformation;

class MotionCompensate {
public:
  MotionCompensate(ros::NodeHandle & nh, ros::NodeHandle nh_private);
  virtual ~MotionCompensate();

double hx, hy, hz, htheta;
double dx, dy, dz, dth;
int img_width, img_height;
int iter;
double cx, cy;

private:
  ros::NodeHandle nh_;   // Node handle used to subscribe to ROS topics
  ros::NodeHandle pnh_;  // Private node handle for reading parameters

  // Callback functions
  void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg);

  // Subscribers
  ros::Subscriber event_sub_;

  // Publishers
  image_transport::Publisher event_count_pub_;
  image_transport::Publisher avg_time_map_pub_;
  image_transport::Publisher mc_event_count_pub_;
  image_transport::Publisher mc_time_map_pub_;
  image_transport::Publisher ground_mask_pub_;
  ros::Publisher model_pub_;
  ros::Publisher grad_pub_;

  cv::Mat event_count_;
  cv::Mat avg_time_map_;
  cv::Mat mc_event_count_;
  cv::Mat mc_time_map_;
  cv::Mat ground_mask_;
  cv::Mat foreground_mask, background_mask;
  cv::Mat rho;
  cv::Mat mc_event_count_pos_;
  cv::Mat mc_event_count_neg_;
  cv::Mat mc_time_map_pos_;
  cv::Mat mc_time_map_neg_;

  double duration;

  ros::Time t0;

  void publishMap();
  ros::Time time_packet_;

  std::deque<dvs_msgs::Event> events_;
  std::vector<dvs_msgs::Event> events_subset_temp;


  double discretization_;
  double acc_threshold_;
  double num_events_map_update_;
  double maxIterations;
  int idx_first_ev_map_;

void findInitialFlow(const std::vector<dvs_msgs::Event>& events_subset);
std::array<double, 4> findBestFlowInRangeBruteForce(const std::vector<dvs_msgs::Event>& events_subset, 
                                                                    const std::array<double, 8>& param_range, double step_xy, double step_zth);

double contrast_f_numerical(const std::vector<dvs_msgs::Event>& events_subset, const double hx_, const double hy_, const double hz_, const double hth_);

cv::Mat computeImageOfWarpedEvents(const std::vector<dvs_msgs::Event>& events_subset, double hx_, double hy_, double hz_, double hth_);

double computeError(
  const double& l_hx,const double& l_hy,const double& l_hz,const double& l_hth
);

double lr_x, lr_y, lr_div, lr_rot;
bool filter_small_compo, use_adam, enable_undistort, save_frames;
double initial_lr_x, initial_lr_y, initial_lr_div, initial_lr_rot = lr_rot;


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

cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
        199.0923665423112, 0.0, 132.1920713777002, 
        0.0, 198.8288204700886, 110.7126600112956, 
        0.0, 0.0, 1.0);
cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << 
        -0.3684363117977873, 0.1509472435566583, 
        -0.0002961305343848646, -0.000759431726241032, 0.0);

cv::Mat R = (cv::Mat_<double>(3,3) <<
                    1.0, 0.0, 0.0, 
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0);

cv::Mat P = (cv::Mat_<double>(3,4) <<
            168.6294097900391, 0.0, 135.348079770296, 0.0, 
            0.0, 178.5641784667969, 113.6189973794753, 0.0, 
            0.0, 0.0, 1.0, 0.0);




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

};

} // namespace
