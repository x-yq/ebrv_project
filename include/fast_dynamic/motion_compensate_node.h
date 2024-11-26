#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <opencv2/core/core.hpp>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <deque>

#include <kindr/minimal/quat-transformation.h>
#include <image_geometry/pinhole_camera_model.h>
// #include <fast_dynamic/minimizer.h>


namespace motion_compensate
{

using Transformation = kindr::minimal::QuatTransformation;
//using Transformation = kindr::minimal::RotationQuaternion;

class MotionCompensate {
public:
  MotionCompensate(ros::NodeHandle & nh, ros::NodeHandle nh_private);
  virtual ~MotionCompensate();

  struct Model {
    double hx, hy, hz, htheta;  // Transformation parameters

    // Default constructor
    Model() : hx(0.0), hy(0.0), hz(0.0), htheta(0.0) {}

    // Parameterized constructor
    Model(double hx_, double hy_, double hz_, double htheta_) 
        : hx(hx_), hy(hy_), hz(hz_), htheta(htheta_) {}

    Model& operator=(const Model& other) {
        if (this != &other) {  // Check for self-assignment
            hx = other.hx;
            hy = other.hy;
            hz = other.hz;
            htheta = other.htheta;
        }
        return *this; // Return *this to allow chained assignments
    }
};

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

  cv::Mat event_count_;
  cv::Mat avg_time_map_;
  cv::Mat mc_event_count_;
  cv::Mat mc_time_map_;

  ros::Time t0;
 
  void publishMap();
  ros::Time time_packet_;

  // Sliding window of events
  std::deque<dvs_msgs::Event> events_;
  std::vector<dvs_msgs::Event> events_subset_temp;


  // Mapping / mosaicing
  double discretization_;
  double acc_threshold_;
  double num_events_map_update_;
  double lr_;
  double maxIterations;
  int idx_first_ev_map_;  // index of first event of processing window
  double avg_contrast;
  std::deque<double> error_history;
  int WINDOW_SIZE = 10.0;

  

void getSlideWindowSize(const std::deque<dvs_msgs::Event>& evs, const double discretization_, int& num_ev);

cv::Point2d warpEvent(
  const Model& model,
  const dvs_msgs::Event& event,
  const double& t_ref
);
void accumulateWarpedEvent(
  const int img_width,
  const int img_height,
  const dvs_msgs::Event& event,
  const cv::Point2d& ev_warped_pt,
  cv::Mat* image_warped,
  cv::Mat* image_event_count
);

void computeImageOfWarpedEvents(
  const Model& model,
  const std::vector<dvs_msgs::Event>& events_subset,
  const int width,
  const int height,
  cv::Mat* image_warped,
  cv::Mat* image_event_count
);

double computeError(
  const Model& modelPrev,
  const Model& modelCurr
);


double lr_x, lr_y, lr_div, lr_rot;

// struct Model {
//     double hx, hy, hz, htheta;  // Transformation parameters

//     // Default constructor
//     Model() : hx(0.0), hy(0.0), hz(0.0), htheta(0.0) {}

//     // Parameterized constructor
//     Model(double hx_, double hy_, double hz_, double htheta_) 
//         : hx(hx_), hy(hy_), hz(hz_), htheta(htheta_) {}

//     Model& operator=(const Model& other) {
//         if (this != &other) {  // Check for self-assignment
//             hx = other.hx;
//             hy = other.hy;
//             hz = other.hz;
//             htheta = other.htheta;
//         }
//         return *this; // Return *this to allow chained assignments
//     }
// };

struct AdamParam {
    double t, m, v;  // Transformation parameters

    // Default constructor
    AdamParam() : t(0.0), m(0.0), v(0.0) {}

    // Parameterized constructor
    AdamParam(double t_, double m_, double v_) 
        : t(t_), m(m_), v(v_) {}

    AdamParam& operator=(const AdamParam& other) {
        if (this != &other) {  // Check for self-assignment
            t = other.t;
            v = other.v;
            m = other.m;
        }
        return *this; // Return *this to allow chained assignments
    }
};

AdamParam ap_x,ap_y,ap_div,ap_rot;
const double beta1 = 0.9, beta2 = 0.999, alpha = 1.0;

double computeError2(const cv::Mat& image);

void updateModel(const Model& pre, 
const double& dx, const double& dy, const double& dz, 
const double& dtheta,Model* cur);

void diffTimeImage(const cv::Mat &time_image, 
                   double &dx_val_r, double &dy_val_r, 
                   double &dz_val_r, double &dth_val_r);

void dZThetaUpdate(const cv::Mat &grad_x, const cv::Mat &grad_y, double &div_r, double &rot_r);

void dXYUpdate(const cv::Mat &grad_x, const cv::Mat &grad_y, double &dx_val_r, double &dy_val_r);

// void updateThresholds(const double& avg_magnitude, double &magnitude_threshold);
// void updateThresholds(const double& error, const cv::Mat& time_image, double &error_threshold);

// void computeAvgMagnitude(const cv::Mat& img, double& magnitude_score,cv::Mat& ref_grad_x, cv::Mat& ref_grad_y);

double updateAdam(const double& grad, AdamParam& ap);

double getDensity(const cv::Mat& event_count);

};

} // namespace
