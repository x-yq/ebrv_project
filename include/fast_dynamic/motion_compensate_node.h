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
// #include <fast_dynamic/minimizer.h>

// #include <dynamic_reconfigure/server.h>
// #include <fast_dynamic/motion_compensateConfig.h>


namespace motion_compensate
{

using Transformation = kindr::minimal::QuatTransformation;
//using Transformation = kindr::minimal::RotationQuaternion;

class MotionCompensate {
public:
  MotionCompensate(ros::NodeHandle & nh, ros::NodeHandle nh_private);
  virtual ~MotionCompensate();

  //   // Dynamic reconfigure
  // void reconfigureCallback(motion_compensate::motion_compensateConfig &config, uint32_t level);
  // boost::shared_ptr<dynamic_reconfigure::Server<motion_compensate::motion_compensateConfig>> server_;
  // dynamic_reconfigure::Server<motion_compensate::motion_compensateConfig>::CallbackType dynamic_reconfigure_callback_;

  struct Grad{
    double dx, dy, dz, dth;  // Transformation parameters

    // Default constructor
    Grad() : dx(0.0), dy(0.0), dz(0.0), dth(0.0) {}

    // Parameterized constructor
    Grad(double dx, double dy, double dz, double dth) 
        : dx(dx), dy(dy), dz(dz), dth(dth) {}

    Grad& operator=(const Grad& other) {
        if (this != &other) {  
            dx = other.dx;
            dy = other.dy;
            dz = other.dz;
            dth = other.dth;
        }
        return *this; 
    }
  };

  struct Model {
    double hx, hy, hz, htheta;  // Transformation parameters

    // Default constructor
    Model() : hx(0.0), hy(0.0), hz(0.0), htheta(0.0) {}

    // Parameterized constructor
    Model(double hx_, double hy_, double hz_, double htheta_) 
        : hx(hx_), hy(hy_), hz(hz_), htheta(htheta_) {}

    Model& operator=(const Model& other) {
        if (this != &other) {  
            hx = other.hx;
            hy = other.hy;
            hz = other.hz;
            htheta = other.htheta;
        }
        return *this; 
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
  ros::Publisher model_pub_;
  ros::Publisher grad_pub_;

  cv::Mat event_count_;
  cv::Mat avg_time_map_;
  cv::Mat mc_event_count_;
  cv::Mat mc_time_map_;

  ros::Time t0;
 
  void publishMap();
  void publishModel(const Model& model, const Grad& grad);
  ros::Time time_packet_;

  std::deque<dvs_msgs::Event> events_;
  std::vector<dvs_msgs::Event> events_subset_temp;


  double discretization_;
  double acc_threshold_;
  double num_events_map_update_;
  double maxIterations;
  int idx_first_ev_map_;

cv::Point2d warpEvent(
  const Model& model,
  const dvs_msgs::Event& event,
  const double& t_ref
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

    AdamOptimizer(double beta1 = 0.9, double beta2 = 0.999, double alpha = 0.1)
        : beta1(beta1), beta2(beta2), alpha(alpha) {}

    void init(double m_init = 0.0, double v_init = 0.0, int t_init = 1) {
        param.m = m_init;
        param.v = v_init;
        param.t = t_init;
    }

    double update(const double& grad) {
        param.m = beta1 * param.m + (1 - beta1) * grad;
        param.v = beta2 * param.v + (1 - beta2) * grad * grad;

        double m_hat = param.m / (1 - std::pow(beta1, param.t));
        double v_hat = param.v / (1 - std::pow(beta2, param.t));

        param.t += 1;

        double scale = alpha * m_hat / (std::sqrt(v_hat) + 1e-7); 

        return scale;
    }
};


AdamOptimizer ap_x,ap_y,ap_div,ap_rot;

bool updateDR = false;


double computeError2(const cv::Mat& image, const cv::Mat& wp_image);

void updateModel(const Model& pre, const Grad& grad,Model* cur, const double& contrast);

void diffTimeImage(const cv::Mat &time_image, Grad* grad);

void dZThetaUpdate(const cv::Mat &grad_x, const cv::Mat &grad_y, double &div_r, double &rot_r);

void dXYUpdate(const cv::Mat &grad_x, const cv::Mat &grad_y, double &dx_val_r, double &dy_val_r);

double getDensity(const cv::Mat& event_count);

void checkDirection(const Grad& old, const Grad& cur, Model* m);

void printInfo(const Grad& grad, const Model& m, const double& error, const double& density);


};

} // namespace
