// event_count
// time_map for avg_time_map
// motion_compensated_event_count
// time_image for minimized ...

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
// #include <glog/logging.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <opencv2/highgui.hpp> //cv::imwrite
#include <fast_dynamic/motion_compensate_node.h>
#include <dvs_msgs/Event.h>
#include <numeric>


namespace motion_compensate
{

MotionCompensate::MotionCompensate(ros::NodeHandle & nh, ros::NodeHandle nh_private)
 : nh_(nh)
 , pnh_("~")
{
  // Get parameters
  nh_private.param<double>("discretization", discretization_, 0.01);
  nh_private.param<double>("num_events_map_update", num_events_map_update_, 10000);
  nh_private.param<double>("acc_threshold", acc_threshold_, 0.0001);
  nh_private.param<double>("tm_max_iter",  maxIterations, 500);
  nh_private.param<double>("lr_x", lr_x, 1.0);
  nh_private.param<double>("lr_y", lr_y, 1.0);
  nh_private.param<double>("lr_div", lr_div, 0.0001);
  nh_private.param<double>("lr_rot", lr_rot, 0.0001);



  // set queue_size to 0 to avoid discarding messages (for correctness).
  event_sub_ = nh_.subscribe("events", 0, &MotionCompensate::eventsCallback, this);

  // Set up publishers
  image_transport::ImageTransport it_(nh_);
  // my topics names: event_count, avg_time_map, motion_compensated_event_count, motion_compensated_time_map
  event_count_pub_ = it_.advertise("event_count", 0);
  avg_time_map_pub_ = it_.advertise("avg_time_map", 0);
  mc_event_count_pub_ = it_.advertise("mc_event_count", 0);
  mc_time_map_pub_ = it_.advertise("mc_time_map", 0);
  model_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/dvs/model", 0);
  grad_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/dvs/grad", 0);

  // // Dynamic reconfigure
  // dynamic_reconfigure_callback_ = boost::bind(&MotionCompensate::reconfigureCallback, this, _1, _2);
  // server_.reset(new dynamic_reconfigure::Server<motion_compensate::motion_compensateConfig>(nh_private));
  // server_->setCallback(dynamic_reconfigure_callback_);


  // Event processing in batches / packets
  idx_first_ev_map_ = 0;   // Index of first event of processing window
  time_packet_ = ros::Time(0);

}


MotionCompensate::~MotionCompensate()
{
  // shut down all publishers
  event_count_pub_.shutdown();
  avg_time_map_pub_.shutdown();
  mc_event_count_pub_.shutdown();
  mc_time_map_pub_.shutdown();
  model_pub_.shutdown();

}

// /**
//  * Interface with the parameters that can be changed online via dynamic reconfigure
//  */
// void MotionCompensate::reconfigureCallback(motion_compensate::motion_compensateConfig &config, uint32_t level)
// {
//   num_events_map_update_ = config.num_events_map_update;
//   acc_threshold_ = config.acc_threshold;
//   maxIterations = config.tm_max_iter;
//   lr_x = config.lr_x;
//   lr_y = config.lr_y;
//   lr_div = config.lr_div;
//   lr_rot = config.lr_rot;
// }


/**
* \brief Function to process event messages received by the ROS node
*/
void MotionCompensate::eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
  // Append events of current message to the queue
  std::cout << "Received an EventArray message with " << msg->events.size() << std::endl;

  static unsigned int packet_number = 0;
  static unsigned long total_event_count = 0;

  if (packet_number == 0)
  {
    t0 = msg->events.front().ts;
  }
  packet_number++;

  for(const dvs_msgs::Event& ev : msg->events) events_.push_back(ev);

  total_event_count += msg->events.size();
  // VLOG(1) << "Packet # " << packet_number << "  event# " << total_event_count << "  queue_size:" << events_.size();

  const double width = msg->width;
  const double height = msg->height;

  ROS_WARN("num_events_map_update_ = %f", num_events_map_update_);

  while (idx_first_ev_map_ + num_events_map_update_ <= events_.size())
  {
    Model M_G_prev(0., 0., 0., 0.);
    Model M_G_new(0., 0., 0., 0.);
    Model M_C_new(0., 0., 0., 0.);
  
    avg_time_map_ = cv::Mat::zeros(height, width,CV_64FC1);
    mc_time_map_ = cv::Mat::zeros(height, width,CV_64FC1);
    event_count_ = cv::Mat::zeros(height, width,CV_64FC1);
    mc_event_count_ = cv::Mat::zeros(height, width,CV_64FC1);

    const std::vector<dvs_msgs::Event> events_subset_ = std::vector<dvs_msgs::Event> (events_.begin() + idx_first_ev_map_,
                                                   events_.begin() + idx_first_ev_map_ + num_events_map_update_);
 
    // // Compute time span of the events
    // ros::Time time_first = events_subset_.front().ts;
    // ros::Time time_last = events_subset_.back().ts;
    // ros::Duration time_dt = time_last - time_first;
    // time_packet_ = time_first + time_dt * 0.5;

    /***
     * Get the average time map and event count map
    */
    ROS_WARN("-----------------AVG TIME MAP AND COUNT MAP-------------------");

    for (const dvs_msgs::Event& event : events_subset_) {
        if (event.x >= 0 && event.x < width && event.y >= 0 && event.y < height) {
            event_count_.at<double>(event.y, event.x)+=1.0;
            avg_time_map_.at<double>(event.y, event.x) += (event.ts.toSec() - t0.toSec());
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          avg_time_map_.at<double>(y, x) /= (event_count_.at<double>(y, x)+1e-7);
        }
    }

    float total_count = cv::sum(event_count_)[0];
    cv::Mat mask = (event_count_ > 1e-5);
    float count = cv::countNonZero(mask); 
    std::cout << "Event count: " << total_count << std::endl;

    /***
     * Timestamp Minimizer
    */
     ROS_WARN("-----------------TIME MAP MINIMIZER-------------------");

    cv::Mat mc_event_count_temp;

    computeImageOfWarpedEvents(M_G_prev, events_subset_, width, height,  &mc_time_map_,&mc_event_count_temp);

    int iter = 0;
    ap_x.init();
    ap_y.init();
    ap_rot.init();
    ap_div.init();
    updateDR = false;

    Grad grad;
    Grad old_grad;

    diffTimeImage(mc_time_map_, &grad);
    double contrast = computeError2(avg_time_map_, mc_time_map_);
    updateModel(M_G_prev, grad, &M_G_new, contrast);
    old_grad = grad;

    double error = computeError(M_G_prev, M_G_new);

    while (true) {
        if (iter > maxIterations){
          ROS_WARN("Reached max iterations %d", iter);
          printInfo(grad, M_G_new, contrast, 0.);
          break;
        }

        // Check if the gradient is close to zero
        if (std::abs(grad.dx) < 1e-6 && std::abs(grad.dy) < 1e-6 && std::abs(grad.dz) < 1e-6) {
            ROS_WARN("Gradient converged due to error change being below threshold after %d", iter);
            printInfo(grad, M_G_new, contrast, 0.);
            break;
        }

        // Check if the error change is below the threshold
        if (contrast <= -2) {
            ROS_WARN("Error converged due to error change being below threshold after %d", iter);
            printInfo(grad, M_G_new, contrast, 0.);
            break;
        }


        // Perform image warping with updated M_G_new
        computeImageOfWarpedEvents(M_G_new, events_subset_, width, height, &mc_time_map_, &mc_event_count_temp);
        
        // Compute time difference and gradient again
        diffTimeImage(mc_time_map_, &grad);
        
        M_G_prev = M_G_new;

        // checkDirection(old_grad, grad, &M_G_prev);

        // Update model parameters using gradient descent
        contrast = computeError2(avg_time_map_,mc_time_map_);
        updateModel(M_G_prev, grad, &M_G_new, contrast);

        error = computeError(M_G_prev, M_G_new);

        // Increment iteration count
        iter++;

        old_grad = grad;

    }

    // publishModel(M_G_new, grad);

   /*
   
   Event count minimizer
   
   */
  ROS_WARN("-----------------EVENT COUNT MINIMIZER-------------------");

  M_C_new = M_G_new;

  std::cout << "x: " << M_C_new.hx << " y:" << M_C_new.hy << " hz:"  << M_C_new.hz << " th: " << M_C_new.htheta << std::endl;

    double D_prime, D;

    cv::Mat mc_time_map_temp;

    computeImageOfWarpedEvents(M_C_new, events_subset_, width, height, &mc_time_map_temp, &mc_event_count_);

    D = getDensity(mc_event_count_);

    D_prime = 0.;
    iter = 0;
    ap_x.init();
    ap_y.init();
    ap_rot.init();
    ap_div.init();

    while (true) {
      if(std::abs(D - D_prime) < 1e-7){
          ROS_WARN("Density converged after %d", iter);
          printInfo(grad, M_C_new, error, D);
          break;
      }

      if (iter > maxIterations){
        ROS_WARN("Reached max iterations %d", iter);
        printInfo(grad, M_C_new, error, D);
        break;
      }

        Model M_C_temp;
        M_C_temp = M_C_new;
        D = D_prime;
        diffTimeImage(mc_time_map_, &grad);

        double deltas[] = {grad.dx, grad.dy, grad.dz, grad.dth};
        // double deltas[] = {M_G_new.hx, M_G_new.hy, M_G_new.hz, M_G_new.htheta};
        double* model_params[] = {&M_C_new.hx, &M_C_new.hy, &M_C_new.hz, &M_C_new.htheta};
        double* model_temp_params[] = {&M_C_temp.hx, &M_C_temp.hy, &M_C_temp.hz, &M_C_temp.htheta};

        for (int i = 0; i < 4; ++i) {

            *model_temp_params[i] = *model_params[i] + deltas[i];
            computeImageOfWarpedEvents(M_C_temp, events_subset_, width, height, &mc_time_map_temp, &mc_event_count_);

            D_prime = getDensity(mc_event_count_temp);

            if (D_prime > D) {
              *model_params[i] += deltas[i];
              computeImageOfWarpedEvents(M_C_new, events_subset_, width, height, &mc_time_map_temp, &mc_event_count_);
              M_C_temp = M_C_new;
            }
            else {
              *model_temp_params[i] = *model_params[i] - deltas[i];
            }
        }
        // ROS_WARN("D prim = %f", D_prime);
        // ROS_WARN("D = %f", D);
        iter++;
    }
        
    publishMap();

    publishModel(M_C_new, grad);

    
    // Slide
    if ( num_events_map_update_ <= events_.size() )
      {
        events_.erase(events_.begin(), events_.begin() + num_events_map_update_);
        idx_first_ev_map_ = 0;
      } else
      {
        idx_first_ev_map_ += num_events_map_update_;
      }

    ROS_WARN("num_events_map_update_ = %f", num_events_map_update_);
  
  }

}

void MotionCompensate::checkDirection(const Grad& old, const Grad& cur, Model* m){
  if(old.dx * cur.dx < 0.){
    m->hx = 0.;
  }
  if(old.dy * cur.dy < 0.){
    m->hy = 0.;
  }
  if(old.dz * cur.dz < 0.){
    m->hz = 0.;
  }
  if(old.dth * cur.dth < 0.){
    m->htheta = 0.;
  }
}

}
