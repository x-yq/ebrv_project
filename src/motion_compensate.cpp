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

namespace motion_compensate
{

MotionCompensate::MotionCompensate(ros::NodeHandle & nh, ros::NodeHandle nh_private)
 : nh_(nh)
 , pnh_("~")
{
  // Get parameters
  nh_private.param<double>("num_events_map_update", num_events_map_update_, 10000);
  nh_private.param<double>("acc_threshold", acc_threshold_, 0.0001);
  nh_private.param<double>("tm_max_iter",  maxIterations, 500);
  nh_private.param<double>("lr_x", lr_x, 1.0);
  nh_private.param<double>("lr_y", lr_y, 1.0);
  nh_private.param<double>("lr_div", lr_div, 0.001);
  nh_private.param<double>("lr_rot", lr_rot, 0.001);
  nh_private.param<double>("filter_threshold", lambda, 0.5);
  nh_private.param<bool>("filter_small_compo", filter_small_compo, false);
  nh_private.param<bool>("use_adam", use_adam, false);
  nh_private.param<bool>("enable_undistort", enable_undistort, false);
  nh_private.param<bool>("save_frames", save_frames, false);



  // set queue_size to 0 to avoid discarding messages (for correctness).
  event_sub_ = nh_.subscribe("events", 0, &MotionCompensate::eventsCallback, this);

  // Set up publishers
  image_transport::ImageTransport it_(nh_);
  // my topics names: event_count, avg_time_map, motion_compensated_event_count, motion_compensated_time_map
  event_count_pub_ = it_.advertise("event_count", 0);
  avg_time_map_pub_ = it_.advertise("avg_time_map", 0);
  mc_event_count_pub_ = it_.advertise("mc_event_count", 0);
  mc_time_map_pub_ = it_.advertise("mc_time_map", 0);
  ground_mask_pub_ = it_.advertise("mask", 0);

  model_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/dvs/model", 0);
  grad_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/dvs/grad", 0);

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

void MotionCompensate::eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
  static unsigned int packet_number = 0;
  static unsigned int slice_number = 0;
  static unsigned long total_event_count = 0;

  if (packet_number == 0)
  {
    t0 = msg->events.front().ts;
    initial_lr_x = lr_x;
    initial_lr_y = lr_y;
    initial_lr_div = lr_div;
    initial_lr_rot = lr_rot;
    img_width = msg->width;
    img_height = msg->height;
  }
  packet_number++;

  for(const dvs_msgs::Event& ev : msg->events) events_.push_back(ev);

  total_event_count += msg->events.size();

  std::cout << "size of image " << img_width << " " << img_height << std::endl;

  while (idx_first_ev_map_ + num_events_map_update_ <= events_.size())
  {
    this->lr_x = initial_lr_x;
    this->lr_y = initial_lr_y;
    this->lr_div = initial_lr_div;
    this->lr_rot = initial_lr_rot;

    this->avg_time_map_ = cv::Mat::zeros(img_height, img_width,CV_64FC1);
    this->mc_time_map_ = cv::Mat::zeros(img_height, img_width,CV_64FC1);
    this->event_count_ = cv::Mat::zeros(img_height, img_width,CV_64FC1);
    this->mc_event_count_ = cv::Mat::zeros(img_height, img_width,CV_64FC1);
    this->ground_mask_ = cv::Mat::zeros(img_height, img_width*2,CV_64FC1);
  
    const std::vector<dvs_msgs::Event> events_subset_temp = std::vector<dvs_msgs::Event> (events_.begin() + idx_first_ev_map_,
                                                   events_.begin() + idx_first_ev_map_ + num_events_map_update_);
 
    /***
     * Get the average time map and event count map
    */
    ROS_WARN("-------------AVG TIME MAP AND COUNT MAP---------------");

    std::vector<dvs_msgs::Event> events_subset_;

    for (const dvs_msgs::Event& event : events_subset_temp) {

      int u_x = event.x;
      int u_y = event.y;
       
      if(enable_undistort){
        std::vector<cv::Point2f> pt = {cv::Point2f(event.x, event.y)};
        std::vector<cv::Point2f> u_pt;
        cv::undistortPoints(pt, u_pt, cameraMatrix, distCoeffs, R, P);
        u_x = u_pt[0].x;
        u_y = u_pt[0].y;
      }

      if (u_x >= 0 && u_x < img_width && u_y >= 0 && u_y < img_height) {
        dvs_msgs::Event ev;
        ev.x = u_x;
        ev.y = u_y;
        ev.ts = ros::Time(event.ts.toSec() - t0.toSec());
        ev.polarity = event.polarity;
        events_subset_.push_back(ev);
      }
    }

    const double slice_first_t = events_subset_.front().ts.toSec();

    for(const auto& event: events_subset_){
      this->event_count_.at<double>(event.y, event.x) += 1.0;
      this->avg_time_map_.at<double>(event.y, event.x) += (event.ts.toSec() - slice_first_t);
    }

    for (int y = 0; y < img_height; y+=1) {
        for (int x = 0; x < img_width; x+=1) {
          float count = event_count_.at<double>(y, x);
          if(count < 1.) continue;
          this->avg_time_map_.at<double>(y, x) /= count;
        }
    }


    if (slice_number == 0){
      findInitialFlow(events_subset_);
    }
    slice_number++;

    ROS_WARN("################slice %d##################", slice_number);
    printInfo(0.,0.,0.);


    /***
     * Timestamp Minimizer
    */
     ROS_WARN("-------------TIME MAP MINIMIZER---------------");

    this->iter = 0;
    ap_x.init();
    ap_y.init();
    ap_rot.init();
    ap_div.init();

    computeImageOfWarpedEvents(events_subset_, this->hx, this->hy, this->hz, this->htheta);
    diffTimeImage(this->mc_time_map_);
    updateModel();
    double l_density = getDensity(this->mc_time_map_, 0.1);
    double cur_density;

    while (true) {

      // if(std::abs(this->lr_x*this->dx) < 1e-4 && std::abs(this->lr_y*this->dy) < 1e-4 &&
      //    std::abs(this->lr_div*this->dz) < 1e-3 && std::abs(this->lr_rot*this->dth) < 1e-1){
      //   ROS_WARN("Error converged after %d", iter);
      //   printInfo(0.,computeContrast(this->mc_time_map_), 0.);
      //   break;
      //   }

        double old_dx = this->dx, old_dy = this->dy, old_dz = this->dz, old_dth = this->dth;
        double l_hx = this->hx, l_hy = this->hy, l_hz = this->hz, l_hth = this->htheta;

        computeImageOfWarpedEvents(events_subset_, this->hx, this->hy, this->hz, this->htheta);

        cur_density = getDensity(this->mc_time_map_, 0.1);
        if(std::abs(cur_density-l_density) < 1e-6){
          ROS_WARN("Density converged after %d", iter);
          printInfo(0.,computeContrast(this->mc_time_map_), cur_density);
          break;
        }
        l_density = cur_density;

        diffTimeImage(this->mc_time_map_);

        updateModel();

        if (this->dx * old_dx < 0)   this->lr_x *= 0.9;
        if (this->dy * old_dy < 0)   this->lr_y *= 0.9;
        if (this->dz * old_dz < 0)   this->lr_rot *= 0.9;
        if (this->dth * old_dth < 0) this->lr_div *= 0.9;

        if(computeError(l_hx, l_hy, l_hz, l_hth) < acc_threshold_){
          ROS_WARN("Error converged after %d", iter);
          printInfo(computeError(l_hx, l_hy, l_hz, l_hth),computeContrast(this->mc_time_map_),0.);
          break;
        }

      
        this->iter+=1;

        if (this->iter > this->maxIterations){
          ROS_WARN("Reached max iterations %d", this->iter);
          printInfo(computeError(l_hx, l_hy, l_hz, l_hth), computeContrast(this->mc_time_map_), 0.);
          break;
        }

    }
    

   /*
   
   Event count minimizer
   
   */
  ROS_WARN("-------------EVENT COUNT MINIMIZER---------------");

    double D_prime, D;

    cv::Mat mc_time_map_temp;

    D = getDensity(mc_event_count_, 1.);

    D_prime = 0.;
    this->iter = 0;
    ap_x.init();
    ap_y.init();
    ap_rot.init();
    ap_div.init();
    double hx_temp = this->hx, hy_temp = this->hy, hz_temp = this->hz, htheta_temp = this->htheta;

    while (true) {
      if(std::abs(D - D_prime) < 1e-8){
          ROS_WARN("Density converged after %d", iter);
          printInfo(0.,computeContrast(this->mc_time_map_), D);
          break;
      }

      D = D_prime;
      diffTimeImage(this->mc_event_count_);

      double deltas[] = {this->dx, this->dy, this->dz, this->dth};
      double lrs[] = {this->lr_x, this->lr_y, this->lr_div, this->lr_rot};
      double* model_params[] = {&this->hx, &this->hy, &this->hz, &this->htheta};
      double* model_temp_params[] = {&hx_temp, &hy_temp, &hz_temp, &htheta_temp};

      for (int i = 0; i < 4; ++i) {

          *model_temp_params[i] = *model_params[i] + lrs[i] * deltas[i];
          computeImageOfWarpedEvents(events_subset_, hx_temp, hy_temp, hz_temp, htheta_temp);

          D_prime = getDensity(this->mc_event_count_, 1.);

          if (D_prime > D) {
            *model_params[i] += lrs[i] * deltas[i];
            computeImageOfWarpedEvents(events_subset_, this->hx, this->hy, this->hz, this->htheta);
          }
          else {
            *model_temp_params[i] = *model_params[i] - lrs[i] * deltas[i];
          }
      }

      this->iter++;

      if (this->iter > maxIterations){
        ROS_WARN("Reached max iterations %d", iter);
        printInfo(0., computeContrast(this->mc_time_map_), D);
        break;
      }

    }

    ROS_WARN("-------------OBJECT DETECTION---------------");
    
    
    duration = events_subset_.back().ts.toSec() - events_subset_.front().ts.toSec();

    detectMovingObjects(this->avg_time_map_, this->mc_time_map_, duration, this->background_mask, this->foreground_mask);

    cv::hconcat(this->foreground_mask, this->background_mask, this->ground_mask_);

    if(save_frames) saveMapsAsMultiChannels();
    publishMap();
    // plotHist(this->avg_time_map_, this->mc_time_map_);

    // Slide
    if ( num_events_map_update_ <= events_.size() )
      {
        events_.erase(events_.begin(), events_.begin() + 10000);
        idx_first_ev_map_ = 0;
      } else
      {
        idx_first_ev_map_ += 10000;
      }
  
  }

}


}
