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
  nh_private.param<double>("tm_max_iter",  maxIterations, 500);
  nh_private.param<double>("lr_x", lr_x, 1.0);
  nh_private.param<double>("lr_y", lr_y, 1.0);
  nh_private.param<double>("lr_div", lr_div, 0.001);
  nh_private.param<double>("lr_rot", lr_rot, 0.001);
  nh_private.param<double>("filter_threshold", lambda, 0.5);

  nh_private.param<double>("depth_x_bin_num", depth_x_bin_num, 10);
  nh_private.param<double>("depth_y_bin_num", depth_y_bin_num, 10);

  nh_private.param<bool>("filter_small_compo", filter_small_compo, false);
  nh_private.param<bool>("use_adam", use_adam, false);
  nh_private.param<bool>("enable_undistort", enable_undistort, false);
  nh_private.param<bool>("plot_hist", plot_hist, false);
  nh_private.param<bool>("better_initial", better_initial, false);
  nh_private.param<bool>("enable_depth", enable_depth, false);
  nh_private.param<bool>("random_initial", random_initial, false);
  nh_private.param<std::string>("bag_args", bag_args, "");
  nh_private.param<int>("bag_ind", bag_ind, 0);

  nh_private.param<int>("contrast_ind", contrast_ind, NORM);
  nh_private.param<int>("optimize_image_type", optimize_image_type, TimeMap);


  // set queue_size to 0 to avoid discarding messages (for correctness).
  if(enable_depth){

    event_sub_ = nh_.subscribe("events", 0, &MotionCompensate::eventsCallback, this);
    depth_image_sub_ = nh_.subscribe("depth_image", 0, &MotionCompensate::depthCallback, this);

    std::string events_topic = "/dvs/events";
    std::string depth_topic;
    if(this->bag_ind == 0){
      depth_topic = "/camera/depth/image_rect_raw";
    }else{
      depth_topic = "/dvs/depthmap";
    }

    double start_time = 0.0;
    double duration = 0.0;
    std::string bag_path;
    std::istringstream stream(bag_args);
    std::string token;


    while (stream >> token) {
        if (token == "-r") {
            continue;
        } else if(token == "-d"){
            continue;
        } else if(token == "--pause"){
            continue;
        } else if (token == "--duration") {
            stream >> duration;
        } else if (token == "-s") {
            stream >> start_time;
        } else {
            bag_path = token;
        }
    }

    rosbag::Bag bag_;
    try {
        bag_.open(bag_path, rosbag::bagmode::Read);
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to open bag file: %s", e.what());
    }

    std::vector<std::string> topics = {events_topic, depth_topic};

    rosbag::View view(bag_, rosbag::TopicQuery(topics));

    ros::Time start_time_ = ros::Time(start_time + view.begin()->getTime().toSec());
    ros::Time end_time_ = ros::Time(start_time + view.begin()->getTime().toSec() + duration);

    if(duration == 0.){
      end_time_ = ros::TIME_MAX;
    }

    for (const rosbag::MessageInstance& m : view) {
        if (m.getTopic() == events_topic) {
          if(m.getTime() >= start_time_ && m.getTime() <= end_time_)
            expected_events_msg_++;
        } else if (m.getTopic() == depth_topic) {
          if(m.getTime() >= start_time_ && m.getTime() <= end_time_)
            expected_depth_msg_++;
        }
    }

    ROS_INFO("Topic '%s' has %d messages.", events_topic.c_str(), expected_events_msg_);
    ROS_INFO("Topic '%s' has %d messages.", depth_topic.c_str(), expected_depth_msg_);

    bag_.close();

  }
  else{
    event_sub_ = nh_.subscribe("events", 0, &MotionCompensate::eventsCallback, this);
  }

  // Set up publishers
  image_transport::ImageTransport it_(nh_);

  event_count_pub_ = it_.advertise("event_count", 0);
  avg_time_map_pub_ = it_.advertise("avg_time_map", 0);
  mc_event_count_pub_ = it_.advertise("mc_event_count", 0);
  mc_time_map_pub_ = it_.advertise("mc_time_map", 0);
  ground_mask_pub_ = it_.advertise("mask", 0);
  depth_map_pub_ = it_.advertise("depth_map",0);

  // Event processing in batches / packets
  idx_first_ev_map_ = 0;   // Index of first event of processing window
  time_packet_ = ros::Time(0);

  packet_number = 0;
  slice_number = 0;
  total_event_count = 0;
  total_depth_map_count = 0;

}


MotionCompensate::~MotionCompensate()
{
  // shut down all publishers
  event_count_pub_.shutdown();
  avg_time_map_pub_.shutdown();
  mc_event_count_pub_.shutdown();
  mc_time_map_pub_.shutdown();
  depth_map_pub_.shutdown();
  ground_mask_pub_.shutdown();

}

void MotionCompensate::depthCallback(const sensor_msgs::ImageConstPtr& depth_msg){

  std::lock_guard<std::mutex> lock(buffer_mutex_);
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_64FC1);
  cv::Mat depth_image = cv_ptr->image;
  depth_maps_.push_back(depth_image);
  total_depth_msg_size_ ++;

  checkAndProcess();
  
}

void MotionCompensate::eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(buffer_mutex_);

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
  // std::cout << "size of image " << img_width << " " << img_height << std::endl;

  total_events_msg_size_ ++;

  checkAndProcess();

}

void MotionCompensate::imuDataCallback(const sensor_msgs::Imu::ConstPtr& imu_msg) {

    std::lock_guard<std::mutex> lock(buffer_mutex_);

    if (velocity_calculated_ == 0) {
      prev_time_ = imu_msg->header.stamp;
      velocity_calculated_ ++;
      return;
    }else if(velocity_calculated_ > 1){
      return;
    }

    ros::Time curr_time = imu_msg->header.stamp;
    // double dt = (curr_time - prev_time_).toSec();
    double dt = 0.05;

    ROS_WARN("dt for imu data: %f", dt);
    
    cv::Vec3d linear_acceleration(
        imu_msg->linear_acceleration.x,
        imu_msg->linear_acceleration.y,
        imu_msg->linear_acceleration.z
    );

    double x = imu_msg->orientation.x, y = imu_msg->orientation.y, z = imu_msg->orientation.z, w = imu_msg->orientation.w;
    cv::Matx33d rotation_matrix = cv::Matx33d(
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
        );

    cv::Vec3d accel_world = rotation_matrix * linear_acceleration;

    accel_world[2] -= 9.81;

    cv::Vec3f initial_velocity_ = accel_world * dt;

    this->angular_vel_x0 = imu_msg->angular_velocity.x;
    this->angular_vel_y0 = imu_msg->angular_velocity.y;
    this->angular_vel_z0 = imu_msg->angular_velocity.z;

    this->linear_vel_x0 = initial_velocity_[0];
    this->linear_vel_y0 = initial_velocity_[1];
    this->linear_vel_z0 = initial_velocity_[2];
    velocity_calculated_ ++;

}


void MotionCompensate::checkAndProcess(){
  if(enable_depth && total_events_msg_size_ >= expected_events_msg_ && total_depth_msg_size_ >= expected_depth_msg_){
        ROS_INFO("All messages received, starting processing...");

        double interval = (events_.back().ts.toSec() - events_.front().ts.toSec()) / depth_maps_.size();
        for (size_t i = 0; i < depth_maps_.size(); ++i) {
            double timestamp = (i + 1) * interval; 
            depth_map_timestamps.push_back(timestamp); 
        }

        // processMessages();
          processMessages_v2();

  }else if(!enable_depth){
    processMessages_v2();
    // processMessages();
  }

}

void MotionCompensate::processMessages() {

  get_intrinsic_params();

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


    cv::Mat invalid_mask = this->event_count_ < 1.0;
    this->avg_time_map_.setTo(0.0, invalid_mask);
    this->event_count_.setTo(0.000001, invalid_mask);
    this->avg_time_map_ = this->avg_time_map_.mul(1.0 / this->event_count_);


    if (slice_number == 0){
      if(better_initial) findInitialFlow(events_subset_);
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
    // double l_density = getDensity(this->mc_time_map_, 0.1);
    // double cur_density;

    while (true) {

      if(std::abs(this->lr_x*this->dx) < 1e-4 && std::abs(this->lr_y*this->dy) < 1e-4 &&
          std::abs(this->lr_rot*this->dth) < 1e-3 && std::abs(this->lr_div*this->dz) < 1e-3){
        ROS_WARN("Error converged after %d", iter);
        printInfo(0.,computeContrast(this->mc_time_map_), 0.);
        break;
        }

        double old_dx = this->dx, old_dy = this->dy, old_dz = this->dz, old_dth = this->dth;
        double l_hx = this->hx, l_hy = this->hy, l_hz = this->hz, l_hth = this->htheta;

        computeImageOfWarpedEvents(events_subset_, this->hx, this->hy, this->hz, this->htheta);

        diffTimeImage(this->mc_time_map_);

        updateModel();

        if (this->dx * old_dx < 0)   this->lr_x *= 0.8;
        if (this->dy * old_dy < 0)   this->lr_y *= 0.8;
        if (this->dz * old_dz < 0)   this->lr_rot *= 0.8;
        if (this->dth * old_dth < 0) this->lr_div *= 0.8;
      
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
          printInfo(0.,computeContrast(this->mc_event_count_), D);
          break;
      }

      D = D_prime;
      diffTimeImage(this->mc_event_count_);

      double deltas[] = {this->dx, this->dy, this->dz, this->dth};
      double lrs[] = {this->lr_x, this->lr_y, this->lr_div, this->lr_rot};
      double* model_params[] = {&this->hx, &this->hy, &this->hz, &this->htheta};
      double* model_temp_params[] = {&hx_temp, &hy_temp, &hz_temp, &htheta_temp};

      for (int i = 0; i < 4; ++i) {
        if(enable_depth && i == 2) continue;

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
        printInfo(0., computeContrast(this->mc_event_count_), D);
        break;
      }

    }

    ROS_WARN("-------------OBJECT DETECTION---------------");
    
    
    duration = events_subset_.back().ts.toSec() - events_subset_.front().ts.toSec();

    detectMovingObjects(this->avg_time_map_, this->mc_time_map_, duration, this->background_mask, this->foreground_mask);

    cv::hconcat(this->foreground_mask, this->background_mask, this->ground_mask_);

    publishMap(0.);
    if(plot_hist) plotHist(this->avg_time_map_, this->mc_time_map_);

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

void MotionCompensate::processMessages_v2() {
  
  get_intrinsic_params();

  while (idx_first_ev_map_ + num_events_map_update_ <= events_.size())
  {

    this->avg_time_map_ = cv::Mat::zeros(img_height, img_width,CV_64FC1);
    this->mc_time_map_ = cv::Mat::zeros(img_height, img_width,CV_64FC1);
    this->event_count_ = cv::Mat::zeros(img_height, img_width,CV_64FC1);
    this->mc_event_count_ = cv::Mat::zeros(img_height, img_width,CV_64FC1);
    this->ground_mask_ = cv::Mat::zeros(img_height, img_width*2,CV_64FC1);
    this->event_depth_map_ = cv::Mat::zeros(img_height, img_width,CV_64FC1);
    this->depth_patches = std::vector<double>(this->depth_x_bin_num*this->depth_y_bin_num);
    // TODO: test first depth map with 0 or 1 for random Z
    this->Z = cv::Mat::zeros(img_height, img_width,CV_64FC1);
  
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
      // this->event_depth_map_.at<double>(event.y, event.x) = 1.;        
    }

    cv::Mat invalid_mask = this->event_count_ < 1.0;
    this->avg_time_map_.setTo(0.0, invalid_mask);
    this->event_depth_map_.setTo(0., invalid_mask);
    this->event_count_.setTo(0.000001, invalid_mask);
    this->avg_time_map_ = this->avg_time_map_.mul(1.0 / this->event_count_);
    this->event_count_.copyTo(this->mc_event_count_);

    // initialize depth patch

    int patch_width = this->img_width / this->depth_x_bin_num;
    int patch_height = this->img_height / this->depth_y_bin_num;
    for (int i = 0; i < this->depth_y_bin_num; ++i) {
        for (int j = 0; j < this->depth_x_bin_num; ++j) {

            int center_x = j * patch_width + patch_width / 2;
            int center_y = i * patch_height + patch_height / 2;

            center_x = std::min(center_x, this->img_width - 1);
            center_y = std::min(center_y, this->img_height - 1);

            double depth_value = this->event_depth_map_.at<double>(center_y, center_x);
            this->depth_patches[i*this->depth_x_bin_num + j] = depth_value;
        }
    }


    // if (slice_number == 0){
    initialize_v2(events_subset_);
    // }
    slice_number++;

    ROS_WARN("################slice %d##################", slice_number);

    /***
     * Timestamp Minimizer
    */
     ROS_WARN("-------------TIME MAP MINIMIZER---------------");

     double total_score = maximizeContrast(events_subset_, ByVel);
    //  std::cout << this->depth_patches[0] << " " << this->depth_patches[30] << std::endl;
     computeImageOfWarpedEvents_v2(events_subset_, TimeMap);
     printInfo_v2(total_score, this->ContrastScore);
     publishMap(slice_first_t);

    /***
     * Event Count Minimizer
    */
     ROS_WARN("-------------Event Count MINIMIZER---------------");

     total_score = maximizeContrast(events_subset_, ByVel);
    //  std::cout << this->depth_patches[0] << " " << this->depth_patches[30] << std::endl;
     computeImageOfWarpedEvents_v2(events_subset_, EventCount);
     printInfo_v2(total_score, this->ContrastScore);
     publishMap(slice_first_t);
    

    // ROS_WARN("-------------OBJECT DETECTION---------------");
    
    
    // duration = events_subset_.back().ts.toSec() - events_subset_.front().ts.toSec();

    // detectMovingObjects(this->avg_time_map_, this->mc_time_map_, duration, this->background_mask, this->foreground_mask);

    // cv::hconcat(this->foreground_mask, this->background_mask, this->ground_mask_);
    
    // computeImageOfWarpedEvents_v2(events_subset_);
    // printInfo_v2(0.,0.);
    // publishMap(slice_first_t);


    if(plot_hist) plotHist(this->avg_time_map_, this->mc_time_map_);

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
