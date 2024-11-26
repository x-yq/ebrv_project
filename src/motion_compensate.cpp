// event_count
// time_map for avg_time_map
// motion_compensated_event_count
// time_image for minimized ...

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <glog/logging.h>
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
  nh_private.param<double>("lr_x", lr_x, 10.0);
  nh_private.param<double>("lr_y", lr_y, 10.0);
  nh_private.param<double>("lr_div", lr_div, 1.0);
  nh_private.param<double>("lr_rot", lr_rot, 1.0);



  // set queue_size to 0 to avoid discarding messages (for correctness).
  event_sub_ = nh_.subscribe("events", 0, &MotionCompensate::eventsCallback, this);

  // Set up publishers
  image_transport::ImageTransport it_(nh_);
  // my topics names: event_count, avg_time_map, motion_compensated_event_count, motion_compensated_time_map
  event_count_pub_ = it_.advertise("event_count", 0);
  avg_time_map_pub_ = it_.advertise("avg_time_map", 0);
  mc_event_count_pub_ = it_.advertise("mc_event_count", 0);
  mc_time_map_pub_ = it_.advertise("mc_time_map", 0);

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

}


/**
* \brief Function to process event messages received by the ROS node
*/
void MotionCompensate::eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
  // Append events of current message to the queue
  std::cout << "Received an EventArray message with " << msg->events.size() << std::endl;


  for(const dvs_msgs::Event& ev : msg->events)
    events_.push_back(ev);
  
  static unsigned int packet_number = 0;
  static unsigned long total_event_count = 0;
  total_event_count += msg->events.size();
  VLOG(1) << "Packet # " << packet_number << "  event# " << total_event_count << "  queue_size:" << events_.size();

  const float width = msg->width;
  const float height = msg->height;
  // if (packet_number == 0)
  // {
  //   avg_time_map_ = cv::Mat::zeros(height, width,CV_64FC1);
  //   mc_time_map_ = cv::Mat::zeros(height, width,CV_64FC1);
  //   event_count_ = cv::Mat::zeros(height, width,CV_32FC1);
  //   mc_event_count_ = cv::Mat::zeros(height, width,CV_32FC1);

  // }
  // packet_number++;
    Model M_G_prev(0., 0., 0., 0.);
    Model M_G_new;
  
    avg_time_map_ = cv::Mat::zeros(height, width,CV_64FC1);
    mc_time_map_ = cv::Mat::zeros(height, width,CV_64FC1);
    event_count_ = cv::Mat::zeros(height, width,CV_32FC1);
    mc_event_count_ = cv::Mat::zeros(height, width,CV_32FC1);

  // getSlideWindowSize(events_, discretization_, num_events_map_update_);
  ROS_WARN("num_events_map_update_ = %f", num_events_map_update_);

  t0 = events_.front().ts;

  while (idx_first_ev_map_ + num_events_map_update_ <= events_.size())
  {

    VLOG(1) << "MAP ev= " << idx_first_ev_map_ << " -- "
            << idx_first_ev_map_ + num_events_map_update_ << ", events_.size()=" << events_.size();
    // Get subset of events
    events_subset_temp = std::vector<dvs_msgs::Event> (events_.begin() + idx_first_ev_map_,
                                                   events_.begin() + idx_first_ev_map_ + num_events_map_update_);

    for(auto& e: events_subset_temp){
      e.ts = ros::Time(e.ts.toSec()-t0.toSec());
    }
    const std::vector<dvs_msgs::Event> events_subset_ = events_subset_temp;
 
    // Compute time span of the events
    ros::Time time_first = events_subset_.front().ts;
    ros::Time time_last = events_subset_.back().ts;
    ros::Duration time_dt = time_last - time_first;
    time_packet_ = time_first + time_dt * 0.5;
    VLOG(2) << "MAP: duration [s]= "<< time_dt.toSec();

    /***
     * Get the average time map and event count map
    */
    event_count_ =  cv::Mat::zeros(height, width,CV_32FC1);
    avg_time_map_ =  cv::Mat::zeros(height, width,CV_64FC1);


    for (const dvs_msgs::Event& event : events_subset_) {
        if (event.x >= 0 && event.x < width && event.y >= 0 && event.y < height) {
            event_count_.at<float>(event.y, event.x)+=1.0;
            double event_time = event.ts.toSec(); 
            avg_time_map_.at<double>(event.y, event.x) += event_time;
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float count = event_count_.at<float>(y, x);
            if (count > 0) {
                avg_time_map_.at<double>(y, x) /= count;
            }
        }
    }


    float total_count = cv::sum(event_count_)[0];
    cv::Mat mask = (event_count_ != 0.0);
    float count = cv::countNonZero(mask); 
    std::cout << "Event count: " << total_count << std::endl;

    /***
     * Timestamp Minimizer
    */

    // cv::Mat count_mask = (mc_time_map_ > 0.0);
    cv::Mat mc_event_count_temp;// = cv::Mat::zeros(height, width, CV_32FC1);
    // mc_event_count_temp.setTo(1.0, count_mask);

    computeImageOfWarpedEvents(M_G_prev, events_subset_, width, height,  &mc_time_map_,&mc_event_count_temp);

    int iter = 0;
    ap_x.t = 0, ap_x.m = 0, ap_x.v = 0;
    ap_y.t = 0, ap_y.m = 0, ap_y.v = 0;
    ap_div.t = 0, ap_div.m = 0, ap_div.v = 0;
    ap_rot.t = 0, ap_rot.m = 0, ap_rot.v = 0;

    double dx, dy, dz, dtheta;
    diffTimeImage(mc_time_map_, dx, dy, dz, dtheta);
    std::cout << "dx: " << dx << " dy: " << dy << " dz: " << dz << " dth: " << dtheta << std::endl;
    updateModel(M_G_prev, dx, dy, dz, dtheta, &M_G_new);
    double old_dx = dx, old_dy = dy, old_dz = dz, old_th = dtheta;

    // double currentError = computeError2(mc_time_map_);
    double error = computeError(M_G_prev, M_G_new);
    std::cout << "current error: " << error << std::endl;

    while (true) {
        if (iter > maxIterations){
          std::cout << "reached max iterations " << iter + 1 << std::endl;
          std::cout << "current model " << M_G_new.hx << " " << M_G_new.hy << " " << M_G_new.hz << " " << M_G_new.htheta << std::endl;
          std::cout << "dx: " << dx << " dy: " << dy << " dz: " << dz << " dth: " << dtheta << std::endl;
          std::cout << "current error: " << error << std::endl;

          break;
        }

        // Check if the gradient is close to zero
        if (std::abs(dx) < 1e-6 && std::abs(dy) < 1e-6 && std::abs(dz) < 1e-6 && std::abs(dtheta) < 1e-6) {
            std::cout << "Gradient converged after " << iter + 1 << " iterations." << std::endl;
            std::cout << "current model " << M_G_new.hx << " " << M_G_new.hy << " " << M_G_new.hz << " " << M_G_new.htheta << std::endl;
            std::cout << "dx: " << dx << " dy: " << dy << " dz: " << dz << " dth: " << dtheta << std::endl;
            std::cout << "current error: " << error << std::endl;
            break;
        }

        // Check if the error change is below the threshold
        if (error <= acc_threshold_) {
            // std::cout << std::abs(1 - currentError / (lastError + 1e-7)) << std::endl;
            std::cout << "Converged due to error change being below threshold after " << iter + 1 << " iterations with following values: "
                      << M_G_new.hx << " " << M_G_new.hy << " " << M_G_new.hz << " " << M_G_new.htheta << std::endl;
            std::cout << "dx: " << dx << " dy: " << dy << " dz: " << dz << " dth: " << dtheta << std::endl;
            std::cout << "current error: " << error << std::endl;
            break;
        }

//  std::cout << "before  " 
//                       << M_G_prev.hx << " " << M_G_prev.hy << " " << M_G_prev.hz << " " << M_G_prev.htheta << std::endl;
            
        M_G_prev = M_G_new;

        //  std::cout << "after  " 
        //               << M_G_prev.hx << " " << M_G_prev.hy << " " << M_G_prev.hz << " " << M_G_prev.htheta << std::endl;
            
    

        // Perform image warping with updated M_G_new
        computeImageOfWarpedEvents(M_G_new, events_subset_, width, height, &mc_time_map_, &mc_event_count_temp);
        
        // Compute time difference and gradient again
        diffTimeImage(mc_time_map_, dx, dy, dz, dtheta);

        // Update model parameters using gradient descent
        updateModel(M_G_prev, dx, dy, dz, dtheta, &M_G_new);
        // std::cout << "dx: " << dx << " dy: " << dy << " dz: " << dz << " dth: " << dtheta << std::endl;
        // std::cout << "current model" << M_G_new.hx << " " << M_G_new.hy << " " << M_G_new.hz << " " << M_G_new.htheta << std::endl;
        
        // currentError = computeError2(mc_time_map_);
        error = computeError(M_G_prev, M_G_new);

        // Increment iteration count
        iter++;
        old_dx = dx, old_dy = dy, old_dz = dz, old_th = dtheta;


        // std::cout << "current error: " << error << "current threshodl"<< acc_threshold_<< std::endl;

    }

    
   /*
   
   Event count minimizer
   
   */
  ROS_WARN("-----------------EVENT COUNT MINIMIZER-------------------");

  Model M_C_new = M_G_new;
  std::cout << "x: " << M_C_new.hx << " y:" << M_C_new.hy << " hz:"  << M_C_new.hz << " th: " << M_C_new.htheta << std::endl;

    cv::Mat mc_time_map_temp;

    double D_prime, D;
    cv::Mat mc_event_count_result;

    computeImageOfWarpedEvents(M_C_new, events_subset_, width, height, &mc_time_map_temp,&mc_event_count_temp);

    D = getDensity(mc_event_count_temp);
    D_prime = 0.;
    mc_event_count_temp.copyTo(mc_event_count_);
    iter = 0;
    ap_x.t = 0, ap_x.m = 0, ap_x.v = 0;
    ap_y.t = 0, ap_y.m = 0, ap_y.v = 0;
    ap_div.t = 0, ap_div.m = 0, ap_div.v = 0;
    ap_rot.t = 0, ap_rot.m = 0, ap_rot.v = 0;

    while (true) {
      if(std::abs(D - D_prime) < 1e-7){
          std::cout << "Density converged after " << iter + 1 << " iterations." << std::endl;
          std::cout << "current model " << M_C_new.hx << " " << M_C_new.hy << " " << M_C_new.hz << " " << M_C_new.htheta << std::endl;
          std::cout << "dx: " << dx << " dy: " << dy << " dz: " << dz << " dth: " << dtheta << std::endl;
          std::cout << "current error: " << error << std::endl;
          break;
      }

      if (iter > maxIterations){
        std::cout << "reached max iterations " << iter + 1 << std::endl;
        std::cout << "current model " << M_C_new.hx << " " << M_C_new.hy << " " << M_C_new.hz << " " << M_C_new.htheta << std::endl;
        std::cout << "dx: " << dx << " dy: " << dy << " dz: " << dz << " dth: " << dtheta << std::endl;
        std::cout << "current error: " << error << std::endl;
        break;
      }

        Model M_C_temp;
        M_C_temp = M_C_new;
        D = D_prime;
        diffTimeImage(mc_time_map_temp, dx, dy, dz, dtheta);
        // std::cout << "dx: " << dx << " dy: " << dy << " dz: " << dz << " dth: " << dtheta << std::endl;

        double deltas[] = {dx, dy, dz, dtheta};
        // double deltas[] = {M_G_new.hx, M_G_new.hy, M_G_new.hz, M_G_new.htheta};
        double* model_params[] = {&M_C_new.hx, &M_C_new.hy, &M_C_new.hz, &M_C_new.htheta};
        double* model_temp_params[] = {&M_C_temp.hx, &M_C_temp.hy, &M_C_temp.hz, &M_C_temp.htheta};

        for (int i = 0; i < 4; ++i) {

            *model_temp_params[i] = *model_params[i] + deltas[i];
            computeImageOfWarpedEvents(M_C_temp, events_subset_, width, height, &mc_time_map_temp, &mc_event_count_temp);

            D_prime = getDensity(mc_event_count_temp);

            if (D_prime > D) {
              *model_params[i] += deltas[i];
              computeImageOfWarpedEvents(M_C_new, events_subset_, width, height, &mc_time_map_, &mc_event_count_);
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

    computeImageOfWarpedEvents(M_C_new, events_subset_, width, height, &mc_time_map_, &mc_event_count_);

    std::cout << "Model hx: " << M_C_new.hx << " hy:" << M_C_new.hy << " hz:"  << M_C_new.hz << " th: " << M_C_new.htheta << std::endl;

    publishMap();

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

    VLOG(1) << "-- num_events_map_update_ = " << num_events_map_update_;

  //  M_G_prev = M_G_new;
  
  }

}

/**
* \brief Compute the current size of slide window
*/
void MotionCompensate::getSlideWindowSize(const std::deque<dvs_msgs::Event>& evs, const double discretization_, int& num_ev) {
    if (evs.size() <= 1) {
        num_ev = 1;
        return;
    }

    double start_time = evs.front().ts.toSec();
    double target_time = start_time + discretization_ ;
    ROS_WARN("target time: %f ",target_time);

    // Find the first index where event time >= target_time
    auto it = std::lower_bound(evs.begin(), evs.end(), target_time, [](const dvs_msgs::Event& event, double time) {
        return event.ts.toSec() < time;
    });

    // Calculate the index
    num_ev = std::distance(evs.begin(), it);

    // for(const dvs_msgs::Event ev : evs)
    //   ROS_WARN("event.ts.toSec() = %f", ev.ts.toSec());
    // ROS_WARN("target time = %f", target_time);
    
}

cv::Point2d MotionCompensate::warpEvent(
  const Model& model,
  const dvs_msgs::Event& event,
  const double& t_ref
){
        cv::Point2d warped_pt;

        double x = event.x;
        double y = event.y;
        double t = event.ts.toSec()  - t_ref; 

        double cosTheta = std::cos(model.htheta);
        double sinTheta = std::sin(model.htheta);

        double rotX = cosTheta * x - sinTheta * y;
        double rotY = sinTheta * x + cosTheta * y;

        double hx = model.hx;
        double hy = model.hy;
        double hz = model.hz;

        warped_pt.x = x - t * (hx + (hz + 1) * rotX - x);
        warped_pt.y = y - t * (hy + (hz + 1) * rotY - y);

        // std::cout << "warped x displacement : " << warped_pt.x - event.x << " warped y displacement: " << warped_pt.y - event.y << std::endl;
        return warped_pt;

}

void MotionCompensate::accumulateWarpedEvent(
  const int img_width,
  const int img_height,
  const dvs_msgs::Event& event,
  const cv::Point2d& ev_warped_pt,
  cv::Mat* image_warped,
  cv::Mat* image_event_count
)
{
  // Accumulate warped events, using bilinear voting (polarity or count)
   int xx = ev_warped_pt.x,
       yy = ev_warped_pt.y;

  if (1 <= xx && xx < img_width-2 && 1 <= yy && yy < img_height-2)
  {

    double time = event.ts.toSec();

      (*image_warped).at<double>(yy, xx) += time;

      (*image_event_count).at<float>(yy, xx) += 1.;

  }
}

void MotionCompensate::computeImageOfWarpedEvents(
  const Model& model,
  const std::vector<dvs_msgs::Event>& events_subset,
  const int width,
  const int height,
  cv::Mat* image_warped,
  cv::Mat* image_event_count
)
{

  *image_event_count = cv::Mat::zeros(height, width, CV_32FC1);
  *image_warped = cv::Mat::zeros(height, width, CV_64FC1);
  const double t_ref = events_subset.front().ts.toSec();

  // Loop through all events
  for (const dvs_msgs::Event& ev : events_subset)
  {
    cv::Point2d warped_pt;
    warped_pt = warpEvent(model, ev, t_ref);
    accumulateWarpedEvent(width, height, ev, warped_pt, image_warped, image_event_count);
    // ROS_WARN("displacement for x %f", ev.x - warped_pt.x);
  }
  for (int y = 0.; y < height; y+=1) {
    for (int x = 0.; x < width; x+=1) {
        float count = image_event_count->at<float>(y, x);
        if (count > 0) {
            image_warped->at<double>(y, x) /= count;
        }
    }
}

}

double MotionCompensate::computeError(
  const Model& modelPrev,
  const Model& modelCurr
){
  return std::sqrt(
                std::pow(modelCurr.hx - modelPrev.hx, 2) +
                std::pow(modelCurr.hy - modelPrev.hy, 2) +
                std::pow(modelCurr.hz - modelPrev.hz, 2) +
                std::pow(modelCurr.htheta - modelPrev.htheta, 2)
            ); 
}

double MotionCompensate::computeError2(const cv::Mat& image)
{
  double contrast = cv::norm(image,cv::NORM_L2SQR) / static_cast<double>(image.rows*image.cols);
  std::cout << contrast << std::endl;
  return contrast;
}

void MotionCompensate::updateModel(const Model& pre, 
const double& dx, const double& dy, const double& dz, 
const double& dtheta,Model* cur){
  cur->hx = pre.hx + dx;
  cur->hy = pre.hy + dy;
  cur->hz = pre.hz + dz;
  cur->htheta = pre.htheta + dtheta;
}

// void MotionCompensate::dZThetaUpdate(const cv::Mat &grad_x, const cv::Mat &grad_y,
//   double &div_r, double &rot_r)
// {

//     // avg_magnitude = computeError2(time_image);
//     // updateThresholds(avg_magnitude, magnitude_threshold);

//     // ROS_WARN("avg_magnitude: %f", avg_magnitude);
//     // ROS_WARN("mag threshold: %f", magnitude_threshold);

//     double rot = 0.0, div = 0.0, pixel_cnt = 0.0;
//     const double EPS = 1e-7;

//     for (int y = 0; y < grad_x.rows; ++y) {
//         for (int x = 0; x < grad_x.cols; ++x) {
//             double dx = grad_x.at<double>(y, x);
//             double dy = grad_y.at<double>(y, x);

//             if (std::abs(dx) < EPS || std::abs(dy) < EPS) continue;

//             double rx = x - grad_x.cols / 2.0;
//             double ry = y - grad_x.rows / 2.0;

//             if (std::abs(rx) < 0.5 || std::abs(ry) < 0.5) continue;

//             rot += rx * dy - ry * dx;
//             div += rx * dx + ry * dy;
//             pixel_cnt++;
//         }
//     }

//     if (pixel_cnt > 10) {
//         rot /= pixel_cnt;
//         div /= pixel_cnt;
//     } else {
//         rot = div = 0.0;
//     }


//     // double scale = std::abs(magnitude_threshold - avg_magnitude + 1e-7);
//     // grad_square_accum_dr += rot * rot + div * div;
//     // std::cout<<"grad square accum dr" << std::sqrt(grad_square_accum_dr) << std::endl;
//     // double scale = 1.0 / (std::sqrt(grad_square_accum_dr) + EPS);

//     // std::cout <<"rot and div"<<std::endl;
//     // std::cout << scale_rot << " " << scale_div << std::endl;


//     cv::Mat curl_x, curl_y;
//     cv::Sobel(grad_y, curl_x, CV_64F, 1, 0, 5); // ∂T_y / ∂x
//     cv::Sobel(grad_x, curl_y, CV_64F, 0, 1, 5); // ∂T_x / ∂y
//     cv::Mat curl = curl_x - curl_y;             // curl = ∂T_y/∂x - ∂T_x/∂y
//     // double mean_curl = cv::mean(curl)[0];
//     double mean_curl = cv::mean(cv::abs(curl))[0];


//     double rotation_threshold = 0.01; 
//     std::cout << mean_curl << std::endl;
//     if (std::abs(mean_curl) > rotation_threshold) {
//       ap_rot.t+=1;
//       ap_div.t+=1;

//       rot_r = lr_rot * updateAdam(rot, ap_rot) * div;
//       div_r = lr_div * updateAdam(div, ap_div) * rot;
//     } else {
//       rot_r = 0.0;
//       div_r = 0.0;
//     }

// }

void MotionCompensate::dZThetaUpdate(const cv::Mat &grad_x, const cv::Mat &grad_y,
                                     double &div_r, double &rot_r) {
    double rot = 0.0, div = 0.0, pixel_cnt = 0.0;
    const double EPS = 1e-7;

    for (int y = 0; y < grad_x.rows; ++y) {
        for (int x = 0; x < grad_x.cols; ++x) {
            double dx = grad_x.at<double>(y, x);
            double dy = grad_y.at<double>(y, x);

            if (std::abs(dx) < EPS || std::abs(dy) < EPS) continue;

            double rx = x - grad_x.cols / 2.0;
            double ry = y - grad_x.rows / 2.0;

            if (std::abs(rx) < 0.5 || std::abs(ry) < 0.5) continue;

            rot += rx * dy - ry * dx;
            div += rx * dx + ry * dy;
            pixel_cnt++;
        }
    }

    if (pixel_cnt > 100) {
        rot /= pixel_cnt;
        div /= pixel_cnt;
    } else {
        rot = div = 0.0;
    }

    // cv::Mat angle_map(grad_x.size(), CV_64F);
    // for (int y = 0; y < grad_x.rows; ++y) {
    //     for (int x = 0; x < grad_x.cols; ++x) {
    //         double dx = grad_x.at<double>(y, x);
    //         double dy = grad_y.at<double>(y, x);
    //         angle_map.at<double>(y, x) = std::atan2(dy, dx);
    //     }
    // }

    // int bins = 36; 
    // cv::Mat hist = cv::Mat::zeros(1, bins, CV_64F);
    // for (int y = 0; y < angle_map.rows; ++y) {
    //     for (int x = 0; x < angle_map.cols; ++x) {
    //         double theta = angle_map.at<double>(y, x);
    //         int bin = static_cast<int>((theta + CV_PI) / (2 * CV_PI) * bins) % bins;
    //         hist.at<double>(0, bin)++;
    //     }
    // }

    // double max_value, min_value;
    // cv::minMaxLoc(hist, &min_value, &max_value);

    // double rotation_threshold = 2.0;
    // std::cout << (max_value - min_value) / (pixel_cnt + EPS) << std::endl;
    // bool has_rotation = (max_value - min_value) / (pixel_cnt + EPS) > rotation_threshold;


    if (!(abs(rot) < 0.01 || abs(div) < 0.01)) {
      ap_rot.t += 1;
      ap_div.t += 1;

      rot_r = lr_rot * updateAdam(rot, ap_rot);
      div_r = lr_div * updateAdam(div, ap_div);
    } else {
      rot_r = 0.0;
      div_r = 0.0;
    }
}



void MotionCompensate::dXYUpdate(const cv::Mat &grad_x,const cv::Mat &grad_y,
                    double &dx_val_r, double &dy_val_r) {
    const double ORIENTATION_THRESHOLD = 45.0;  

    cv::Mat magnitude, orientation;
    cv::cartToPolar(grad_x, grad_y, magnitude, orientation, true);
    const double EPS = 1e-7;

    double dx_sum = 0.0, dy_sum = 0.0;
    int dx_count = 0, dy_count = 0, cnt = 0;
    double dx_, dy_;

    for (int y = 0; y < grad_x.rows; ++y) {
        for (int x = 0; x < grad_x.cols; ++x) {
            double theta = orientation.at<double>(y, x);

            dx_ += grad_x.at<double>(y, x);
            dy_ += grad_y.at<double>(y, x);
            cnt ++;

            if ((theta >= -ORIENTATION_THRESHOLD && theta < ORIENTATION_THRESHOLD) || 
                (theta > 180 - ORIENTATION_THRESHOLD && theta <= 180 + ORIENTATION_THRESHOLD)) {
                dx_sum += grad_x.at<double>(y, x);
                dx_count++;
            } 
            else if ((theta > 90 - ORIENTATION_THRESHOLD && theta < 90 + ORIENTATION_THRESHOLD) || 
                     (theta > 270 - ORIENTATION_THRESHOLD && theta < 270 + ORIENTATION_THRESHOLD)) {
                dy_sum += grad_y.at<double>(y, x);
                dy_count++;
            }
        }
    }

    if (dx_count > 0 && dy_count>0) { 
        dx_sum /= dx_count;
        dy_sum /= dy_count;
        dx_ /= cnt;
        dy_ /=cnt;

        double ratio = std::abs(dx_sum/(dy_sum+1e-7));

        if(ratio > 20.0){
          ap_x.t+=1;
          double ada = updateAdam(dx_, ap_x);
          // std::cout << "adam x" << ada << std::endl;
          dx_val_r = ada * lr_x; //* dx_sum;
          dy_val_r = 0.;
        }
        else if(ratio < 0.05){
          dx_val_r = 0.;
          ap_y.t+=1;
          double ada = updateAdam(dy_, ap_y);
          // std::cout << "adam y" << ada << std::endl;
          dy_val_r = lr_y * ada;// * dy_sum;
        }
        else{
          ap_x.t+=1;
          ap_y.t+=1;
          double ada = updateAdam(dx_, ap_x);
          // std::cout << "adam x" << ada << std::endl;
          dx_val_r = ada * lr_x; //* dx_sum;

          ada = updateAdam(dy_, ap_y);
          // std::cout << "adam y" << ada << std::endl;
          dy_val_r = lr_y * ada;// * dy_sum;
        }
        
        // std::cout << "dx and dy" << std::endl;
        // std::cout << dx_sum << " " << dy_sum << std::endl;
    } else {
        dx_val_r = dy_val_r = 0.0;
    }
}


// void MotionCompensate::updateThresholds(const double& error, const cv::Mat& time_image, double &error_threshold) {
//    double contrast = computeError2(time_image);
//     if(std::abs(contrast)+0.01 < std::abs(avg_contrast)){
//       error_history.push_back(error);
//       if (error_history.size() > WINDOW_SIZE) error_history.pop_front();
//       double avg_error = std::accumulate(error_history.begin(), error_history.end(), 0.0) / error_history.size();
//       error_threshold = avg_error - 0.1;
//     }
//     else{
//       error_threshold = error_threshold;
//     }

// }

void MotionCompensate::diffTimeImage(const cv::Mat &time_image,
                   double &dx_val_r, double &dy_val_r, 
                   double &dz_val_r, double &dth_val_r) {
    dx_val_r = dy_val_r = dz_val_r = dth_val_r = 0.0;

    cv::Mat time_image_blurred;
    cv::GaussianBlur(time_image, time_image_blurred, cv::Size(0, 0), 1.0);

    cv::Mat grad_x, grad_y;
    cv::Sobel(time_image_blurred, grad_x, CV_64FC1, 1, 0, 3);
    cv::Sobel(time_image_blurred, grad_y, CV_64FC1, 0, 1, 3);

    dXYUpdate(grad_x, grad_y, dx_val_r, dy_val_r);
    dZThetaUpdate(grad_x, grad_y, dz_val_r, dth_val_r);
}

double MotionCompensate::updateAdam(const double& grad, AdamParam& ap){

  ap.m = beta1 * ap.m + (1 - beta1) * grad;
  ap.v = beta2 * ap.v + (1 - beta2) * (grad * grad);

  double m_hat = ap.m / (1 - std::pow(beta1, ap.t));
  double v_hat = ap.v / (1 - std::pow(beta2, ap.t));

  double scale = alpha * m_hat / (std::sqrt(v_hat) + 1e-7);

  return scale;

}

double MotionCompensate::getDensity(const cv::Mat& event_count){
  double total_count = cv::sum(event_count)[0];
  cv::Mat mask = (event_count != 0.0);
  double count = cv::countNonZero(mask); 
  double density = total_count / count;
  return density;
}


}
