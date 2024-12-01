#include <fast_dynamic/motion_compensate_node.h>
#include <geometry_msgs/PoseStamped.h>
// #include <glog/logging.h>
#include <fast_dynamic/image_util.h>

namespace motion_compensate
{

/**
* \brief Publish several variables related to the mapping (mosaicing) part
*/
void MotionCompensate::publishMap()
{
  // Publish the current map state
  // VLOG(1) << "publishMap()";

  cv_bridge::CvImage cv_image_time;
  cv_image_time.header.stamp = ros::Time::now();
  cv_image_time.encoding = "mono8";

    cv_bridge::CvImage cv_image_time_mc;
  cv_image_time_mc.header.stamp = ros::Time::now();
  cv_image_time_mc.encoding = "mono8";


if ( mc_time_map_pub_.getNumSubscribers() > 0){
  // float min_val, max_val;
  // image_util::minMaxLocRobust(avg_time_map_, min_val, max_val, 10.); 
  // float min_val_mc, max_val_mc;
  // image_util::minMaxLocRobust(mc_time_map_, min_val_mc, max_val_mc, 10.); 

  // float global_min = std::min(min_val, min_val_mc);
  // float global_max = std::max(max_val, max_val_mc);
  // cv::Mat normalized_time_map, normalized_mc_time_map;

  //   image_util::normalize(avg_time_map_, global_min, global_max, normalized_time_map, 10.);
  //   cv::Mat avg_time_map_color_mapped, avg_time_map_enhanced_map;

  //   cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
  //   clahe->setClipLimit(3.0);
  //   clahe->apply(normalized_time_map, avg_time_map_enhanced_map);
  //   cv::applyColorMap(avg_time_map_enhanced_map, avg_time_map_color_mapped, cv::COLORMAP_JET);
  //   cv::Mat mask = (normalized_time_map == 0);
  //   avg_time_map_color_mapped.setTo(cv::Vec3b(0, 0, 0), mask);
  //   avg_time_map_color_mapped.copyTo(cv_image_time.image);
  //   avg_time_map_pub_.publish(cv_image_time.toImageMsg());
  
  
  //   image_util::normalize(mc_time_map_, global_min, global_max, normalized_mc_time_map, 10.);
  //   cv::Mat mc_time_map_color_mapped, mc_time_map_enhanced;
  //   clahe = cv::createCLAHE();
  //   clahe->setClipLimit(3.0);
  //   clahe->apply(normalized_mc_time_map, mc_time_map_enhanced);
  //   cv::applyColorMap(mc_time_map_enhanced, mc_time_map_color_mapped, cv::COLORMAP_JET);
  //   mask = (normalized_mc_time_map == 0);
  //   mc_time_map_color_mapped.setTo(cv::Vec3b(0, 0, 0), mask);
  //   mc_time_map_color_mapped.copyTo(cv_image_time.image);
  //   mc_time_map_pub_.publish(cv_image_time.toImageMsg());

  // float min_val, max_val;
  // image_util::minMaxLocRobust(avg_time_map_, min_val, max_val, 10.); 
  // float min_val_mc, max_val_mc;
  // image_util::minMaxLocRobust(mc_time_map_, min_val_mc, max_val_mc, 10.); 

  // float global_min = std::min(min_val, min_val_mc);
  // float global_max = std::max(max_val, max_val_mc);
  // cv::Mat normalized_time_map, normalized_mc_time_map;

  // // Normalize avg_time_map_
  // image_util::normalize(avg_time_map_, global_min, global_max, normalized_time_map, 10.);

  // cv::Mat mask = (normalized_time_map == 0);
  // normalized_time_map.setTo(0, mask); // Set masked areas to black (0)
  // // normalized_time_map.copyTo(cv_image_time.image);
  // // avg_time_map_pub_.publish(cv_image_time.toImageMsg());

  // // Normalize mc_time_map_
  // image_util::normalize(mc_time_map_, global_min, global_max, normalized_mc_time_map, 10.);

  // mask = (normalized_mc_time_map == 0);
  // normalized_mc_time_map.setTo(0, mask); // Set masked areas to black (0)

  // cv::Mat to_publish;
  // cv::hconcat(normalized_time_map, normalized_mc_time_map, to_publish);
  // to_publish.copyTo(cv_image_time_mc.image);
  // mc_time_map_pub_.publish(cv_image_time_mc.toImageMsg());

    cv::Mat image_stacked, normalized_stacked_image;
    cv::hconcat(avg_time_map_, mc_time_map_, image_stacked);
    // float min_val, max_val;
    // image_util::minMaxLocRobust(image_stacked, min_val, max_val, 1.);
    image_util::normalize(image_stacked, normalized_stacked_image, 1.);

    normalized_stacked_image.copyTo(cv_image_time.image);
    mc_time_map_pub_.publish(cv_image_time.toImageMsg());

}

  cv_bridge::CvImage cv_image;
  cv_image.header.stamp = ros::Time::now();
  cv_image.encoding = "mono8";

  if ( mc_event_count_pub_.getNumSubscribers() > 0)
  {
    // float min_val, max_val;
    // image_util::minMaxLocRobust(event_count_, min_val, max_val, 1.);
    // float min_val_mc, max_val_mc;
    // image_util::minMaxLocRobust(mc_event_count_, min_val_mc, max_val_mc, 1.);

    cv::Mat image_stacked, normalized_stacked_image;
    cv::hconcat(event_count_, mc_event_count_, image_stacked);
    float min_val, max_val;
    image_util::minMaxLocRobust(image_stacked, min_val, max_val, 1.);
    float scale = ((max_val != min_val) ? 255.f / (max_val - min_val) : 10.f);
    normalized_stacked_image = scale * (image_stacked - min_val);
    normalized_stacked_image.convertTo(normalized_stacked_image, CV_8UC1);

    cv_image.image = normalized_stacked_image;
    mc_event_count_pub_.publish(cv_image.toImageMsg());


    // float global_min = std::min(min_val, min_val_mc);
    // float global_max = std::max(max_val, max_val_mc);

    // cv::Mat normalized_ec_map, normalized_mc_ec_map;

    // float scale = ((global_max != global_min) ? 255.f / (global_max - global_min) : 10.f);
    // normalized_ec_map = scale * (event_count_ - global_min);
    // normalized_ec_map.convertTo(normalized_ec_map, CV_8UC1);
    // // cv_image.image = normalized_ec_map;
    // // event_count_pub_.publish(cv_image.toImageMsg());

    // scale = ((global_max != global_min) ? 255.f / (global_max - global_min) : 10.f);
    // normalized_mc_ec_map = scale * (mc_event_count_ - global_min);
    // normalized_mc_ec_map.convertTo(normalized_mc_ec_map, CV_8UC1);
    // // cv_image.image = normalized_mc_ec_map;
    // // mc_event_count_pub_.publish(cv_image.toImageMsg());

    // int equality = normalized_mc_ec_map == normalized_ec_map;
    // std::cout << equality << std::endl;
  }

}

void MotionCompensate::publishModel(const Model& model, const Grad& grad){
  // VLOG(1) << "publishModel()";
  geometry_msgs::PointStamped model_msg;
  model_msg.point.x = model.hx;
  model_msg.point.y = model.hy;
  model_msg.point.z = model.htheta;
  model_msg.header.stamp = time_packet_;
  model_pub_.publish(model_msg);

  geometry_msgs::PointStamped grad_msg;
  grad_msg.point.x = grad.dx;
  grad_msg.point.y = grad.dy;
  grad_msg.point.z = grad.dth;
  grad_msg.header.stamp = time_packet_;
  grad_pub_.publish(grad_msg);
}

} // namespace motion_compensate


