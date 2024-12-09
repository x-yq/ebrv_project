#include <fast_dynamic/motion_compensate_node.h>
#include <geometry_msgs/PoseStamped.h>
// #include <glog/logging.h>
#include <fast_dynamic/image_util.h>

namespace motion_compensate
{

cv::Mat MotionCompensate::denoiseTimeMap(const cv::Mat &time_map, int n, int window_size) {
    cv::Mat result = time_map.clone();

    int radius = window_size / 2;

    for (int y = radius; y < time_map.rows - radius; ++y) {
        for (int x = radius; x < time_map.cols - radius; ++x) {
            if (time_map.at<uchar>(y, x) > 0) {
                int non_zero_count = 0;

                for (int wy = -radius; wy <= radius; ++wy) {
                    for (int wx = -radius; wx <= radius; ++wx) {
                        if (time_map.at<uchar>(y + wy, x + wx) > 0) {
                            ++non_zero_count;
                        }
                    }
                }

                if (non_zero_count <= n) {
                    result.at<uchar>(y, x) = 0;
                }
            }
        }
    }

    return result;
}

/**
* \brief Publish several variables related to the mapping (mosaicing) part
*/
void MotionCompensate::publishMap()
{

  cv_bridge::CvImage cv_image_time;
  cv_image_time.header.stamp = ros::Time::now();
  cv_image_time.encoding = "mono8";

    cv_bridge::CvImage cv_image_time_mc;
  cv_image_time_mc.header.stamp = ros::Time::now();
  cv_image_time_mc.encoding = "mono8";


if ( mc_time_map_pub_.getNumSubscribers() > 0){

    cv::Mat image_stacked, normalized_stacked_image;
    cv::hconcat(avg_time_map_, mc_time_map_, image_stacked);
    image_util::normalize(image_stacked, normalized_stacked_image, 15.);
    cv::Mat denoised;
    cv::bilateralFilter(normalized_stacked_image, denoised, 9, 75, 75);
    denoised = denoiseTimeMap(denoised, 6, 3);

    denoised.copyTo(cv_image_time.image);
    mc_time_map_pub_.publish(cv_image_time.toImageMsg());
}

  cv_bridge::CvImage cv_image;
  cv_image.header.stamp = ros::Time::now();
  cv_image.encoding = "mono8";

  if ( mc_event_count_pub_.getNumSubscribers() > 0)
  {

    cv::Mat image_stacked, normalized_stacked_image;
    cv::hconcat(event_count_, mc_event_count_, image_stacked);
    image_util::normalize(image_stacked, normalized_stacked_image, 15.);
    // cv::medianBlur(normalized_stacked_image,normalized_stacked_image,3);

    cv_image.image = normalized_stacked_image;
    mc_event_count_pub_.publish(cv_image.toImageMsg());

  }

  
  cv_bridge::CvImage cv_image_mask;
  cv_image_mask.header.stamp = ros::Time::now();
  cv_image_mask.encoding = "mono8";

if ( ground_mask_pub_.getNumSubscribers() > 0){

    cv::Mat normalized_stacked_image;
    image_util::normalize(ground_mask_, normalized_stacked_image, 1.);

    normalized_stacked_image.copyTo(cv_image_mask.image);
    ground_mask_pub_.publish(cv_image_mask.toImageMsg());

}

}

// void MotionCompensate::publishModel(const Model& model, const Grad& grad){
//   // VLOG(1) << "publishModel()";
//   geometry_msgs::PointStamped model_msg;
//   model_msg.point.x = model.hx;
//   model_msg.point.y = model.hy;
//   model_msg.point.z = model.htheta;
//   model_msg.header.stamp = time_packet_;
//   model_pub_.publish(model_msg);

//   geometry_msgs::PointStamped grad_msg;
//   grad_msg.point.x = grad.dx;
//   grad_msg.point.y = grad.dy;
//   grad_msg.point.z = grad.dth;
//   grad_msg.header.stamp = time_packet_;
//   grad_pub_.publish(grad_msg);
// }

} // namespace motion_compensate


