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
void MotionCompensate::publishMap(double t_ref)
{

  cv_bridge::CvImage cv_image_time;
  cv_image_time.header.stamp = ros::Time::now();
  cv_image_time.encoding = "mono8";

if ( mc_time_map_pub_.getNumSubscribers() > 0){

    cv::Mat image_stacked, normalized_stacked_image;
    cv::hconcat(this->avg_time_map_, this->mc_time_map_, image_stacked);
    image_util::normalize(image_stacked, normalized_stacked_image, 15.);
    cv::Mat denoised;
    cv::bilateralFilter(normalized_stacked_image, denoised, 9, 75, 75);
    denoised = denoiseTimeMap(denoised, 5, 3);

    denoised.copyTo(cv_image_time.image);
    mc_time_map_pub_.publish(cv_image_time.toImageMsg());
}

  cv_bridge::CvImage cv_image_count;
  cv_image_count.header.stamp = ros::Time::now();
  cv_image_count.encoding = "mono8";

  if ( mc_event_count_pub_.getNumSubscribers() > 0)
  {

    cv::Mat image_stacked, normalized_stacked_image;
    cv::hconcat(this->event_count_, this->mc_event_count_, image_stacked);
    image_util::normalize(image_stacked, normalized_stacked_image, 15.);

    cv_image_count.image = normalized_stacked_image;
    mc_event_count_pub_.publish(cv_image_count.toImageMsg());

  }

  
//   cv_bridge::CvImage cv_image_mask;
//   cv_image_mask.header.stamp = ros::Time::now();
//   cv_image_mask.encoding = "mono8";

// if ( ground_mask_pub_.getNumSubscribers() > 0){

//     cv::Mat normalized_stacked_image;
//     image_util::normalize(ground_mask_, normalized_stacked_image, 1.);

//     normalized_stacked_image.copyTo(cv_image_mask.image);
//     ground_mask_pub_.publish(cv_image_mask.toImageMsg());

// }

  cv_bridge::CvImage cv_depth_map;
  cv_depth_map.header.stamp = ros::Time::now();
  cv_depth_map.encoding = "bgr8";

if ( ground_mask_pub_.getNumSubscribers() > 0){

    cv::Mat image_stacked, normalized_stacked_image;
    cv::Mat depth_;
    if(random_initial){
        depth_ = this->Z;
    }else{
        depth_ = getGTDepthMap_v2(t_ref);
    }
    this->grad_Z = cv::Mat::zeros(this->Z.rows, this->Z.cols, CV_64FC1);
    cv::hconcat(this->grad_Z, this->Z, image_stacked);
    image_util::normalize(image_stacked, normalized_stacked_image, 1.);
    // cv::Mat denoised;
    // cv::bilateralFilter(normalized_stacked_image, denoised, 9, 75, 75);
    // denoised = denoiseTimeMap(denoised, 5, 3);

    cv::Mat colored_denoised;
    cv::applyColorMap(normalized_stacked_image, colored_denoised, cv::COLORMAP_JET);

    colored_denoised.copyTo(cv_depth_map.image);
    ground_mask_pub_.publish(cv_depth_map.toImageMsg());

}

}


} // namespace motion_compensate


