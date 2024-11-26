#pragma once

#include <opencv2/core/core.hpp>

namespace image_util
{

void minMaxLocRobust(const cv::Mat& image, float& rmin, float& rmax,
                     const float& percentage_pixels_to_discard);

void normalize(const cv::Mat& src, const float& rmin_val, const float& rmax_val, cv::Mat& dst, const float& percentage_pixels_to_discard);

} // namespace
