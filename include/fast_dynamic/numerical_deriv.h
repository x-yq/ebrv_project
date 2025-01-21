#pragma once

#include <gsl/gsl_vector.h>
#include <opencv2/core/core.hpp>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

double vs_gsl_Gradient_ForwardDiff (
    const cv::Mat& depth_map,
    const std::vector<dvs_msgs::Event>& events_subset,
    const gsl_vector * x, /**< [in] Point at which the gradient is to be evaluated */
    double (*func_f)(const gsl_vector * x, const std::vector<dvs_msgs::Event>& events_subset, const cv::Mat& depth_map), /**< [in] User-supplied routine that returns the value of the function at x */
    gsl_vector * J,       /**< [out] Gradient vector (same length as x) */
    double dh      /**< [in] Increment in variable for numerical differentiation */
    );