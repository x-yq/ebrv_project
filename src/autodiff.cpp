// #include <cv_bridge/cv_bridge.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
// #include <math.h>
// #include <iostream>
// #include <opencv2/highgui.hpp> 
// #include <fast_dynamic/motion_compensate_node.h>
// #include <dvs_msgs/Event.h>
// #include <numeric>


// using namespace cv;
// using namespace std;
// using namespace Eigen;

// namespace motion_compensate
// {

// // template <typename T>
// // MatrixXd MotionCompensate::A_v2_autodiff(T x, T y) {
// //     T fx = static_cast<T>(cameraMatrix.at<double>(0, 0));
// //     T fy = static_cast<T>(cameraMatrix.at<double>(1, 1));
// //     T cxx = static_cast<T>(cameraMatrix.at<double>(0, 2));
// //     T cyy = static_cast<T>(cameraMatrix.at<double>(1, 2));

// //     Eigen::Matrix<T, 2, 3> matrix;
// //     matrix << fx, T(0), -(x - cxx),
// //               T(0), fy, -(y - cyy);
// //     return matrix;
// // }

// // template <typename T>
// // MatrixXd MotionCompensate::B_v2_autodiff(T x, T y) {
// //     T fx = static_cast<T>(cameraMatrix.at<double>(0, 0));
// //     T fy = static_cast<T>(cameraMatrix.at<double>(1, 1));
// //     T cxx = static_cast<T>(cameraMatrix.at<double>(0, 2));
// //     T cyy = static_cast<T>(cameraMatrix.at<double>(1, 2));

// //     Eigen::Matrix<T, 2, 3> matrix;
// //     matrix << (x - cxx) * (y - cyy) / fx, -(fx * fx + (x - cxx) * (x - cxx)) / fx, y - cyy,
// //               (fy * fy + (y - cyy) * (y - cyy)) / fy, -(x - cxx) * (y - cyy) / fx, -(x - cxx);
// //     return matrix;
// // }

// // MatrixXd MotionCompensate::convertToAutodiffMatrix(const cv::Mat& mat) {
// //     MatrixXd result(mat.rows, mat.cols);
// //     for (int i = 0; i < mat.rows; ++i) {
// //         for (int j = 0; j < mat.cols; ++j) {
// //             result(i, j) = autodiff::real(mat.at<double>(i, j));
// //         }
// //     }
// //     return result;
// // }

// // // 使用 real 类型计算图像
// // template <typename T>
// // cv::Mat MotionCompensate::computeImage_v2_autodiff(const std::vector<dvs_msgs::Event>& events_subset,
// //                                  const VectorXreal& linear,
// //                                  const VectorXreal& angular,
// //                                  const MatrixXd& depthMap) {
// //     cv::Mat temp_mc_time_map_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);
// //     cv::Mat temp_mc_event_count_ = cv::Mat::zeros(img_height, img_width, CV_64FC1);

// //     const T t_ref = T(events_subset.front().ts.toSec());
// //     for (const dvs_msgs::Event& ev : events_subset) {
// //         T xx = T(ev.x);
// //         T yy = T(ev.y);
// //         T dt = T(ev.ts.toSec()) - t_ref;

// //         T depth = depthMap(int(yy), int(xx));  // 使用 Eigen 访问深度值

// //         Eigen::Matrix<autodiff::real, 2, 3> A = A_v2_autodiff<autodiff::real>(xx, yy);
// //         Eigen::Matrix<autodiff::real, 2, 3> B = B_v2_autodiff<autodiff::real>(xx, yy);

// //         Eigen::Matrix<T, 2, 1> v;
// //         if (depth < T(0)) {
// //             v.setZero();
// //         } else {
// //             v = (T(1.0) / depth) * A * linear + B * angular;
// //         }

// //         T w_x = xx + v[0] * dt;
// //         T w_y = yy + v[1] * dt;

// //         if (T(0) <= w_x && w_x < T(img_width) && T(0) <= w_y && w_y < T(img_height)) {
// //             int row = static_cast<int>(w_y);  // Cast w_y to integer
// //             int col = static_cast<int>(w_x);  // Cast w_x to integer
// //             temp_mc_time_map_.at<double>(row, col) += static_cast<double>(dt);
// //             temp_mc_event_count_.at<double>(row, col) += 1.0;
// //         }
// //     }

// //     return temp_mc_time_map_;
// // }

// // template <typename T>
// // T MotionCompensate::computeLoss_v2_autodiff(const std::vector<dvs_msgs::Event>& events_subset,
// //                           const VectorXreal& linear,
// //                           const VectorXreal& angular,
// //                           const MatrixXd& depthMap) {
// //     cv::Mat image = computeImage_v2_autodiff(events_subset, linear, angular, depthMap);

// //     cv::Mat grad_x, grad_y;
// //     cv::Sobel(image, grad_x, CV_64FC1, 1, 0, 3);
// //     cv::Sobel(image, grad_y, CV_64FC1, 0, 1, 3);

// //     cv::Mat magnitude;
// //     cv::magnitude(grad_x, grad_y, magnitude);
// //     cv::Scalar mean_value = cv::mean(magnitude);

// //     return -T(mean_value[0]);
// // }


// // dual MotionCompensate::computeLoss_v2_autodiff(const MatrixXd& grad_x, const MatrixXd& grad_y) {

// //     double valid_pixel = 0.;

// //     MatrixXd magnitude(img_height, img_width);
// //     for (int i = 0; i < rows; i++) {
// //         for (int j = 0; j < cols; j++) {
// //             if(grad_x(i, j) < 1e-7 || grad_y(i, j) < 1e-7) continue;
// //             valid_pixel ++;
// //             magnitude(i, j) = sqrt(pow(grad_x(i, j), 2) + pow(grad_y(i, j), 2));
// //         }
// //     }

// //     double mean_mag = magnitude.sum() / valid_pixel; 

// //     return -dual(mean_mag);
// // }

// dual MotionCompensate::computeCost4SinglePixel_autodiff(const dual& grad_x, const dual& grad_y) {

//     return (grad_x*grad_x + grad_y*grad_y).sqrt();

// }



// // 计算梯度的函数
// void MotionCompensate::computeGradients_v2_autodiff(const std::vector<dvs_msgs::Event>& events_subset, 
//                                         const cv::Mat& depthMap, 
//                                         const cv::Vec3f& linear_vel_cam, 
//                                         const cv::Vec3f& angular_vel_cam) {

//     cv::Mat image = computeImageOfWarpedEvents_v2(events_subset);
//     cv::Mat grad_x, grad_y;
//     cv::Sobel(image, grad_x, CV_64FC1, 1, 0, 3);
//     cv::Sobel(image, grad_y, CV_64FC1, 0, 1, 3);
    
//     dual gradX, gradY;

//     auto lossFunc = [this](const dual& gradX, const dual& gradY) {
//         return this->computeCost4SinglePixel_autodiff(gradX, gradY);
//     };

//     dual dcdgx_sum, dcdgy_sum;
//     double valid_pixel = 0.;

//     for (int i = 0; i < grad_x.rows; ++i) {
//         for (int j = 0; j < grad_x.cols; ++j) {
//             gradX = dual(grad_x.at<double>(i, j));
//             gradY = dual(grad_y.at<double>(i, j));
//             dual dcdgx = derivative(lossFunc, wrt(gradX), at(gradX, gradY));
//             dual dcdgy = derivative(lossFunc, wrt(gradX), at(gradX, gradY));
//             if(dcdgx<dual(1e-7) || dcdgy<dual(1e-7)) continue;
//             dcdgx_sum += dcdgx;
//             dcdgy_sum += dcdgy;
//             valid_pixel ++;
//         }
//     }

//     double dgx = double(dcdgx_sum) / valid_pixel;
//     double dgy = double(dcdgy_sum) / valid_pixel;

//     std::cout << "Derivative wrt gradX: " << dgx << std::endl;
//     std::cout << "Derivative wrt gradY: " << dgy << std::endl;


//     // auto depthMap_autodiff = convertToAutodiffMatrix(depthMap);

//     // // 初始化线性和角速度变量为真实的 linear_vel_cam 和 angular_vel_cam
//     // autodiff::real linear_x = linear_vel_cam[0], linear_y = linear_vel_cam[1], linear_z = linear_vel_cam[2];
//     // autodiff::real angular_x = angular_vel_cam[0], angular_y = angular_vel_cam[1], angular_z = angular_vel_cam[2];

//     // // Eigen::Matrix<autodiff::real, 3, 1> linear(linear_x, linear_y, linear_z);
//     // // Eigen::Matrix<autodiff::real, 3, 1> angular(angular_x, angular_y, angular_z);
//     // VectorXreal linear(3); 
//     // VectorXreal angular(3);
//     // linear << linear_x, linear_y, linear_z;
//     // angular << angular_x, angular_y, angular_z;

//     // // 定义目标函数
//     // autodiff::real loss_func = [&](const VectorXreal& lin, 
//     //                      const VectorXreal& ang, 
//     //                      const Eigen::Matrix<autodiff::real, Eigen::Dynamic, Eigen::Dynamic>& depth) {
//     //     return computeLoss_v2_autodiff(events_subset, lin, ang, depth);
//     // };

//     // // 计算梯度
    
//     // // autodiff::real3d grad_linear, grad_angular;
//     // VectorXd grad_linear, grad_angular;
//     // grad_linear = derivative(loss_func, wrt(linear), at(linear, angular, depthMap_autodiff));
//     // grad_angular = derivative(loss_func, wrt(angular), at(linear, angular, depthMap_autodiff));
//     // MatrixXd grad_depth(depthMap.rows, depthMap.cols);
//     // grad_depth = derivative(loss_func, wrt(depthMap_autodiff), at(linear, angular, depthMap_autodiff));

//     // // 转换 grad_linear 和 grad_angular 为 cv::Vec3f
//     // cv::Vec3f grad_linear_cv(grad_linear(0), grad_linear(1), grad_linear(2));
//     // cv::Vec3f grad_angular_cv(grad_angular(0), grad_angular(1), grad_angular(2));

//     // // 转换 grad_depth 为 cv::Mat
//     // cv::Mat grad_depth_cv(depthMap.rows, depthMap.cols, CV_64FC1);
//     // for (int i = 0; i < depthMap.rows; ++i) {
//     //     for (int j = 0; j < depthMap.cols; ++j) {
//     //         grad_depth_cv.at<double>(i, j) = grad_depth(i, j);
//     //     }
//     // }
// }



// }