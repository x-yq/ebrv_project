#include <ros/ros.h>
#include <fast_dynamic/motion_compensate_node.h>

#include <gflags/gflags.h>
// #include <glog/logging.h>

int main(int argc, char* argv[])
{
  // google::InitGoogleLogging(argv[0]);
  // google::ParseCommandLineFlags(&argc, &argv, true);
  // google::InstallFailureSignalHandler();
  // FLAGS_alsologtostderr = true;
  // FLAGS_colorlogtostderr = true;

  ros::init(argc, argv, "motion_compensate");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  motion_compensate::MotionCompensate mc(nh, nh_private);

  ros::spin();

  return 0;
}
