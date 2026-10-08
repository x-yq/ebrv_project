# Event-Based Robot Vision

A ROS-based computer vision project for **motion compensation and moving-object detection using event cameras**.

The project processes asynchronous event streams, estimates camera motion, compensates event motion, and detects moving objects.

## 1. Project Background

Event cameras record pixel-level brightness changes asynchronously instead of conventional image frames. When the camera moves, background events become spatially misaligned.

This project investigates motion compensation for event-based robot vision using:

- 2-DoF, 4-DoF and 6-DoF motion models
- Time Surface and Event Count representations
- Contrast-based optimization
- Optional depth information
- Moving-object detection after motion compensation

### Pipeline

```text
Event Stream
     ↓
Motion Estimation
     ↓
Event Warping
     ↓
Motion-Compensated Representation
     ↓
Moving Object Detection
```

## 2. Environment Setup

Tested layout: **Ubuntu 20.04 + ROS Noetic** (any ROS1 distro with a C++17 compiler should work).

### 2.1 Requirements

| Dependency | Used for |
|---|---|
| ROS1 (`roscpp`, `rosbag`, `cv_bridge`, `image_transport`, `image_geometry`, `sensor_msgs`, `geometry_msgs`) | node, bag reading, image publishing |
| `catkin_simple` | build system (`catkin_simple()` / `cs_add_executable`) |
| `minkindr` (kindr) | transformations |
| `dvs_msgs`, `dvs_renderer` (from `rpg_dvs_ros`) | event messages, event rendering in the launch file |
| `rqt_gui` | dashboard opened by the launch file |
| OpenCV | image operations |
| Eigen3 | linear algebra |
| GSL | numerical minimization |
| `autodiff` (header-only) | automatic differentiation |
| Ceres Solver | included by the source files |
| gflags | included by the node |
| `cnpy` | saving `.npy` files |
| C++17 compiler | `-std=c++17` is set in `CMakeLists.txt` |

### 2.2 Install system packages

```bash
sudo apt update
sudo apt install -y \
  git wget build-essential cmake \
  ros-noetic-desktop-full ros-noetic-catkin python3-catkin-tools \
  ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-image-geometry \
  ros-noetic-rqt ros-noetic-rqt-gui \
  libopencv-dev libeigen3-dev libgsl-dev libceres-dev libgflags-dev zlib1g-dev
```

(If ROS is not installed yet, follow <http://wiki.ros.org/noetic/Installation/Ubuntu>.)

### 2.3 Create the workspace and fetch ROS dependencies

```bash
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# build helper used by CMakeLists.txt
git clone https://github.com/catkin/catkin_simple.git

# event camera messages + renderer
git clone https://github.com/uzh-rpg/rpg_dvs_ros.git

# transformations (follow its README for its own dependencies, e.g. eigen_catkin)
git clone https://github.com/ethz-asl/minkindr.git

# this project
git clone <YOUR_REPOSITORY_URL> fast_dynamic
```

> `rpg_dvs_ros` also contains camera driver packages that need `libcaer`. If you only replay bags, skip them at build time (see 2.6).

### 2.4 Install `autodiff` (header-only)

```bash
cd ~
git clone https://github.com/autodiff/autodiff.git
cd autodiff && mkdir build && cd build
cmake .. -DAUTODIFF_BUILD_TESTS=OFF -DAUTODIFF_BUILD_PYTHON=OFF \
         -DAUTODIFF_BUILD_EXAMPLES=OFF -DAUTODIFF_BUILD_DOCS=OFF
sudo make install
```

### 2.5 Build `cnpy` and fix the path in `CMakeLists.txt`

```bash
cd ~/catkin_ws/src
git clone https://github.com/rogersce/cnpy.git
cd cnpy && mkdir build && cd build
cmake .. && make
```

`CMakeLists.txt` currently hardcodes the author's machine path. Replace **both** occurrences of `/home/x-yq/catkin_ws/src/cnpy` with your own path:

```cmake
include_directories(/home/<YOUR_USER>/catkin_ws/src/cnpy)
link_directories(/home/<YOUR_USER>/catkin_ws/src/cnpy/build)
```

If the node fails at start-up with `libcnpy.so: cannot open shared object file`, add the build folder to the library path:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/catkin_ws/src/cnpy/build
```

### 2.6 Build

```bash
cd ~/catkin_ws
catkin_make -DCATKIN_BLACKLIST_PACKAGES="davis_ros_driver;dvs_ros_driver;dvxplorer_ros_driver"
source devel/setup.bash
```

Make sure the package's `package.xml` lists the dependencies above, because `catkin_simple(ALL_DEPS_REQUIRED)` reads them from it.

## 3. Docker (all-in-one setup)

The `Dockerfile` installs every dependency (ROS Noetic, OpenCV, GSL, Ceres, `autodiff`, `cnpy`, `catkin_simple`, `minkindr` and its dependencies, `rpg_dvs_ros`), patches the hardcoded paths, and builds the package. Put it in the repository root (next to `package.xml`).

```bash
# 1. build the image
docker build -t fast_dynamic:noetic .

# 2. put bags in ./bags (see "Download the ROS Bags"), allow X11, run
mkdir -p bags
xhost +local:docker
docker run --rm -it --net=host \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $PWD/bags:/catkin_ws/bags \
  fast_dynamic:noetic
```

Or with Compose: `docker compose up`.

Notes:

- The default command runs `roslaunch fast_dynamic motion_compensate.launch`. Inside the container, bags live in `/catkin_ws/bags` (the launch file's `/home/x-yq/catkin_ws/bags` is rewritten to this path), so the launch file expects e.g. `/catkin_ws/bags/slider_depth.bag`. Adjust the bag file name and `bag_ind` in the launch file, or override the command:
  ```bash
  docker run --rm -it --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PWD/bags:/catkin_ws/bags fast_dynamic:noetic \
    bash -c "roscore & sleep 3; rosrun fast_dynamic fast_dynamic events:=/dvs/events _bag_ind:=0 & rosbag play /catkin_ws/bags/slider_depth.bag"
  ```
- A display is needed for `rqt_gui` and OpenCV windows (Linux/X11 as shown; on macOS/Windows use an X server such as XQuartz/VcXsrv). For headless runs, run the node and `rosbag play` manually as above and skip `rqt_gui`.
- To use your own launch/source edits without rebuilding the whole image, rebuild with `docker build` (dependency layers are cached, so only the last layers rerun).
- The Docker build was written from the project's source files and dependency manifests; if a dependency version drifts, pin its git tag in the `Dockerfile`.

## 4. Download the ROS Bags

Create a folder for the data:

```bash
mkdir -p ~/catkin_ws/bags && cd ~/catkin_ws/bags
```

### Slider sequences (public, UZH RPG Event-Camera Dataset)

Dataset page: <http://rpg.ifi.uzh.ch/davis_data.html> (Mueggler et al., IJRR 2017). The bags are in the RPG DVS ROS driver format.

```bash
wget http://rpg.ifi.uzh.ch/datasets/davis/slider_depth.bag
wget http://rpg.ifi.uzh.ch/datasets/davis/slider_far.bag
```

If a direct link is unavailable, download the file from the dataset page's table instead.

### Other sequences

| Sequence | Source |
|---|---|
| `What_is_Background` | Moving-object dataset accompanying Mitrokhin et al., IROS 2018. Place it as `bags/IROS_Dataset/what_is_background/background.bag`. |
| `Test_vins` | Recorded sequence with a depth topic. Add your own download link here. |
| `simulation_3planes` | Simulated sequence. Add your own download link here. |
| `Drone_Sequence_with_a_Ball` | Add your own download link here. |

Check what a bag contains before running:

```bash
rosbag info ~/catkin_ws/bags/slider_depth.bag
```

The bag must contain `/dvs/events` (`dvs_msgs/EventArray`).

## 5. Running

The node reads its dataset-specific camera intrinsics from `bag_ind`. Set it to match the bag you play:

| `bag_ind` | Sequence |
|---|---|
| 0 | `slider_depth` |
| 1 | `slider_far` |
| 2 | `what_is_background` |
| 3 | `test_vins` |
| 4 | `simulation_3planes` |

Any other dataset needs a new intrinsics case in `get_intrinsic_params()` (`utils.cpp`). Otherwise default 240×180 intrinsics are used.

### Option A: launch file (recommended)

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch fast_dynamic motion_compensate.launch
```

The launch file starts `roscore` automatically and launches:

- `rosbag play` (currently with a hardcoded path and `-r 0.1`)
- the `fast_dynamic` node with all parameters and the `events` → `/dvs/events` remap
- `rqt_gui` with `motion_compensate.perspective`
- an event viewer node for the raw event view

**Before the first run, edit `launch/motion_compensate.launch`:**

0. The viewer node is declared as `pkg="dvs_displayer" type="dvs_displayer"`, which no longer exists in `rpg_dvs_ros` (it was renamed `dvs_renderer`). Change it to `pkg="dvs_renderer" type="dvs_renderer"` and replace the `event_image` remap with `<remap from="dvs_rendering" to="event_image" />` so the rqt perspective (which reads `/event_image`) keeps working.

1. Change the bag path in the `rosbag play` node (`args="-r 0.1 -d 1.0 -s 0. <PATH_TO_BAG>"`).
2. Set `bag_ind` to the matching value from the table above.
3. If `enable_depth` is `true`, set the `bag_args` parameter to the same bag path (see Depth below).

### Option B: manual start

```bash
# terminal 1
roscore

# terminal 2
source ~/catkin_ws/devel/setup.bash
rosrun fast_dynamic fast_dynamic events:=/dvs/events _bag_ind:=0

# terminal 3
rosbag play ~/catkin_ws/bags/slider_depth.bag
```

The `events:=/dvs/events` remap is required: the node subscribes to the relative topic name `events`, so without it no events arrive. Parameters not passed on the command line use the defaults from `motion_compensate.cpp`.

### Topics

Input:

```text
/dvs/events          (dvs_msgs/EventArray)
```

Optional depth input (only when `enable_depth` is `true`):

```text
/camera/depth/image_rect_raw   (when bag_ind = 3, test_vins)
/dvs/depthmap                  (all other bag_ind values)
```

Depth mode scans the bag file itself to count messages and waits until all events and depth images have been received before processing. The `bag_args` parameter must therefore contain the real bag path and must refer to the same bag that is played.

Output images (`image_transport`):

```text
event_count, avg_time_map, mc_event_count, mc_time_map, mask, depth_map
```

View them with `rqt_image_view` or the provided rqt perspective.

### Launch parameters

| Parameter | Meaning |
|---|---|
| `num_events_map_update` | Event batch size per map update |
| `optimize_image_type` | `0` TimeMap, `1` EventCount |
| `contrast_ind` | `0` norm, `1` variance, `2` gradient magnitude |
| `lr_x`, `lr_y`, `lr_div`, `lr_rot` | Learning rates (translation, divergence/scale, rotation) |
| `use_adam` | Use Adam instead of plain gradient steps |
| `tm_max_iter` | Maximum optimization iterations |
| `filter_threshold` | Threshold for the moving-object mask |
| `filter_small_compo` | Keep only the largest valid foreground component |
| `better_initial`, `random_initial` | Initialization options |
| `enable_depth` | Use depth information |
| `depth_x_bin_num`, `depth_y_bin_num` | Depth patch grid size |
| `enable_undistort` | Undistort events with the camera calibration |
| `plot_hist` | Show a histogram window (blocks until a key is pressed) |
| `bag_ind` | Selects the camera intrinsics (see table above) |
| `bag_args` | Bag path and options, used only in depth mode |

> **Note:** The motion model (4-DoF vs. 6-DoF pipeline) is not a launch parameter. It is selected in `checkAndProcess()` in `motion_compensate.cpp` by calling `processMessages()` or `processMessages_v2()`. Rebuild after changing it.

## 6. Example

The project includes experiments on several event-camera sequences:

- `Slider_Far`
- `Slider_Depth`
- `What_is_Background`
- `Drone_Sequence_with_a_Ball`
- `Test_vins`

A typical processing flow is:

```text
ROS Bag
  ↓
Event Stream
  ↓
Event Processing
  ↓
Motion Estimation
  ↓
Event Warping
  ↓
Motion-Compensated Event Representation
  ↓
Moving Object Detection
```

## 7. Results & Visualization

Example Time Surface results:

![2-DoF Time Surface](figures/2dof/test_vins_slice10_ts_mag.png)

![4-DoF Time Surface](figures/4dof/test_vins_slice10_ts_mag.png)

Example 6-DoF results:

![6-DoF Time Map](figures/6dof/tv_gt_time_map.png)

![6-DoF Event Count](figures/6dof/tv_gt_event_count.png)

Moving-object detection example:

![6-DoF Detection](figures/6dof/tv_gt_detect.png)

Additional visualizations are available under:

```text
figures/
├── 2dof/
├── 4dof/
├── 6dof/
└── motion_seg/
```

The results demonstrate how motion compensation can align background events and provide a better representation for subsequent moving-object detection.

## 8. Key Takeaways

- Event-based computer vision
- Camera motion estimation and compensation
- Multi-DoF motion models
- Depth-aware event processing
- Numerical optimization
- ROS-based perception
- Event-stream processing
- Moving-object detection

## 9. Citation

If you use this project or its related methods, please also cite:

```bibtex
@inproceedings{mitrokhin2018event,
  title={Event-based moving object detection and tracking},
  author={Mitrokhin, Anton and Fermuller, Cornelia and Parameshwara, Chaithanya and Aloimonos, Yiannis},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2018}
}

@article{mueggler2017event,
  title={The event-camera dataset and simulator: Event-based data for pose estimation, visual odometry, and SLAM},
  author={Mueggler, Elias and Rebecq, Henri and Gallego, Guillermo and Delbruck, Tobi and Scaramuzza, Davide},
  journal={The International Journal of Robotics Research},
  volume={36},
  number={2},
  pages={142--149},
  year={2017}
}
```

The project also builds upon the event-camera datasets and methods referenced in the accompanying report.
