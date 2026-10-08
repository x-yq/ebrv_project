# syntax=docker/dockerfile:1
# Build context = repository root (the folder containing package.xml, src/, include/, launch/).
FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive \
    WS=/catkin_ws \
    CNPY_DIR=/opt/cnpy

# ---------- system packages ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ca-certificates build-essential cmake \
      python3-catkin-tools python3-rosdep \
      libopencv-dev libeigen3-dev libgsl-dev libceres-dev \
      libgflags-dev libgoogle-glog-dev zlib1g-dev \
      ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-image-geometry \
      ros-noetic-rqt ros-noetic-rqt-gui ros-noetic-rqt-image-view ros-noetic-rqt-plot \
    && rm -rf /var/lib/apt/lists/*

# ---------- autodiff (header-only, needs C++17) ----------
RUN git clone --depth 1 https://github.com/autodiff/autodiff.git /tmp/autodiff \
 && cmake -S /tmp/autodiff -B /tmp/autodiff/build \
      -DAUTODIFF_BUILD_TESTS=OFF -DAUTODIFF_BUILD_PYTHON=OFF \
      -DAUTODIFF_BUILD_EXAMPLES=OFF -DAUTODIFF_BUILD_DOCS=OFF \
 && cmake --build /tmp/autodiff/build --target install -j"$(nproc)" \
 && rm -rf /tmp/autodiff

# ---------- cnpy ----------
RUN git clone --depth 1 https://github.com/rogersce/cnpy.git ${CNPY_DIR} \
 && cmake -S ${CNPY_DIR} -B ${CNPY_DIR}/build \
 && cmake --build ${CNPY_DIR}/build -j"$(nproc)"
ENV LD_LIBRARY_PATH=${CNPY_DIR}/build:${LD_LIBRARY_PATH}

# ---------- catkin workspace + ROS source dependencies ----------
WORKDIR ${WS}/src
RUN git clone --depth 1 https://github.com/catkin/catkin_simple.git \
 && git clone --depth 1 https://github.com/uzh-rpg/rpg_dvs_ros.git \
 && git clone --depth 1 https://github.com/ethz-asl/minkindr.git \
 && git clone --depth 1 https://github.com/ethz-asl/eigen_catkin.git \
 && git clone --depth 1 https://github.com/ethz-asl/eigen_checks.git \
 && git clone --depth 1 https://github.com/ethz-asl/glog_catkin.git \
 && git clone --depth 1 https://github.com/ethz-asl/gflags_catkin.git

# ---------- this project ----------
COPY . ${WS}/src/fast_dynamic

# Docker-specific path fixes (the repo itself keeps the author's local paths):
#  * cnpy include/link dirs in CMakeLists.txt
#  * bag folder in the launch file -> /catkin_ws/bags (mount your bags there)
#  * dvs_displayer was renamed dvs_renderer in rpg_dvs_ros; publish it as /event_image
RUN sed -i "s#/home/x-yq/catkin_ws/src/cnpy#${CNPY_DIR}#g" ${WS}/src/fast_dynamic/CMakeLists.txt \
 && sed -i "s#/home/x-yq/catkin_ws/bags#${WS}/bags#g" ${WS}/src/fast_dynamic/launch/motion_compensate.launch \
 && sed -i '/pkg="dvs_displayer"/,/<\/node>/{s#pkg="dvs_displayer" type="dvs_displayer"#pkg="dvs_renderer" type="dvs_renderer"#;s#<remap from="event_image" to="event_image" />#<remap from="dvs_rendering" to="event_image" />#}' \
      ${WS}/src/fast_dynamic/launch/motion_compensate.launch

# ---------- build (skip camera driver packages; they need libcaer) ----------
WORKDIR ${WS}
RUN . /opt/ros/noetic/setup.sh \
 && catkin_make -DCMAKE_BUILD_TYPE=Release \
      -DCATKIN_BLACKLIST_PACKAGES="davis_ros_driver;dvs_ros_driver;dvxplorer_ros_driver;dvs_calibration;dvs_calibration_gui;dvs_file_writer"

# ---------- runtime ----------
RUN mkdir -p ${WS}/bags
VOLUME ["/catkin_ws/bags"]

RUN printf '#!/bin/bash\nsource /opt/ros/noetic/setup.bash\nsource /catkin_ws/devel/setup.bash\nexec "$@"\n' \
      > /ros_entrypoint.sh && chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["roslaunch", "fast_dynamic", "motion_compensate.launch"]
