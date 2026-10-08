# syntax=docker/dockerfile:1
# Build context = repository root (the folder containing package.xml, src/, include/, launch/).
# FROM osrf/ros:noetic-desktop-full
FROM arm64v8/ros:noetic

ENV DEBIAN_FRONTEND=noninteractive \
    WS=/catkin_ws \
    CNPY_DIR=/opt/cnpy

# ---------- system packages ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ca-certificates build-essential cmake \
      python3-catkin-tools python3-rosdep \
      libopencv-dev libeigen3-dev libgsl-dev libceres-dev \
      libgflags-dev libgoogle-glog-dev zlib1g-dev \
      ros-noetic-cv-bridge \
      ros-noetic-image-transport \
      ros-noetic-image-geometry \
      ros-noetic-camera-info-manager \
      ros-noetic-rqt \
      ros-noetic-rqt-gui \
      ros-noetic-rqt-image-view \
      ros-noetic-rqt-plot \
      ros-noetic-tf-conversions \
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
ENV LD_LIBRARY_PATH=${CNPY_DIR}/build
ENV LIBRARY_PATH=${CNPY_DIR}/build

# ---------- catkin workspace + ROS source dependencies ----------
WORKDIR ${WS}/src
RUN git clone --depth 1 https://github.com/catkin/catkin_simple.git \
 && git clone --depth 1 https://github.com/uzh-rpg/rpg_dvs_ros.git \
 && git clone --depth 1 https://github.com/ethz-asl/minkindr.git \
 && git clone --depth 1 https://github.com/ethz-asl/eigen_catkin.git \
 && git clone --depth 1 https://github.com/ethz-asl/eigen_checks.git \
 && git clone --depth 1 https://github.com/ethz-asl/glog_catkin.git \
 && git clone --depth 1 https://github.com/ethz-asl/gflags_catkin.git \
 && git clone --depth 1 https://github.com/ethz-asl/catkin_boost_python_buildtool.git \
 && git clone --depth 1 https://github.com/ethz-asl/numpy_eigen.git


# --------------------------------------------------
# Bags directory
# --------------------------------------------------
RUN mkdir -p ${WS}/bags

VOLUME ["${WS}/bags"]

# --------------------------------------------------
# ROS entrypoint
# --------------------------------------------------
RUN printf '#!/bin/bash\n\
source /opt/ros/noetic/setup.bash\n\
if [ -f /catkin_ws/devel/setup.bash ]; then\n\
    source /catkin_ws/devel/setup.bash\n\
fi\n\
exec "$@"\n' > /ros_entrypoint.sh \
    && chmod +x /ros_entrypoint.sh

WORKDIR ${WS}

ENTRYPOINT ["/ros_entrypoint.sh"]

CMD ["bash"]
