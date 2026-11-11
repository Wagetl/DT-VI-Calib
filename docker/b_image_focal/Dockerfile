FROM ubuntu:20.04

# 离线 bag 标定不需要 RealSense 驱动或额外的 LLVM 软件源。
RUN apt-get -o Acquire::Retries=3 update \
    && DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
    build-essential clang cmake git ca-certificates ccache \
    libtbb-dev libglew-dev libjpeg-dev libpng-dev liblz4-dev libbz2-dev \
    libboost-regex-dev libboost-filesystem-dev libboost-date-time-dev \
    libboost-program-options-dev libopencv-dev libgtest-dev \
    libsuitesparse-dev libfmt-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=clang CXX=clang++
WORKDIR /workspace
