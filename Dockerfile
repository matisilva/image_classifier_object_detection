FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get -y install \
    python3-dev \
    python3-pip \
    cmake \
    gcc \
    git \
    build-essential \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev \
    libavcodec-dev \
    libavformat-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    wget \
    vim \
    unzip \
    autoconf \
    automake \
    libtool \
    curl \
    make \
    g++ \
    unzip \
    libusb-1.0-0-dev \
    libglfw3-dev \
    libglfw3 \
    software-properties-common \
    udev

RUN apt-get install -y libtbb2 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-dev

RUN pip3 install --upgrade pip==9.0.3 \
    && pip3 install cython==0.29.1 setuptools

RUN pip3 install ipython==7.2.0 \
    scipy==1.1.0 \
    imutils \
    sklearn \
    opencv-python==3.4.4.19 \
    keras \
    tensorflow-gpu==1.11.0

RUN cd /opt/ && git clone https://github.com/pjreddie/darknet \
	&& cd /opt/darknet \
	&& make GPU=1 CUDNN=0 OPENCV=0 OPENMP=0 \
	&& sed -i 's/print r/print(r)/g' /opt/darknet/python/darknet.py >> /opt/darknet/python/darknet.py \
    && export DARKNET_DIR=/opt/darknet

ENV PYTHONPATH $PYTHONPATH:/opt/darknet/python
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/darknet
