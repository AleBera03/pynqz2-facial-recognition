FROM nvidia/cudagl:10.0-devel-ubuntu18.04 AS base

# reinstall gpg keys
RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get install -y wget && \
    apt-key del 7fa2af80 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

# install libcudnn7
RUN wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb && \
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb && \
    dpkg -i libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb && \
    dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb && \
    rm *.deb

# install some deps and tools
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
    git \
    sudo \
    wget \
    unzip \
    vim \
    pkg-config \
    lsb-release \
    graphviz \
    libncurses5 \
    build-essential \
    autoconf \
    libtool \
    libgoogle-glog-dev \
    libopenblas-dev \
    libgflags-dev \
    libopencv-dev \
    protobuf-compiler \
    libprotobuf-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libhdf5-dev \
    libsnappy-dev \
    libboost-all-dev \
    libssl-dev \
    python3-dev \
    python3-opencv \
    python3-numpy \
    libboost-python-dev \
    libboost-thread-dev \
    libcudnn7 \
    libcudnn7-dev \
    cmake \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME
ARG PASSWORD
ARG USER_HOME="/home/${USERNAME}"
ARG DNNDK_PATH="${USER_HOME}/xilinx_dnndk"

# create and set user
RUN useradd --create-home --home-dir "${USER_HOME}" --shell=/bin/bash --user-group "${USERNAME}" --groups sudo && \
    chown "${USERNAME}:${USERNAME}" "${USER_HOME}" && \
    echo "${USERNAME}:${PASSWORD}" | chpasswd
USER ${USERNAME}


FROM base AS runner

WORKDIR ${USER_HOME}

# copy dnndk to image
COPY --chown=${USERNAME}:${USERNAME} ./xilinx_dnndk_v3.1_190809/xilinx_dnndk_v3.1 ${DNNDK_PATH}

# install and init anaconda
RUN wget "https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh" -O anaconda.sh -v && \
    chmod +x anaconda.sh && \
    bash anaconda.sh -b -p ${USER_HOME}/anaconda3 && \
    rm anaconda.sh

ENV PATH="${USER_HOME}/anaconda3/bin:${PATH}"

# install tensorflow and other deps
RUN conda create -y -n decent pip python=3.6 && \
    cd ${DNNDK_PATH}/host_x86/decent-tf/ubuntu18.04 && \
    conda run -n decent pip install ./tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl && \
    conda run -n decent pip install numpy scikit-image opencv-python==4.6.0.66 scikit-learn==0.24.2 scipy progressbar2 keras==2.2.4 && \
    conda run -n decent pip install pillow && \
    conda run -n decent pip install 'h5py==2.10.0' --force-reinstall

# install caffe
ARG MODE
COPY --chown=${USERNAME}:${USERNAME} "./caffe-makefiles/${MODE}/Makefile.config" "Makefile.config"
RUN git clone https://github.com/BVLC/caffe.git caffe && \
    cp Makefile.config caffe/Makefile.config && \
    cd caffe && \
    make all -j$(nproc) && \
    make pycaffe -j$(nproc) && \
    echo "export PYTHONPATH=${USER_HOME}/caffe/python:\$PYTHONPATH" >> ${USER_HOME}/.bashrc && \
    echo "export LD_LIBRARY_PATH=${USER_HOME}/caffe/.build_release/lib:\$LD_LIBRARY_PATH" >> ${USER_HOME}/.bashrc && \
    echo "export PATH=${USER_HOME}/caffe/build/tools:\$PATH" >> ${USER_HOME}/.bashrc


USER root
RUN /bin/bash -c 'if [ "${MODE}" = "gpu" ]; then \
    mkdir -p /usr/local/cuda && \
    echo "CUDA Version 10.0.130" > /usr/local/cuda/version.txt && \
    mkdir -p /usr/local/cuda/include && \
    echo "#define CUDNN_MAJOR 7" > /usr/local/cuda/include/cudnn.h && \
    echo "#define CUDNN_MINOR 4" >> /usr/local/cuda/include/cudnn.h && \
    echo "#define CUDNN_PATCHLEVEL 1" >> /usr/local/cuda/include/cudnn.h; \
    fi'


# install dnndk
USER root
RUN cd ${DNNDK_PATH}/host_x86 && \
    chmod a+x ./install.sh && \
    conda run -n decent ./install.sh PynqZ2

USER ${USERNAME}
RUN conda init bash

WORKDIR ${USER_HOME}

