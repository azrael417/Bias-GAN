#FROM gitlab-master.nvidia.com:5005/dl/dgx/pytorch:19.10-py3-devel
FROM nvcr.io/nvidia/pytorch:20.01-py3

#Install gds system prereqs
RUN apt-get update && apt-get install --assume-yes apt-utils \
    && apt-get install --assume-yes libudev-dev \
    && apt-get install --assume-yes liburcu-dev \
    && apt-get install --assume-yes libmount-dev \
    && apt-get install --assume-yes libnuma-dev \
    && apt-get install --assume-yes libjsoncpp-dev \
    && apt-get install --assume-yes libssl-dev

#install and update NCCL
COPY ./sys/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb /opt/
RUN cd /opt && dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb \
    && rm nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb \
    && apt-get update && apt-get install --upgrade --assume-yes libnccl2 libnccl-dev

##Install DALI prereqs
#RUN apt-get install --assume-yes libavformat-dev \
#    && apt-get install --assume-yes libavfilter-dev \
#    && apt-get install --assume-yes libtiff-dev

#Install conda prereqs
#ENV PROJ_LIB /opt/conda/share/proj
RUN conda config --add channels conda-forge \
    && conda install matplotlib basemap basemap-data-hires
ENV PROJ_LIB /opt/conda/share/proj

#install horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod

##install Torch2TRT
#RUN cd /opt; git clone https://github.com/NVIDIA-AI-IOT/torch2trt \
#    && cd torch2trt && python setup.py install

#create folders and copy stuff
RUN mkdir -p /opt
COPY ./src /opt

#create additional folders for mapping data in
RUN mkdir -p /data && mkdir -p /data && mkdir -p /data/output

