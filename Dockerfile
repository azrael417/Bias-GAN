#FROM gitlab-master.nvidia.com:5005/dl/dgx/pytorch:19.10-py3-devel
FROM nvcr.io/nvidia/pytorch:20.01-py3

ARG gds_version=20200220

## Install OpenMPI w/ IBVerbs support.  Builds *without* CUDA-aware MPI features on purpose as
## DLFW don't use those features and it adds overhead for checking every pointer's location in MPI
#ARG OPENMPI_VERSION=4.0.3
#ENV OPENMPI_VERSION=${OPENMPI_VERSION}
#RUN wget -q -O - https://www.open-mpi.org/software/ompi/v$(echo "${OPENMPI_VERSION}" | cut -d . -f 1-2)/downloads/openmpi-${OPENMPI_VERSION}.tar.gz | tar -xzf - \
#    && cd openmpi-${OPENMPI_VERSION} \
#    && ln -sf /usr/include/slurm-wlm /usr/include/slurm \
#    && ./configure --enable-orterun-prefix-by-default --with-verbs \
#                   --with-pmi --with-pmix=internal \
#		   --with-cuda=/usr/local/cuda \
#		   --prefix=/usr/local/mpi --disable-getpwuid \
#    && make -j"$(nproc)" install \
#    && cd .. && rm -rf openmpi-${OPENMPI_VERSION} \
#    && echo "/usr/local/mpi/lib" >> /etc/ld.so.conf.d/openmpi.conf \
#    && rm -f /usr/lib/libibverbs.so /usr/lib/libibverbs.a \
#    && ldconfig
#ENV PATH /usr/local/mpi/bin:$PATH

#Install gds system prereqs
RUN apt-get update && apt-get install --assume-yes apt-utils \
    && apt-get install --assume-yes libudev-dev \
    && apt-get install --assume-yes liburcu-dev \
    && apt-get install --assume-yes libmount-dev \
    && apt-get install --assume-yes libnuma-dev \
    && apt-get install --assume-yes libjsoncpp-dev \
    && apt-get install --assume-yes libssl-dev

#install GDS
COPY ./sys/gds-alpha-${gds_version}/lib/libcufile.so /usr/lib/x86_64-linux-gnu/
COPY ./sys/gds-alpha-${gds_version}/lib/cufile.h /usr/include/
ENV CUFILE_EXPERIMENTAL_FS 1

#install GDS examples
RUN mkdir -p /opt/gds_examples
COPY ./sys/gds-alpha-${gds_version}/tools/samples /opt/gds_examples

#install and update NCCL
COPY ./sys/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb /opt/
RUN cd /opt && dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb \
    && rm nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb \
    && apt-get update && apt-get install --reinstall --assume-yes libnccl2 libnccl-dev

#Install conda prereqs
RUN conda config --add channels conda-forge \
    && conda install matplotlib basemap basemap-data-hires pillow
ENV PROJ_LIB /opt/conda/share/proj

#install horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod

#install mpi4py
RUN pip install mpi4py

#install other python stuff necessary
RUN pip install netcdf4 ecmwf-api-client cdsapi wandb ruamel.yaml

#install numpy with pytorch extensions
RUN mkdir -p /opt
#numpy
COPY ./src/numpy_reader /opt/numpy_reader
RUN cd /opt/numpy_reader && python setup.py install

#local condv 2d
COPY ./src/conv2d_local /opt/conv2d_local
RUN cd /opt/conv2d_local && python setup.py install

##install Torch2TRT
#RUN cd /opt; git clone https://github.com/NVIDIA-AI-IOT/torch2trt \
#    && cd torch2trt && python setup.py install

#copy additional stuff
COPY ./src/deepCam /opt/deepCam
COPY ./src/utils /opt/utils

#init empty git repo so that wandb works
#RUN cd /opt/deepCam && git init

#copy cert:
RUN mkdir -p /certs
COPY no-git/ecmwf_cert.key /certs/.ecmwfapirc
COPY no-git/copernicus_cert.key /certs/.cdsapirc
COPY no-git/wandb_cert_era.key /certs/.wandbirc
COPY no-git/wandb_cert_gpsro.key /certs/.wandbirc_gpsro

#create additional folders for mapping data in
RUN mkdir -p /data && mkdir -p /data && mkdir -p /data/output
