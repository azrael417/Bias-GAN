#FROM gitlab-master.nvidia.com:5005/dl/dgx/pytorch:19.10-py3-devel
FROM nvcr.io/nvidia/pytorch:20.09-py3

ARG gds_version=20200220

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

#install mpi4py
RUN pip install mpi4py

#install other python stuff necessary
RUN pip install --upgrade wandb && \
    pip install netcdf4 ecmwf-api-client cdsapi
RUN conda install -y -c conda-forge ruamel.yaml

# raytune:
RUN pip install -U ray && pip install ray[tune]

#install numpy with pytorch extensions
RUN mkdir -p /opt
#numpy
COPY ./src/numpy_reader /opt/numpy_reader
RUN cd /opt/numpy_reader && python setup.py install

#copy additional stuff
COPY ./src/deepCam /opt/deepCam
COPY ./src/utils /opt/utils

#copy cert:
RUN mkdir -p /certs
COPY no-git/ecmwf_cert.key /certs/.ecmwfapirc
COPY no-git/copernicus_cert.key /certs/.cdsapirc
COPY no-git/wandb_cert_era.key /certs/.wandbirc
COPY no-git/wandb_cert_gpsro.key /certs/.wandbirc_gpsro

#create additional folders for mapping data in
RUN mkdir -p /data && mkdir -p /data && mkdir -p /data/output
