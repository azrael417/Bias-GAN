FROM nvcr.io/nvidia/pytorch:20.10-py3

#Install gds system prereqs
RUN apt-get update && apt-get install --assume-yes apt-utils \
    && apt-get install --assume-yes libudev-dev \
    && apt-get install --assume-yes liburcu-dev \
    && apt-get install --assume-yes libmount-dev \
    && apt-get install --assume-yes libnuma-dev \
    && apt-get install --assume-yes libjsoncpp-dev \
    && apt-get install --assume-yes libssl-dev

#install GDS
COPY sys/gds_0.8.0_amd64.deb /tmp/gds_0.8.0_amd64.deb
RUN dpkg --force-all -i /tmp/gds_0.8.0_amd64.deb && \
    mv /usr/local/gds/lib/*.so /usr/lib/x86_64-linux-gnu/ && \
    mv /usr/local/gds/lib/*.h /usr/include/x86_64-linux-gnu/

##install GDS examples
#RUN mkdir -p /opt/gds_examples
#COPY ./sys/gds-alpha-${gds_version}/tools/samples /opt/gds_examples

#install mpi4py
RUN pip install mpi4py

#install other python stuff necessary
RUN pip install --upgrade wandb && \
    pip install netcdf4 ecmwf-api-client cdsapi
RUN conda install -y -c conda-forge ruamel.yaml

# raytune:
RUN pip install -U ray && pip install ray[tune] && pip install hyperopt

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
