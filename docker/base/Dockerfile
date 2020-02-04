FROM nvidia/cuda:10.2-base

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
      && ln -s /usr/bin/python3 python \
        && pip3 install --upgrade pip

RUN apt-get install -y vim nmap


COPY .vimrc /root
ADD .vim /root/.vim

# install pytorch 
RUN pip install torch
RUN pip install torchvision


# install sklearn
RUN pip install scikit-learn 


RUN mkdir /data
RUN mkdir /src
RUN mkdir /log

WORKDIR /src


# copy your source files into /src
# COPY resnet.py /src
# COPY main.py /src
# COPY dataset.py /src
# COPY imagedata.py /src


