#!/usr/bin/bash
docker build -f Dockerfile . -t alectio/mnist_lenet:latest
docker push alectio/mnist_lenet:latest
