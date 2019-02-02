#!/bin/bash

# Set image name
IMAGE="yolo-v1:latest"

# Get project root dicrectory
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# Buld docker image from project root directory
cd ${PROJ_DIR} && \
nvidia-docker build -t ${IMAGE} --build-arg UID=`id -u` -f ${THIS_DIR}/Dockerfile .
