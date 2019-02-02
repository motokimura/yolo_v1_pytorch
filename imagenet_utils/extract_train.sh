#!/bin/bash

TRAIN_DIR=data/imagenet/train

mkdir -p ${TRAIN_DIR}
mv ILSVRC2012_img_train.tar ${TRAIN_DIR}
cd ${TRAIN_DIR}

tar -xvf ILSVRC2012_img_train.tar
rm -f ILSVRC2012_img_train.tar

find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
