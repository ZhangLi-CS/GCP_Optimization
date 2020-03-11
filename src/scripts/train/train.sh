#!/bin/sh
now=$(date +"%Y%m%d_%H%M%S")
jobname=ResNet50_GCP_LRnorm
log_dir=logs/${jobname}
if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

python  -u main.py \
	-a resnet50_mpncov \
	--lr_mode LRnorm \
	--wd 1e-4 \
        -p 500 \
        -j 32 \
        --resume checkpoint.pth.tar  \
	/home/sdc1/dataset/ILSVRC2012/images | tee ./logs/${jobname}/record-train-${now}.txt \

