#!/bin/sh
now=$(date +"%Y%m%d_%H%M%S")

jobname=ResNet18_GAP_LRnorm_LandScape_0.1_75

log_dir=logs/${jobname}
if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

python  -u main.py \
	-a resnet18 \
	--lr 0.1 \
	--wd 1e-4 \
	-p 500 \
	--epochs 100 \
	--b 256 \
	--resume /home/ubuntu/Projects/CodeSort/landscape/ResNet/checkpoint.pth.tar \
        --resume_temporary /home/ubuntu/Projects/CodeSort/landscape/ResNet/temporary.pth.tar \
	/home/sdc1/dataset/ILSVRC2012/images   | tee ./logs/${jobname}/record-train-${now}.txt \


