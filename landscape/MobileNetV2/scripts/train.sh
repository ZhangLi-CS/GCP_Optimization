#!/bin/sh
now=$(date +"%Y%m%d_%H%M%S")

jobname=MobileNetV2_GCP_LRnorm_LandScape_0.045_1.5

log_dir=logs/${jobname}
if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

python  -u main.py \
	-a mobilenet_v2 \
	--lr 0.045 \
	--wd 4e-5 \
	-p 1000 \
	--epochs 400 \
	--b 96 \
	--resume /home/ubuntu/Projects/CodeSort/landscape/MobileNetV2/checkpoint.pth.tar \
        --resume_temporary /home/ubuntu/Projects/CodeSort/landscape/MobileNetV2/temporary.pth.tar \
        /home/sdc1/dataset/ILSVRC2012/images  | tee ./logs/${jobname}/record-train-${now}.txt \


