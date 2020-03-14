#!/bin/sh
now=$(date +"%Y%m%d_%H%M%S")
jobname=ShuffleNet_V2_x1_0_LRnorm
log_dir=logs/${jobname}
if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

python  -u main_shufflenet.py \
	-a shufflenet_v2_X1_0 \
	--lr_mode LRnorm \
        -p 500 \
        -j 32 \
        --resume checkpoint.pth.tar  \
	/home/sdc1/dataset/ILSVRC2012/images | tee ./logs/${jobname}/record-train-${now}.txt \

