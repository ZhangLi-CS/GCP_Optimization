----------------------------------------------parameter modification----------------------------------------------

now=$(date +"%Y%m%d_%H%M%S")

### create the floder to save the log file
jobname=ResNet50_GCP_LRnorm
log_dir=logs/${jobname}
if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

### train or val 
python  -u main.py \
        -a resnet50_mpncov \		#models
        --lr_mode LRnorm \		#Learning rate mode setting
        -p 500 \			#print frequency
        -j 32 \				#worker
        --resume checkpoint.pth.tar  \	#path to latest checkpoint
        /home/sdc1/dataset/ILSVRC2012/images | tee ./logs/${jobname}/record-train-${now}.txt \	#dataset path | output the log file


------------------------------------------------------------------------------------------------------------------



