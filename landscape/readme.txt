----------------------------------------------parameter modification----------------------------------------------

now=$(date +"%Y%m%d_%H%M%S")

### create the floder to save the log file
jobname=ResNet18_GAP_LRnorm_LandScape_0.1_75

log_dir=logs/${jobname}
if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

### train
python  -u main.py \
        -a resnet18 \				#models
        --lr 0.1 \				#learning rate
        --wd 1e-4 \				#weight decay
        -p 500 \				#print frequency
        --epochs 100 \				#total epoch
        --b 256 \				#batchsize
        --resume checkpoint.pth.tar \		#path to latest checkpoint
        --resume_temporary temporary.pth.tar \	#path to temporary parameter
        /home/sdc1/dataset/ILSVRC2012/images   | tee ./logs/${jobname}/record-train-${now}.txt \	#dataset path | output the log file


------------------------------------------------------------------------------------------------------------------



