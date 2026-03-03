export CXX=g++
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

conf_name='GeoMAR'

ROOT_PATH='./experiments/' # The path for saving model and logs

gpus='4,5,6,7,'
# gpus='0,'

#P: pretrain SL: soft learning
node_n=1

timestamp=$(date "+%Y-%m-%d_%H-%M-%S")

nohup python -u main_GeoMAR.py \
--root-path $ROOT_PATH \
--base "configs/${conf_name}.yaml" \
-t True \
--gpus $gpus \
--num-nodes $node_n \
> logs/${conf_name}_${timestamp}.log 2>&1 &

