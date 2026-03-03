#!/bin/bash

root='results'
out_root='results/metrics'

if [ ! -d $root ];then
    mkdir -p $root
fi

if [ ! -d $out_root ];then
    mkdir -p $out_root
fi


dataset_name_array=('test_e14_linerk+1_0118_lfw_kqv')

dataset_location_array=('LFW-Test/cropped_faces')
# dataset_location_array=('Child')
# dataset_location_array=('WebPhoto_cropped_faces/WebPhoto_cropped_faces')
# dataset_location_array=('Wider-Test')
# dataset_location_array=('celeba_512_test')


checkpoint='/data/ganlu33/GeoMAR/experiments/logs/2026-01-18T20-38-10_GeoMAR/checkpoints/epoch=014-val_fid=58.088-val_niqe=3.904-val_BCE_loss=4.894-val_Rec_loss=0.232.ckpt'
config='./configs/GeoMAR.yaml'
output_name='GeoMAR'
GPU='7'

# echo ${0}
echo ${checkpoint}
# echo ${config}
echo ${output_name}
echo $GPU
echo $dataset_name_array


outdir=$root'/'$output_name'_'${dataset_name_array[0]}
align_test_path='/data/'${dataset_location_array[0]}



CUDA_VISIBLE_DEVICES=$GPU python -u scripts/test.py \
--outdir $outdir \
-r $checkpoint \
-c $config \
--test_path $align_test_path \
--aligned



outdir=$output_name'_'${dataset_name_array[0]}

test_image=$outdir'/restored_faces'

out_name=$outdir


need_post=0


# CelebAHQ_GT='/data/celeba_512_validation'

# FID
CUDA_VISIBLE_DEVICES=$GPU python -u scripts/metrics/cal_fid.py \
$root'/'$test_image \
--fid_stats 'experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth' \
--save_name $out_root'/'$out_name'_fid.txt' \

CUDA_VISIBLE_DEVICES=$GPU python scripts/metrics/cal_niqe.py \
$root'/'$test_image \
--save_name $out_root'/'$out_name'_niqe.txt' \


# CUDA_VISIBLE_DEVICES=$GPU python scripts/metrics/cal_clipiqa.py \
# $root'/'$test_image \
# --save_name $out_root'/'$out_name'_clipiqa.txt' \

# CUDA_VISIBLE_DEVICES=$GPU python scripts/metrics/cal_musiq.py \
# $root'/'$test_image \
# --save_name $out_root'/'$out_name'_musiq.txt' \


CUDA_VISIBLE_DEVICES=$GPU python scripts/metrics/cal_maniqa.py \
$root'/'$test_image \
--save_name $out_root'/'$out_name'_maniqa.txt' \

if [ -d $CelebAHQ_GT ]
then
# PSRN SSIM LPIPS
CUDA_VISIBLE_DEVICES=$GPU python -u scripts/metrics/cal_psnr_ssim.py \
"/data/gl/DiffusionReward/output/celeba255" \
--gt_folder $CelebAHQ_GT \
--save_name $out_root'/'$out_name'_psnr_ssim_lpips.txt' \
--need_post $need_post \

else
    echo 'The path of GT does not exist'
fi