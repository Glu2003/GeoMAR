exp_name='GeoMAR'

root_path='experiments'
out_root_path='results'

tag='test_e28_0106_celeba144_error'
align_test_path="/data/celeba_512_validation_lq/"
outdir=$out_root_path'/'$exp_name'_'$tag

if [ ! -d $outdir ];then
    mkdir $outdir
fi

python -u scripts/test_cpa.py \
--outdir $outdir \
-r /data/ganlu33/GeoMAR/experiments/logs/2026-01-06T13-07-09_GeoMAR_newtext+contrativeloss/checkpoints/epoch=028-val_fid=14.905-val_niqe=3.848-val_BCE_loss=33.999-val_Rec_loss=0.104.ckpt \
-c 'configs/GeoMAR.yaml' \
--test_path $align_test_path \
--aligned \
--save_features \

