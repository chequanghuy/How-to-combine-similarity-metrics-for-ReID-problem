# Experiment all tricks with center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('veri')" DATASETS.ROOT_DIR "('/home/ceec/huycq/track2_aicity_2021/lib/data/datasets')" MODEL.PRETRAIN_CHOICE "('imagenet')" MODEL.PRETRAIN_PATH "('/home/ceec/chuong/reid-strong-baseline/veri_weights/final_2_80.pth')" OUTPUT_DIR "('veri_weights')"