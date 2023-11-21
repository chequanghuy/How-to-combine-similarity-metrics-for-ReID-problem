# Experiment all tricks without center loss with re-ranking : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
# with re-ranking
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('veri')" DATASETS.ROOT_DIR "('/home/ceec/huycq/track2_aicity_2021/lib/data/datasets')" TEST.RE_RANKING "('no')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('/home/ceec/chuong/reid/veri_weights/final_2_80.pth')"