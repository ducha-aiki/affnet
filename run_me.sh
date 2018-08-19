#!/bin/bash
mkdir dataset
#mkdir dataset/HP_HessianPatches
#wget http://cmp.felk.cvut.cz/~mishkdmy/datasets/HPatches_HessianPatches/_test.pt
#mv _test.pt dataset/HP_HessianPatches/_test.pt
mkdir dataset/6Brown
python -utt gen_ds.py
mv dataset/*.pt dataset/6Brown
python -utt train_AffNet_test_on_graffity.py --gpu-id=0 --dataroot=dataset/6Brown --lr=0.005 --n-pairs=10000000 --batch-size=1024 --descriptor=HardNet --arch=AffNetFast --loss=HardNegC --epochs=20 --expname=AffNetFast_lr005_10M_20ep_aswap  2>&1 | tee  affnet.log &
