#!/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python "${BASEDIR}/scripts/train.py" --encoder_name resnet10 --encoder_depth 5 --decoder_channels 256,128,64,32,16 --results /results/medical/resnet10 --learning_rate 0.0005 --focal

#python"${BASEDIR}/export_model.py" --ckptpath /results/medical/resnet10/checkpoints/epoch_best.ckpt --configfile /data/msd/01_2d/config.pkl --outdir /results/segmentation/2dUnet/medical/resnet10 --prefix BraTr_01_2d  --onnx --torchckpt --hparamfile

python "${BASEDIR}/scripts/train.py" --encoder_name resnet18 --encoder_depth 5 --encoder_pretrained_source imagenet --decoder_channels 256,128,64,32,16 --results /results/medical/resnet18 --learning_rate 0.0005 --focal

#python "${BASEDIR}/export_model.py" --ckptpath /results/medical/resnet18/checkpoints/epoch\=97.ckpt --configfile /data/msd/01_2d/config.pkl --outdir /results/segmentation/2dUnet/medical/resnet18 --prefix BraTr_01_2d  --onnx --torchckpt --hparamfile

python "${BASEDIR}/scripts/train.py" --encoder_name resnet34 --encoder_depth 5 --encoder_pretrained_source imagenet --decoder_channels 256,128,64,32,16 --results /results/medical/resnet34 --learning_rate 0.0005 --focal

#python "${BASEDIR}/export_model.py" --ckptpath /results/medical/resnet34/checkpoints/epoch\=97.ckpt --configfile /data/msd/01_2d/config.pkl --outdir /results/segmentation/2dUnet/medical/resnet34 --prefix BraTr_01_2d  --onnx --torchckpt --hparamfile

python "${BASEDIR}/scripts/train.py" --encoder_name resnet50 --encoder_depth 5 --encoder_pretrained_source imagenet --decoder_channels 1024,512,256,128,32 --results /results/medical/resnet50 --learning_rate 0.0005 --focal

#python "${BASEDIR}/export_model.py" --ckptpath /results/medical/resnet50/checkpoints/epoch\=163.ckpt --configfile /data/msd/01_2d/config.pkl --outdir /results/segmentation/2dUnet/medical/resnet50 --prefix BraTr_01_2d  --onnx --torchckpt --hparamfile

python "${BASEDIR}/scripts/train.py" --encoder_name resnet101 --encoder_depth 5 --encoder_pretrained_source imagenet --decoder_channels 1024,512,256,128,32 --results /results/medical/resnet101 --learning_rate 0.0005 --batch_size 32 --focal

#...

python "${BASEDIR}/scripts/train.py" --encoder_name vgg16 --encoder_depth 5 --encoder_pretrained_source imagenet --decoder_channels 256,256,128,64,32 --data /data/msd/01_2d  --results /results/medical/vgg16 --learning_rate 0.0005 --batch_size 64 --focal

python "${BASEDIR}/scripts/train.py" --encoder_name vgg19 --encoder_depth 5 --encoder_pretrained_source imagenet --decoder_channels 256,256,128,64,32 --data /data/msd/01_2d  --results /results/medical/vgg19 --learning_rate 0.0005 --batch_size 64 --focal

python "${BASEDIR}/scripts/train.py" --encoder_name resnet10 --encoder_depth 5  --data /data/dagm/01_2d  --results /results/industrial/resnet10 --learning_rate 0.0001 --batch_size 16 --focal

python "${BASEDIR}/scripts/train.py" --encoder_name resnet18 --encoder_depth 5  --data /data/dagm/01_2d  --results /results/industrial/resnet18 --learning_rate 0.0001 --batch_size 16 --focal

python "${BASEDIR}/scripts/train.py" --encoder_name resnet34 --encoder_depth 5  --data /data/dagm/01_2d  --results /results/industrial/resnet34 --learning_rate 0.0001 --batch_size 16 --focal

python "${BASEDIR}/scripts/train.py" --encoder_name vgg16 --encoder_depth 5 --encoder_pretrained_source imagenet --decoder_channels 256,256,128,64,32 --data /data/dagm/01_2d  --results /results/industrial/vgg16 --learning_rate 0.0001 --batch_size 8 --focal

python "${BASEDIR}/scripts/train.py" --encoder_name vgg19 --encoder_depth 5 --encoder_pretrained_source imagenet --decoder_channels 256,256,128,64,32 --data /data/dagm/01_2d  --results /results/industrial/vgg19 --learning_rate 0.0001 --batch_size 8 --focal

