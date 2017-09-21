#!/usr/bin/env sh

set -e
/disk1/chengw/caffe-ssd/build/tools/caffe train \
	 --solver=models/VGGNet/face/SSD_300x300/solver-2.prototxt \
	 --weights=models/VGGNet/face/SSD_300x300/_iter_787.caffemodel
	 #--weights=models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \
