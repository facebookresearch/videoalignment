# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/sh

CLIMBING_PATH=/path/to/climbing

# Download Climbing videos
mkdir -p ${CLIMBING_PATH}
wget -r --no-parent -nH -nv -A .mp4 http://pascal.inrialpes.fr/data2/evve/videos_align/climbing/ -P ${CLIMBING_PATH}
mv ${CLIMBING_PATH}/data2/evve/videos_align/climbing/ ${CLIMBING_PATH}/videos && rm -rf ${CLIMBING_PATH}/data2

# Download Climbing groundtruth
wget -nv http://pascal.inrialpes.fr/data/evve/align_data/gt_climbing.align -P ${CLIMBING_PATH}

# Download pre-computed features
wget http://dl.fbaipublicfiles.com/videoalignment/climbing_rmac_resnet34_29.zip -P ${CLIMBING_PATH} && \
 unzip -q ${CLIMBING_PATH}/climbing_rmac_resnet34_29.zip -d ${CLIMBING_PATH}

# You can also edit manually videoalignment/my_data.py to put the right location of Climbing dataset
sed -i -e "s#\/datasets01\/climbing\/112017#${CLIMBING_PATH}#g" videoalignment/my_data.py
