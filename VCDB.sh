# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# See http://www.yugangjiang.info/research/VCDB/index.html
# You MUST download files from Google Drive first.
# You should have the files core_dataset.zip, and core_dataset.z01 to core_dataset.z11

# Set this variable to the directory containing the files core_dataset.*
CORE_DATASET_DIR="."
# Set this variable to the target directory.
# You must also set it in the file videoalignment/my_data.py.
VCDB_DATASET_DIR="."

# Check files downloaded from Google Drive.
for i in {1..9}
do
    if [ ! -f "${CORE_DATASET_DIR}/core_dataset.z0${i}" ]; then
        echo "The file ${CORE_DATASET_DIR}/core_dataset.z0${i} doesn't exist. You must download it from Google Drive. See http://www.yugangjiang.info/research/VCDB/index.html"
        exit 1
    fi
done

for i in {10..11}
do
    if [ ! -f "${CORE_DATASET_DIR}/core_dataset.z${i}" ]; then
        echo "The file ${CORE_DATASET_DIR}/core_dataset.z${i} doesn't exist. You must download it from Google Drive. See http://www.yugangjiang.info/research/VCDB/index.html"
        exit 1
    fi
done

if [ ! -f "${CORE_DATASET_DIR}/core_dataset.zip" ]; then
    echo "The file ${CORE_DATASET_DIR}/core_dataset.zip doesn't exist. You must download it from Google Drive. See http://www.yugangjiang.info/research/VCDB/index.html"
    exit 1
fi

# Unzip
mkdir -p ${VCDB_DATASET_DIR}
zip -s 0 ${CORE_DATASET_DIR}/core_dataset.zip  --out ${CORE_DATASET_DIR}/core.zip && \
 unzip ${CORE_DATASET_DIR}/core.zip -d ${VCDB_DATASET_DIR} && \
 rm ${CORE_DATASET_DIR}/core.zip

# Move files to have the right directory structure
mv ${VCDB_DATASET_DIR}/core_dataset ${VCDB_DATASET_DIR}/videos

# Fix some annotation files
sed -i -e "/38f11a4162d8e94227ac644f117f942735b9a504.mp4,bf582249cfc79d691195a8681961029cc5149a76.flv,00:02:35,00:03:15,00:01:04,00:03:18/d" "${VCDB_DATASET_DIR}/annotation/mr_and_mrs_smith_tango.txt"
sed -i -e "/14c262ea09b4ca66feb7e88cf57e0faaeacc301f.mp4,f150e062960b477adcac3f12ef4543337f5a91a4.flv,00:00:19,00:00:33,00:00:00,00:00:14/d" "${VCDB_DATASET_DIR}/annotation/obama_kicks_door.txt"
sed -i -e "/067fb2aa9623905a42a2b0b286de1386e45c5bf8.flv,8f19329946455ae5c2c7b788ea6f6513bf5e1c9a.flv,00:01:06,00:02:18,00:00:05,00:01:17/d" "${VCDB_DATASET_DIR}/annotation/run_forrest_fun.txt"


# Download pre-computed features
wget http://dl.fbaipublicfiles.com/videoalignment/VCDB_rmac_resnet34_29.zip -P ${VCDB_DATASET_DIR} && \
 unzip -q ${VCDB_DATASET_DIR}/VCDB_rmac_resnet34_29.zip -d ${VCDB_DATASET_DIR}
