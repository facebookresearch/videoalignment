# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/sh

EVAL_DIR=/path/to/eval
mkdir -p ${EVAL_DIR}

for model in CTE TMK_Poullot TMK
do
    for dataset in Climbing Madonna EVVE VCDB
    do
        echo "Evaluate model ${model} on dataset ${dataset}"
        python main.py --model=${model} --dataset_test=${dataset} --output_dir=${EVAL_DIR}
    done
done

echo "Evaluation done"