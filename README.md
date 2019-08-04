# Video Alignment

This repository contains the code of different video alignment algorithms, to reproduce
some of the results of the paper [LAMV: Learning to Align and Match Videos with Kernelized Temporal Layers](#lamv).

Facebook also open-sourced a [C++ version of TMK](https://github.com/facebook/ThreatExchange/tree/master/hashing/tmk), see [this post](https://newsroom.fb.com/news/2019/08/open-source-photo-video-matching/) for more information.


## Installation

Use the [environment.yml](environment.yml) file to create a conda environment with the required packages:
```bash
git clone https://github.com/facebookresearch/videoalignment.git
cd videoalignment
conda env create --file=environment.yml
source activate videoalignment
```

Alternatively, you can install each package individually, with `conda` or `pip`:
```bash
# Create conda env
conda create -n videoalignment python=3.6 -y
source activate videoalignment

# Install required packages
conda install pytorch torchvision -c pytorch -y
conda install scikit-learn -y
conda install -c conda-forge ffmpeg -y
pip install imageio imageio-ffmpeg progressbar
```

## How to reproduce the results

The fastest way to reproduce the results is to use the precomputed features we provide.
Once the dataset and its pre-computed features are correctly setup, you can use [main.py](main.py)
to run the evaluation.
```bash
# valid dataset names are: `Climbing`, `Madonna`, `EVVE` and `VCDB`
# valid models names are: `TMK`, `TMK_Poullot` and `CTE`
python main.py --dataset_test Climbing --model TMK --output_dir=${EVAL_DIR}
```
The script [run_all.sh](run_all.sh) will run all models on all datasets.


### Data setup

The datasets we have used for research are:
* [Climbing, for temporal video alignment](http://pascal.inrialpes.fr/data/evve/index_align.html). 
You can use the script [climbing.sh](climbing.sh) to download and setup correctly this dataset.
* [VCDB, for video copy detection](http://www.yugangjiang.info/research/VCDB/index.html). 
You can use the script [VCDB.sh](VCDB.sh) to setup correctly this dataset (download has 
to be done before, using the Google Drive link on the dataset page)
* [Madonna, for temporal video alignment](http://pascal.inrialpes.fr/data/evve/index_align.html)
* [EVVE, for event retrieval](http://pascal.inrialpes.fr/data/evve/)

You can specify the path to the data by modifying the file [my_data.py](./videoalignment/my_data.py).   

If you want to use the code on your own dataset, you should write a `VideoDataset` class
under [datasets.py](./videoalignment/datasets.py). The `EVVE` class is a good example to start with.

For each dataset, we expect the following directory tree:
```
root_dir (eg. EVVE) 
│
└───annotations (for EVVE) or annotation (VCDB)
│   │   some gt file or sub-folder with gt
│   │   ...
│
└───gt_climbing.align or gt_madonna.align (for Climbing and Madonna)
│
└───rmac_resnet34_29 (provided pre-computed features) 
│   │   video frame features.pkl
│   │   ...
│
└───rmac_resnet34_30 (optional, if you extracted features with another model)
│   │   video frame features.pkl
│   │   ...
│
└───videos
│   │   file011.mp4 or sub-folder with videos
│   │   ...   
```

### Expected results

In the following tables:
- `TMK_Poullot` is the model refereed as `TMK [26]` in the paper
- `TMK (norm=feat_freq)` is not in the paper 
- Frame descriptor is RMAC for all models
- The `Gist` links show the expected output of the evaluation script.

#### Madonna

See Table 2 in [LAMV paper](#lamv).
Metric is percentage of queries localized better than a threshold (0.1s, 1s, 10s)

|                     | @0.1s | @1s   | @10s  | Gist  |
|:--------------------|:-----:|:-----:|:-----:|:-----:|
| TMK_Poullot         | 9.13  | 36.52 | 64.78 | [link](https://gist.github.com/LowikC/51b27531bbc1565b1a5ecee189f63861)|
| TMK (norm=feat_freq)|cis 33.91 | 70.00 | 76.96 | [link](https://gist.github.com/LowikC/8bcd68b3e9db4ec5ef97938c9876684c)|

#### Climbing

See Table 2 in [LAMV paper](#lamv).
Metric is percentage of queries localized better than a threshold (0.1s, 1s, 10s)

|                     | @0.1s | @1s   | @10s  | Gist  |
|:--------------------|:-----:|:-----:|:-----:|:-----:|
| TMK_Poullot         | 00.00 |  3.37 | 22.47 | [link](https://gist.github.com/LowikC/ee1eba338f51c153c790070fbada5b58)|
| TMK (norm=feat_freq)| 14.61 | 28.09 | 34.83 | [link](https://gist.github.com/LowikC/81bf5371c04854e9a5dd280ffed3eb83)|

#### VCDB

See Table 3 in [LAMV paper](#lamv).

|                     | F1 Score | Gist  |
|:------------------- |:--------:|:-----:|
| TMK_Poullot         | 66.02    | [link](https://gist.github.com/LowikC/a5b8807f61a5ce6ea2dacb0fbc209f51)| 
| TMK (norm=feat_freq)| 67.30    | [link](https://gist.github.com/LowikC/394bc50344a8030fc69b6b945cdd0833)|


#### EVVE

See Table 4 in [LAMV paper](#lamv).

|                     | mean mAP | Gist  |
|:--------------------|:--------:|:-----:|
| TMK_Poullot         | 51.53    | [link](https://gist.github.com/LowikC/ff24ba09142349c8c8047d4845447866)|
| TMK (norm=feat_freq)| 53.43    | [link](https://gist.github.com/LowikC/23e62afe090b103ece6df6a8fa01319d)|
| TMK (norm=feat_freq) + Query expansion (N1=10)| 58.36    | [link](https://gist.github.com/LowikC/b425bd5eabb6e116a82e442314f6e99e)|



## Frame-level feature extraction

By design, the computation of frame level features is separated from the rest of the code,
and the core components are agnostic of the feature choice. Therefore, any frame-level feature can be used.

Our choice has been [RMACs](#rmac)
They can be extracted from any layer of a Convolutional Neural Network, they are compact,
they can be compared with dot products, and they work really well.
We used ResNet-34 as our CNN, and after running a complete search on all its layer,
we found that layer 29th is the best for Video Copy Detection.
In the following, we will assume to use layer 29, even though most of the code can handle 
a different layer if needed.

The extraction of RMAC features is handled by [rmac_features.py](./rmac_features.py).

We provide some pre-computed features on the datasets Madonna, Climbing, EVVE and VCDB.
```bash
cd /path/to/madonna
# replace madonna (1.4GB) by climbing (620MB), EVVE (19GB) or VCDB (2.6GB)
wget http://dl.fbaipublicfiles.com/videoalignment/madonna_rmac_resnet34_29.zip && \
 unzip madonna_rmac_resnet34_29.zip
```
See the [Data setup](#data-setup) section for more details on the expected location of each file.  


### PCA+Whitening data

RMAC has a PCA+Whitening stage, which is trained on a dataset separated from the one used for evaluation.
You can compute PCA+Whitening data on a dataset of choice by calling the provided script with `--train_pca`.

However, if you don't really need to use a specific dataset, you can use the PCA+W data we have computed
on the entire [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/),
using Resnet34/Layer29 which is large enough to give “good” PCA matrices.

To download the PCA weights:
```bash
PCA_PATH="/path/to/YFCC100M_pca/"
mkdir -p ${PCA_PATH}
cd ${PCA_PATH}
wget http://dl.fbaipublicfiles.com/videoalignment/yfcc100m_resnet34_29_pca.zip && \
    unzip yfcc100m_resnet34_29_pca.zip && \
    rm yfcc100m_resnet34_29_pca.zip
```
The provided features have already been processed with this PCA matrix.


### Getting RMAC features for a video

If you want to compute the features yourself, you can use [rmac_features.py](./rmac_features.py):
```
PCA_PATH="/path/to/YFCC100M_pca/"
python rmac_features.py         \
    /path/to/video.ext          \
    ${PCA_PATH}/YFCC100M_pca    \
    /path/to/output/folder      \
    --fps=15                    \
    --resnet_level=29           \
    --workers=4                 \
    --b_s=32
```
where:
* the first argument is the path to the video. Internally, it will use `ffmpeg` to read the video
 and get the raw frames from it.
* the second argument is the “prefix” for PCA+W data.
 The script will add `_mean_resnet34_29.t7` and `_pca_Dvt_resnet34_29.t7` by itself.
* the third argument is the directory where to output the features.
 This will save a .pkl file with frame-level features.
* --fps specifies the frame rate at which the script will get frames from the video.
 Notice that this* is not *the frame rate of the video. A good choice is 15 fps,
 but anyway you should use the same value for all the videos you want to compare,
 otherwise TMK won't work.
* --workers is the number of threads that will be used for scaling and normalizing frames.
* --b_s is the batch size


## Package structure and features

The computation of the descriptors, the matching of different descriptors, and
performance evaluation are all managed by main.py, even though they live in separated parts of the package.
We will first take a quick look at those parts.

### Datasets

See [the Data setup section](#data-setup) to specify the path to your data, and the datasets we used.


### Models

Available models are implemented in [models.py](videoalignment/models.py):
- [`CTE`](#cte): Circulant temporal encoding
- [`TMK_Poullot`](#tmk): the original TMK by Poullot
- [`TMK`](#lamv): the TMK with our normalization, but without the trainable part described in the paper.

All models share some common methods. In particular:
* `model.single_fv` takes as input a tensor with shape (`b_s`, `T`, `d`) where
 `b_s` is the batch size,`T` is the maximum length of the video, and `d` is the size of
 the frame level descriptor,
 and a tensor with shape (`b_s`, `T`) containing the real-valued timestamps of each descriptor,
 and produces as output the video descriptor (in the case of TMK, this is the TMK descriptor).
 Videos shorter than T can be zero-padded without affecting the final descriptor.
* `model.score_pair` matches two descriptors, for a given list of offsets.
 In the most common case, this will be all the possible offsets in the range (`-T`, `T`) 
 where `T` is the maximum length of the two videos.


### Evaluators

The code supports different evaluators:

* Localization errors (Madonna/Climbing)
* Segment-level precision/recall (as defined in the VCDB paper)
* mAP for Event detection

These are all implemented inside [eval.py](videoalignment/eval.py),
and are called by the test method inside [main.py](main.py).
The use of evaluators requires to have defined an appropriate dataset class,
so if you plan to use this code for evaluation, you should firstly set up a dataset.


## Testing a model on a dataset, and evaluating its performance

All the components described above are called and arranged by [main.py](videoalignment/main.py),
which also supports some command line arguments for the most frequent operations.
For example, to run a model, test it on a dataset, and evaluate its performance,
you just need to call:

```bash
python main.py --dataset_test EVVE --model TMK_Poullot
```

where:

* --model is the name of the model (e.g. the name of the class in [models.py](videoalignment/models.py))
* --dataset_test is the test dataset (again, the name of the class in [datasets.py](videoalignment/datasets.py))

Depending on the test dataset, different evaluators will be called. In the case of EVVE,
the mAP will be computed. You can take a look at the test function in [main.py](main.py)
to see how this works.


## What if I have my own data, and just want to run a model outside of this structure?

Here is a toy example to use the models on any data.

```python
from typing import NamedTuple
import torch
from videoalignment.models import TMK_Poullot

class Args(NamedTuple):
    m: int = 16

model = TMK_Poullot(Args())

d = 512
n_frames_a = 300
frame_features_a = torch.zeros(1, n_frames_a, d).float()
timestamps_a = torch.zeros(1, n_frames_a).float()

n_frames_b = 400
frame_features_b = torch.zeros(1, n_frames_b, d).float()
timestamps_b = torch.zeros(1, n_frames_b).float()

tmk_fv_a = model.single_fv(frame_features_a, timestamps_a)
tmk_fv_b = model.single_fv(frame_features_b, timestamps_b)

offsets = torch.arange(-n_frames_b, n_frames_b).view(1, -1).float()
scores = model.score_pair(tmk_fv_a, tmk_fv_b, offsets)
```

1. We load a model, in this case `TMK_Poullot`.
2. Since all models expect to receive command line arguments from main.py,
 we simulate that by building an Args class with the required parameters (here, `m` is the number of frequencies).
3. We load the frame features and the timestamps of two videos (for the sake of simplicity, all zeros)
4. We call `single_fv` to get the TMK features from the two videos.
   Notice that, although the two videos had different lengths, the two TMK feature vectors have the same size. 
5. We call `score_pair` to actually match the two videos, where `offsets` are all the
 deltas for which the TMK scores is computed.
 Notice that you can specify different offsets for different videos in the batch.
 The last calls returns a (1, 800) tensor containing the TMK score for every offset.

From here, you can:

* Compute the maximum score for all offsets, (i.e. `torch.max(scores, 1)[0]`),
 which can be used as a measure of similarity between the two videos.
* Compute the argmax of the scores (i.e.  `torch.max(scores, 1)[1]`),
 which gives you the predicted alignment point.

The unit tests in [test_models.py](videoalignment/test_models.py) should give you a good
understanding of the API.

## References

### LAMV 

[LAMV: Learning to Align and Match Videos with Kernelized Temporal Layers](https://research.fb.com/publications/lamv-learning-to-align-and-match-videos-with-kernelized-temporal-layers/)
```citation
@article{Baraldi2018LAMVLT,
  title={LAMV: Learning to Align and Match Videos with Kernelized Temporal Layers},
  author={Lorenzo Baraldi and Matthijs Douze and Rita Cucchiara and Herv{\'e} J{\'e}gou},
  journal={2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2018},
  pages={7804-7813}
}
```

### TMK

[Temporal Matching Kernel with Explicit Feature Maps](https://hal.inria.fr/hal-01842277)
```
@inproceedings{poullot:hal-01842277,
  TITLE = {{Temporal Matching Kernel with Explicit Feature Maps}},
  AUTHOR = {Poullot, S{\'e}bastien and Tsukatani, Shunsuke and Phuong Nguyen, Anh and J{\'e}gou, Herv{\'e} and Satoh, Shin'ichi},
  URL = {https://hal.inria.fr/hal-01842277},
  BOOKTITLE = {{ACM Multimedia 2018}},
  ADDRESS = {Brisbane, Australia},
  PUBLISHER = {{ACM Press}},
  PAGES = {1-10},
  YEAR = {2015},
  MONTH = Oct,
  DOI = {10.1145/2733373.2806228},
  PDF = {https://hal.inria.fr/hal-01842277/file/p381-poullot.pdf},
  HAL_ID = {hal-01842277},
  HAL_VERSION = {v1},
}
```

### CTE

[Circulant Temporal Encoding for Video Retrieval and Temporal Alignment](https://arxiv.org/abs/1506.02588)
```
@article{Douze2015CirculantTE,
  title={Circulant Temporal Encoding for Video Retrieval and Temporal Alignment},
  author={Matthijs Douze and J{\'e}r{\^o}me Revaud and Jakob J. Verbeek and Herv{\'e} J{\'e}gou and Cordelia Schmid},
  journal={International Journal of Computer Vision},
  year={2015},
  volume={119},
  pages={291-306}
}
```

### RMAC

[Particular object retrieval with integral max-pooling of CNN activations (RMAC)](https://arxiv.org/abs/1511.05879)
```
@article{Tolias2016ParticularOR,
  title={Particular object retrieval with integral max-pooling of CNN activations},
  author={Giorgos Tolias and Ronan Sicre and Herv{\'e} J{\'e}gou},
  journal={CoRR},
  year={2016},
  volume={abs/1511.05879}
}
```

## License

This repository is licensed under the CC BY-NC 4.0.
