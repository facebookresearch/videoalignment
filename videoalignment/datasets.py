# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import itertools
import os
import pickle as pkl
import warnings

import imageio
import numpy as np
from scipy.sparse.csgraph import connected_components, csgraph_from_dense
from torch.utils.data import Dataset

from .my_data import DATASETS

VALID_DATASETS = {"Climbing", "Madonna", "VCDB", "EVVE"}


class DatasetException(Exception):
    pass


def raise_dataset_error(dataset, message):
    raise DatasetException(
        f"The dataset {dataset} is not configured properly.\n"
        f"Check that you define the correct path in the file videoalignment/my_data.py."
        f"Reason: {message}"
    )


def check_data(dataset):
    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"{dataset} is not a valid dataset. Valid datasets are {VALID_DATASETS}"
        )

    if dataset not in DATASETS:
        raise_dataset_error(dataset, f"'{dataset}' is not in `DATASETS`")
    root_dir = DATASETS[dataset]
    if not os.path.isdir(root_dir):
        raise_dataset_error(
            dataset, f"The specified directory ({root_dir}) doesn't exist."
        )
    if not os.path.isdir(os.path.join(root_dir, "videos")):
        raise_dataset_error(
            dataset, f"`videos` dir doesn't exist in the specified directory"
        )

    if dataset == "Climbing":
        if not os.path.isfile(os.path.join(root_dir, "gt_climbing.align")):
            raise_dataset_error(
                dataset, "gt_climbing.align doesn't exist in the specified directory"
            )
    elif dataset == "Madonna":
        if not os.path.isfile(os.path.join(root_dir, "gt_madonna.align")):
            raise_dataset_error(
                dataset, "gt_madonna.align doesn't exist in the specified directory"
            )
    elif dataset == "VCDB":
        if not os.path.isdir(os.path.join(root_dir, "annotation")):
            raise_dataset_error(
                dataset, "`annotation` dir doesn't exist in the specified directory"
            )
    elif dataset == "EVVE":
        if not os.path.isdir(os.path.join(root_dir, "annotations")):
            raise_dataset_error(
                dataset, "`annotations` dir doesn't exist in the specified directory"
            )
    else:
        # will never reach this
        pass


def compute_offset(video_a, video_b):
    return video_a["global"] + video_a["begin"] - (video_b["global"] + video_b["begin"])


def read_gt(dataset):
    check_data(dataset)
    root_dir = DATASETS[dataset]
    if dataset in ["Climbing", "Madonna"]:
        videos = []
        gt_file = os.path.join(root_dir, f"gt_{dataset.lower()}.align")
        for line in open(gt_file, "r"):
            chunks = line.split(" ")
            video = chunks[0]
            if dataset == "Madonna":
                video = video.split("/")[1].split(".")[0]
            else:
                video = video.split(".")[0]
            videos.append(
                {
                    "video": video,
                    "begin": float(chunks[1]),
                    "end": float(chunks[2]),
                    "global": float(chunks[3]),
                }
            )
        pairs = []
        for video_a in videos:
            video_o = videos[videos.index(video_a) + 1 :]
            for video_b in video_o:
                extent1 = np.around(
                    np.arange(
                        video_a["global"] + video_a["begin"],
                        video_a["global"] + video_a["end"],
                        0.01,
                    ),
                    2,
                )
                extent2 = np.around(
                    np.arange(
                        video_b["global"] + video_b["begin"],
                        video_b["global"] + video_b["end"],
                        0.01,
                    ),
                    2,
                )
                offset = compute_offset(video_a, video_b)
                if np.intersect1d(extent1, extent2).size > 0:
                    pairs.append({"videos": [video_a, video_b], "offset": offset})

    elif dataset == "VCDB":
        annotation_dir = os.path.join(root_dir, "annotation")
        videos = []
        pairs = []
        for filename in os.listdir(annotation_dir):
            video_file = filename.split(".")[0]
            for line_i, line in enumerate(open("%s/%s" % (annotation_dir, filename))):

                chunks = line.split(",")
                begin_a = chunks[2].split(":")[::-1]
                begin_a = sum(int(t) * 60 ** i for (i, t) in enumerate(begin_a))
                end_a = chunks[3].split(":")[::-1]
                end_a = sum(int(t) * 60 ** i for (i, t) in enumerate(end_a))
                begin_b = chunks[4].split(":")[::-1]
                begin_b = sum(int(t) * 60 ** i for (i, t) in enumerate(begin_b))
                end_b = chunks[5].split(":")[::-1]
                end_b = sum(int(t) * 60 ** i for (i, t) in enumerate(end_b))

                video_a = {
                    "video": "%s/%s" % (video_file, chunks[0].split(".")[0]),
                    "global": -begin_a,
                    "begin": begin_a,
                    "end": end_a,
                }
                video_b = {
                    "video": "%s/%s" % (video_file, chunks[1].split(".")[0]),
                    "global": -begin_b,
                    "begin": begin_b,
                    "end": end_b,
                }

                offset = compute_offset(video_a, video_b)

                if video_a not in videos:
                    videos.append(video_a)
                else:
                    video_a = videos[videos.index(video_a)]
                if video_b not in videos:
                    videos.append(video_b)
                else:
                    video_b = videos[videos.index(video_b)]

                pairs.append({"videos": [video_a, video_b], "offset": offset})
    elif dataset == "EVVE":
        annotation_dir = os.path.join(root_dir, "annotations")
        videos = []
        pairs = []
        durations = os.path.join(annotation_dir, "durations.pkl")
        durations = pkl.load(open(durations, "rb"))
        for fname in os.listdir(annotation_dir):
            if not fname.endswith(".dat"):
                continue
            event = fname.split(".")[0]
            this_videos = []
            for l_i, l in enumerate(open(os.path.join(annotation_dir, fname), "r")):
                vidname, gt, split = l.split()
                vidname = "%s/%s" % (event, vidname)
                duration = durations[vidname]
                gt = int(gt)
                video = {
                    "video": vidname,
                    "begin": 0,
                    "end": duration,
                    "global": 0,
                    "split": split,
                    "pos_null": gt,
                    "event": event,
                }
                this_videos.append(video)

            videos.extend(this_videos)

            for video_a in this_videos:
                video_o = this_videos[this_videos.index(video_a) + 1 :]
                for video_b in video_o:
                    if (
                        video_a["split"] == video_b["split"]
                        and video_a["pos_null"] == 1
                        and video_b["pos_null"] == 1
                    ):
                        pairs.append({"videos": [video_a, video_b], "offset": -1})
    else:
        raise NotImplementedError
    return videos, pairs


class VideoDataset(Dataset):
    fps = 15
    short_length = 1500
    max_regions = 20
    n_folds = 1

    def __init__(
        self,
        phase,
        args,
        get_entire_videos=False,
        get_single_videos=False,
        get_all_pairs=False,
        get_nonoverlapping_too=False,
        chunk_randomly=False,
        pad=True,
    ):
        super(VideoDataset, self).__init__()
        assert phase in ["train", "val", "all"]
        assert (
            get_entire_videos
            ^ get_single_videos
            ^ get_all_pairs
            ^ get_nonoverlapping_too
        ) or not (
            get_entire_videos
            and get_single_videos
            and get_all_pairs
            and get_nonoverlapping_too
        )
        self.phase = phase
        self.args = args
        self.get_entire_videos = get_entire_videos
        self.get_single_videos = get_single_videos
        self.get_all_pairs = get_all_pairs
        self.get_triplets = get_nonoverlapping_too
        self.chunk_randomly = chunk_randomly
        self.pad = pad
        if self.chunk_randomly:
            self.length = self.short_length
        else:
            self.length = self.max_length

        if self.args.fold_index > self.n_folds:
            raise NotImplementedError

        self.fps_cache = dict()
        self.features = None
        self.triplets = None
        self.split_train_val()

    def __str__(self):
        return "%s dataset (fold %d, phase %s)" % (
            self.__class__.__name__,
            self.args.fold_index,
            self.phase,
        )

    def annotate_connected_components(self):
        graph_matrix = (
            np.ones((len(self.gt_all_videos), len(self.gt_all_videos))) * np.inf
        )
        for p in self.gt_all_overlapping_pairs:
            graph_matrix[
                self.gt_all_videos.index(p["videos"][0]),
                self.gt_all_videos.index(p["videos"][1]),
            ] = 1
            graph_matrix[
                self.gt_all_videos.index(p["videos"][1]),
                self.gt_all_videos.index(p["videos"][0]),
            ] = 1
        graph = csgraph_from_dense(np.ma.masked_invalid(graph_matrix))
        ncc, labels = connected_components(graph, directed=False)

        for video_i, label in enumerate(labels):
            self.gt_all_videos[video_i]["label"] = int(label)
            for i in range(len(self.gt_all_overlapping_pairs)):
                if (
                    self.gt_all_videos[video_i]
                    in self.gt_all_overlapping_pairs[i]["videos"]
                ):
                    self.gt_all_overlapping_pairs[i]["label"] = int(label)

        return ncc

    def select_hardest_ops(self, n):
        print("Pruning of %s..." % self)
        fvs = np.zeros((len(self.videos), 512))
        for v_i, v in enumerate(self.videos):
            fv = self.get_single_feature(v, pad=False)[0]
            fv = np.sum(fv, 0)
            fvs[v_i] = fv / (10e-8 + np.sqrt(np.sum(fv ** 2)))

        scores = np.zeros((len(self.overlapping_pairs),))
        for op_i, op in enumerate(self.overlapping_pairs):
            va_i = self.videos.index(op["videos"][0])
            vb_i = self.videos.index(op["videos"][1])
            scores[op_i] = np.inner(fvs[va_i], fvs[vb_i])

        selected_i = np.argsort(scores)[:n]
        self.overlapping_pairs = [self.overlapping_pairs[i] for i in selected_i]

    def video_path(self, v):
        raise NotImplementedError

    def split_train_val(self):
        raise NotImplementedError

    @property
    def all_pairs(self):
        pairs = []
        for v_a, v_b in itertools.combinations(self.videos, 2):
            pairs.append({"videos": [v_a, v_b]})
        return pairs

    def get_fps(self, v):
        if self.video_path(v) not in self.fps_cache.keys():
            try:
                vid = imageio.get_reader(self.video_path(v), "ffmpeg")
                self.fps_cache[self.video_path(v)] = vid.get_meta_data()["fps"]
            except Exception as e:
                print(e)
                warnings.warn(
                    "Unable to get metadata for video %s" % self.video_path(v)
                )
                self.fps_cache[self.video_path(v)] = 25
        return self.fps_cache[self.video_path(v)]

    def get_video_feature_entire(self, v):
        if self.features_string.endswith(".pkl"):
            if not self.features:
                self.features = pkl.load(
                    open(self.features_string % self.args.resnet_level, "rb")
                )

            if v["video"].endswith("longvid"):
                ts = self.features["frames/" + v["video"] + ".avi"].astype(np.float32)
            else:
                ts = self.features["frames/" + v["video"]].astype(np.float32)
        else:
            if v["video"] == "albert_ixus70/longvid":
                vv = "albert_ixus70/longvid.avi"
            else:
                vv = v["video"]
            fname = self.features_string % self.args.resnet_level + vv + ".pkl"
            # incompatibility of pickled numpy array,
            # see https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
            # TODO (lowik): Re-serialize all files with pickle from Python3
            ts = pkl.load(open(fname, "rb"), encoding="latin1").astype(np.float32)
        return ts

    def get_video_feature(self, v):
        ts = self.get_video_feature_entire(v)
        if self.__class__ != EVVE:
            begin = int(v["begin"] * self.fps)
            end = int(v["end"] * self.fps)
            if begin < end:
                ts = ts[begin:end]
        return ts

    def get_single_feature_entire(self, v):
        fps = self.get_fps(v)
        ts = self.get_video_feature_entire(v)

        if self.pad:
            if ts.shape[0] > self.length:
                # warnings.warn('I have rescaled a feature vector!')
                ts = ts[: self.length]

            offset = self.length - ts.shape[0]
            if offset > 0:
                ts = np.concatenate([ts, np.zeros([offset] + list(ts.shape[1:]))])

        xs = get_timestamps(fps, self.fps, ts.shape[0]) * self.fps
        return ts, xs

    def get_single_feature(self, v, pad=True):
        fps = self.get_fps(v)
        ts = self.get_video_feature(v)

        if self.pad and pad:
            if ts.shape[0] > self.length:
                # warnings.warn('I have rescaled a feature vector!')
                ts = ts[: self.length]

            offset = self.length - ts.shape[0]
            if offset > 0:
                ts = np.concatenate([ts, np.zeros([offset] + list(ts.shape[1:]))])

        xs = get_timestamps(fps, self.fps, ts.shape[0]) * self.fps
        return ts.astype(np.float32), xs.astype(np.float32)

    def get_video_length(self, v):
        begin = int(np.around(v["begin"] * self.fps))
        end = int(np.around(v["end"] * self.fps))
        return min(self.max_length, end - begin)

    def get_pair_feature(self, p):
        ts_a, xs_a = self.get_single_feature(p["videos"][0])
        ts_b, xs_b = self.get_single_feature(p["videos"][1])

        offset = -compute_offset(p["videos"][0], p["videos"][1]) * self.fps

        return ts_a, ts_b, xs_a, xs_b, offset

    def get_triplet_feature(self, t):
        ts_a, xs_a = self.get_single_feature(t["videos"][0])
        ts_b, xs_b = self.get_single_feature(t["videos"][1])
        ts_c, xs_c = self.get_single_feature(t["videos"][2])

        offsetb = -compute_offset(t["videos"][0], t["videos"][1]) * self.fps

        return ts_a, ts_b, ts_c, xs_a, xs_b, xs_c, offsetb, 0

    def __getitem__(self, item):
        if self.get_entire_videos:
            return self.get_single_feature_entire(self.videos[item])
        elif self.get_single_videos:
            return self.get_single_feature(self.videos[item])
        elif self.get_all_pairs:
            return self.get_pair_feature(self.all_pairs[item])
        elif self.get_triplets:
            return self.get_triplet_feature(self.triplets[item])
        else:
            return list(self.get_pair_feature(self.overlapping_pairs[item]))

    def __len__(self):
        if self.get_entire_videos:
            return len(self.videos)
        elif self.get_single_videos:
            return len(self.videos)
        elif self.get_all_pairs:
            return len(self.all_pairs)
        elif self.get_triplets:
            return len(self.triplets)
        else:
            return len(self.overlapping_pairs)


class Madonna(VideoDataset):
    features_string = os.path.join(DATASETS["Madonna"], "rmac_resnet34_%d/")
    max_length = 9085
    short_length = 500
    n_folds = 5
    is_localization = True

    def __init__(self, *args, **kwargs):
        check_data("Madonna")
        self.gt_all_videos, self.gt_all_overlapping_pairs = read_gt("Madonna")
        super(Madonna, self).__init__(*args, **kwargs)

    def video_path(self, v):
        return os.path.join(DATASETS["Madonna"], "videos", "%s.mp4" % v["video"])

    def split_train_val(self):
        # Split overlapping pairs according to connected components
        ncc = self.annotate_connected_components()
        videos_train = []
        videos_val = []
        pairs_train = []
        pairs_val = []

        cc = list(range(ncc))
        val_cc = cc[
            (ncc // self.n_folds)
            * self.args.fold_index : (ncc // self.n_folds)
            * (self.args.fold_index + 1)
        ]
        train_cc = (
            cc[: (ncc // self.n_folds) * self.args.fold_index]
            + cc[(ncc // self.n_folds) * (self.args.fold_index + 1) :]
        )

        for cc in val_cc:
            videos_val.extend([v for v in self.gt_all_videos if v["label"] == cc])
            pairs_val.extend(
                [p for p in self.gt_all_overlapping_pairs if p["label"] == cc]
            )

        for cc in train_cc:
            videos_train.extend([v for v in self.gt_all_videos if v["label"] == cc])
            pairs_train.extend(
                [p for p in self.gt_all_overlapping_pairs if p["label"] == cc]
            )

        if self.phase == "train":
            self.videos = videos_train
            self.overlapping_pairs = pairs_train
        elif self.phase == "val":
            self.videos = videos_val
            self.overlapping_pairs = pairs_val
        elif self.phase == "all":
            self.videos = self.gt_all_videos
            self.overlapping_pairs = self.gt_all_overlapping_pairs


class Climbing(VideoDataset):
    features_string = os.path.join(DATASETS["Climbing"], "rmac_resnet34_%d/")
    max_length = 18886
    is_localization = True

    def __init__(self, *args, **kwargs):
        check_data("Climbing")
        self.gt_all_videos, self.gt_all_overlapping_pairs = read_gt("Climbing")
        super(Climbing, self).__init__(*args, **kwargs)

    def video_path(self, v):
        if v["video"].endswith("longvid"):
            return os.path.join(
                DATASETS["Climbing"], "videos", "%s.avi.mp4" % v["video"]
            )
        else:
            return os.path.join(DATASETS["Climbing"], "videos", "%s.mp4" % v["video"])

    def split_train_val(self):
        self.videos = self.gt_all_videos
        self.overlapping_pairs = self.gt_all_overlapping_pairs


class VCDB(VideoDataset):
    features_string = os.path.join(DATASETS["VCDB"], "rmac_resnet34_%d/")
    max_length = 2000  # 39930
    n_folds = 5
    short_length = 60
    is_localization = True

    def __init__(self, *args, **kwargs):
        check_data("VCDB")
        self.gt_all_videos, self.gt_all_overlapping_pairs = read_gt("VCDB")
        super(VCDB, self).__init__(*args, **kwargs)

    def video_path(self, v):
        extensions = ["flv", "mp4"]
        for ext in extensions:
            fname = os.path.join(
                DATASETS["VCDB"], "videos", "%s.%s" % (v["video"], ext)
            )
            if os.path.isfile(fname):
                return fname

    def split_train_val(self):
        cc = list(set([v["video"].split("/")[0] for v in self.gt_all_videos]))
        ncc = len(cc)

        val_cc = cc[
            (ncc // self.n_folds)
            * self.args.fold_index : (ncc // self.n_folds)
            * (self.args.fold_index + 1)
        ]
        train_cc = (
            cc[: (ncc // self.n_folds) * self.args.fold_index]
            + cc[(ncc // self.n_folds) * (self.args.fold_index + 1) :]
        )

        videos_train = []
        videos_val = []

        for c in train_cc:
            videos_train.extend(
                [v for v in self.gt_all_videos if v["video"].split("/")[0] == c]
            )

        for c in val_cc:
            videos_val.extend(
                [v for v in self.gt_all_videos if v["video"].split("/")[0] == c]
            )

        pairs_train = [
            p for p in self.gt_all_overlapping_pairs if p["videos"][0] in videos_train
        ]
        pairs_val = [
            p for p in self.gt_all_overlapping_pairs if p["videos"][0] in videos_val
        ]

        if self.phase == "train":
            self.videos = videos_train
            self.overlapping_pairs = pairs_train
        elif self.phase == "val":
            self.videos = videos_val
            self.overlapping_pairs = pairs_val
        elif self.phase == "all":
            self.videos = self.gt_all_videos
            self.overlapping_pairs = self.gt_all_overlapping_pairs


class EVVE(VideoDataset):
    features_string = os.path.join(DATASETS["EVVE"], "rmac_resnet34_%d/")
    max_length = 16000
    is_localization = False

    def __init__(self, *args, **kwargs):
        check_data("EVVE")
        self.gt_all_videos, self.gt_all_overlapping_pairs = read_gt("EVVE")
        super(EVVE, self).__init__(*args, **kwargs)

    def video_path(self, v):
        extensions = ["flv", "mp4"]
        for ext in extensions:
            fname = os.path.join(
                DATASETS["EVVE"], "videos", "%s.%s" % (v["video"], ext)
            )
            if os.path.isfile(fname):
                return fname

    def split_train_val(self):
        if self.phase == "all":
            self.videos = self.gt_all_videos
            self.overlapping_pairs = self.gt_all_overlapping_pairs
        elif self.phase == "train":
            videos_train = [v for v in self.gt_all_videos if v["split"] == "database"]
            pairs_train = [
                p
                for p in self.gt_all_overlapping_pairs
                if p["videos"][0] in videos_train and p["videos"][1] in videos_train
            ]
            self.videos = videos_train
            self.overlapping_pairs = pairs_train
        elif self.phase == "val":
            videos_val = [v for v in self.gt_all_videos if v["split"] == "query"]

            pairs_val = [
                p
                for p in self.gt_all_overlapping_pairs
                if p["videos"][0] in videos_val and p["videos"][1] in videos_val
            ]
            self.videos = videos_val
            self.overlapping_pairs = pairs_val


def is_the_same_pair(a, b):
    c1 = a["videos"][0] == b["videos"][0] and a["videos"][1] == b["videos"][1]
    c2 = a["videos"][0] == b["videos"][1] and a["videos"][1] == b["videos"][0]
    return c1 or c2


def get_timestamps(ori_fps, dst_fps, n_frames_dst):
    actual_timestamps = (
        np.round(np.arange(0, n_frames_dst) * ori_fps / dst_fps) / ori_fps
    )
    return actual_timestamps


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]
