# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle as pkl
import tempfile
from subprocess import call

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.models import resnet34

DESCRIPTION = """
"RMAC features extractor"
"""


class VideoDataset(Dataset):
    """
    A simple Dataset class for loading images inside a folder.
    """

    def __init__(self, video_path, transforms, fps, image_area):
        self.video_path = video_path
        self.transforms = transforms
        self.root_dir = tempfile.TemporaryDirectory()
        self.root = self.root_dir.name

        FNULL = open(os.devnull, "w")
        call(
            [
                "ffmpeg",
                "-i",
                self.video_path,
                "-filter:v",
                "fps=%f" % fps,
                os.path.join(self.root + "/%07d.jpg"),
            ],
            stderr=FNULL,
        )

        self.images = sorted(os.listdir(self.root))

        self.images = [os.path.join(self.root, i) for i in self.images]
        self.transforms.transforms[0].size = get_scale(self.images[0], image_area)

    def __getitem__(self, item):
        path = self.images[item]
        img = default_loader(path)
        if self.transforms:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.images)


class PCA(object):
    """
    Fits and applies PCA whitening
    """

    def __init__(self, n_components, device):
        self.n_components = n_components
        self.mean = None
        self.DVt = None
        self.device = device

    def fit(self, X):
        mean = X.mean(axis=0)
        X -= mean
        self.mean = torch.from_numpy(mean).view(1, -1).to(self.device)

        Xcov = np.dot(X.T, X)
        d, V = np.linalg.eigh(Xcov)
        idx = np.argsort(d)[::-1][: self.n_components]
        d = d[idx]
        V = V[:, idx]
        D = np.diag(1.0 / np.sqrt(d + 10e-8))
        self.DVt = torch.from_numpy(np.dot(D, V.T)).to(self.device)

    def apply(self, X):
        X -= self.mean.expand_as(X)
        num = torch.mm(self.DVt, X.transpose(0, 1)).transpose(0, 1)
        out = num / torch.sqrt(torch.sum(num ** 2, dim=1, keepdim=True))
        return out


def get_rmac_region_coordinates(H, W, L):
    # Almost verbatim from Tolias et al Matlab implementation.
    # Could be heavily pythonized, but really not worth it...
    # Desired overlap of neighboring regions
    ovr = 0.4
    # Possible regions for the long dimension
    steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
    w = np.minimum(H, W)

    b = (np.maximum(H, W) - w) / (steps - 1)
    # steps(idx) regions for long dimension. The +1 comes from Matlab
    # 1-indexing...
    idx = np.argmin(np.abs(((w ** 2 - w * b) / w ** 2) - ovr)) + 1

    # Region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx
    elif H > W:
        Hd = idx

    regions_xywh = []
    for l in range(1, L + 1):
        wl = np.floor(2 * w / (l + 1))
        wl2 = np.floor(wl / 2 - 1)
        # Center coordinates
        if l + Wd - 1 > 0:
            b = (W - wl) / (l + Wd - 1)
        else:
            b = 0
        cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
        # Center coordinates
        if l + Hd - 1 > 0:
            b = (H - wl) / (l + Hd - 1)
        else:
            b = 0
        cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

        for i_ in cenH:
            for j_ in cenW:
                regions_xywh.append([j_, i_, wl, wl])

    # Round the regions. Careful with the borders!
    for i in range(len(regions_xywh)):
        for j in range(4):
            regions_xywh[i][j] = int(round(regions_xywh[i][j]))
        if regions_xywh[i][0] + regions_xywh[i][2] > W:
            regions_xywh[i][0] -= (regions_xywh[i][0] + regions_xywh[i][2]) - W
        if regions_xywh[i][1] + regions_xywh[i][3] > H:
            regions_xywh[i][1] -= (regions_xywh[i][1] + regions_xywh[i][3]) - H
    return np.array(regions_xywh).astype(np.float32)


class ResNet34_ith(nn.Module):
    """
    ResNet-34 cut at the i-th layer
    """

    def __init__(self, n):
        super(ResNet34_ith, self).__init__()
        self.n = n
        self.model = resnet34(pretrained=True)
        self.model.eval()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        counter = 1
        if counter == self.n:
            return x
        x = self.model.maxpool(x)
        for block in [
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        ]:
            for layer in block._modules.values():
                residual = x
                x = layer.conv1(x)
                x = layer.bn1(x)
                x = layer.relu(x)
                counter += 1
                if counter == self.n:
                    break

                x = layer.conv2(x)
                x = layer.bn2(x)
                if layer.downsample is not None:
                    residual = layer.downsample(residual)

                x += residual
                x = layer.relu(x)
                counter += 1
                if counter == self.n:
                    break
        return x


def get_scale(image_path, n_pixels):
    image = default_loader(image_path)
    factor = np.sqrt((image.width * image.height) / n_pixels)
    return int(np.rint(min(image.width, image.height) / factor))


def get_rmac_descriptors(cnn, args, video_path, aggregated=True, pca=None):
    image_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print("Create video dataset")
    dataset = VideoDataset(video_path, image_transforms, args.fps, args.image_area)
    print("Extract features")
    dataloader = DataLoader(dataset, batch_size=args.b_s, num_workers=args.workers)
    devices_ids = list(range(torch.cuda.device_count()))
    if devices_ids:
        print(f"Use devices: {devices_ids}")
        parallel_cnn = nn.DataParallel(cnn, device_ids=devices_ids)
    else:
        parallel_cnn = cnn

    all_rmac_descriptors = []
    dataloader_iter = iter(dataloader)
    with torch.no_grad():
        for it, x_images in enumerate(dataloader_iter):
            images = x_images.float().to(args.device)
            features = parallel_cnn(images)
            nc = features.size()[1]
            args.pca_dimensions = nc
            rmac_regions = get_rmac_region_coordinates(
                features.size()[2], features.size()[3], args.rmac_levels
            ).astype(np.int)

            rmac_descriptors = []
            for region in rmac_regions:
                desc = torch.max(
                    torch.max(
                        features[
                            :,
                            :,
                            region[1] : (region[3] + region[1]),
                            region[0] : (region[2] + region[0]),
                        ],
                        2,
                        keepdim=True,
                    )[0],
                    3,
                    keepdim=True,
                )[0]
                rmac_descriptors.append(desc.view(-1, 1, nc))
            rmac_descriptors = torch.cat(rmac_descriptors, 1)
            nr = rmac_descriptors.size()[1]

            # L2-norm
            rmac_descriptors = rmac_descriptors.view(-1, nc)
            rmac_descriptors = rmac_descriptors / torch.sqrt(
                torch.sum(rmac_descriptors ** 2, dim=1, keepdim=True)
            )

            if args.regional_descriptors:
                rmac_descriptors = rmac_descriptors.view(-1, nr, nc)

            if not aggregated:
                rmac_descriptors = rmac_descriptors.cpu().data.numpy()
                all_rmac_descriptors.append(rmac_descriptors)

            else:
                # PCA whitening
                rmac_descriptors = pca.apply(rmac_descriptors)

                # Sum aggregation and L2-normalization
                rmac_descriptors = torch.sum(
                    rmac_descriptors.view(-1, nr, args.pca_dimensions), 1
                )
                rmac_descriptors = rmac_descriptors / torch.sqrt(
                    torch.sum(rmac_descriptors ** 2, dim=1, keepdim=True)
                )
                rmac_descriptors = rmac_descriptors.detach().cpu().numpy()
                all_rmac_descriptors.append(rmac_descriptors)

        return np.concatenate(all_rmac_descriptors)


def train_pca(cnn, args):
    out_dir = args.out_folder
    sum_pooling = args.sum_before_pca
    directory = os.path.join(out_dir, "tmp_%d/" % args.resnet_level)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    outfile = directory + os.path.basename(args.dataset_folder) + ".pkl"

    if os.path.exists(outfile):
        return

    descriptors = []
    for dir, subdirs, files in os.walk(args.dataset_folder):
        if files and not subdirs:
            for file in files:
                file = os.path.join(dir, file)
                print("Computing descriptors for %s" % file)
                try:
                    desc = get_rmac_descriptors(cnn, args, file, aggregated=False)
                    if sum_pooling:
                        desc = np.sum(desc, 1)
                        desc = desc / np.sqrt(
                            np.sum(desc ** 2, axis=-1, keepdims=True) + 10e-8
                        )
                    descriptors.append(desc)
                except:
                    print("Unable to process %s" % file)

    descriptors = np.concatenate(descriptors)
    pca = PCA(args.pca_dimensions, args.device)
    pca.fit(descriptors)
    if sum_pooling:
        pca_prefix = args.pca_files_prefix + "_sum"
    else:
        pca_prefix = args.pca_files_prefix
    torch.save(
        pca.DVt,
        os.path.join(
            args.dataset_folder, pca_prefix + "_Dvt_resnet34_%d.t7" % args.resnet_level
        ),
    )
    torch.save(
        pca.mean,
        os.path.join(
            args.dataset_folder, pca_prefix + "_mean_resnet34_%d.t7" % args.resnet_level
        ),
    )
    print("Done.")


def compute_features(cnn, args):
    pca = PCA(args.pca_dimensions, args.device)
    pca.DVt = torch.load(
        os.path.join(
            args.dataset_folder,
            args.pca_files_prefix + "_Dvt_resnet34_%d.t7" % args.resnet_level,
        ),
        map_location="cpu",
    ).to(args.device)
    pca.mean = torch.load(
        os.path.join(
            args.dataset_folder,
            args.pca_files_prefix + "_mean_resnet34_%d.t7" % args.resnet_level,
        ),
        map_location="cpu",
    ).to(args.device)

    in_filename = args.dataset_folder
    print("Computing descriptors for %s" % in_filename)
    descriptors = get_rmac_descriptors(cnn, args, in_filename, pca=pca)

    filename = os.path.basename(in_filename)
    filename, _ = os.path.splitext(filename)
    out_folder = os.path.join(args.out_folder, f"rmac_resnet34_{args.resnet_level}")
    fname_desc = os.path.join(out_folder, filename + ".pkl")
    os.makedirs(out_folder, exist_ok=True)
    pkl.dump(descriptors, open(fname_desc, "wb"), -1)


def main(args):
    cnn = ResNet34_ith(args.resnet_level)
    cnn = cnn.to(args.device)

    if args.train_pca:
        train_pca(cnn, args)
    else:
        compute_features(cnn, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "dataset_folder",
        help="path to dataset (to compute the PCA), or to a single video file",
    )
    parser.add_argument("pca_files_prefix", help="Prefix for files for PCA")
    parser.add_argument("out_folder", help="Output folder for descriptors files")
    parser.add_argument("--train_pca", action="store_true")
    parser.add_argument("--regional_descriptors", action="store_true")
    parser.add_argument("--image_area", default=12000, type=int)
    parser.add_argument("--resnet_level", default=30, type=int)
    parser.add_argument("--rmac_levels", default=3, type=int)
    parser.add_argument("--pca_dimensions", default=512, type=int)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--b_s", default=200, type=int)
    parser.add_argument("--fps", default=15, type=float)
    parser.add_argument("--sum_before_pca", action="store_true")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running with args={args}")

    main(args)
