# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys

from setuptools import setup

if sys.version_info < (3, 6):
    raise RuntimeError("videoalignment requires Python 3.6")

setup(
    name="videoalignment",
    version="0.1.0",
    description="Temporal Match Kernel for Video Alignment",
    url="https://github.com/facebookresearch/videoalignment",
    author="Facebook AI",
    license="CC-BY-NC",
    packages=["videoalignment"],
)
