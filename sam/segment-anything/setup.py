# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="segment_anything",
    version="1.0",
    install_requires=[],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python", "onnx", "onnxruntime"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)
# python helpers/extract_embeddings.py --checkpoint-path sam_vit_h_4b8939.pth --dataset-folder data 
# python helpers/generate_onnx.py --checkpoint-path sam_vit_h_4b8939.pth --onnx-model-path ./sam_onnx.onnx --orig-im-size 360 360

# # cd到项目2的主目录下
# python helpers\generate_onnx.py --checkpoint-path D:\segment\segment-anything\sam-epoch=35-train_loss=0.0158.ckpt --onnx-model-path ./sam_onnx1.onnx --orig-im-size 360 360