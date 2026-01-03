# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

here = Path(__file__).resolve().parent
ext_src_root = Path("_ext_src")
ext_sources = glob.glob(str((here / ext_src_root / "src") / "*.cpp")) + glob.glob(
    str((here / ext_src_root / "src") / "*.cu")
)
ext_sources = [str(Path(p).resolve().relative_to(here)) for p in ext_sources]
ext_include = str((here / ext_src_root / "include").resolve())

py_modules = [
    p.stem
    for p in here.glob("*.py")
    if p.name not in {"setup.py", "__init__.py"} and not p.name.endswith("_test.py")
]

setup(
    name="pointnet2",
    version="0.1.0",
    description="PointNet++ ops and utilities",
    packages=["pointnet2"],
    package_dir={"pointnet2": "pointnet2"},
    py_modules=py_modules,
    ext_modules=[
        CUDAExtension(
            name="pointnet2._ext",
            sources=ext_sources,
            include_dirs=[ext_include],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
