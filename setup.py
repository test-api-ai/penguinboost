"""Build script for PenguinBoost C++ extension.

Usage:
    pip install -e .          # editable install (builds C++ in-place)
    pip install .             # normal install
"""

import os
import sys
import platform
from setuptools import setup, Extension, find_packages

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    raise RuntimeError(
        "pybind11 is required to build PenguinBoost.\n"
        "Install it with:  pip install pybind11"
    )

# ---- Compiler flags ----
extra_compile_args = ["-O3", "-std=c++17", "-ffast-math"]
extra_link_args = []

if platform.system() == "Darwin":
    # macOS: suppress deprecation warnings from the SDK
    extra_compile_args += ["-Wno-deprecated-declarations"]
    extra_link_args    += ["-Wl,-undefined,dynamic_lookup"]
elif platform.system() == "Linux":
    extra_compile_args += ["-march=native"]

ext = Extension(
    name="penguinboost._core",
    sources=["penguinboost/cpp/_core.cpp"],
    include_dirs=[pybind11_include],
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    ext_modules=[ext],
    packages=find_packages(),
)
