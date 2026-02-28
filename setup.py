"""Build script for PenguinBoost C++ extension.

Usage:
    pip install -e .          # editable install (builds C++ in-place)
    pip install .             # normal install
"""

import os
import sys
import platform
import subprocess
import tempfile
from setuptools import setup, Extension, find_packages

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    raise RuntimeError(
        "pybind11 is required to build PenguinBoost.\n"
        "Install it with:  pip install pybind11"
    )

_system = platform.system()

# ---- コンパイラフラグ（OS / コンパイラ別） ----
if _system == "Windows":
    # MSVC
    extra_compile_args = ["/O2", "/std:c++17", "/EHsc"]
    extra_link_args    = []
else:
    # GCC / Clang (Linux / macOS)
    extra_compile_args = ["-O3", "-std=c++17", "-ffast-math"]
    extra_link_args    = []
    if _system == "Darwin":
        extra_compile_args += ["-Wno-deprecated-declarations"]
        extra_link_args    += ["-Wl,-undefined,dynamic_lookup"]
    # -march=native は配布用 wheel では使用しない（ビルドマシン固有命令を避ける）


# ---- OpenMP ヘルパー ----

def _find_homebrew_libomp():
    """arm64: /opt/homebrew/opt/libomp  /  x86: /usr/local/opt/libomp"""
    for prefix in ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]:
        if os.path.exists(os.path.join(prefix, "include", "omp.h")):
            return prefix
    return None


def _try_compile_openmp(compile_args, link_args, inc_dirs, lib_dirs, libs):
    """最小限の OpenMP プログラムをコンパイルして成否を返す。"""
    src = "#include <omp.h>\nint main(){return omp_get_max_threads();}\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "omp_test.c")
        out_path = os.path.join(tmpdir, "omp_test")
        with open(src_path, "w") as f:
            f.write(src)
        cc = os.environ.get("CC", "cc")
        cmd = [cc]
        for d in inc_dirs:
            cmd += [f"-I{d}"]
        cmd += compile_args + [src_path, "-o", out_path]
        for d in lib_dirs:
            cmd += [f"-L{d}"]
        for lib in libs:
            cmd += [f"-l{lib}"]
        cmd += link_args
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=30)
            return r.returncode == 0
        except Exception:
            return False


def _configure_openmp():
    """(compile_args, link_args, inc_dirs, lib_dirs, libs) または None を返す。"""

    # 環境変数で明示的に無効化（CI macOS wheel ビルド等）
    if os.environ.get("DISABLE_OPENMP"):
        return None

    if _system == "Windows":
        # MSVC は /openmp を組み込みサポート（追加ライブラリ不要）
        return ["/openmp"], [], [], [], []

    if _system == "Darwin":
        prefix = _find_homebrew_libomp()
        if prefix is None:
            return None
        inc_dir = os.path.join(prefix, "include")
        lib_dir = os.path.join(prefix, "lib")
        c_args  = ["-Xpreprocessor", "-fopenmp"]
        l_args  = [f"-Wl,-rpath,{lib_dir}"]
        inc_dirs = [inc_dir]
        lib_dirs = [lib_dir]
        libs     = ["omp"]
        if _try_compile_openmp(c_args, l_args, inc_dirs, lib_dirs, libs):
            return c_args, l_args, inc_dirs, lib_dirs, libs
        return None

    if _system == "Linux":
        c_args = ["-fopenmp"]
        l_args = ["-fopenmp"]
        if _try_compile_openmp(c_args, l_args, [], [], []):
            return c_args, l_args, [], [], []
        return None

    return None


# ---- OpenMP が利用可能なら有効化 ----
omp_config = _configure_openmp()
ext_include_dirs = [pybind11_include]
ext_lib_dirs     = []
ext_libs         = []

if omp_config is not None:
    omp_compile, omp_link, omp_inc, omp_libdirs, omp_libs = omp_config
    extra_compile_args += omp_compile
    extra_link_args    += omp_link
    ext_include_dirs   += omp_inc
    ext_lib_dirs        = omp_libdirs
    ext_libs            = omp_libs
    print(f"[PenguinBoost] OpenMP enabled ({_system})")
else:
    print("[PenguinBoost] OpenMP not found – building single-threaded")

ext = Extension(
    name="penguinboost._core",
    sources=["penguinboost/cpp/_core.cpp"],
    include_dirs=ext_include_dirs,
    library_dirs=ext_lib_dirs,
    libraries=ext_libs,
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    ext_modules=[ext],
    packages=find_packages(),
)
