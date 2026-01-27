from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _get_include_dirs(pkg_root: Path) -> list[str]:
    return [
        str(pkg_root / "tbik_kernel" / "include"),
        str(pkg_root / "tbik_kernel" / "csrc" / "allreduce"),
    ]


def _get_sources(pkg_root: Path) -> list[str]:
    return [
        str(pkg_root / "csrc" / "common_ops_allreduce.cc"),
        str(pkg_root / "tbik_kernel" / "csrc" / "allreduce" / "custom_all_reduce.cu"),
    ]


pkg_root = Path(__file__).resolve().parent

setup(
    name="mini-allreduce",
    version="0.1.0",
    packages=["mini_allreduce", "tbik_kernel"],
    package_dir={"mini_allreduce": ".", "tbik_kernel": "tbik_kernel"},
    ext_modules=[
        CUDAExtension(
            name="tbik_kernel.common_ops",
            sources=_get_sources(pkg_root),
            include_dirs=_get_include_dirs(pkg_root),
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)