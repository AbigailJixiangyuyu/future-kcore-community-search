"""Build the original CPU sampler without installing packages or replacing binaries."""
from pathlib import Path

from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
setup(
    name="swift-local-sampler",
    ext_modules=[
        Extension(
            "sampler_core",
            [str(ROOT / "sampler/sampler_core.cpp")],
            include_dirs=[
                str(ROOT / "swift-dgl/third_party/tensorpipe/third_party/pybind11/include")
            ],
            extra_compile_args=["-O2", "-std=c++14", "-fopenmp"],
            extra_link_args=["-fopenmp"],
            language="c++",
        )
    ],
)
