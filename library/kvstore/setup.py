from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="kvstore",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="kvstore",
            sources=["kvstore.cc"],
            include_dirs=["."],
            extra_compile_args=["-std=c++17", "-g0", "-O3", "-D_GNU_SOURCE","-fopenmp"],
            extra_link_args=['-fopenmp', '-O3'],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
