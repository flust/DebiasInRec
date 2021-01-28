from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("recommend",
    sources=["_recommend.pyx", "greedy.cpp"],
    language="c++",
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"],
    include_dirs=[numpy.get_include()])],
)

