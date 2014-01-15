# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:37:17 2013

@author: khayyam
"""
from numpy.distutils.misc_util import get_numpy_include_dirs
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("convex", ["src/SparseMatrix.cpp", "src/linearprogramming.cpp", "convex.pyx", "src/denoising.cpp",  "src/derivatives.cpp",  "src/morphological.cpp", "src/onlm.cpp", "src/ornlm.cpp", "src/topological.cpp", "src/TotalVariation.cpp", "src/bits.cpp"],include_dirs=get_numpy_include_dirs(), extra_compile_args=["-Iinclude"], language="c++")]
setup(
  name = 'Algorithms for convex optimization',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
