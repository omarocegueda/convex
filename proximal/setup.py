from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Proximal methods for image denoising',
  ext_modules = cythonize(["proximal/derivatives.pyx", "proximal/total_variation.pyx"]),
)