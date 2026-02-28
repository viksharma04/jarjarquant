from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("jarjarquant._cython.opt_threshold", [
              "jarjarquant/_cython/opt_threshold.pyx"]),
    Extension("jarjarquant._cython.bar_permute", [
              "jarjarquant/_cython/bar_permute.pyx"]),
    Extension("jarjarquant._cython.indicators", [
              "jarjarquant/_cython/indicators.pyx"]),
]

setup(
    packages=find_packages(),
    install_requires=[],
    ext_modules=cythonize(extensions, compiler_directives={
                          'language_level': "3"}),
    include_dirs=[np.get_include()],
    setup_requires=['Cython', 'numpy'],
)
