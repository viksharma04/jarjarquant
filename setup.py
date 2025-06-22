from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("jarjarquant.cython_utils.opt_threshold", [
              "jarjarquant/cython_utils/opt_threshold.pyx"]),
    Extension("jarjarquant.cython_utils.bar_permute", [
              "jarjarquant/cython_utils/bar_permute.pyx"]),
    Extension("jarjarquant.cython_utils.indicators", [
              "jarjarquant/cython_utils/indicators.pyx"]),
]

setup(
    packages=find_packages(),
    install_requires=[],
    ext_modules=cythonize(extensions, compiler_directives={
                          'language_level': "3"}),
    include_dirs=[np.get_include()],
    setup_requires=['Cython', 'numpy'],
)
