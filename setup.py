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
    name='jarjarquant',
    version='0.1.0',
    description='Implements common financial ML techniques',
    author='Vik Sharma',
    url='https://github.com/viksharma04/jarjarquant.git',
    packages=find_packages(),
    install_requires=[],
    ext_modules=cythonize(extensions, compiler_directives={
                          'language_level': "3"}),
    include_dirs=[np.get_include()],
    setup_requires=['Cython', 'numpy'],
)
