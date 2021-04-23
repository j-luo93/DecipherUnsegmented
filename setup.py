from distutils.core import setup

import numpy as np
from setuptools import find_packages, setup

setup(
    name='du',
    version='0.0',
    packages=find_packages(),
    include_dirs=[np.get_include()]
)
