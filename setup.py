import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext_modules = Extension("artsm.utils.angles", ["artsm/utils/angles.pyx"], include_dirs=[np.get_include()])

setup(
    name="artsm",
    version="1.0",
    packages=find_packages(),

    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(ext_modules),
)
