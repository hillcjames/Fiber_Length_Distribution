'''

This should build and 

Created on Apr 7, 2016

@author: james
'''

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Image stuff in C',
  ext_modules = cythonize("./quickFuncs.pyx"),
)