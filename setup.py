from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(name='MultiMarketABM',
      packages=['mmabm'],
      ext_modules=cythonize('**/*.pyx', compiler_directives={'language_level': 3}),
      include_dirs=[numpy.get_include()],
      )
