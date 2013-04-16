from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(name = 'cython_example',
      author = 'Yarden Katz',
      # Cython extensions
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("scores", ["scores.pyx"])])
