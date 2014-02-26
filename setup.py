import distutils
from distutils.core import setup, Extension
import distutils.ccompiler
import glob
import os
import os.path
import sys
import numpy as np

if sys.argv[1] == "clean":
    print "Cleaning files..."
    #os.system("rm -rf ./build/")
    #os.system("rm -rf mylapack/*.c")
    #os.system("rm -rf mylapack/*.so")

if sys.version_info > (3, 0):
    options["use_2to3"] = True

# This forces distutils to place the data files
# in the directory where the Py packages are installed
# (usually 'site-packages'). This is unfortunately
# required since there's no good way to retrieve
# data_files= from setup() in a platform independent
# way.
from distutils.command.install import INSTALL_SCHEMES
for scheme in INSTALL_SCHEMES.values():
        scheme['data'] = scheme['purelib']

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
      
# Include our headers and numpy's headers
include_dirs = [os.path.join(CURRENT_DIR, "include")]

# Lapack functions extension
# Source files
c_source_dir = os.path.join(CURRENT_DIR, "src")
lapack_sources = \
  glob.glob(os.path.join(c_source_dir, "lapack", "*.c"))
f2c_sources = \
  glob.glob(os.path.join(c_source_dir, "f2c", "*.c"))
blas_sources = \
  glob.glob(os.path.join(c_source_dir, "blas", "*.c"))
# Include numpy headers
all_c_sources = \
  lapack_sources + blas_sources + f2c_sources

lapack_ext = Extension("mylapack.lapack",
                       all_c_sources + ["mylapack/lapack.pyx"],
                       include_dirs=include_dirs)
my_extensions = [lapack_ext]

##
## Handle creation of source distribution. Here we definitely
## need to use Cython. This creates the *.c files from *.pyx
## files.
##
cmdclass = {}
from distutils.command.sdist import sdist as _sdist
class sdist(_sdist):
    """
    Override sdist command to use cython
    """
    def run(self):
        try:
            from Cython.Build import cythonize
        except ImportError:
            raise Exception, "Cannot create source distribution without Cython."
        print "Cythonizing"
        extensions = cythonize(my_extensions)
        _sdist.run(self)
cmdclass['sdist'] = sdist


##
## Handle Cython sources. Determine whether or not to use Cython
##
extensions = []
def no_cythonize(extensions, **_ignore):
    new_extensions = []
    for extension in extensions:
        ext_copy = \
          Extension(extension.name,
                    extension.sources,
                    include_dirs=extension.include_dirs,
                    library_dirs=extension.library_dirs)
        sources = []
        for sfile in ext_copy.sources:
            path, ext = os.path.splitext(sfile)
            new_sfile = sfile
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                new_sfile = path + ext
            sources.append(new_sfile)
        ext_copy.sources[:] = sources
        new_extensions.append(ext_copy)
    return new_extensions


# Whether or not to use Cython
USE_CYTHON = False

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


## Force to not use Cython, unless we're making
## a source distribution with 'sdist'
if sys.argv[1] == "sdist":
    USE_CYTHON = True
else:
    USE_CYTHON = False

if USE_CYTHON:
    print "Using Cython."
    extensions = cythonize(my_extensions)
    cmdclass.update({'build_ext': build_ext})
else:
    extensions = no_cythonize(my_extensions)
    from distutils.command import build_ext
    print "Not using Cython."


setup(name = 'cython_example',
      author = "Yarden Katz",
      author_email = "yarden@mit.edu",
      cmdclass = cmdclass,
      ext_modules = extensions,
      include_dirs = [np.get_include()],
      packages = ['mylapack'],
      install_requires = [
          "numpy >= 1.5.0",
          ],
      )

