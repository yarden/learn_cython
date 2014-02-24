from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(name = 'cython_example',
      author = 'Yarden Katz',
      # Cython extensions
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("scores", ["scores.pyx"])])


# from distutils.core import setup
# from Cython.Build import cythonize

# ext = Extension(
#         "cpp_test",                 # name of extension
#             ["cpp_test.pyx",
#              "Rectangle.cpp"],           # filename of our Pyrex/Cython source
#                 language="c++",              # this causes Pyrex/Cython to create C++ source
#                                 cmdclass = {'build_ext': build_ext}
#                                     )

# setup(name="call_cpp",
#       cmdclass = {'build_ext': build_ext},
#       ext_modules = [ext])

