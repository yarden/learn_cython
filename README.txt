Run the example using:

Step 1:

# Run Cython to get the *.c files
$ python setup.py sdist

Step 2: 

# Compile/install package
$ python setup.py build_ext

or alternatively:

$ pip install .

Step 3:

# Run code
$ python run.py
importing mylapack
<module 'mylapack' from 'mylapack/__init__.pyc'>
LAPACK interface
Calling dgemm
** On entry to DGEMM , parameter number 10 had an illegal value
