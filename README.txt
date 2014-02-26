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
Traceback (most recent call last):
  File "run.py", line 5, in <module>
    import mylapack.lapack
ImportError: mylapack/lapack.so: undefined symbol: cdgemm

