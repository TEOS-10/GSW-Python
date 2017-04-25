'''
Minimal setup.py for building gswc.
'''

from __future__ import print_function

import os
import sys

from setuptools import Extension, setup

import numpy as np

# Check Python version.
if sys.version_info < (3, 5):
    pip_message = ('This may be due to an out of date pip. '
                   'Make sure you have pip >= 9.0.1.')
    try:
        import pip
        pip_version = tuple([int(x) for x in pip.__version__.split('.')[:3]])
        if pip_version < (9, 0, 1):
            pip_message = ('Your pip version is out of date, '
                           'please install pip >= 9.0.1. '
                           'pip {} detected.').format(pip.__version__)
        else:
            # pip is new enough - it must be something else.
            pip_message = ''
    except Exception:
        pass

    error = """
Latest gsw does not support Python < 3.5.
When using Python 2.7 please install the last pure Python version
of gsw available at PyPI (3.0.6).
Python {py} detected.
{pip}
""".format(py=sys.version_info, pip=pip_message)

    print(error, file=sys.stderr)
    sys.exit(1)

if os.name in ('nt', 'dos'):
    srcdir = 'src2'
    c_ext = 'cpp'
else:
    srcdir = 'src'
    c_ext = 'c'


rootpath = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    return open(os.path.join(rootpath, *parts), 'r').read()


def extract_version():
    version = None
    fname = os.path.join(rootpath, 'gsw', '__init__.py')
    with open(fname) as f:
        for line in f:
            if (line.startswith('__version__')):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Remove quotation characters
                break
    return version

LICENSE = read('LICENSE')
long_description = read('README.rst')

config = dict(
    name='gsw',
    version=extract_version(),
    packages=['gsw'],
    author=['Eric Firing', 'Filipe Fernandes'],
    author_email='efiring@hawaii.edu',
    description='Gibbs Seawater Oceanographic Package of TEOS-10',
    long_description=long_description,
    license=LICENSE,
    url='https://github.com/TEOS-10/GSW-python',
    download_url='https://pypi.python.org/pypi/gsw/',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.5',
    platforms='any',
    keywords=['oceanography', 'seawater', 'TEOS-10'],
    setup_requires=['numpy'],
    ext_modules=[
        Extension('gsw._gsw_ufuncs',
                  [srcdir + '/_ufuncs.c',
                   srcdir + '/c_gsw/gsw_oceanographic_toolbox.' + c_ext,
                   srcdir + '/c_gsw/gsw_saar.' + c_ext])],
    include_dirs=[np.get_include(),
                  os.path.join(rootpath, srcdir, 'c_gsw')],
)

setup(**config)
