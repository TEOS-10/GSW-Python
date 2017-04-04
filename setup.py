'''
Minimal setup.py for building gswc.
'''
import os

from setuptools import Extension, setup

import numpy as np

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
long_description = read('README')

config = dict(
    name='gsw',
    version=extract_version(),
    packages=['gsw', 'gsw/tests', 'gsw/ice'],
    author=['Eric Firing', 'Filipe Fernandes'],
    author_email='efiring@hawaii.edu',
    description='Gibbs Seawater Oceanographic Package of TEOS-10',
    long_description=long_description,
    license=LICENSE,
    # url='https://github.com/TEOS-10/GSW-python',
    # download_url='https://pypi.python.org/pypi/gsw/',
    platforms='any',
    keywords=['oceanography', 'seawater', 'TEOS-10'],
    setup_requires=['numpy'],
    ext_modules=[
        Extension('_gsw_ufuncs',
                  ['src/_ufuncs.c',
                   'src/c_gsw/gsw_oceanographic_toolbox.c',
                   'src/c_gsw/gsw_saar.c'])],
    include_dirs=[np.get_include(),
                  os.path.join(rootpath, 'src', 'c_gsw')],
)

setup(**config)
