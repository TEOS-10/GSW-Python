'''
Minimal setup.py for building gswc.
'''

from __future__ import print_function

import os
import sys
import shutil

import pkg_resources
from setuptools import Extension, setup
from distutils.command.build_ext import build_ext as _build_ext

import versioneer


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


rootpath = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    return open(os.path.join(rootpath, *parts), 'r').read()


class build_ext(_build_ext):
    # Extention builder from pandas without the cython stuff
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

        for ext in self.extensions:
            if hasattr(ext, 'include_dirs') and not numpy_incl in ext.include_dirs:
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


LICENSE = read('LICENSE')
long_description = read('README.rst')

cmdclass = versioneer.get_cmdclass()
cmdclass.update({'build_ext': build_ext})

# MSVC can't handle C complex, and distutils doesn't seem to be able to
# let us force C++ compilation of .c files, so we use the following hack for
# Windows.
if sys.platform == 'win32':
    cext = 'cpp'
    shutil.copy('src/c_gsw/gsw_oceanographic_toolbox.c',
                'src/c_gsw/gsw_oceanographic_toolbox.cpp')
    shutil.copy('src/c_gsw/gsw_saar.c', 'src/c_gsw/gsw_saar.cpp')
else:
    cext = 'c'

ufunc_src_list = ['src/_ufuncs.c',
                  'src/c_gsw/gsw_oceanographic_toolbox.' + cext,
                  'src/c_gsw/gsw_saar.' + cext]

config = dict(
    name='gsw',
    version=versioneer.get_version(),
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
    install_requires=['numpy'],
    setup_requires=['numpy'],
    ext_modules=[Extension('gsw._gsw_ufuncs', ufunc_src_list)],
    include_dirs=[os.path.join(rootpath, 'src', 'c_gsw')],
    cmdclass=cmdclass,
)

setup(**config)
