'''
Minimal setup.py for building gswc.
'''


import os
import sys
import shutil

import pkg_resources
from setuptools import Extension, setup
from distutils.command.build_ext import build_ext as _build_ext


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

config = {
    "use_scm_version": {
        "write_to": "gsw/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
    "ext_modules": [Extension('gsw._gsw_ufuncs', ufunc_src_list)],
    "include_dirs": [os.path.join(rootpath, 'src', 'c_gsw')],
    "cmdclass": {"build_ext": build_ext},
}

setup(**config)
