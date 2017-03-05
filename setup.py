'''
    setup.py file for logit.c
    Note that since this is a numpy extension
    we use numpy.distutils instead of
    distutils from the python standard library.

    Calling
    $python setup.py build_ext --inplace
    will build the extension library in the current file.

    Calling
    $python setup.py build
    will build a file that looks like ./build/lib*, where
    lib* is a file that begins with lib. The library will
    be in this file and end with a C library extension,
    such as .so

    Calling
    $python setup.py install
    will install the module in your site-packages file.

    See the distutils section of
    'Extending and Embedding the Python Interpreter'
    at docs.python.org  and the documentation
    on numpy.distutils for more information.
'''

import numpy
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup


attributes = dict(name='gswc',
                  packages=['gswc'],
                  package_dir={'gswc':'gsw'},
                  description="Python-wrapped Gibbs Seawater Toolkit",
                  )

def configuration(parent_package='', top_path=None):

    config = Configuration('gswc',
                           parent_package,
                           top_path,
                           **attributes)

    config.add_extension('_gsw_ufuncs',
                         ['src/_ufuncs.c',
                          'src/c_gsw/gsw_oceanographic_toolbox.c',
                          'src/c_gsw/gsw_saar.c',
                          ],
                         include_dirs=['src/c_gsw'],
                         )

    return config

if __name__ == "__main__":
    setup(configuration=configuration)

