'''
Minimal setup.py for building gswc.
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

