"""
This is a Python implementation of the Gibbs SeaWater (GSW) Oceanographic
Toolbox of TEOS-10.  Extensive documentation is available from
http://www.teos-10.org/; users of this Python package are strongly
encouraged to study the documents posted there.

This implementation is based on GSW-C for core functions, with
additional functions written in Python.  GSW-C is the
work of Frank Delahoyde and Glenn Hyland (author of GSW-Fortran,
on which GSW-C is based), who translated and re-implemented the
algorithms originally written in Matlab by David Jackett,
Trevor McDougall, and Paul Barker.

The present Python library has an interface that is similar to the
original Matlab code, but with a few important differences:

- Many functions in the GSW-Matlab toolbox are not yet available here.
- Taking advantage of Python namespaces, we omit the "gsw" prefix
  from the function names.
- Missing values may be handled using `numpy.ma` masked arrays, or
  using `nan` values.
- All functions follow numpy broadcasting rules; function arguments
  must be broadcastable to the dimensions of the highest-dimensioned
  argument.  Recall that with numpy broadcasting, extra dimensions
  are automatically added as needed on the left, but must be added
  explicitly as needed on the right.
- Functions such as `Nsquared` that operate on profiles rather than
  scalars have an `axis` keyword argument to specify the index that
  is incremented along the pressure (depth) axis.

"""


from ._fixed_wrapped_ufuncs import *

from .stability import *
from .geostrophy import *
from .utility import *
from . import geostrophy
from . import utility
from . import stability
from . import density
from . import energy
from . import conversions

from . import ice

from .conversions import t90_from_t68

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
