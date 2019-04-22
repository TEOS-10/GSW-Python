.. gsw documentation master file, created by
   sphinx-quickstart on Mon Mar 13 15:27:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GSW-Python
==========
This Python implementation of the Thermodynamic Equation of
Seawater 2010 (`TEOS-10 <http://TEOS-10.org>`__) is based
primarily on numpy ufunc wrappers of
the `GSW-C <https://github.com/TEOS-10/python-gsw/>`__ implementation.
We expect it to replace the original
`python-gsw <https://github.com/TEOS-10/python-gsw/>`__
pure-python implementation after a brief overlap period.
The primary reasons for this change are that by building on the
C implementation we reduce code duplication and we gain an immediate
update to the 75-term equation.  Additional benefits include a
major increase in speed, a reduction in memory usage, and the
inclusion of more functions.  The penalty is that a C (or MSVC C++ for
Windows) compiler is required to build the package from source.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   intro
   install
   gsw
   gsw_flat


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
