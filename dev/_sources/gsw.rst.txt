Subpackages
===========

All functions are available in the base module namespace, and
via the index to these web pages.

Subsets of functions are grouped in subpackages, each of which corresponds
approximately to one or more of the groups in the table on pages
16-19 of http://www.teos-10.org/pubs/Getting_Started.pdf.  These
subpackages are particularly useful for finding functions using
tab-completion in IPython.

When importing functions in a module or script, however, it is safer
to import them directly
from the ``gsw`` namespace; it is more concise and future-proof;
the organization of the subpackages is subject to change.


.. toctree::
   :maxdepth: 4

   conversions
   density
   energy
   stability
   geostrophy
   ice
   freezing
