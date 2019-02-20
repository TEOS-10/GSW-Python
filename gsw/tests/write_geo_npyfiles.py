"""
Generate our own test values for geostrophic calculations.

We are presently using simple pchip interpolation for
geostrophy rather than trying to mimic the ever-changing
Matlab functions. Therefore we have to make our own test
values, using the current test cast inputs.

This is a minimal script for that purpose, to be run in
the tests directory in which it lives.  It should be run
only if we change to a different calculation algorithm,
or we update the cast input and general check value file.
"""

import numpy as np
import gsw
from gsw._utilities import Bunch

cv = Bunch(np.load('gsw_cv_v3_0.npz'))

dyn_height = gsw.geo_strf_dyn_height(cv.SA_chck_cast,
                                     cv.CT_chck_cast,
                                     cv.p_chck_cast,
                                     cv.pr)
np.save('geo_strf_dyn_height.npy', dyn_height)

lon = cv.long_chck_cast
lat = cv.lat_chck_cast
p = cv.p_chck_cast
CT = cv.CT_chck_cast
SA = cv.SA_chck_cast
strf = gsw.geo_strf_dyn_height(SA, CT, p)
geovel, midlon, midlat = gsw.geostrophic_velocity(strf, lon, lat)
np.save('geo_strf_velocity.npy', geovel)
# midlon, midlat are OK
