/*
**  $Id: gsw_saar.c,v b04abca68ac0 2015/09/13 17:47:28 fdelahoyde $
**  $Version: 3.05.0-2 $
**
**  GSW TEOS-10 V3.05
*/
#include "gswteos-10.h"
#include "gsw_internal_const.h"
#include "gsw_saar_data.c"

#ifdef _MSC_VER
#   include <float.h>
#   if _MSC_VER < 1900
#       define isnan(x) _isnan(x)
#   endif
#endif

static double
gsw_sum(double *x, int n)
{
	int	i;
	double	val;

	for (val=0.0, i=0; i<n; val += x[i], i++);
	return (val);
}

#define max(a,b)	(((a)>(b))?(a):(b))
#define sum(x)		gsw_sum(x, sizeof (x)/sizeof (double))

/*
!==========================================================================
function gsw_saar(p,long,lat)
!==========================================================================

! Calculates the Absolute Salinity Anomaly Ratio, SAAR at a geographic point.
!
! p      : sea pressure                                    [dbar]
! long   : longitude                                       [deg E]
! lat    : latitude                                        [deg N]
!
! gsw_saar : Absolute Salinity Anomaly Ratio               [unitless]
*/
double
gsw_saar(double p, double lon, double lat)
{
	GSW_SAAR_DATA;
	int	nx=gsw_nx, ny=gsw_ny, nz=gsw_nz;
	int	indx0, indy0, indz0, k, ndepth_index;
	double	saar[4], saar_old[4];
	double	sa_upper, sa_lower, dlong, dlat;
	double	r1, s1, t1, ndepth_max, return_value;


	return_value	 = GSW_INVALID_VALUE;

    if (isnan(lat) || isnan(lon) || isnan(p))
        return (return_value);

	if (lat  <  -86.0  ||  lat  >  90.0)
	    return (return_value);

	if (lon  <  0.0)
	    lon	+= 360.0;

	dlong	= longs_ref[1]-longs_ref[0];
	dlat	= lats_ref[1]-lats_ref[0];

	indx0	= floor(0 + (nx-1)*(lon-longs_ref[0])/(longs_ref[nx-1]-
		    longs_ref[0]));
	if (indx0 == nx-1)
	    indx0	= nx-2;

	indy0 = floor(0 + (ny-1)*(lat-lats_ref[0])/(lats_ref[ny-1]-
		    lats_ref[0]));
	if(indy0 == ny-1)
	    indy0	= ny-2;
/*
! Look for the maximum valid "ndepth_ref" value around our point.
! Note: invalid "ndepth_ref" values are NaNs (a hangover from the codes
! Matlab origins), but we have replaced the NaNs with a value of "9e90",
! hence we need an additional upper-limit check in the code below so they
! will not be recognised as valid values.
*/
	ndepth_max	= -1.0;
	for (k=0; k < 4; k++) {
	    ndepth_index	= indy0+delj[k]+(indx0+deli[k])*ny;
	    if (ndepth_ref[ndepth_index] > 0.0 &&
	        ndepth_ref[ndepth_index] < 1e90)
		ndepth_max = max(ndepth_max, ndepth_ref[ndepth_index]);
	}
/*
! If we are a long way from the ocean then there will be no valid "ndepth_ref"
! values near the point (ie. surrounded by NaNs) - so just return SAAR = 0.0
*/
	if (ndepth_max == -1.0)
	    return (0.0);

	if (p > p_ref[(int)(ndepth_max)-1])
	    p	= p_ref[(int)(ndepth_max)-1];
	indz0	= gsw_util_indx(p_ref,nz,p);

	r1	= (lon-longs_ref[indx0])/(longs_ref[indx0+1]-longs_ref[indx0]);
	s1	= (lat-lats_ref[indy0])/(lats_ref[indy0+1]-lats_ref[indy0]);
	t1	= (p-p_ref[indz0])/(p_ref[indz0+1]-p_ref[indz0]);

	for (k=0; k<4; k++)
	    saar[k]	= saar_ref[indz0+nz*(indy0+delj[k]+(indx0+deli[k])*ny)];

	if (longs_pan[0] <= lon && lon <= longs_pan[npan-1]-0.001 &&
	    lats_pan[npan-1] <= lat && lat <= lats_pan[0]) {
	    memmove(saar_old,saar,4*sizeof (double));
	    gsw_add_barrier(saar_old,lon,lat,longs_ref[indx0],
			lats_ref[indy0],dlong,dlat,saar);
	} else if (fabs(sum(saar))  >=  GSW_ERROR_LIMIT) {
	    memmove(saar_old,saar,4*sizeof (double));
	    gsw_add_mean(saar_old,saar);
	}

	sa_upper	= (1.0-s1)*(saar[0] + r1*(saar[1]-saar[0])) +
			s1*(saar[3] + r1*(saar[2]-saar[3]));

	for (k=0; k<4; k++)
	    saar[k]	= saar_ref[indz0+1+nz*(indy0+delj[k]+
				(indx0+deli[k])*ny)];

	if (longs_pan[0] <= lon && lon <= longs_pan[npan-1]-0.001 &&
	    lats_pan[npan-1] <= lat && lat <= lats_pan[0]) {
	    memmove(saar_old,saar,4*sizeof (double));
	    gsw_add_barrier(saar_old,lon,lat,longs_ref[indx0],
				lats_ref[indy0],dlong,dlat,saar);
	} else if (fabs(sum(saar))  >=  GSW_ERROR_LIMIT) {
	    memmove(saar_old,saar,4*sizeof (double));
	    gsw_add_mean(saar_old,saar);
	}

	sa_lower	= (1.0-s1)*(saar[0] + r1*(saar[1]-saar[0])) +
				s1*(saar[3] + r1*(saar[2]-saar[3]));
	if (fabs(sa_lower)  >=  GSW_ERROR_LIMIT)
	    sa_lower	= sa_upper;
	return_value	= sa_upper + t1*(sa_lower-sa_upper);

	if (fabs(return_value) >= GSW_ERROR_LIMIT)
	    return_value	= GSW_INVALID_VALUE;

	return (return_value);
}

/*
!==========================================================================
function gsw_deltasa_atlas(p,lon,lat)
!==========================================================================

! Calculates the Absolute Salinity Anomaly atlas value, delta_SA_atlas.
!
! p      : sea pressure                                    [dbar]
! lon    : longiture                                       [deg E]
! lat    : latitude                                        [deg N]
!
! deltasa_atlas : Absolute Salinity Anomaly atlas value    [g/kg]
*/
double
gsw_deltasa_atlas(double p, double lon, double lat)
{
	GSW_SAAR_DATA;
	int	nx=gsw_nx, ny=gsw_ny, nz=gsw_nz;
	int	indx0, indy0, indz0, k, ndepth_index;
	double	dsar[4], dsar_old[4];
	double	dlong, dlat;
	double	return_value, sa_upper, sa_lower;
	double	r1, s1, t1, ndepth_max;

	return_value	= GSW_INVALID_VALUE;

    if (isnan(lat) || isnan(lon) || isnan(p))
        return (return_value);

	if (lat < -86.0  ||  lat  >  90.0)
	    return (return_value);

	if (lon < 0.0)
	    lon	+= 360.0;

	dlong	= longs_ref[1]-longs_ref[0];
	dlat	= lats_ref[1]-lats_ref[0];

	indx0	= floor(0 + (nx-1)*(lon-longs_ref[0])/
			(longs_ref[nx-1]-longs_ref[0]));
	if (indx0 == nx-1)
	    indx0	= nx-2;

	indy0	= floor(0 + (ny-1)*(lat-lats_ref[0])/
			(lats_ref[ny-1]-lats_ref[0]));
	if (indy0 == ny-1)
	    indy0	= ny-2;
/*
! Look for the maximum valid "ndepth_ref" value around our point.
! Note: invalid "ndepth_ref" values are NaNs (a hangover from the codes
! Matlab origins), but we have replaced the NaNs with a value of "9e90",
! hence we need an additional upper-limit check in the code below so they
! will not be recognised as valid values.
*/
	ndepth_max	= -1;
	for (k=0; k<4; k++) {
	    ndepth_index	= indy0+delj[k]+(indx0+deli[k])*ny;
	    if (ndepth_ref[ndepth_index] > 0.0 &&
	        ndepth_ref[ndepth_index] < 1e90)
		ndepth_max	= max(ndepth_max, ndepth_ref[ndepth_index]);
	}
/*
! If we are a long way from the ocean then there will be no valid "ndepth_ref"
! values near the point (ie. surrounded by NaNs) - so deltasa_atlas = 0.0
*/
	if (ndepth_max == -1.0)
	    return (0.0);

	if (p > p_ref[(int)(ndepth_max)-1])
	    p	= p_ref[(int)(ndepth_max)-1];
	indz0	= gsw_util_indx(p_ref,nz,p);

	r1	= (lon-longs_ref[indx0])/
			(longs_ref[indx0+1]-longs_ref[indx0]);
	s1	= (lat-lats_ref[indy0])/
			(lats_ref[indy0+1]-lats_ref[indy0]);
	t1	= (p-p_ref[indz0])/
			(p_ref[indz0+1]-p_ref[indz0]);

	for (k=0; k < 4; k++)
	    dsar[k]	= delta_sa_ref[indz0+nz*(indy0+delj[k]+
				(indx0+deli[k])*ny)];

	if (longs_pan[0] <= lon && lon <= longs_pan[npan-1]-0.001 &&
            lats_pan[npan-1] <= lat && lat <= lats_pan[0]) {
	    memmove(dsar_old,dsar,4*sizeof (double));
	    gsw_add_barrier(dsar_old,lon,lat,longs_ref[indx0],
				lats_ref[indy0],dlong,dlat,dsar);
	} else if (fabs(sum(dsar)) >= GSW_ERROR_LIMIT) {
	    memmove(dsar_old,dsar,4*sizeof (double));
	    gsw_add_mean(dsar_old,dsar);
	}

	sa_upper	= (1.0-s1)*(dsar[0] + r1*(dsar[1]-dsar[0])) +
				s1*(dsar[3] + r1*(dsar[2]-dsar[3]));

	for (k=0; k<4; k++)
	    dsar[k]	= delta_sa_ref[indz0+1+nz*(indy0+delj[k]+
				(indx0+deli[k])*ny)];

	if (longs_pan[0] <= lon && lon <= longs_pan[npan-1] &&
	    lats_pan[npan-1] <= lat && lat <= lats_pan[0]) {
	    memmove(dsar_old,dsar,4*sizeof (double));
	    gsw_add_barrier(dsar_old,lon,lat,longs_ref[indx0],
				lats_ref[indy0],dlong,dlat,dsar);
	} else if (fabs(sum(dsar)) >= GSW_ERROR_LIMIT) {
	    memmove(dsar_old,dsar,4*sizeof (double));
	    gsw_add_mean(dsar_old,dsar);
	}

	sa_lower	= (1.0-s1)*(dsar[0] + r1*(dsar[1]-dsar[0])) +
				s1*(dsar[3] + r1*(dsar[2]-dsar[3]));
	if (fabs(sa_lower) >= GSW_ERROR_LIMIT)
	    sa_lower	= sa_upper;
	return_value	= sa_upper + t1*(sa_lower-sa_upper);

	if (fabs(return_value) >= GSW_ERROR_LIMIT)
	    return (GSW_INVALID_VALUE);

	return (return_value);
}
/*
**  The End
*/
