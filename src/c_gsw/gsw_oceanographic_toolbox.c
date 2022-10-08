/*
**  $Id: gsw_oceanographic_toolbox-head,v c61271a7810d 2016/08/19 20:04:03 fdelahoyde $
**  Version: 3.05.0-3
**
**  This is a translation of the original f90 source code into C
**  by the Shipboard Technical Support Computing Resources group
**  at Scripps Institution of Oceanography -- sts-cr@sio.ucsd.edu.
**  The original notices follow.
**
==========================================================================
 Gibbs SeaWater (GSW) Oceanographic Toolbox of TEOS-10 (Fortran)
==========================================================================

 This is a subset of functions contained in the Gibbs SeaWater (GSW)
 Oceanographic Toolbox of TEOS-10.

 Version 1.0 written by David Jackett
 Modified by Paul Barker (version 3.02)
 Modified by Glenn Hyland (version 3.04+)

 For help with this Oceanographic Toolbox email: help@teos-10.org

 This software is available from http://www.teos-10.org

==========================================================================

 gsw_data_v3_0.nc
 NetCDF file that contains the global data set of Absolute Salinity Anomaly
 Ratio, the global data set of Absolute Salinity Anomaly atlas, and check
 values and computation accuracy values for use in gsw_check_function.
 The data set gsw_data_v3_0.nc must not be tampered with.

 gsw_check_function.f90
 Contains the check functions. We suggest that after downloading, unzipping
 and installing the toolbox the user runs this program to ensure that the
 toolbox is installed correctly and there are no conflicts. This toolbox has
 been tested to compile and run with gfortran.

 cd test
 make
 ./gsw_check

 Note that gfortran is the name of the GNU Fortran project, developing a
 free Fortran 95/2003/2008 compiler for GCC, the GNU Compiler Collection.
 It is available from http://gcc.gnu.org/fortran/

==========================================================================
*/
#include "gswteos-10.h"

#ifdef __cplusplus
#       define DCOMPLEX std::complex<double>
#else
#   define DCOMPLEX double complex
#       define real(x) creal(x)
#       define log(x) clog(x)
#endif

#include "gsw_internal_const.h"


/*
!==========================================================================
subroutine gsw_add_barrier(input_data,lon,lat,long_grid,lat_grid,dlong_grid,dlat_grid,output_data)
!==========================================================================

!  Adds a barrier through Central America (Panama) and then averages
!  over the appropriate side of the barrier
!
!  data_in      :  data                                         [unitless]
!  lon          :  Longitudes of data degrees east              [0 ... +360]
!  lat          :  Latitudes of data degrees north              [-90 ... +90]
!  longs_grid   :  Longitudes of regular grid degrees east      [0 ... +360]
!  lats_grid    :  Latitudes of regular grid degrees north      [-90 ... +90]
!  dlongs_grid  :  Longitude difference of regular grid degrees [deg longitude]
!  dlats_grid   :  Latitude difference of regular grid degrees  [deg latitude]
!
!  output_data  : average of data depending on which side of the
!                 Panama canal it is on                         [unitless]
*/
void
gsw_add_barrier(double *input_data, double lon, double lat,
                double long_grid, double lat_grid, double dlong_grid,
                double dlat_grid, double *output_data)
{
        GSW_SAAR_DATA;
        int     above_line[4];
        int     k, nmean, above_line0, kk;
        double  r, lats_line, data_mean;

        k               = gsw_util_indx(longs_pan,npan,lon);
                        /*   the lon/lat point */
        r               = (lon-longs_pan[k])/(longs_pan[k+1]-longs_pan[k]);
        lats_line       = lats_pan[k] + r*(lats_pan[k+1]-lats_pan[k]);

        above_line0     = (lats_line <= lat);

        k               = gsw_util_indx(longs_pan,npan,long_grid);
                        /*the 1 & 4 lon/lat points*/
        r               = (long_grid-longs_pan[k])/
                                (longs_pan[k+1]-longs_pan[k]);
        lats_line       = lats_pan[k] + r*(lats_pan[k+1]-lats_pan[k]);

        above_line[0]   = (lats_line <= lat_grid);
        above_line[3]   = (lats_line <= lat_grid+dlat_grid);

        k               = gsw_util_indx(longs_pan,6,long_grid+dlong_grid);
                        /*the 2 & 3 lon/lat points */
        r               = (long_grid+dlong_grid-longs_pan[k])/
                        (longs_pan[k+1]-longs_pan[k]);
        lats_line       = lats_pan[k] + r*(lats_pan[k+1]-lats_pan[k]);

        above_line[1]   = (lats_line <= lat_grid);
        above_line[2]   = (lats_line <= lat_grid+dlat_grid);

        nmean           = 0;
        data_mean       = 0.0;

        for (kk=0; kk<4; kk++) {
            if ((fabs(input_data[kk]) <= 100.0) &&
                above_line0 == above_line[kk]) {
                nmean   = nmean+1;
                data_mean       = data_mean+input_data[kk];
            }
        }
        if (nmean == 0)
            data_mean   = 0.0;  /*errorreturn*/
        else
            data_mean   = data_mean/nmean;

        for (kk=0; kk<4; kk++) {
            if ((fabs(input_data[kk]) >= 1.0e10) ||
                above_line0 != above_line[kk])
                output_data[kk] = data_mean;
            else
                output_data[kk] = input_data[kk];
        }

        return;
}
/*
!==========================================================================
subroutine gsw_add_mean(data_in,data_out)
!==========================================================================

! Replaces NaN's with non-nan mean of the 4 adjacent neighbours
!
! data_in   : data set of the 4 adjacent neighbours
!
! data_out : non-nan mean of the 4 adjacent neighbours     [unitless]
*/
void
gsw_add_mean(double *data_in, double *data_out)
{
        int     k, nmean;
        double  data_mean;

        nmean           = 0;
        data_mean       = 0.0;

        for (k=0; k<4; k++) {
            if (fabs(data_in[k]) <= 100.0) {
                nmean++;
                data_mean       = data_mean+data_in[k];
            }
        }

        if (nmean == 0.0)
            data_mean   = 0.0;    /*errorreturn*/
        else
            data_mean   = data_mean/nmean;

        for (k=0; k<4; k++) {
            if (fabs(data_in[k]) >= 100.0)
                data_out[k]     = data_mean;
            else
                data_out[k]     = data_in[k];
        }
        return;
}
/*
!==========================================================================
function gsw_adiabatic_lapse_rate_from_ct(sa,ct,p)
!==========================================================================

! Calculates the adiabatic lapse rate from Conservative Temperature
!
! sa     : Absolute Salinity                                 [g/kg]
! ct     : Conservative Temperature                          [deg C]
! p      : sea pressure                                      [dbar]
!
! gsw_adiabatic_lapse_rate_from_ct : adiabatic lapse rate    [K/Pa]
*/
double
gsw_adiabatic_lapse_rate_from_ct(double sa, double ct, double p)
{
        int     n0=0, n1=1, n2=2;
        double  pt0, pr0=0.0, t;

        pt0     = gsw_pt_from_ct(sa,ct);
        t       = gsw_pt_from_t(sa,pt0,pr0,p);

        return (-gsw_gibbs(n0,n1,n1,sa,t,p)/gsw_gibbs(n0,n2,n0,sa,t,p));

}
/*
!==========================================================================
elemental function gsw_adiabatic_lapse_rate_ice (t, p)
!==========================================================================
!
!  Calculates the adiabatic lapse rate of ice.
!
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!    Note.  The output is in unit of degrees Celsius per Pa,
!      (or equivilently K/Pa) not in units of K/dbar.
!--------------------------------------------------------------------------
*/
double
gsw_adiabatic_lapse_rate_ice(double t, double p)
{
        return (-gsw_gibbs_ice(1,1,t,p)/gsw_gibbs_ice(2,0,t,p));
}
/*
!==========================================================================
function gsw_alpha(sa,ct,p)
!==========================================================================

!  Calculates the thermal expansion coefficient of seawater with respect to
!  Conservative Temperature using the computationally-efficient 48-term
!  expression for density in terms of SA, CT and p (IOC et al., 2010)
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature                        [deg C]
! p      : sea pressure                                    [dbar]
!
! gsw_alpha : thermal expansion coefficient of seawater (48 term equation)
*/
double
gsw_alpha(double sa, double ct, double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  xs, ys, z, v_ct_part;

        xs      = sqrt(gsw_sfac*sa + offset);
        ys      = ct*0.025;
        z       = p*1e-4;

        v_ct_part = a000
                + xs*(a100 + xs*(a200 + xs*(a300 + xs*(a400 + a500*xs))))
                + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 + a410*xs)))
                + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs)) + ys*(a030
                + xs*(a130 + a230*xs) + ys*(a040 + a140*xs + a050*ys ))))
                + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 + a401*xs)))
                + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) + ys*(a021
                + xs*(a121 + a221*xs) + ys*(a031 + a131*xs + a041*ys)))
                + z*(a002 + xs*(a102 + xs*(a202 + a302*xs)) + ys*(a012
                + xs*(a112 + a212*xs) + ys*(a022 + a122*xs + a032*ys))
                + z*(a003 + a103*xs + a013*ys + a004*z)));

        return (0.025*v_ct_part/gsw_specvol(sa,ct,p));
}
/*
!==========================================================================
function gsw_alpha_on_beta(sa,ct,p)
!==========================================================================

!  Calculates alpha divided by beta, where alpha is the thermal expansion
!  coefficient and beta is the saline contraction coefficient of seawater
!  from Absolute Salinity and Conservative Temperature.  This function uses
!  the computationally-efficient expression for specific volume in terms of
!  SA, CT and p (Roquet et al., 2014).
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature                        [deg C]
! p      : sea pressure                                    [dbar]
!
! alpha_on_beta
!        : thermal expansion coefficient with respect to   [kg g^-1 K^-1]
!          Conservative Temperature divided by the saline
!          contraction coefficient at constant Conservative
!          Temperature
*/
double
gsw_alpha_on_beta(double sa, double ct, double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  xs, ys, z, v_ct_part, v_sa_part;

        xs      = sqrt(gsw_sfac*sa + offset);
        ys      = ct*0.025;
        z       = p*1e-4;

        v_ct_part = a000
                + xs*(a100 + xs*(a200 + xs*(a300 + xs*(a400 + a500*xs))))
                + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 + a410*xs)))
                + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs)) + ys*(a030
                + xs*(a130 + a230*xs) + ys*(a040 + a140*xs + a050*ys ))))
                + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 + a401*xs)))
                + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) + ys*(a021
                + xs*(a121 + a221*xs) + ys*(a031 + a131*xs + a041*ys)))
                + z*(a002 + xs*(a102 + xs*(a202 + a302*xs)) + ys*(a012
                + xs*(a112 + a212*xs) + ys*(a022 + a122*xs + a032*ys))
                + z*(a003 + a103*xs + a013*ys + a004*z)));

        v_sa_part = b000
                + xs*(b100 + xs*(b200 + xs*(b300 + xs*(b400 + b500*xs))))
                + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 + b410*xs)))
                + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs)) + ys*(b030
                + xs*(b130 + b230*xs) + ys*(b040 + b140*xs + b050*ys))))
                + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 + b401*xs)))
                + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) + ys*(b021
                + xs*(b121 + b221*xs) + ys*(b031 + b131*xs + b041*ys)))
                + z*(b002 + xs*(b102 + xs*(b202 + b302*xs))+ ys*(b012
                + xs*(b112 + b212*xs) + ys*(b022 + b122*xs + b032*ys))
                + z*(b003 +  b103*xs + b013*ys + b004*z)));

        return (-(v_ct_part*xs)/(20.0*gsw_sfac*v_sa_part));
}
/*
!==========================================================================
function gsw_alpha_wrt_t_exact(sa,t,p)
!==========================================================================

! Calculates thermal expansion coefficient of seawater with respect to
! in-situ temperature
!
! sa     : Absolute Salinity                               [g/kg]
! t      : insitu temperature                              [deg C]
! p      : sea pressure                                    [dbar]
!
! gsw_alpha_wrt_t_exact : thermal expansion coefficient    [1/K]
!                         wrt (in-situ) temperature
*/
double
gsw_alpha_wrt_t_exact(double sa, double t, double p)
{
        int     n0=0, n1=1;

        return (gsw_gibbs(n0,n1,n1,sa,t,p)/gsw_gibbs(n0,n0,n1,sa,t,p));
}
/*
!==========================================================================
elemental function gsw_alpha_wrt_t_ice (t, p)
!==========================================================================
!
!  Calculates the thermal expansion coefficient of ice with respect to
!  in-situ temperature.
!
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  alpha_wrt_t_ice  =  thermal expansion coefficient of ice with respect
!                      to in-situ temperature                       [ 1/K ]
!--------------------------------------------------------------------------
*/
double
gsw_alpha_wrt_t_ice(double t, double p)
{
        return (gsw_gibbs_ice(1,1,t,p)/gsw_gibbs_ice(0,1,t,p));
}
/*
!==========================================================================
function gsw_beta(sa,ct,p)
!==========================================================================

!  Calculates the saline (i.e. haline) contraction coefficient of seawater
!  at constant Conservative Temperature using the computationally-efficient
!  expression for specific volume in terms of SA, CT and p
!  (Roquet et al., 2014).
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature (ITS-90)               [deg C]
! p      : sea pressure                                    [dbar]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
! beta   : saline contraction coefficient of seawater      [kg/g]
!          at constant Conservative Temperature
*/
double
gsw_beta(double sa, double ct, double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  xs, ys, z, v_sa_part;

        xs      = sqrt(gsw_sfac*sa + offset);
        ys      = ct*0.025;
        z       = p*1e-4;

        v_sa_part = b000
                + xs*(b100 + xs*(b200 + xs*(b300 + xs*(b400 + b500*xs))))
                + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 + b410*xs)))
                + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs)) + ys*(b030
                + xs*(b130 + b230*xs) + ys*(b040 + b140*xs + b050*ys))))
                + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 + b401*xs)))
                + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) + ys*(b021
                + xs*(b121 + b221*xs) + ys*(b031 + b131*xs + b041*ys)))
                + z*(b002 + xs*(b102 + xs*(b202 + b302*xs))+ ys*(b012
                + xs*(b112 + b212*xs) + ys*(b022 + b122*xs + b032*ys))
                + z*(b003 +  b103*xs + b013*ys + b004*z)));

        return (-v_sa_part*0.5*gsw_sfac/(gsw_specvol(sa,ct,p)*xs));
}
/*
!==========================================================================
function gsw_beta_const_t_exact(sa,t,p)
!==========================================================================

! Calculates saline (haline) contraction coefficient of seawater at
! constant in-situ temperature.
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
!
! beta_const_t_exact : haline contraction coefficient      [kg/g]
*/
double
gsw_beta_const_t_exact(double sa, double t, double p)
{
        int     n0=0, n1=1;

        return (-gsw_gibbs(n1,n0,n1,sa,t,p)/gsw_gibbs(n0,n0,n1,sa,t,p));
}
/*
!==========================================================================
function gsw_c_from_sp(sp,t,p)
!==========================================================================

!  Calculates conductivity, C, from (SP,t,p) using PSS-78 in the range
!  2 < SP < 42.  If the input Practical Salinity is less than 2 then a
!  modified form of the Hill et al. (1986) fomula is used for Practical
!  Salinity.  The modification of the Hill et al. (1986) expression is to
!  ensure that it is exactly consistent with PSS-78 at SP = 2.
!
!  The conductivity ratio returned by this function is consistent with the
!  input value of Practical Salinity, SP, to 2x10^-14 psu over the full
!  range of input parameters (from pure fresh water up to SP = 42 psu).
!  This error of 2x10^-14 psu is machine precision at typical seawater
!  salinities.  This accuracy is achieved by having four different
!  polynomials for the starting value of Rtx (the square root of Rt) in
!  four different ranges of SP, and by using one and a half iterations of
!  a computationally efficient modified Newton-Raphson technique (McDougall
!  and Wotherspoon, 2012) to find the root of the equation.
!
!  Note that strictly speaking PSS-78 (Unesco, 1983) defines Practical
!  Salinity in terms of the conductivity ratio, R, without actually
!  specifying the value of C(35,15,0) (which we currently take to be
!  42.9140 mS/cm).
!
! sp     : Practical Salinity                               [unitless]
! t      : in-situ temperature [ITS-90]                     [deg C]
! p      : sea pressure                                     [dbar]
!
! c      : conductivity                                     [ mS/cm ]
*/
double
gsw_c_from_sp(double sp, double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SP_COEFFICIENTS;
        double  p0 = 4.577801212923119e-3,      p1 = 1.924049429136640e-1,
                p2 = 2.183871685127932e-5,      p3 = -7.292156330457999e-3,
                p4 = 1.568129536470258e-4,      p5 = -1.478995271680869e-6,
                p6 = 9.086442524716395e-4,      p7 = -1.949560839540487e-5,
                p8 = -3.223058111118377e-6,     p9 = 1.175871639741131e-7,
                p10 = -7.522895856600089e-5,    p11 = -2.254458513439107e-6,
                p12 = 6.179992190192848e-7,     p13 = 1.005054226996868e-8,
                p14 = -1.923745566122602e-9,    p15 = 2.259550611212616e-6,
                p16 = 1.631749165091437e-7,     p17 = -5.931857989915256e-9,
                p18 = -4.693392029005252e-9,    p19 = 2.571854839274148e-10,
                p20 = 4.198786822861038e-12,
                q0 = 5.540896868127855e-5,      q1 = 2.015419291097848e-1,
                q2 = -1.445310045430192e-5,     q3 = -1.567047628411722e-2,
                q4 = 2.464756294660119e-4,      q5 = -2.575458304732166e-7,
                q6 = 5.071449842454419e-3,      q7 = -9.081985795339206e-5,
                q8 = -3.635420818812898e-6,     q9 = 2.249490528450555e-8,
                q10 = -1.143810377431888e-3,    q11 = 2.066112484281530e-5,
                q12 = 7.482907137737503e-7,     q13 = 4.019321577844724e-8,
                q14 = -5.755568141370501e-10,   q15 = 1.120748754429459e-4,
                q16 = -2.420274029674485e-6,    q17 = -4.774829347564670e-8,
                q18 = -4.279037686797859e-9,    q19 = -2.045829202713288e-10,
                q20 = 5.025109163112005e-12,
                s0 = 3.432285006604888e-3,      s1 = 1.672940491817403e-1,
                s2 = 2.640304401023995e-5,      s3 = 1.082267090441036e-1,
                s4 = -6.296778883666940e-5,     s5 = -4.542775152303671e-7,
                s6 = -1.859711038699727e-1,     s7 = 7.659006320303959e-4,
                s8 = -4.794661268817618e-7,     s9 = 8.093368602891911e-9,
                s10 = 1.001140606840692e-1,     s11 = -1.038712945546608e-3,
                s12 = -6.227915160991074e-6,    s13 = 2.798564479737090e-8,
                s14 = -1.343623657549961e-10,   s15 = 1.024345179842964e-2,
                s16 = 4.981135430579384e-4,     s17 = 4.466087528793912e-6,
                s18 = 1.960872795577774e-8,     s19 = -2.723159418888634e-10,
                s20 = 1.122200786423241e-12,
                u0 = 5.180529787390576e-3,      u1 = 1.052097167201052e-3,
                u2 = 3.666193708310848e-5,      u3 = 7.112223828976632e0,
                u4 = -3.631366777096209e-4,     u5 = -7.336295318742821e-7,
                u6 = -1.576886793288888e+2,     u7 = -1.840239113483083e-3,
                u8 = 8.624279120240952e-6,      u9 = 1.233529799729501e-8,
                u10 = 1.826482800939545e+3,     u11 = 1.633903983457674e-1,
                u12 = -9.201096427222349e-5,    u13 = -9.187900959754842e-8,
                u14 = -1.442010369809705e-10,   u15 = -8.542357182595853e+3,
                u16 = -1.408635241899082e0,     u17 = 1.660164829963661e-4,
                u18 = 6.797409608973845e-7,     u19 = 3.345074990451475e-10,
                u20 = 8.285687652694768e-13;

        double  t68, ft68, x, rtx=0.0, dsp_drtx, sqrty,
                part1, part2, hill_ratio, sp_est,
                rtx_old, rt, aa, bb, cc, dd, ee, ra,r, rt_lc, rtxm,
                sp_hill_raw;

        t68     = t*1.00024e0;
        ft68    = (t68 - 15e0)/(1e0 + k*(t68 - 15e0));

        x       = sqrt(sp);

    /*
     |--------------------------------------------------------------------------
     ! Finding the starting value of Rtx, the square root of Rt, using four
     ! different polynomials of SP and t68.
     !--------------------------------------------------------------------------
    */

        if (sp >= 9.0) {
            rtx = p0 + x*(p1 + p4*t68 + x*(p3 + p7*t68 + x*(p6
                  + p11*t68 + x*(p10 + p16*t68 + x*p15))))
                  + t68*(p2+ t68*(p5 + x*x*(p12 + x*p17) + p8*x
                  + t68*(p9 + x*(p13 + x*p18)+ t68*(p14 + p19*x + p20*t68))));
        } else if (sp >= 0.25 && sp < 9.0) {
            rtx = q0 + x*(q1 + q4*t68 + x*(q3 + q7*t68 + x*(q6
                  + q11*t68 + x*(q10 + q16*t68 + x*q15))))
                  + t68*(q2+ t68*(q5 + x*x*(q12 + x*q17) + q8*x
                  + t68*(q9 + x*(q13 + x*q18)+ t68*(q14 + q19*x + q20*t68))));
        } else if (sp >= 0.003 && sp < 0.25) {
            rtx =  s0 + x*(s1 + s4*t68 + x*(s3 + s7*t68 + x*(s6
                  + s11*t68 + x*(s10 + s16*t68 + x*s15))))
                  + t68*(s2+ t68*(s5 + x*x*(s12 + x*s17) + s8*x
                  + t68*(s9 + x*(s13 + x*s18)+ t68*(s14 + s19*x + s20*t68))));
        } else if (sp < 0.003) {
            rtx =  u0 + x*(u1 + u4*t68 + x*(u3 + u7*t68 + x*(u6
                  + u11*t68 + x*(u10 + u16*t68 + x*u15))))
                  + t68*(u2+ t68*(u5 + x*x*(u12 + x*u17) + u8*x
                  + t68*(u9 + x*(u13 + x*u18)+ t68*(u14 + u19*x + u20*t68))));
        }

    /*
     !--------------------------------------------------------------------------
     ! Finding the starting value of dSP_dRtx, the derivative of SP with respect
     ! to Rtx.
     !--------------------------------------------------------------------------
    */
        dsp_drtx        =  a1 + (2e0*a2 + (3e0*a3 +
                                (4e0*a4 + 5e0*a5*rtx)*rtx)*rtx)*rtx
                          + ft68*(b1 + (2e0*b2 + (3e0*b3 + (4e0*b4 +
                                5e0*b5*rtx)*rtx)*rtx)*rtx);

        if (sp < 2.0) {
            x           = 400e0*(rtx*rtx);
            sqrty       = 10.0*rtx;
            part1       = 1e0 + x*(1.5e0 + x);
            part2       = 1e0 + sqrty*(1e0 + sqrty*(1e0 + sqrty));
            hill_ratio  = gsw_hill_ratio_at_sp2(t);
            dsp_drtx    = dsp_drtx
                          + a0*800e0*rtx*(1.5e0 + 2e0*x)/(part1*part1)
                          + b0*ft68*(10e0 + sqrty*(20e0 + 30e0*sqrty))/
                                (part2*part2);
            dsp_drtx    = hill_ratio*dsp_drtx;
        }

    /*
     !--------------------------------------------------------------------------
     ! One iteration through the modified Newton-Raphson method (McDougall and
     ! Wotherspoon, 2012) achieves an error in Practical Salinity of about
     ! 10^-12 for all combinations of the inputs.  One and a half iterations of
     ! the modified Newton-Raphson method achevies a maximum error in terms of
     ! Practical Salinity of better than 2x10^-14 everywhere.
     !
     ! We recommend one and a half iterations of the modified Newton-Raphson
     ! method.
     !
     ! Begin the modified Newton-Raphson method.
     !--------------------------------------------------------------------------
    */
        sp_est  = a0 + (a1 + (a2 + (a3 + (a4 + a5*rtx)*rtx)*rtx)*rtx)*rtx
                + ft68*(b0 + (b1 + (b2+ (b3 + (b4 + b5*rtx)*rtx)*rtx)*rtx)*rtx);
        if (sp_est <  2.0) {
            x           = 400e0*(rtx*rtx);
            sqrty       = 10e0*rtx;
            part1       = 1e0 + x*(1.5e0 + x);
            part2       = 1e0 + sqrty*(1e0 + sqrty*(1e0 + sqrty));
            sp_hill_raw = sp_est - a0/part1 - b0*ft68/part2;
            hill_ratio  = gsw_hill_ratio_at_sp2(t);
            sp_est      = hill_ratio*sp_hill_raw;
        }

        rtx_old = rtx;
        rtx     = rtx_old - (sp_est - sp)/dsp_drtx;

        rtxm    = 0.5e0*(rtx + rtx_old); /*This mean value of Rtx, Rtxm, is the
                  value of Rtx at which the derivative dSP_dRtx is evaluated.*/

        dsp_drtx=  a1 + (2e0*a2 + (3e0*a3 + (4e0*a4 +
                                5e0*a5*rtxm)*rtxm)*rtxm)*rtxm
                   + ft68*(b1 + (2e0*b2 + (3e0*b3 + (4e0*b4 +
                                5e0*b5*rtxm)*rtxm)*rtxm)*rtxm);
        if (sp_est <  2.0) {
            x   = 400e0*(rtxm*rtxm);
            sqrty       = 10e0*rtxm;
            part1       = 1e0 + x*(1.5e0 + x);
            part2       = 1e0 + sqrty*(1e0 + sqrty*(1e0 + sqrty));
            dsp_drtx    = dsp_drtx
                          + a0*800e0*rtxm*(1.5e0 + 2e0*x)/(part1*part1)
                          + b0*ft68*(10e0 + sqrty*(20e0 + 30e0*sqrty))/
                                (part2*part2);
            hill_ratio  = gsw_hill_ratio_at_sp2(t);
            dsp_drtx    = hill_ratio*dsp_drtx;
        }

    /*
     !--------------------------------------------------------------------------
     ! The line below is where Rtx is updated at the end of the one full
     ! iteration of the modified Newton-Raphson technique.
     !--------------------------------------------------------------------------
    */
        rtx     = rtx_old - (sp_est - sp)/dsp_drtx;
    /*
     !--------------------------------------------------------------------------
     ! Now we do another half iteration of the modified Newton-Raphson
     ! technique, making a total of one and a half modified N-R iterations.
     !--------------------------------------------------------------------------
    */
        sp_est  = a0 + (a1 + (a2 + (a3 + (a4 + a5*rtx)*rtx)*rtx)*rtx)*rtx
                + ft68*(b0 + (b1 + (b2+ (b3 + (b4 + b5*rtx)*rtx)*rtx)*rtx)*rtx);
        if (sp_est <  2.0) {
            x           = 400e0*(rtx*rtx);
            sqrty       = 10e0*rtx;
            part1       = 1e0 + x*(1.5e0 + x);
            part2       = 1e0 + sqrty*(1e0 + sqrty*(1e0 + sqrty));
            sp_hill_raw = sp_est - a0/part1 - b0*ft68/part2;
            hill_ratio  = gsw_hill_ratio_at_sp2(t);
            sp_est      = hill_ratio*sp_hill_raw;
        }
        rtx     = rtx - (sp_est - sp)/dsp_drtx;

    /*
     !--------------------------------------------------------------------------
     ! Now go from Rtx to Rt and then to the conductivity ratio R at pressure p.
     !--------------------------------------------------------------------------
    */
        rt      = rtx*rtx;

        aa      = d3 + d4*t68;
        bb      = 1e0 + t68*(d1 + d2*t68);
        cc      = p*(e1 + p*(e2 + e3*p));
    /* rt_lc (i.e. rt_lower_case) corresponds to rt as defined in
       the UNESCO 44 (1983) routines. */
        rt_lc   = c0 + (c1 + (c2 + (c3 + c4*t68)*t68)*t68)*t68;

        dd      = bb - aa*rt_lc*rt;
        ee      = rt_lc*rt*aa*(bb + cc);
        ra      = sqrt(dd*dd + 4e0*ee) - dd;
        r       = 0.5e0*ra/aa;

    /*
     ! The dimensionless conductivity ratio, R, is the conductivity input, C,
     ! divided by the present estimate of C(SP=35, t_68=15, p=0) which is
     ! 42.9140 mS/cm (=4.29140 S/m^).
    */
        return (gsw_c3515*r);
}
/*
!==========================================================================
function gsw_cabbeling(sa,ct,p)
!==========================================================================

!  Calculates the cabbeling coefficient of seawater with respect to
!  Conservative Temperature.  This function uses the computationally-
!  efficient expression for specific volume in terms of SA, CT and p
!  (Roquet et al., 2014).
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature (ITS-90)               [deg C]
! p      : sea pressure                                    [dbar]
!
! cabbeling  : cabbeling coefficient with respect to       [1/K^2]
!              Conservative Temperature.
*/
double
gsw_cabbeling(double sa, double ct, double p)
{
        double  alpha_ct, alpha_on_beta, alpha_sa, beta_sa, rho,
                v_sa, v_ct, v_sa_sa, v_sa_ct, v_ct_ct;

        gsw_specvol_first_derivatives(sa,ct,p,&v_sa,&v_ct, NULL);

        gsw_specvol_second_derivatives(sa,ct,p,&v_sa_sa,&v_sa_ct,&v_ct_ct,
                                        NULL, NULL);

        rho             = gsw_rho(sa,ct,p);

        alpha_ct        = rho*(v_ct_ct - rho*v_ct*v_ct);

        alpha_sa        = rho*(v_sa_ct - rho*v_sa*v_ct);

        beta_sa         = -rho*(v_sa_sa - rho*v_sa*v_sa);

        alpha_on_beta   = gsw_alpha_on_beta(sa,ct,p);

        return (alpha_ct +
                alpha_on_beta*(2.0*alpha_sa - alpha_on_beta*beta_sa));
}
/*
!==========================================================================
elemental function gsw_chem_potential_water_ice (t, p)
!==========================================================================
!
!  Calculates the chemical potential of water in ice from in-situ
!  temperature and pressure.
!
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  chem_potential_water_ice  =  chemical potential of ice          [ J/kg ]
!--------------------------------------------------------------------------
*/
double
gsw_chem_potential_water_ice(double t, double p)
{
        return (gsw_gibbs_ice(0,0,t,p));
}
/*
!==========================================================================
elemental function gsw_chem_potential_water_t_exact (sa, t, p)
!==========================================================================
!
!  Calculates the chemical potential of water in seawater.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  t   =  in-situ temperature (ITS-90)                            [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  chem_potential_water_t_exact  =  chemical potential of water in seawater
!                                                                   [ J/g ]
!--------------------------------------------------------------------------
*/
double
gsw_chem_potential_water_t_exact(double sa, double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        double  g03_g, g08_g, g_sa_part, x, x2, y, z, kg2g = 1e-3;

        x2 = gsw_sfac*sa;
        x = sqrt(x2);
        y = t*0.025;
        z = p*1e-4;

        g03_g = 101.342743139674 + z*(100015.695367145 +
    z*(-2544.5765420363 + z*(284.517778446287 +
    z*(-33.3146754253611 + (4.20263108803084 - 0.546428511471039*z)*z)))) +
    y*(5.90578347909402 + z*(-270.983805184062 +
    z*(776.153611613101 + z*(-196.51255088122 +
    (28.9796526294175 - 2.13290083518327*z)*z))) +
    y*(-12357.785933039 + z*(1455.0364540468 +
    z*(-756.558385769359 + z*(273.479662323528 +
    z*(-55.5604063817218 + 4.34420671917197*z)))) +
    y*(736.741204151612 + z*(-672.50778314507 +
    z*(499.360390819152 + z*(-239.545330654412 +
    (48.8012518593872 - 1.66307106208905*z)*z))) +
    y*(-148.185936433658 + z*(397.968445406972 +
    z*(-301.815380621876 + (152.196371733841 - 26.3748377232802*z)*z)) +
    y*(58.0259125842571 + z*(-194.618310617595 +
    z*(120.520654902025 + z*(-55.2723052340152 + 6.48190668077221*z))) +
    y*(-18.9843846514172 + y*(3.05081646487967 - 9.63108119393062*z) +
    z*(63.5113936641785 + z*(-22.2897317140459 + 8.17060541818112*z))))))));

        g08_g = x2*(1416.27648484197 +
    x*(-2432.14662381794 + x*(2025.80115603697 +
    y*(543.835333000098 + y*(-68.5572509204491 +
    y*(49.3667694856254 + y*(-17.1397577419788 +
    2.49697009569508*y))) - 22.6683558512829*z) +
    x*(-1091.66841042967 - 196.028306689776*y +
    x*(374.60123787784 - 48.5891069025409*x +
    36.7571622995805*y) + 36.0284195611086*z) +
    z*(-54.7919133532887 + (-4.08193978912261 -
    30.1755111971161*z)*z)) +
    z*(199.459603073901 + z*(-52.2940909281335 +
    (68.0444942726459 - 3.41251932441282*z)*z)) +
    y*(-493.407510141682 + z*(-175.292041186547 +
    (83.1923927801819 - 29.483064349429*z)*z) +
    y*(-43.0664675978042 + z*(383.058066002476 +
    z*(-54.1917262517112 + 25.6398487389914*z)) +
    y*(-10.0227370861875 - 460.319931801257*z +
    y*(0.875600661808945 + 234.565187611355*z))))) +
    y*(168.072408311545));

        g_sa_part = 8645.36753595126 +
    x*(-7296.43987145382 + x*(8103.20462414788 +
    y*(2175.341332000392 + y*(-274.2290036817964 +
    y*(197.4670779425016 + y*(-68.5590309679152 +
    9.98788038278032*y))) - 90.6734234051316*z) +
    x*(-5458.34205214835 - 980.14153344888*y +
    x*(2247.60742726704 - 340.1237483177863*x +
    220.542973797483*y) + 180.142097805543*z) +
    z*(-219.1676534131548 + (-16.32775915649044 -
    120.7020447884644*z)*z)) +
    z*(598.378809221703 + z*(-156.8822727844005 +
    (204.1334828179377 - 10.23755797323846*z)*z)) +
    y*(-1480.222530425046 + z*(-525.876123559641 +
    (249.57717834054571 - 88.449193048287*z)*z) +
    y*(-129.1994027934126 + z*(1149.174198007428 +
    z*(-162.5751787551336 + 76.9195462169742*z)) +
    y*(-30.0682112585625 - 1380.9597954037708*z +
    y*(2.626801985426835 + 703.695562834065*z))))) +
    y*(1187.3715515697959);

        return (kg2g*(g03_g + g08_g - 0.5*x2*g_sa_part));
}
/*
!==========================================================================
elemental function gsw_cp_ice (t, p)
!==========================================================================
!
!  Calculates the isobaric heat capacity of seawater.
!
!  t   =  in-situ temperature (ITS-90)                            [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!          ( i.e. absolute pressure - 10.1325 dbar )
!
!  gsw_cp_ice  =  heat capacity of ice                       [J kg^-1 K^-1]
!--------------------------------------------------------------------------
*/
double
gsw_cp_ice(double t, double p)
{
        GSW_TEOS10_CONSTANTS;

        return (-(t + gsw_t0)*gsw_gibbs_ice(2,0,t,p));
}
/*
!==========================================================================
function gsw_cp_t_exact(sa,t,p)
!==========================================================================

! Calculates isobaric heat capacity of seawater
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
!
! gsw_cp_t_exact : heat capacity                           [J/(kg K)]
*/
double
gsw_cp_t_exact(double sa, double t, double p)
{
        int     n0, n2;

        n0 = 0;
        n2 = 2;

        return (-(t+273.15e0)*gsw_gibbs(n0,n2,n0,sa,t,p));
}
/*
!==========================================================================
elemental subroutine gsw_ct_first_derivatives (sa, pt, ct_sa, ct_pt)
!==========================================================================
!
!  Calculates the following two derivatives of Conservative Temperature
!  (1) CT_SA, the derivative with respect to Absolute Salinity at
!      constant potential temperature (with pr = 0 dbar), and
!   2) CT_pt, the derivative with respect to potential temperature
!      (the regular potential temperature which is referenced to 0 dbar)
!      at constant Absolute Salinity.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  pt  =  potential temperature (ITS-90)                          [ deg C ]
!         (whose reference pressure is 0 dbar)
!
!  CT_SA  =  The derivative of Conservative Temperature with respect to
!            Absolute Salinity at constant potential temperature
!            (the regular potential temperature which has reference
!            sea pressure of 0 dbar).
!            The CT_SA output has units of:                     [ K/(g/kg)]
!  CT_pt  =  The derivative of Conservative Temperature with respect to
!            potential temperature (the regular one with pr = 0 dbar)
!            at constant SA. CT_pt is dimensionless.           [ unitless ]
!--------------------------------------------------------------------------
*/
void
gsw_ct_first_derivatives(double sa, double pt, double *ct_sa, double *ct_pt)
{
        GSW_TEOS10_CONSTANTS;
        double  abs_pt, g_sa_mod, g_sa_t_mod, x, y_pt;

        abs_pt = gsw_t0 + pt ;

        if (ct_pt != NULL)
            *ct_pt = -(abs_pt*gsw_gibbs_pt0_pt0(sa,pt))/gsw_cp0;

        if (ct_sa == NULL)
            return;

        x = sqrt(gsw_sfac*sa);
        y_pt = 0.025*pt;

        g_sa_t_mod = 1187.3715515697959 + x*(-1480.222530425046
            + x*(2175.341332000392 + x*(-980.14153344888
            + 220.542973797483*x) + y_pt*(-548.4580073635929
            + y_pt*(592.4012338275047 + y_pt*(-274.2361238716608
            + 49.9394019139016*y_pt)))) + y_pt*(-258.3988055868252
            + y_pt*(-90.2046337756875 + y_pt*10.50720794170734)))
            + y_pt*(3520.125411988816  + y_pt*(-1351.605895580406
            + y_pt*(731.4083582010072  + y_pt*(-216.60324087531103
            + 25.56203650166196*y_pt))));
        g_sa_t_mod = 0.5*gsw_sfac*0.025*g_sa_t_mod;

        g_sa_mod = 8645.36753595126 + x*(-7296.43987145382
            + x*(8103.20462414788 + y_pt*(2175.341332000392
            + y_pt*(-274.2290036817964 + y_pt*(197.4670779425016
            + y_pt*(-68.5590309679152 + 9.98788038278032*y_pt))))
            + x*(-5458.34205214835 - 980.14153344888*y_pt
            + x*(2247.60742726704 - 340.1237483177863*x
            + 220.542973797483*y_pt))) + y_pt*(-1480.222530425046
            + y_pt*(-129.1994027934126 + y_pt*(-30.0682112585625
            + y_pt*(2.626801985426835 ))))) + y_pt*(1187.3715515697959
            + y_pt*(1760.062705994408 + y_pt*(-450.535298526802
            + y_pt*(182.8520895502518 + y_pt*(-43.3206481750622
            + 4.26033941694366*y_pt)))));
        g_sa_mod = 0.5*gsw_sfac*g_sa_mod;

        *ct_sa = (g_sa_mod - abs_pt*g_sa_t_mod)/gsw_cp0;
}
/*
!==========================================================================
elemental subroutine gsw_ct_first_derivatives_wrt_t_exact (sa, t, p, &
                                       ct_sa_wrt_t, ct_t_wrt_t, ct_p_wrt_t)
!==========================================================================
!
!  Calculates the following three derivatives of Conservative Temperature.
!  These derivatives are done with respect to in-situ temperature t (in the
!  case of CT_T_wrt_t) or at constant in-situ tempertature (in the cases of
!  CT_SA_wrt_t and CT_P_wrt_t).
!   (1) CT_SA_wrt_t, the derivative of CT with respect to Absolute Salinity
!       at constant t and p, and
!   (2) CT_T_wrt_t, derivative of CT with respect to in-situ temperature t
!       at constant SA and p.
!   (3) CT_P_wrt_t, derivative of CT with respect to pressure P (in Pa) at
!       constant SA and t.
!
!  This function uses the full Gibbs function. Note that this function
!  avoids the NaN that would exist in CT_SA_wrt_t at SA = 0 if it were
!  evaluated in the straightforward way from the derivatives of the Gibbs
!  function function.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  t   =  in-situ temperature (ITS-90)                            [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar)
!
!  CT_SA_wrt_t  =  The first derivative of Conservative Temperature with
!                  respect to Absolute Salinity at constant t and p.
!                                              [ K/(g/kg)]  i.e. [ K kg/g ]
!  CT_T_wrt_t  =  The first derivative of Conservative Temperature with
!                 respect to in-situ temperature, t, at constant SA and p.
!                                                              [ unitless ]
!  CT_P_wrt_t  =  The first derivative of Conservative Temperature with
!                 respect to pressure P (in Pa) at constant SA and t.
!                                                                  [ K/Pa ]
!--------------------------------------------------------------------------
*/
void
gsw_ct_first_derivatives_wrt_t_exact(double sa, double t, double p,
        double *ct_sa_wrt_t, double *ct_t_wrt_t, double *ct_p_wrt_t)
{
        GSW_TEOS10_CONSTANTS;
        double g_sa_mod, g_sa_t_mod, pt0, x, y, y_pt, z;

        pt0 = gsw_pt0_from_t(sa,t,p);

        if (ct_sa_wrt_t != NULL) {

            x = sqrt(gsw_sfac*sa);
            y = 0.025*t;
            y_pt = 0.025*pt0;
            z = rec_db2pa*p;
                /* the input pressure (p) is sea pressure in units of dbar */

            g_sa_t_mod = 1187.3715515697959 + z*(1458.233059470092 +
            z*(-687.913805923122 + z*(249.375342232496 +
            z*(-63.313928772146 + 14.09317606630898*z)))) +
            x*(-1480.222530425046 + x*(2175.341332000392 +
            x*(-980.14153344888 + 220.542973797483*x) +
            y*(-548.4580073635929 + y*(592.4012338275047 +
            y*(-274.2361238716608 + 49.9394019139016*y))) -
            90.6734234051316*z) + z*(-525.876123559641 +
            (249.57717834054571 - 88.449193048287*z)*z) +
            y*(-258.3988055868252 + z*(2298.348396014856 +
            z*(-325.1503575102672 + 153.8390924339484*z)) +
            y*(-90.2046337756875 - 4142.8793862113125*z +
            y*(10.50720794170734 + 2814.78225133626*z)))) +
            y*(3520.125411988816 + y*(-1351.605895580406 +
            y*(731.4083582010072 + y*(-216.60324087531103 +
            25.56203650166196*y) + z*(-2381.829935897496 +
            (597.809129110048 - 291.8983352012704*z)*z)) +
            z*(4165.4688847996085 + z*(-1229.337851789418 +
            (681.370187043564 - 66.7696405958478*z)*z))) +
            z*(-3443.057215135908 + z*(1349.638121077468 +
            z*(-713.258224830552 +
            (176.8161433232 - 31.68006188846728*z)*z))));
            g_sa_t_mod = 0.5*gsw_sfac*0.025*g_sa_t_mod;

            g_sa_mod = 8645.36753595126 +
            x*(-7296.43987145382 + x*(8103.20462414788 +
            y_pt*(2175.341332000392 + y_pt*(-274.2290036817964 +
            y_pt*(197.4670779425016 + y_pt*(-68.5590309679152 +
            9.98788038278032*y_pt)))) +
            x*(-5458.34205214835 - 980.14153344888*y_pt +
            x*(2247.60742726704 - 340.1237483177863*x +
            220.542973797483*y_pt))) +
            y_pt*(-1480.222530425046 +
            y_pt*(-129.1994027934126 +
            y_pt*(-30.0682112585625 + y_pt*(2.626801985426835 ))))) +
            y_pt*(1187.3715515697959 +
            y_pt*(1760.062705994408 + y_pt*(-450.535298526802 +
            y_pt*(182.8520895502518 + y_pt*(-43.3206481750622 +
            4.26033941694366*y_pt)))));
            g_sa_mod = 0.5*gsw_sfac*g_sa_mod;

            *ct_sa_wrt_t = (g_sa_mod - (gsw_t0+pt0)*g_sa_t_mod)/gsw_cp0;

        }

        if (ct_t_wrt_t != NULL)
            *ct_t_wrt_t = -(gsw_t0+pt0)*gsw_gibbs(0,2,0,sa,t,p)/gsw_cp0;

        if (ct_p_wrt_t != NULL)
            *ct_p_wrt_t = -(gsw_t0+pt0)*gsw_gibbs(0,1,1,sa,t,p)/gsw_cp0;
}
/*
!==========================================================================
elemental function gsw_ct_freezing (sa, p, saturation_fraction)
!==========================================================================
!
!  Calculates the Conservative Temperature at which seawater freezes.  The
!  Conservative Temperature freezing point is calculated from the exact
!  in-situ freezing temperature which is found by a modified Newton-Raphson
!  iteration (McDougall and Wotherspoon, 2013) of the equality of the
!  chemical potentials of water in seawater and in ice.
!
!  An alternative GSW function, gsw_CT_freezing_poly, it is based on a
!  computationally-efficient polynomial, and is accurate to within -5e-4 K
!  and 6e-4 K, when compared with this function.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  saturation_fraction = the saturation fraction of dissolved air in
!                        seawater
!
!  CT_freezing = Conservative Temperature at freezing of seawater [ deg C ]
!--------------------------------------------------------------------------
*/
double
gsw_ct_freezing(double sa, double p, double saturation_fraction)
{
        double  t_freezing;

        t_freezing = gsw_t_freezing(sa,p,saturation_fraction);
        return (gsw_ct_from_t(sa,t_freezing,p));
}
/*
!==========================================================================
elemental subroutine gsw_ct_freezing_first_derivatives (sa, p, &
                          saturation_fraction, ctfreezing_sa, ctfreezing_p)
!==========================================================================
!
!  Calculates the first derivatives of the Conservative Temperature at
!  which seawater freezes, with respect to Absolute Salinity SA and
!  pressure P (in Pa).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  saturation_fraction = the saturation fraction of dissolved air in
!                        seawater
!
!  CTfreezing_SA = the derivative of the Conservative Temperature at
!                  freezing (ITS-90) with respect to Absolute Salinity at
!                  fixed pressure              [ K/(g/kg) ] i.e. [ K kg/g ]
!
!  CTfreezing_P  = the derivative of the Conservative Temperature at
!                  freezing (ITS-90) with respect to pressure (in Pa) at
!                  fixed Absolute Salinity                         [ K/Pa ]
!--------------------------------------------------------------------------
*/
void
gsw_ct_freezing_first_derivatives(double sa, double p,
       double saturation_fraction, double *ctfreezing_sa, double *ctfreezing_p)
{
        double  tf_sa, tf_p, ct_sa_wrt_t, ct_t_wrt_t, ct_p_wrt_t, tf;

        tf = gsw_t_freezing(sa,p,saturation_fraction);

        if (ctfreezing_sa != NULL && ctfreezing_p != NULL) {

            gsw_t_freezing_first_derivatives(sa,p,saturation_fraction,
                                                  &tf_sa,&tf_p);
            gsw_ct_first_derivatives_wrt_t_exact(sa,tf,p,
                                &ct_sa_wrt_t,&ct_t_wrt_t,&ct_p_wrt_t);

            *ctfreezing_sa = ct_sa_wrt_t + ct_t_wrt_t*tf_sa;
            *ctfreezing_p  = ct_p_wrt_t  + ct_t_wrt_t*tf_p;

        } else if (ctfreezing_sa != NULL && ctfreezing_p == NULL) {

            gsw_t_freezing_first_derivatives(sa,p,saturation_fraction,
                                                  &tf_sa, NULL);
            gsw_ct_first_derivatives_wrt_t_exact(sa,tf,p,
                                &ct_sa_wrt_t,&ct_t_wrt_t,NULL);

            *ctfreezing_sa = ct_sa_wrt_t + ct_t_wrt_t*tf_sa;

        } else if (ctfreezing_sa == NULL && ctfreezing_p != NULL) {

            gsw_t_freezing_first_derivatives(sa,p,saturation_fraction,
                                                  NULL, &tf_p);
            gsw_ct_first_derivatives_wrt_t_exact(sa,tf,p,
                                NULL,&ct_t_wrt_t,&ct_p_wrt_t);

            *ctfreezing_p  = ct_p_wrt_t  + ct_t_wrt_t*tf_p;

        }
}
/*
!==========================================================================
elemental subroutine gsw_ct_freezing_first_derivatives_poly (sa, p, &
                          saturation_fraction, ctfreezing_sa, ctfreezing_p)
!==========================================================================
!
!  Calculates the first derivatives of the Conservative Temperature at
!  which seawater freezes, with respect to Absolute Salinity SA and
!  pressure P (in Pa) of the comptationally efficient polynomial fit of the
!  freezing temperature (McDougall et al., 2014).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  saturation_fraction = the saturation fraction of dissolved air in
!                        seawater
!
!  CTfreezing_SA = the derivative of the Conservative Temperature at
!                  freezing (ITS-90) with respect to Absolute Salinity at
!                  fixed pressure              [ K/(g/kg) ] i.e. [ K kg/g ]
!
!  CTfreezing_P  = the derivative of the Conservative Temperature at
!                  freezing (ITS-90) with respect to pressure (in Pa) at
!                  fixed Absolute Salinity                         [ K/Pa ]
!--------------------------------------------------------------------------
*/
void
gsw_ct_freezing_first_derivatives_poly(double sa, double p,
                double saturation_fraction, double *ctfreezing_sa,
                double *ctfreezing_p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_FREEZING_POLY_COEFFICIENTS;
        double  p_r, sa_r, x, d = -a - a*b - 2.4*b/gsw_sso,
                e = 2.0*a*b/gsw_sso;

        sa_r = sa*1e-2;
        x = sqrt(sa_r);
        p_r = p*1e-4;

        if (ctfreezing_sa != NULL) *ctfreezing_sa =
            (c1 + x*(1.5*c2 + x*(2.0*c3 + x*(2.5*c4 + x*(3.0*c5
                + 3.5*c6*x)))) + p_r*(c10 + x*(1.5*c11 + x*(2.0*c13
                + x*(2.5*c16 + x*(3.0*c19 + 3.5*c22*x))))
                + p_r*(c12 + x*(1.5*c14 + x*(2.0*c17 + 2.5*c20*x))
                + p_r*(c15 + x*(1.5*c18 + 2.0*c21*x)))))*1e-2
                - saturation_fraction*1e-3*(d - sa*e);

        if (ctfreezing_p != NULL) *ctfreezing_p =
            (c7 + sa_r*(c10 + x*(c11 + x*(c13 + x*(c16 + x*(c19 + c22*x)))))
                + p_r*(2.0*c8 + sa_r*(2.0*c12 + x*(2.0*c14 + x*(2.0*c17
                + 2.0*c20*x))) + p_r*(3.0*c9 + sa_r*(3.0*c15 + x*(3.0*c18
                + 3.0*c21*x)))))*1e-8;
}
/*
!==========================================================================
elemental function gsw_ct_freezing_poly (sa, p, saturation_fraction)
!==========================================================================
!
!  Calculates the Conservative Temperature at which seawater freezes.
!  The error of this fit ranges between -5e-4 K and 6e-4 K when compared
!  with the Conservative Temperature calculated from the exact in-situ
!  freezing temperature which is found by a Newton-Raphson iteration of the
!  equality of the chemical potentials of water in seawater and in ice.
!  Note that the Conservative temperature freezing temperature can be found
!  by this exact method using the function gsw_CT_freezing.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  saturation_fraction = the saturation fraction of dissolved air in
!                        seawater
!
!  CT_freezing = Conservative Temperature at freezing of seawater [ deg C ]
!                That is, the freezing temperature expressed in
!                terms of Conservative Temperature (ITS-90).
!--------------------------------------------------------------------------
*/
double
gsw_ct_freezing_poly(double sa, double p, double saturation_fraction)
{
        GSW_TEOS10_CONSTANTS;
        GSW_FREEZING_POLY_COEFFICIENTS;
        double  p_r, sa_r, x, return_value;

        sa_r    = sa*1.0e-2;
        x       = sqrt(sa_r);
        p_r     = p*1.0e-4;

        return_value = c0
    + sa_r*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + c6*x)))))
    + p_r*(c7 + p_r*(c8 + c9*p_r)) + sa_r*p_r*(c10 + p_r*(c12
    + p_r*(c15 + c21*sa_r)) + sa_r*(c13 + c17*p_r + c19*sa_r)
    + x*(c11 + p_r*(c14 + c18*p_r) + sa_r*(c16 + c20*p_r + c22*sa_r)));

        /* Adjust for the effects of dissolved air */
        return_value = return_value - saturation_fraction*
                 (1e-3)*(2.4 - a*sa)*(1.0 + b*(1.0 - sa/gsw_sso));

        return (return_value);
}
/*
!==========================================================================
elemental function gsw_ct_from_enthalpy (sa, h, p)
!==========================================================================
!
!  Calculates the Conservative Temperature of seawater, given the Absolute
!  Salinity, specific enthalpy, h, and pressure p.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  h   =  specific enthalpy                                        [ J/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325d0 dbar )
!
!  CT  =  Conservative Temperature ( ITS-90)                      [ deg C ]
!--------------------------------------------------------------------------
*/
double
gsw_ct_from_enthalpy(double sa, double h, double p)
{
        GSW_TEOS10_CONSTANTS;
        double  ct, ct_freezing, ct_mean, ct_old, f, h_freezing, h_ct, h_40,
                ct_40 = 40.0;

        ct_freezing = gsw_ct_freezing(sa,p,0.0);

        h_freezing = gsw_enthalpy(sa,ct_freezing,p);
        if (h < (h_freezing - gsw_cp0)) {
            /*
            ! The input, seawater enthalpy h, is less than the enthalpy at the
            ! freezing temperature, i.e. the water is frozen.
            */
            return (GSW_INVALID_VALUE);
        }

        h_40 = gsw_enthalpy(sa,ct_40,p);
        if (h > h_40) {
            /*
            ! The input seawater enthalpy is greater than the enthalpy
            ! when CT is 40C
            */
            return (GSW_INVALID_VALUE);
        }

        /* first guess of ct */
        ct = ct_freezing + (ct_40 - ct_freezing)*(h - h_freezing)/
                                (h_40 - h_freezing);
        gsw_enthalpy_first_derivatives(sa,ct,p,NULL,&h_ct);

        /*
        !------------------------------------------------------
        ! Begin the modified Newton-Raphson iterative procedure
        !------------------------------------------------------
        */

        ct_old = ct;
        f = gsw_enthalpy(sa,ct_old,p) - h;
        ct = ct_old - f/h_ct;
        ct_mean = 0.5*(ct + ct_old);
        gsw_enthalpy_first_derivatives(sa,ct_mean,p,NULL,&h_ct);
        ct = ct_old - f/h_ct;

        ct_old = ct;
        f = gsw_enthalpy(sa,ct_old,p) - h;
        ct = ct_old - f/h_ct;
        /*
        ! After 1.5d0 iterations of this modified Newton-Raphson iteration,
        ! the error in CT is no larger than 4x10^-13 degrees C, which
        ! is machine precision for this calculation.
        */
        return (ct);
}
/*
!==========================================================================
elemental function gsw_ct_from_enthalpy_exact (sa, h, p)
!==========================================================================
!
!  Calculates the Conservative Temperature of seawater, given the Absolute
!  Salinity, specific enthalpy, h, and pressure p.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  h   =  specific enthalpy                                        [ J/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325d0 dbar )
!
!  CT  =  Conservative Temperature ( ITS-90)                      [ deg C ]
!--------------------------------------------------------------------------
*/
double
gsw_ct_from_enthalpy_exact(double sa, double h, double p)
{
        GSW_TEOS10_CONSTANTS;
        double  ct, ct_freezing, ct_mean, ct_old, f, h_freezing, h_ct, h_40,
                ct_40 = 40.0;

        ct_freezing = gsw_ct_freezing(sa,p,0.0);

        h_freezing = gsw_enthalpy_ct_exact(sa,ct_freezing,p);
        if (h < (h_freezing - gsw_cp0)) {
            /*
            ! The input, seawater enthalpy h, is less than the enthalpy at the
            ! freezing temperature, i.e. the water is frozen.
            */
            return (GSW_INVALID_VALUE);
        }

        h_40 = gsw_enthalpy_ct_exact(sa,ct_40,p);
        if (h > h_40) {
            /*
            ! The input seawater enthalpy is greater than the enthalpy
            ! when CT is 40C
            */
            return (GSW_INVALID_VALUE);
        }

        /* First guess of ct */
        ct = ct_freezing + (ct_40 - ct_freezing)*(h - h_freezing)/
                                (h_40 - h_freezing);
        gsw_enthalpy_first_derivatives_ct_exact(sa,ct,p,NULL,&h_ct);
        /*
        !------------------------------------------------------
        ! Begin the modified Newton-Raphson iterative procedure
        !------------------------------------------------------
        */
        ct_old = ct;
        f = gsw_enthalpy_ct_exact(sa,ct_old,p) - h;
        ct = ct_old - f/h_ct;
        ct_mean = 0.5*(ct + ct_old);
        gsw_enthalpy_first_derivatives_ct_exact(sa,ct_mean,p,NULL,&h_ct);
        ct = ct_old - f/h_ct;
        /*
        ! After 1 iteration of this modified Newton-Raphson iteration,
        ! the error in CT is no larger than 5x10^-14 degrees C, which
        ! is machine precision for this calculation.
        */
        return (ct);
}
/*
!=========================================================================
elemental function gsw_ct_from_entropy (sa, entropy)
!=========================================================================
!
!  Calculates Conservative Temperature with entropy as an input variable.
!
!  SA       =  Absolute Salinity                                   [ g/kg ]
!  entropy  =  specific entropy                                   [ deg C ]
!
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!--------------------------------------------------------------------------
*/
double
gsw_ct_from_entropy(double sa, double entropy)
{
        double  pt;

        pt = gsw_pt_from_entropy(sa,entropy);
        return (gsw_ct_from_pt(sa,pt));
}
/*
!==========================================================================
function gsw_ct_from_pt(sa,pt)
!==========================================================================

! Calculates Conservative Temperature from potential temperature of seawater
!
! sa      : Absolute Salinity                              [g/kg]
! pt      : potential temperature with                     [deg C]
!           reference pressure of 0 dbar
!
! gsw_ct_from_pt : Conservative Temperature                [deg C]
*/
double
gsw_ct_from_pt(double sa, double pt)
{
        GSW_TEOS10_CONSTANTS;
        double  x2, x, y, pot_enthalpy;

        x2              = gsw_sfac*sa;
        x               = sqrt(x2);
        y               = pt*0.025e0;   /*! normalize for F03 and F08 */
        pot_enthalpy    =  61.01362420681071e0 + y*(168776.46138048015e0 +
             y*(-2735.2785605119625e0 + y*(2574.2164453821433e0 +
             y*(-1536.6644434977543e0 + y*(545.7340497931629e0 +
             (-50.91091728474331e0 - 18.30489878927802e0*y)*y))))) +
             x2*(268.5520265845071e0 + y*(-12019.028203559312e0 +
             y*(3734.858026725145e0 + y*(-2046.7671145057618e0 +
             y*(465.28655623826234e0 + (-0.6370820302376359e0 -
             10.650848542359153e0*y)*y)))) +
             x*(937.2099110620707e0 + y*(588.1802812170108e0+
             y*(248.39476522971285e0 + (-3.871557904936333e0-
             2.6268019854268356e0*y)*y)) +
             x*(-1687.914374187449e0 + x*(246.9598888781377e0 +
             x*(123.59576582457964e0 - 48.5891069025409e0*x)) +
             y*(936.3206544460336e0 +
             y*(-942.7827304544439e0 + y*(369.4389437509002e0 +
             (-33.83664947895248e0 - 9.987880382780322e0*y)*y))))));

        return (pot_enthalpy/gsw_cp0);
}
/*
!==========================================================================
elemental subroutine gsw_ct_from_rho (rho, sa, p, ct, ct_multiple)
! =========================================================================
!
!  Calculates the Conservative Temperature of a seawater sample, for given
!  values of its density, Absolute Salinity and sea pressure (in dbar).
!
!  rho  =  density of a seawater sample (e.g. 1026 kg/m^3)       [ kg/m^3 ]
!   Note. This input has not had 1000 kg/m^3 subtracted from it.
!     That is, it is 'density', not 'density anomaly'.
!  SA   =  Absolute Salinity                                       [ g/kg ]
!  p    =  sea pressure                                            [ dbar ]
!          ( i.e. absolute pressure - 10.1325 dbar )
!
!  CT  =  Conservative Temperature  (ITS-90)                      [ deg C ]
!  CT_multiple  =  Conservative Temperature  (ITS-90)             [ deg C ]
!    Note that at low salinities, in brackish water, there are two possible
!      Conservative Temperatures for a single density.  This programme will
!      output both valid solutions.  To see this second solution the user
!      must call the programme with two outputs (i.e. [CT,CT_multiple]), if
!      there is only one possible solution and the programme has been
!      called with two outputs the second variable will be set to NaN.
!--------------------------------------------------------------------------
*/
void
gsw_ct_from_rho(double rho, double sa, double p, double *ct,
                double *ct_multiple)
{
        int     number_of_iterations;
        double  a, alpha_freezing, alpha_mean, b, c, ct_a, ct_b, ct_diff,
                ct_freezing, ct_max_rho, ct_mean, ct_old,
                delta_ct, delta_v, factor, factorqa, factorqb,
                rho_40, rho_extreme, rho_freezing, rho_max, rho_mean,
                rho_old, sqrt_disc, top, v_ct, v_lab;

        /*alpha_limit is the positive value of the thermal expansion coefficient
        ! which is used at the freezing temperature to distinguish between
        ! salty and fresh water.*/
        double  alpha_limit = 1e-5;

        /*rec_half_rho_TT is a constant representing the reciprocal of half the
        ! second derivative of density with respect to temperature near the
        ! temperature of maximum density.*/
        double  rec_half_rho_tt = -110.0;

        rho_40 = gsw_rho(sa,40.0,p);
        if (rho < rho_40) {
            *ct = GSW_INVALID_VALUE;
            if (ct_multiple != NULL) *ct_multiple = *ct;
            return;
        }

        ct_max_rho = gsw_ct_maxdensity(sa,p);
        rho_max = gsw_rho(sa,ct_max_rho,p);
        rho_extreme = rho_max;

        /*Assumes that the seawater is always unsaturated with air*/
        ct_freezing = gsw_ct_freezing_poly(sa,p,0.0);

        gsw_rho_alpha_beta(sa,ct_freezing,p,&rho_freezing,&alpha_freezing,NULL);

        /*reset the extreme values*/
        if (ct_freezing > ct_max_rho) rho_extreme = rho_freezing;

        if (rho > rho_extreme) {
            *ct = GSW_INVALID_VALUE;
            if (ct_multiple != NULL) *ct_multiple = *ct;
            return;
        }

        if (alpha_freezing > alpha_limit) {

            ct_diff = 40.0 - ct_freezing;
            top = rho_40 - rho_freezing + rho_freezing*alpha_freezing*ct_diff;
            a = top/(ct_diff*ct_diff);
            b = -rho_freezing*alpha_freezing;
            c = rho_freezing - rho;
            sqrt_disc = sqrt(b*b - 4*a*c);
            *ct = ct_freezing + 0.5*(-b - sqrt_disc)/a;

        } else {

            ct_diff = 40.0 - ct_max_rho;
            factor = (rho_max - rho)/(rho_max - rho_40);
            delta_ct = ct_diff*sqrt(factor);

            if (delta_ct > 5.0)
                *ct = ct_max_rho + delta_ct;
            else {
                /*Set the initial value of the quadratic solution roots.*/
                ct_a = ct_max_rho + sqrt(rec_half_rho_tt*(rho - rho_max));
                for (number_of_iterations = 1; number_of_iterations <= 7;
                    number_of_iterations++) {
                    ct_old = ct_a;
                    rho_old = gsw_rho(sa,ct_old,p);
                    factorqa = (rho_max - rho)/(rho_max - rho_old);
                    ct_a = ct_max_rho + (ct_old - ct_max_rho)*sqrt(factorqa);
                }

                if ((ct_freezing - ct_a) < 0.0) {
                    *ct = GSW_INVALID_VALUE;
                    if (ct_multiple != NULL) *ct_multiple = *ct;
                    return;
                }

                *ct = ct_a;
                if (ct_multiple == NULL) return;

                /*Set the initial value of the quadratic solution roots.*/
                ct_b = ct_max_rho - sqrt(rec_half_rho_tt*(rho - rho_max));
                for (number_of_iterations = 1; number_of_iterations <= 7;
                    number_of_iterations++) {
                    ct_old = ct_b;
                    rho_old = gsw_rho(sa,ct_old,p);
                    factorqb = (rho_max - rho)/(rho_max - rho_old);
                    ct_b = ct_max_rho + (ct_old - ct_max_rho)*sqrt(factorqb);
                }
                /*
                ! After seven iterations of this quadratic iterative procedure,
                ! the error in rho is no larger than 4.6x10^-13 kg/m^3.
                */
                if ((ct_freezing - ct_b) < 0.0) {
                    *ct = GSW_INVALID_VALUE;
                    *ct_multiple = *ct;
                    return;
                }
                *ct_multiple = ct_b;
                return;
            }
        }

        /*Begin the modified Newton-Raphson iterative method*/

        v_lab = 1.0/rho;
        gsw_rho_alpha_beta(sa,*ct,p,&rho_mean,&alpha_mean,NULL);
        v_ct = alpha_mean/rho_mean;

        for (number_of_iterations = 1; number_of_iterations <= 3;
            number_of_iterations++) {
            ct_old = *ct;
            delta_v = gsw_specvol(sa,ct_old,p) - v_lab;
            *ct = ct_old - delta_v/v_ct;
            ct_mean = 0.5*(*ct + ct_old);
            gsw_rho_alpha_beta(sa,ct_mean,p,&rho_mean,&alpha_mean,NULL);
            v_ct = alpha_mean/rho_mean;
            *ct = ct_old - delta_v/v_ct ;
        }
        /*
        ! After three iterations of this modified Newton-Raphson iteration,
        ! the error in rho is no larger than 1.6x10^-12 kg/m^3.
        */
        if (ct_multiple != NULL) *ct_multiple = GSW_INVALID_VALUE;
        return;
}
/*
!==========================================================================
function gsw_ct_from_t(sa,t,p)
!==========================================================================

! Calculates Conservative Temperature from in-situ temperature
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
!
! gsw_ct_from_t : Conservative Temperature                 [deg C]
*/
double
gsw_ct_from_t(double sa, double t, double p)
{
        double  pt0;

        pt0     = gsw_pt0_from_t(sa,t,p);
        return (gsw_ct_from_pt(sa,pt0));
}
/*
!==========================================================================
elemental function gsw_ct_maxdensity (sa, p)
!==========================================================================
!
!  Calculates the Conservative Temperature of maximum density of seawater.
!  This function returns the Conservative temperature at which the density
!  of seawater is a maximum, at given Absolute Salinity, SA, and sea
!  pressure, p (in dbar).
!
!  SA =  Absolute Salinity                                         [ g/kg ]
!  p  =  sea pressure                                              [ dbar ]
!        ( i.e. absolute pressure - 10.1325 dbar )
!
!  CT_maxdensity  =  Conservative Temperature at which            [ deg C ]
!                    the density of seawater is a maximum for
!                    given Absolute Salinity and pressure.
!--------------------------------------------------------------------------
*/
double
gsw_ct_maxdensity(double sa, double p)
{
        int     number_of_iterations;
        double  alpha, ct, ct_mean, ct_old, dalpha_dct,
                dct = 0.001;

        ct = 3.978 - 0.22072*sa;         /*the initial guess of ct.*/

        dalpha_dct = 1.1e-5;             /*the initial guess for dalpha_dct.*/

        for (number_of_iterations = 1; number_of_iterations <= 3;
            number_of_iterations++) {
            ct_old = ct;
            alpha = gsw_alpha(sa,ct_old,p);
            ct = ct_old - alpha/dalpha_dct;
            ct_mean = 0.5*(ct + ct_old);
            dalpha_dct = (gsw_alpha(sa,ct_mean+dct,p)
                          - gsw_alpha(sa,ct_mean-dct,p))/(dct + dct);
            ct = ct_old - alpha/dalpha_dct;
        }
        /*
        ! After three iterations of this modified Newton-Raphson (McDougall and
        ! Wotherspoon, 2012) iteration, the error in CT_maxdensity is typically
        ! no larger than 1x10^-15 degrees C.
        */
        return (ct);
}
/*
!==========================================================================
elemental subroutine gsw_ct_second_derivatives (sa, pt, ct_sa_sa, ct_sa_pt, &
                                                ct_pt_pt)
!==========================================================================
!
!  Calculates the following three, second-order derivatives of Conservative
!  Temperature
!   (1) CT_SA_SA, the second derivative with respect to Absolute Salinity
!       at constant potential temperature (with p_ref = 0 dbar),
!   (2) CT_SA_pt, the derivative with respect to potential temperature
!       (the regular potential temperature which is referenced to 0 dbar)
!       and Absolute Salinity, and
!   (3) CT_pt_pt, the second derivative with respect to potential
!       temperature (the regular potential temperature which is referenced
!       to 0 dbar) at constant Absolute Salinity.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  pt  =  potential temperature (ITS-90)                          [ deg C ]
!         (whose reference pressure is 0 dbar)
!
!  CT_SA_SA  =  The second derivative of Conservative Temperature with
!               respect to Absolute Salinity at constant potential
!               temperature (the regular potential temperature which
!               has reference sea pressure of 0 dbar).
!               CT_SA_SA has units of:                     [ K/((g/kg)^2) ]
!  CT_SA_pt  =  The derivative of Conservative Temperature with
!               respect to potential temperature (the regular one with
!               p_ref = 0 dbar) and Absolute Salinity.
!               CT_SA_pt has units of:                        [ 1/(g/kg) ]
!  CT_pt_pt  =  The second derivative of Conservative Temperature with
!               respect to potential temperature (the regular one with
!               p_ref = 0 dbar) at constant SA.
!               CT_pt_pt has units of:                              [ 1/K ]
!--------------------------------------------------------------------------
*/
void
gsw_ct_second_derivatives(double sa, double pt, double *ct_sa_sa,
        double *ct_sa_pt, double *ct_pt_pt)
{
        double  ct_pt_l, ct_pt_u, ct_sa_l, ct_sa_u, pt_l, pt_u, sa_l, sa_u,
                delta, dsa = 1e-3, dpt = 1e-2;

        if ((ct_sa_sa != NULL)) {

            sa_u = sa + dsa;
            sa_l = sa - dsa;
            if (sa_l < 0.0) {
                sa_l = 0.0;
                delta = sa_u;
            } else {
                delta = 2 * dsa;
            }

            gsw_ct_first_derivatives(sa_l,pt,&ct_sa_l,NULL);
            gsw_ct_first_derivatives(sa_u,pt,&ct_sa_u,NULL);

            *ct_sa_sa = (ct_sa_u - ct_sa_l)/delta;

        }

        if ((ct_sa_pt != NULL) || (ct_pt_pt != NULL)) {

            pt_l = pt - dpt;
            pt_u = pt + dpt;
            delta = 2 * dpt;

            if ((ct_sa_pt != NULL) && (ct_pt_pt != NULL)) {

                gsw_ct_first_derivatives(sa,pt_l,&ct_sa_l,&ct_pt_l);
                gsw_ct_first_derivatives(sa,pt_u,&ct_sa_u,&ct_pt_u);

                *ct_sa_pt = (ct_sa_u - ct_sa_l)/delta;
                *ct_pt_pt = (ct_pt_u - ct_pt_l)/delta;

            } else if ((ct_sa_pt != NULL) && (ct_pt_pt == NULL)) {

                gsw_ct_first_derivatives(sa,pt_l,&ct_sa_l,NULL);
                gsw_ct_first_derivatives(sa,pt_u,&ct_sa_u,NULL);

                *ct_sa_pt = (ct_sa_u - ct_sa_l)/delta;

            } else if ((ct_sa_pt == NULL) && (ct_pt_pt != NULL)) {

                gsw_ct_first_derivatives(sa,pt_l,NULL,&ct_pt_l);
                gsw_ct_first_derivatives(sa,pt_u,NULL,&ct_pt_u);

                *ct_pt_pt = (ct_pt_u - ct_pt_l)/delta;

            }
        }
}
/*
!==========================================================================
function gsw_deltasa_from_sp(sp,p,lon,lat)
!==========================================================================

! Calculates Absolute Salinity Anomaly, deltaSA, from Practical Salinity, SP.
!
! sp     : Practical Salinity                              [unitless]
! p      : sea pressure                                    [dbar]
! lon    : longitude                                       [deg E]
! lat    : latitude                                        [deg N]
!
! gsw_deltasa_from_sp : Absolute Salinty Anomaly           [g/kg]
*/
double
gsw_deltasa_from_sp(double sp, double p, double lon, double lat)
{
        double  res;

        res     = gsw_sa_from_sp(sp,p,lon,lat) - gsw_sr_from_sp(sp);
        if (res > GSW_ERROR_LIMIT)
            res = GSW_INVALID_VALUE;
        return (res);
}
/*
!==========================================================================
elemental function gsw_dilution_coefficient_t_exact (sa, t, p)
!==========================================================================
!
!  Calculates the dilution coefficient of seawater.  The dilution
!  coefficient of seawater is defined as the Absolute Salinity times the
!  second derivative of the Gibbs function with respect to Absolute
!  Salinity, that is, SA.*g_SA_SA.
!
!  SA =  Absolute Salinity                                         [ g/kg ]
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!        ( i.e. absolute pressure - 10.1325 dbar )
!
!  dilution_coefficient_t_exact  =  dilution coefficient   [ (J/kg)(kg/g) ]
!--------------------------------------------------------------------------
*/
double
gsw_dilution_coefficient_t_exact(double sa, double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        double  g08, x, x2, y, z;

        x2 = gsw_sfac*sa;
        x = sqrt(x2);
        y = t*0.025;
        z = p*1e-4;
            /*note.the input pressure (p) is sea pressure in units of dbar.*/

        g08 = 2.0*(8103.20462414788 +
                  y*(2175.341332000392 +
                      y*(-274.2290036817964 +
                          y*(197.4670779425016 +
                              y*(-68.5590309679152 + 9.98788038278032*y))) -
                  90.6734234051316*z) +
                      1.5*x*(-5458.34205214835 - 980.14153344888*y +
                          (4.0/3.0)*x*(2247.60742726704 -
                          340.1237483177863*1.25*x + 220.542973797483*y) +
                      180.142097805543*z) +
                  z*(-219.1676534131548 +
                      (-16.32775915649044 - 120.7020447884644*z)*z));

        g08 = x2*g08 +
                  x*(-7296.43987145382 +
                      z*(598.378809221703 +
                          z*(-156.8822727844005 +
                              (204.1334828179377 - 10.23755797323846*z)*z)) +
                      y*(-1480.222530425046 +
                          z*(-525.876123559641 +
                              (249.57717834054571 - 88.449193048287*z)*z) +
                          y*(-129.1994027934126 +
                              z*(1149.174198007428 +
                                  z*(-162.5751787551336 + 76.9195462169742*z)) +
                          y*(-30.0682112585625 - 1380.9597954037708*z +
                              y*(2.626801985426835 + 703.695562834065*z))))) +
              11625.62913253464 + 1702.453469893412*y;

        return (0.25*gsw_sfac*g08);
/*
! Note that this function avoids the singularity that occurs at SA = 0 if
! the straightforward expression for the dilution coefficient of seawater,
! SA*g_SA_SA is simply evaluated as SA.*gsw_gibbs(2,0,0,SA,t,p).
*/
}
/*
!==========================================================================
function gsw_dynamic_enthalpy(sa,ct,p)
!==========================================================================

!  Calculates dynamic enthalpy of seawater using the computationally-
!  efficient expression for specific volume in terms of SA, CT and p
!  (Roquet et al., 2014).  Dynamic enthalpy is defined as enthalpy minus
|  potential enthalpy (Young, 2010).
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature (ITS-90)               [deg C]
! p      : sea pressure                                    [dbar]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
! dynamic_enthalpy  :  dynamic enthalpy                    [J/kg]
*/
double
gsw_dynamic_enthalpy(double sa, double ct, double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  dynamic_enthalpy_part, xs, ys, z;

        xs      = sqrt(gsw_sfac*sa + offset);
        ys      = ct*0.025;
        z       = p*1e-4;

        dynamic_enthalpy_part =
       z*(h001 + xs*(h101 + xs*(h201 + xs*(h301 + xs*(h401
    + xs*(h501 + h601*xs))))) + ys*(h011 + xs*(h111 + xs*(h211 + xs*(h311
    + xs*(h411 + h511*xs)))) + ys*(h021 + xs*(h121 + xs*(h221 + xs*(h321
    + h421*xs))) + ys*(h031 + xs*(h131 + xs*(h231 + h331*xs)) + ys*(h041
    + xs*(h141 + h241*xs) + ys*(h051 + h151*xs + h061*ys))))) + z*(h002
    + xs*(h102 + xs*(h202 + xs*(h302 + xs*(h402 + h502*xs)))) + ys*(h012
    + xs*(h112 + xs*(h212 + xs*(h312 + h412*xs))) + ys*(h022 + xs*(h122
    + xs*(h222 + h322*xs)) + ys*(h032 + xs*(h132 + h232*xs) + ys*(h042
    + h142*xs + h052*ys)))) + z*(h003 + xs*(h103 + xs*(h203 + xs*(h303
    + h403*xs))) + ys*(h013 + xs*(h113 + xs*(h213 + h313*xs)) + ys*(h023
    + xs*(h123 + h223*xs) + ys*(h033 + h133*xs + h043*ys))) + z*(h004
    + xs*(h104 + h204*xs) + ys*(h014 + h114*xs + h024*ys) + z*(h005
    + h105*xs + h015*ys + z*(h006 + h007*z))))));

        return (dynamic_enthalpy_part*db2pa*1e4);
}
/*
!==========================================================================
function gsw_enthalpy(sa,ct,p)
!==========================================================================

!  Calculates specific enthalpy of seawater using the computationally-
!  efficient expression for specific volume in terms of SA, CT and p
!  (Roquet et al., 2014).
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature (ITS-90)               [deg C]
! p      : sea pressure                                    [dbar]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
! enthalpy  :  specific enthalpy of seawater               [J/kg]
*/
double
gsw_enthalpy(double sa, double ct, double p)
{
        GSW_TEOS10_CONSTANTS;
        return (gsw_cp0*ct + gsw_dynamic_enthalpy(sa,ct,p));
}
/*
!==========================================================================
elemental function gsw_enthalpy_ct_exact (sa, ct, p)
!==========================================================================
!
!  Calculates specific enthalpy of seawater from Absolute Salinity and
!  Conservative Temperature and pressure.
!
!  Note that this function uses the full Gibbs function.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  enthalpy_CT_exact  =  specific enthalpy                         [ J/kg ]
!--------------------------------------------------------------------------
*/
double
gsw_enthalpy_ct_exact(double sa, double ct, double p)
{
        double  t;

        t = gsw_t_from_ct(sa,ct,p);
        return (gsw_enthalpy_t_exact(sa,t,p));
}
/*
!==========================================================================
elemental function gsw_enthalpy_diff (sa, ct, p_shallow, p_deep)
!==========================================================================
!
!  Calculates the difference of the specific enthalpy of seawater between
!  two different pressures, p_deep (the deeper pressure) and p_shallow
!  (the shallower pressure), at the same values of SA and CT.  This
!  function uses the computationally-efficient expression for specific
!  volume in terms of SA, CT and p (Roquet et al., 2014).  The output
!  (enthalpy_diff_CT) is the specific enthalpy evaluated at (SA,CT,p_deep)
!  minus the specific enthalpy at (SA,CT,p_shallow).
!
!  SA         =  Absolute Salinity                                 [ g/kg ]
!  CT         =  Conservative Temperature (ITS-90)                [ deg C ]
!  p_shallow  =  upper sea pressure                                [ dbar ]
!                ( i.e. shallower absolute pressure - 10.1325 dbar )
!  p_deep     =  lower sea pressure                                [ dbar ]
!                ( i.e. deeper absolute pressure - 10.1325 dbar )
!
!  enthalpy_diff_CT  =  difference of specific enthalpy            [ J/kg ]
!                       (deep minus shallow)
!--------------------------------------------------------------------------
*/
double
gsw_enthalpy_diff(double sa, double ct, double p_shallow, double p_deep)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  dynamic_enthalpy_shallow, dynamic_enthalpy_deep,
                part_1, part_2, part_3, part_4, part_5, xs, ys,
                z_deep, z_shallow;

        xs = sqrt(gsw_sfac*sa + offset);
        ys = ct*0.025;
        z_shallow = p_shallow*1e-4;
        z_deep = p_deep*1e-4;

        part_1 = h001 + xs*(h101 + xs*(h201 + xs*(h301 + xs*(h401
            + xs*(h501 + h601*xs))))) + ys*(h011 + xs*(h111 + xs*(h211+ xs*(h311
            + xs*(h411 + h511*xs)))) + ys*(h021 + xs*(h121 + xs*(h221 + xs*(h321
            + h421*xs))) + ys*(h031 + xs*(h131 + xs*(h231 + h331*xs)) + ys*(h041
            + xs*(h141 + h241*xs) + ys*(h051 + h151*xs + h061*ys)))));

        part_2 = h002 + xs*(h102 + xs*(h202 + xs*(h302 + xs*(h402 + h502*xs))))
            + ys*(h012 + xs*(h112 + xs*(h212 + xs*(h312 + h412*xs))) + ys*(h022
            + xs*(h122 + xs*(h222 + h322*xs)) + ys*(h032 + xs*(h132 + h232*xs)
            + ys*(h042 + h142*xs + h052*ys))));

        part_3 = h003 + xs*(h103 + xs*(h203 + xs*(h303 + h403*xs))) + ys*(h013
            + xs*(h113 + xs*(h213 + h313*xs)) + ys*(h023 + xs*(h123 + h223*xs)
            + ys*(h033 + h133*xs + h043*ys)));

        part_4 = h004 + xs*(h104 + h204*xs) + ys*(h014 + h114*xs + h024*ys);

        part_5 = h005 + h105*xs + h015*ys;

        dynamic_enthalpy_shallow =  z_shallow*(part_1 + z_shallow*(part_2
            + z_shallow*(part_3 + z_shallow*(part_4 + z_shallow*(part_5
            + z_shallow*(h006 + h007*z_shallow))))));

        dynamic_enthalpy_deep = z_deep*(part_1 + z_deep*(part_2 + z_deep*(part_3
            + z_deep*(part_4+z_deep*(part_5 + z_deep*(h006 + h007*z_deep))))));

        return ((dynamic_enthalpy_deep - dynamic_enthalpy_shallow)*db2pa*1e4);
}
/*
!==========================================================================
elemental subroutine gsw_enthalpy_first_derivatives (sa, ct, p, h_sa, h_ct)
!==========================================================================
!
!  Calculates the following two derivatives of specific enthalpy (h) of
!  seawater using the computationally-efficient expression for
!  specific volume in terms of SA, CT and p (Roquet et al., 2014).
!   (1) h_SA, the derivative with respect to Absolute Salinity at
!       constant CT and p, and
!   (2) h_CT, derivative with respect to CT at constant SA and p.
!  Note that h_P is specific volume (1/rho) it can be calculated by calling
!  gsw_specvol(SA,CT,p).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  h_SA  =  The first derivative of specific enthalpy with respect to
!           Absolute Salinity at constant CT and p.
!                                            [ J/(kg (g/kg))]  i.e. [ J/g ]
!  h_CT  =  The first derivative of specific enthalpy with respect to
!           CT at constant SA and p.                           [ J/(kg K) ]
!--------------------------------------------------------------------------
*/
void
gsw_enthalpy_first_derivatives(double sa, double ct, double p, double *h_sa,
                                double *h_ct)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  dynamic_h_ct_part, dynamic_h_sa_part, xs, ys, z;

        xs = sqrt(gsw_sfac*sa + offset);
        ys = ct*0.025;
        z = p*1e-4;

        if (h_sa != NULL) {

            dynamic_h_sa_part =  z*(h101 + xs*(2.0*h201 + xs*(3.0*h301
                + xs*(4.0*h401 + xs*(5.0*h501 + 6.0*h601*xs)))) + ys*(h111
                + xs*(2.0*h211 + xs*(3.0*h311 + xs*(4.0*h411
                + 5.0*h511*xs))) + ys*(h121 + xs*(2.0*h221 + xs*(3.0*h321
                + 4.0*h421*xs)) + ys*(h131 + xs*(2.0*h231 + 3.0*h331*xs)
                + ys*(h141 + 2.0*h241*xs + h151*ys)))) + z*(h102
                + xs*(2.0*h202 + xs*(3.0*h302 + xs*(4.0*h402
                + 5.0*h502*xs))) + ys*(h112 + xs*(2.0*h212 + xs*(3.0*h312
                + 4.0*h412*xs)) + ys*(h122 + xs*(2.0*h222 + 3.0*h322*xs)
                + ys*(h132 + 2.0*h232*xs + h142*ys ))) + z*(h103 + xs*(2.0*h203
                + xs*(3.0*h303 + 4.0*h403*xs)) + ys*(h113 + xs*(2.0*h213
                + 3.0*h313*xs) + ys*(h123 + 2.0*h223*xs + h133*ys))
                + z*(h104 + 2.0*h204*xs + h114*ys + h105*z))));

            *h_sa = 1e8*0.5*gsw_sfac*dynamic_h_sa_part/xs;

        }

        if (h_ct != NULL) {

            dynamic_h_ct_part = z*(h011 + xs*(h111
                + xs*(h211 + xs*(h311 + xs*(h411
                + h511*xs)))) + ys*(2.0*(h021 + xs*(h121 + xs*(h221 + xs*(h321
                + h421*xs)))) + ys*(3.0*(h031 + xs*(h131 + xs*(h231 + h331*xs)))
                + ys*(4.0*(h041 + xs*(h141 + h241*xs)) + ys*(5.0*(h051
                + h151*xs) + 6.0*h061*ys)))) + z*(h012 + xs*(h112 + xs*(h212
                + xs*(h312 + h412*xs))) + ys*(2.0*(h022 + xs*(h122 + xs*(h222
                + h322*xs))) + ys*(3.0*(h032 + xs*(h132 + h232*xs))
                + ys*(4.0*(h042 + h142*xs) + 5.0*h052*ys))) + z*(h013
                + xs*(h113 + xs*(h213 + h313*xs)) + ys*(2.0*(h023 + xs*(h123
                + h223*xs)) + ys*(3.0*(h033 + h133*xs) + 4.0*h043*ys))
                + z*(h014 + h114*xs + 2.0*h024*ys + h015*z ))));

            *h_ct = gsw_cp0 + 1e8*0.025*dynamic_h_ct_part;

        }
}
/*
!==========================================================================
elemental subroutine gsw_enthalpy_first_derivatives_ct_exact (sa, ct, p, &
                                                              h_sa, h_ct)
!==========================================================================
!
!  Calculates the following two derivatives of specific enthalpy (h)
!   (1) h_SA, the derivative with respect to Absolute Salinity at
!       constant CT and p, and
!   (2) h_CT, derivative with respect to CT at constant SA and p.
!  Note that h_P is specific volume (1/rho) it can be calculated by calling
!  gsw_specvol_CT_exact(SA,CT,p). This function uses the full Gibbs function.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  h_SA  =  The first derivative of specific enthalpy with respect to
!           Absolute Salinity at constant CT and p.
!                                            [ J/(kg (g/kg))]  i.e. [ J/g ]
!  h_CT  =  The first derivative of specific enthalpy with respect to
!           CT at constant SA and p.                           [ J/(kg K) ]
!--------------------------------------------------------------------------
*/
void
gsw_enthalpy_first_derivatives_ct_exact(double sa, double ct, double p,
                double *h_sa, double *h_ct)
{
        GSW_TEOS10_CONSTANTS;
        double  g_sa_mod_pt, g_sa_mod_t, pt0, t, temp_ratio, x, y, y_pt, z;

        t = gsw_t_from_ct(sa,ct,p);
        pt0 = gsw_pt_from_ct(sa,ct);

        temp_ratio = (gsw_t0 + t)/(gsw_t0 + pt0);

        if (h_ct != NULL) *h_ct = gsw_cp0*temp_ratio;

        if (h_sa == NULL) return;

        x = sqrt(gsw_sfac*sa);
        y = 0.025*t;
        z = rec_db2pa*p;
             /*note.the input pressure (p) is sea pressure in units of dbar.*/

        g_sa_mod_t = 8645.36753595126 + z*(-6620.98308089678 +
            z*(769.588305957198 + z*(-193.0648640214916 +
            (31.6816345533648 - 5.24960313181984*z)*z))) +
            x*(-7296.43987145382 + x*(8103.20462414788 +
            y*(2175.341332000392 + y*(-274.2290036817964 +
            y*(197.4670779425016 + y*(-68.5590309679152 + 9.98788038278032*y)))
            - 90.6734234051316*z) +
            x*(-5458.34205214835 - 980.14153344888*y +
            x*(2247.60742726704 - 340.1237483177863*x + 220.542973797483*y)
            + 180.142097805543*z) +
            z*(-219.1676534131548 + (-16.32775915649044
            - 120.7020447884644*z)*z)) +
            z*(598.378809221703 + z*(-156.8822727844005 + (204.1334828179377
            - 10.23755797323846*z)*z)) +
            y*(-1480.222530425046 + z*(-525.876123559641 + (249.57717834054571
            - 88.449193048287*z)*z) +
            y*(-129.1994027934126 + z*(1149.174198007428 +
            z*(-162.5751787551336 + 76.9195462169742*z)) +
            y*(-30.0682112585625 - 1380.9597954037708*z + y*(2.626801985426835
            + 703.695562834065*z))))) +
            y*(1187.3715515697959 + z*(1458.233059470092 +
            z*(-687.913805923122 + z*(249.375342232496 + z*(-63.313928772146
            + 14.09317606630898*z)))) +
            y*(1760.062705994408 + y*(-450.535298526802 +
            y*(182.8520895502518 + y*(-43.3206481750622 + 4.26033941694366*y) +
            z*(-595.457483974374 + (149.452282277512 - 72.9745838003176*z)*z)) +
            z*(1388.489628266536 + z*(-409.779283929806 + (227.123395681188
            - 22.2565468652826*z)*z))) +
            z*(-1721.528607567954 + z*(674.819060538734 +
            z*(-356.629112415276 + (88.4080716616 - 15.84003094423364*z)*z)))));

        g_sa_mod_t = 0.5*gsw_sfac*g_sa_mod_t;

        y_pt = 0.025*pt0;

        g_sa_mod_pt = 8645.36753595126 +
            x*(-7296.43987145382 + x*(8103.20462414788 +
            y_pt*(2175.341332000392 + y_pt*(-274.2290036817964 +
            y_pt*(197.4670779425016 + y_pt*(-68.5590309679152
            + 9.98788038278032*y_pt)))) +
            x*(-5458.34205214835 - 980.14153344888*y_pt +
            x*(2247.60742726704 - 340.1237483177863*x
            + 220.542973797483*y_pt))) +
            y_pt*(-1480.222530425046 + y_pt*(-129.1994027934126 +
            y_pt*(-30.0682112585625 + y_pt*2.626801985426835)))) +
            y_pt*(1187.3715515697959 + y_pt*(1760.062705994408
            + y_pt*(-450.535298526802 +
            y_pt*(182.8520895502518 + y_pt*(-43.3206481750622
            + 4.26033941694366*y_pt)))));

        g_sa_mod_pt = 0.5*gsw_sfac*g_sa_mod_pt;

        *h_sa = g_sa_mod_t - temp_ratio*g_sa_mod_pt;
}
/*
!==========================================================================
elemental function gsw_enthalpy_ice (t, p)
!==========================================================================
!
! Calculates the specific enthalpy of ice (h_Ih).
!
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!        ( i.e. absolute pressure - 10.1325 dbar )
!
!  gsw_enthalpy_ice  :  specific enthalpy of ice                   [ J/kg ]
!--------------------------------------------------------------------------
*/
double
gsw_enthalpy_ice(double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_GIBBS_ICE_COEFFICIENTS;
        double  tau, dzi, g0;
        DCOMPLEX r2, sqtau_t1, sqtau_t2, g;

        tau = (t + gsw_t0)*rec_tt;

        dzi = db2pa*p*rec_pt;

        g0 = g00 + dzi*(g01 + dzi*(g02 + dzi*(g03 + g04*dzi)));

        r2 = r20 + dzi*(r21 + r22*dzi);

        sqtau_t1 = (tau*tau)/(t1*t1);
        sqtau_t2 = (tau*tau)/(t2*t2);

        g = r1*t1*(log(1.0 - sqtau_t1) + sqtau_t1)
            + r2*t2*(log(1.0 - sqtau_t2) + sqtau_t2);

        return (g0 + tt*real(g));
}
/*
!==========================================================================
elemental subroutine gsw_enthalpy_second_derivatives (sa, ct, p, h_sa_sa, &
                                                      h_sa_ct, h_ct_ct)
! =========================================================================
!
!  Calculates the following three second-order derivatives of specific
!  enthalpy (h),using the computationally-efficient expression for
!  specific volume in terms of SA, CT and p (Roquet et al., 2014).
!   (1) h_SA_SA, second-order derivative with respect to Absolute Salinity
!       at constant CT & p.
!   (2) h_SA_CT, second-order derivative with respect to SA & CT at
!       constant p.
!   (3) h_CT_CT, second-order derivative with respect to CT at constant SA
!       and p.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  h_SA_SA  =  The second derivative of specific enthalpy with respect to
!              Absolute Salinity at constant CT & p.    [ J/(kg (g/kg)^2) ]
!  h_SA_CT  =  The second derivative of specific enthalpy with respect to
!              SA and CT at constant p.                  [ J/(kg K(g/kg)) ]
!  h_CT_CT  =  The second derivative of specific enthalpy with respect to
!              CT at constant SA and p.                      [ J/(kg K^2) ]
!--------------------------------------------------------------------------
*/
void
gsw_enthalpy_second_derivatives(double sa, double ct, double p,
                double *h_sa_sa, double *h_sa_ct, double *h_ct_ct)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  dynamic_h_ct_ct_part, dynamic_h_sa_ct_part,
                dynamic_h_sa_sa_part, xs, xs2, ys, z;

        xs = sqrt(gsw_sfac*sa + offset);
        ys = ct*0.025;
        z = p*1e-4;

        if (h_sa_sa != NULL) {

            xs2 = pow(xs,2);
            dynamic_h_sa_sa_part = z*(-h101 + xs2*(3.0*h301 + xs*(8.0*h401
                + xs*(15.0*h501 + 24.0*h601*xs))) + ys*(- h111
                + xs2*(3.0*h311 + xs*(8.0*h411 + 15.0*h511*xs)) + ys*(-h121
                + xs2*(3.0*h321 + 8.0*h421*xs) + ys*(-h131 + 3.0*h331*xs2
                + ys*(-h141 - h151*ys)))) + z*(-h102 + xs2*(3.0*h302
                + xs*(8.0*h402 + 15.0*h502*xs)) + ys*(-h112 + xs2*(3.0*h312
                + 8.0*h412*xs) + ys*(-h122 + 3.0*h322*xs2 + ys*(-h132
                - h142*ys ))) + z*(xs2*(8.0*h403*xs + 3.0*h313*ys)
                + z*(-h103 + 3.0*h303*xs2 + ys*(-h113 + ys*(-h123 - h133*ys))
                + z*(-h104 - h114*ys - h105*z)))));

            *h_sa_sa = 1e8*0.25*gsw_sfac*gsw_sfac*dynamic_h_sa_sa_part/
                        pow(xs,3);

        }

        if (h_sa_ct != NULL) {

            dynamic_h_sa_ct_part = z*(h111 + xs*(2.0*h211 + xs*(3.0*h311
                + xs*(4.0*h411 + 5.0*h511*xs))) + ys*(2.0*h121
                + xs*(4.0*h221 + xs*(6.0*h321 + 8.0*h421*xs))
                + ys*(3.0*h131 + xs*(6.0*h231 + 9.0*h331*xs)
                + ys*(4.0*h141 + 8.0*h241*xs + 5.0*h151*ys ))) + z*(h112
                + xs*(2.0*h212 + xs*(3.0*h312 + 4.0*h412*xs))
                + ys*(2.0*h122 + xs*(4.0*h222 + 6.0*h322*xs)
                + ys*(3.0*h132 + 6.0*h232*xs + 4.0*h142*ys)) + z*(h113
                + xs*(2.0*h213 + 3.0*h313*xs) + ys*(2.0*h123
                + 4.0*h223*xs + 3.0*h133*ys) + h114*z)));

            *h_sa_ct = 1e8*0.025*0.5*gsw_sfac*dynamic_h_sa_ct_part/xs;

        }

        if (h_ct_ct != NULL) {

            dynamic_h_ct_ct_part = z*(2.0*h021 + xs*(2.0*h121 + xs*(2.0*h221
                + xs*(2.0*h321 + 2.0*h421*xs))) + ys*(6.0*h031
                + xs*(6.0*h131 + xs*(6.0*h231 + 6.0*h331*xs))
                + ys*(12.0*h041 + xs*(12.0*h141 + 12.0*h241*xs)
                + ys*(20.0*h051 + 20.0*h151*xs + 30.0*h061*ys)))
                + z*(2.0*h022 + xs*(2.0*h122 + xs*(2.0*h222
                + 2.0*h322*xs)) + ys*(6.0*h032 + xs*(6.0*h132
                + 6.0*h232*xs) + ys*(12.0*h042 + 12.0*h142*xs
                + 20.0*h052*ys)) + z*(2.0*h023 + xs*(2.0*h123
                + 2.0*h223*xs) + ys*(6.0*h133*xs + 6.0*h033
                + 12.0*h043*ys) + 2.0*h024*z)));

            *h_ct_ct = 1e8*6.25e-4*dynamic_h_ct_ct_part;

        }
}
/*
!==========================================================================
elemental subroutine gsw_enthalpy_second_derivatives_ct_exact (sa, ct, p, &
                                                 h_sa_sa, h_sa_ct, h_ct_ct)
!==========================================================================
!
!  Calculates three second-order derivatives of specific enthalpy (h).
!  Note that this function uses the full Gibbs function.
!
!  sa  =  Absolute Salinity                                        [ g/kg ]
!  ct  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  h_sa_sa  =  The second derivative of specific enthalpy with respect to
!              Absolute Salinity at constant ct & p.    [ J/(kg (g/kg)^2) ]
!  h_sa_ct  =  The second derivative of specific enthalpy with respect to
!              sa and ct at constant p.                  [ J/(kg K(g/kg)) ]
!  h_ct_ct  =  The second derivative of specific enthalpy with respect to
!              ct at constant sa and p.                      [ J/(kg K^2) ]
!--------------------------------------------------------------------------
*/
void
gsw_enthalpy_second_derivatives_ct_exact(double sa, double ct, double p,
                double *h_sa_sa, double *h_sa_ct, double *h_ct_ct)
{
        GSW_TEOS10_CONSTANTS;
        double  factor, gsa_pt0, gsat_pt0, gsat, part_b, pt0, h_ct_ct_val,
                rec_abs_pt0, rec_gtt_pt0, rec_gtt, t, temp_ratio,
                gsasa, gsasa_pt0, pr0 = 0.0, sa_small = 1e-100;
        int     n0=0, n1=1, n2=2;

        pt0 = gsw_pt_from_ct(sa,ct);
        rec_abs_pt0 = 1.0/(gsw_t0 + pt0);
        t = gsw_pt_from_t(sa,pt0,pr0,p);
        temp_ratio = (gsw_t0 + t)*rec_abs_pt0;

        rec_gtt_pt0 = 1.0/gsw_gibbs(n0,n2,n0,sa,pt0,pr0);
        rec_gtt = 1.0/gsw_gibbs(n0,n2,n0,sa,t,p);

        /*h_ct_ct is naturally well-behaved as sa approaches zero.*/
        h_ct_ct_val = gsw_cp0*gsw_cp0*
            (temp_ratio*rec_gtt_pt0 - rec_gtt)*(rec_abs_pt0*rec_abs_pt0);

        if (h_ct_ct != NULL) *h_ct_ct = h_ct_ct_val;

        if ((h_sa_sa == NULL) && (h_sa_ct == NULL)) return;

        gsat_pt0 = gsw_gibbs(n1,n1,n0,sa,pt0,pr0);
        gsat = gsw_gibbs(n1,n1,n0,sa,t,p);
        gsa_pt0 = gsw_gibbs(n1,n0,n0,sa,pt0,pr0);

        part_b = (temp_ratio*gsat_pt0*rec_gtt_pt0 - gsat*rec_gtt)*rec_abs_pt0;
        factor = gsa_pt0/gsw_cp0;

        if (h_sa_sa != NULL) {

            gsasa = gsw_gibbs(n2,n0,n0,sa,t,p);
            gsasa_pt0 = gsw_gibbs(n2,n0,n0,sa,pt0,pr0);
          /*
          ! h_sa_sa has a singularity at sa = 0, and blows up as sa
          ! approaches zero.
         */
            *h_sa_sa = gsasa - temp_ratio*gsasa_pt0
                + temp_ratio*gsat_pt0*gsat_pt0*rec_gtt_pt0
                - gsat*gsat*rec_gtt
                - 2.0*gsa_pt0*part_b + (factor*factor)*h_ct_ct_val;

        }
        if (h_sa_ct == NULL) return;

        /*
        ! h_sa_ct should not blow up as sa approaches zero.
        ! The following lines of code ensure that the h_sa_ct output
        ! of this function does not blow up in this limit.
        ! That is, when sa < 1e-100 g/kg, we force the h_sa_ct
        ! output to be the same as if sa = 1e-100 g/kg.
        */
        if (sa < sa_small) {
            rec_gtt_pt0 = 1.0/gsw_gibbs(n0,n2,n0,sa_small,pt0,pr0);
            rec_gtt = 1.0/gsw_gibbs(n0,n2,n0,sa_small,t,p);
            gsat_pt0 = gsw_gibbs(n1,n1,n0,sa_small,pt0,pr0);
            gsat = gsw_gibbs(n1,n1,n0,sa_small,t,p);
            gsa_pt0 = gsw_gibbs(n1,n0,n0,sa_small,pt0,pr0);
            part_b = (temp_ratio*gsat_pt0*rec_gtt_pt0
                        - gsat*rec_gtt)*rec_abs_pt0;
            factor = gsa_pt0/gsw_cp0;
        }

        *h_sa_ct  = gsw_cp0*part_b - factor*h_ct_ct_val;
}
/*
!==========================================================================
function gsw_enthalpy_sso_0(p)
!==========================================================================

!  This function calculates enthalpy at the Standard Ocean Salinity, SSO,
!  and at a Conservative Temperature of zero degrees C, as a function of
!  pressure, p, in dbar, using a streamlined version of the
!  computationally-efficient expression for specific volume, that is, a
!  streamlined version of the code "gsw_enthalpy(SA,CT,p)".
!
! p      : sea pressure                                    [dbar]
!
! enthalpy_sso_0 : enthalpy(sso,0,p)
*/
double
gsw_enthalpy_sso_0(double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  dynamic_enthalpy_sso_0_p, z;

        z = p*1.0e-4;

        dynamic_enthalpy_sso_0_p =
                        z*( 9.726613854843870e-4 + z*(-2.252956605630465e-5
                        + z*( 2.376909655387404e-6 + z*(-1.664294869986011e-7
                        + z*(-5.988108894465758e-9 + z*(h006 + h007*z))))));
        return (dynamic_enthalpy_sso_0_p*db2pa*1.0e4);
}
/*
!==========================================================================
function gsw_enthalpy_t_exact(sa,t,p)
!==========================================================================

! Calculates the specific enthalpy of seawater
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
!
! gsw_enthalpy_t_exact : specific enthalpy                 [J/kg]
*/
double
gsw_enthalpy_t_exact(double sa, double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        int     n0=0, n1=1;

        return (gsw_gibbs(n0,n0,n0,sa,t,p) -
                (t+gsw_t0)*gsw_gibbs(n0,n1,n0,sa,t,p));
}
/*
!==========================================================================
elemental subroutine gsw_entropy_first_derivatives (sa, ct, eta_sa, eta_ct)
! =========================================================================
!
!  Calculates the following two partial derivatives of specific entropy
!  (eta)
!   (1) eta_SA, the derivative with respect to Absolute Salinity at
!       constant Conservative Temperature, and
!   (2) eta_CT, the derivative with respect to Conservative Temperature at
!       constant Absolute Salinity.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!
!  eta_SA =  The derivative of specific entropy with respect to
!            Absolute Salinity (in units of g kg^-1) at constant
!            Conservative Temperature.
!            eta_SA has units of:         [ J/(kg K(g/kg))]  or [ J/(g K) ]
!  eta_CT =  The derivative of specific entropy with respect to
!            Conservative Temperature at constant Absolute Salinity.
!            eta_CT has units of:                            [ J/(kg K^2) ]
!--------------------------------------------------------------------------
*/
void
gsw_entropy_first_derivatives(double sa, double ct, double *eta_sa,
        double *eta_ct)
{
        GSW_TEOS10_CONSTANTS;
        double  pt, pr0 = 0.0;
        int     n0=0, n1=1;

        pt = gsw_pt_from_ct(sa,ct);

        if (eta_sa != NULL)
            *eta_sa = -(gsw_gibbs(n1,n0,n0,sa,pt,pr0))/(gsw_t0 + pt);

        if (eta_ct != NULL)
            *eta_ct = gsw_cp0/(gsw_t0 + pt);
}
/*
!=========================================================================
elemental function gsw_entropy_from_ct (sa, ct)
!=========================================================================
!
!  Calculates specific entropy of seawater from Conservative Temperature.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!
!  entropy  =  specific entropy                                   [ deg C ]
!--------------------------------------------------------------------------
*/
double
gsw_entropy_from_ct(double sa, double ct)
{
        double  pt0;

        pt0 = gsw_pt_from_ct(sa, ct);
        return (-gsw_gibbs(0,1,0,sa,pt0,0));
}
/*
!==========================================================================
elemental function gsw_entropy_from_pt (sa, pt)
!==========================================================================
!
!  Calculates specific entropy of seawater.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  pt  =  potential temperature (ITS-90)                          [ deg C ]
!
!  entropy  =  specific entropy                                [ J/(kg*K) ]
!--------------------------------------------------------------------------
*/
double
gsw_entropy_from_pt(double sa, double pt)
{
        int     n0 = 0, n1 = 1;
        double  pr0 = 0.0;

        return (-gsw_gibbs(n0,n1,n0,sa,pt,pr0));
}
/*
!==========================================================================
function gsw_entropy_from_t(sa,t,p)
!==========================================================================

! Calculates the specific entropy of seawater
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
!
! gsw_entropy_from_t : specific entropy                    [J/(kg K)]
*/
double
gsw_entropy_from_t(double sa, double t, double p)
{
        int     n0=0, n1=1;

        return (-gsw_gibbs(n0,n1,n0,sa,t,p));

}
/*
!==========================================================================
elemental function gsw_entropy_ice (t, p)
!==========================================================================
!
!  Calculates specific entropy of ice.
!
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  ice_entropy  =  specific entropy of ice                 [ J kg^-1 K^-1 ]
!--------------------------------------------------------------------------
*/
double
gsw_entropy_ice(double t, double p)
{
        return (-gsw_gibbs_ice(1,0,t,p));
}
/*
!==========================================================================
function gsw_entropy_part(sa,t,p)
!==========================================================================

! entropy minus the terms that are a function of only SA
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
!
! entropy_part : entropy part
*/
double
gsw_entropy_part(double sa, double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        double  x2, x, y, z, g03, g08;

        x2      = gsw_sfac*sa;
        x       = sqrt(x2);
        y       = t*0.025;
        z       = p*1e-4;

        g03     = z*(-270.983805184062 +
                z*(776.153611613101 + z*(-196.51255088122 +
                   (28.9796526294175 - 2.13290083518327*z)*z))) +
                y*(-24715.571866078 + z*(2910.0729080936 +
                z*(-1513.116771538718 + z*(546.959324647056 +
                   z*(-111.1208127634436 + 8.68841343834394*z)))) +
                y*(2210.2236124548363 + z*(-2017.52334943521 +
                z*(1498.081172457456 + z*(-718.6359919632359 +
                   (146.4037555781616 - 4.9892131862671505*z)*z))) +
                y*(-592.743745734632 + z*(1591.873781627888 +
                z*(-1207.261522487504 + (608.785486935364 -
                   105.4993508931208*z)*z)) +
                y*(290.12956292128547 + z*(-973.091553087975 +
                z*(602.603274510125 + z*(-276.361526170076 +
                   32.40953340386105*z))) +
                y*(-113.90630790850321 + y*(21.35571525415769 -
                   67.41756835751434*z) +
                z*(381.06836198507096 + z*(-133.7383902842754 +
                   49.023632509086724*z)))))));

        g08     = x2*(z*(729.116529735046 +
                z*(-343.956902961561 + z*(124.687671116248 +
                   z*(-31.656964386073 + 7.04658803315449*z)))) +
                x*( x*(y*(-137.1145018408982 + y*(148.10030845687618 +
                   y*(-68.5590309679152 + 12.4848504784754*y))) -
                22.6683558512829*z) + z*(-175.292041186547 +
                   (83.1923927801819 - 29.483064349429*z)*z) +
                y*(-86.1329351956084 + z*(766.116132004952 +
                   z*(-108.3834525034224 + 51.2796974779828*z)) +
                y*(-30.0682112585625 - 1380.9597954037708*z +
                   y*(3.50240264723578 + 938.26075044542*z)))) +
                y*(1760.062705994408 + y*(-675.802947790203 +
                y*(365.7041791005036 + y*(-108.30162043765552 +
                   12.78101825083098*y) +
                z*(-1190.914967948748 + (298.904564555024 -
                   145.9491676006352*z)*z)) +
                z*(2082.7344423998043 + z*(-614.668925894709 +
                   (340.685093521782 - 33.3848202979239*z)*z))) +
                z*(-1721.528607567954 + z*(674.819060538734 +
                z*(-356.629112415276 + (88.4080716616 -
                   15.84003094423364*z)*z)))));

        return (-(g03 + g08)*0.025);
}

/*
!==========================================================================
function gsw_entropy_part_zerop(sa,pt0)
!==========================================================================

! entropy part evaluated at the sea surface
!
! sa     : Absolute Salinity                               [g/kg]
! pt0    : insitu temperature                              [deg C]
!
! entropy_part_zerop : entropy part at the sea surface
*/
double
gsw_entropy_part_zerop(double sa, double pt0)
{
        GSW_TEOS10_CONSTANTS;
        double  x2, x, y, g03, g08;

        x2      = gsw_sfac*sa;
        x       = sqrt(x2);
        y       = pt0*0.025;

        g03     = y*(-24715.571866078 + y*(2210.2236124548363 +
                y*(-592.743745734632 + y*(290.12956292128547 +
                y*(-113.90630790850321 + y*21.35571525415769)))));

        g08     = x2*(x*(x*(y*(-137.1145018408982 + y*(148.10030845687618 +
                y*(-68.5590309679152 + 12.4848504784754*y)))) +
                y*(-86.1329351956084 + y*(-30.0682112585625 +
                   y*3.50240264723578))) +
                y*(1760.062705994408 + y*(-675.802947790203 +
                y*(365.7041791005036 + y*(-108.30162043765552 +
                   12.78101825083098*y)))));

        return (-(g03 + g08)*0.025);
}
/*
!==========================================================================
elemental subroutine gsw_entropy_second_derivatives (sa, ct, eta_sa_sa, &
                                                     eta_sa_ct, eta_ct_ct)
! =========================================================================
!
!  Calculates the following three second-order partial derivatives of
!  specific entropy (eta)
!   (1) eta_SA_SA, the second derivative with respect to Absolute
!       Salinity at constant Conservative Temperature, and
!   (2) eta_SA_CT, the derivative with respect to Absolute Salinity and
!       Conservative Temperature.
!   (3) eta_CT_CT, the second derivative with respect to Conservative
!       Temperature at constant Absolute Salinity.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!
!  eta_SA_SA =  The second derivative of specific entropy with respect
!               to Absolute Salinity (in units of g kg^-1) at constant
!               Conservative Temperature.
!               eta_SA_SA has units of:                 [ J/(kg K(g/kg)^2)]
!  eta_SA_CT =  The second derivative of specific entropy with respect
!               to Conservative Temperature at constant Absolute
!               Salinity. eta_SA_CT has units of:     [ J/(kg (g/kg) K^2) ]
!  eta_CT_CT =  The second derivative of specific entropy with respect
!               to Conservative Temperature at constant Absolute
!               Salinity.  eta_CT_CT has units of:           [ J/(kg K^3) ]
!--------------------------------------------------------------------------
*/
void
gsw_entropy_second_derivatives(double sa, double ct,
        double *eta_sa_sa, double *eta_sa_ct, double *eta_ct_ct)
{
        GSW_TEOS10_CONSTANTS;
        double  abs_pt, ct_pt, ct_sa, pt, ct_ct, pr0 = 0.0;
        int     n0=0, n1=1, n2=2;

        pt = gsw_pt_from_ct(sa,ct);
        abs_pt = gsw_t0 + pt;

        ct_pt = -(abs_pt*gsw_gibbs(n0,n2,n0,sa,pt,pr0))/gsw_cp0;

        ct_ct = -gsw_cp0/(ct_pt*abs_pt*abs_pt);

        if ((eta_sa_ct != NULL) || (eta_sa_sa != NULL)) {

            ct_sa = (gsw_gibbs(n1,n0,n0,sa,pt,pr0) -
                       (abs_pt*gsw_gibbs(n1,n1,n0,sa,pt,pr0)))/gsw_cp0;

            if (eta_sa_ct != NULL) *eta_sa_ct = -ct_sa*ct_ct;

            if (eta_sa_sa != NULL)
                *eta_sa_sa = -gsw_gibbs(n2,n0,n0,sa,pt,pr0)/abs_pt +
                                             ct_sa*ct_sa*ct_ct;
        }

        if (eta_ct_ct != NULL) *eta_ct_ct = ct_ct;
}
/*
!==========================================================================
function gsw_fdelta(p,lon,lat)
!==========================================================================

! Calculates fdelta.
!
! p      : sea pressure                                    [dbar]
! lon    : longitude                                       [deg E]
! lat    : latitude                                        [deg N]
!
! gsw_fdelta : Absolute Salinty Anomaly                    [unitless]
*/
double
gsw_fdelta(double p, double lon, double lat)
{
        double  sa, saar;

        saar    = gsw_saar(p,lon,lat);
        if (saar >= GSW_ERROR_LIMIT)
            sa  = GSW_INVALID_VALUE;
        else
            sa  = ((1.0 + 0.35)*saar)/(1.0 - 0.35*saar);
        return (sa);
}
/*
!==========================================================================
elemental subroutine gsw_frazil_properties (sa_bulk, h_bulk, p, &
                                            sa_final, ct_final, w_ih_final)
!==========================================================================
!
!  Calculates the mass fraction of ice (mass of ice divided by mass of ice
!  plus seawater), w_Ih_final, which results from given values of the bulk
!  Absolute Salinity, SA_bulk, bulk enthalpy, h_bulk, occurring at pressure
!  p.  The final values of Absolute Salinity, SA_final, and Conservative
!  Temperature, CT_final, of the interstitial seawater phase are also
!  returned.  This code assumes that there is no dissolved air in the
!  seawater (that is, saturation_fraction is assumed to be zero
!  throughout the code).
!
!  When the mass fraction w_Ih_final is calculated as being a positive
!  value, the seawater-ice mixture is at thermodynamic equlibrium.
!
!  This code returns w_Ih_final = 0 when the input bulk enthalpy, h_bulk,
!  is sufficiently large (i.e. sufficiently "warm") so that there is no ice
!  present in the final state.  In this case the final state consists of
!  only seawater rather than being an equlibrium mixture of seawater and
!  ice which occurs when w_Ih_final is positive.  Note that when
!  w_Ih_final = 0, the final seawater is not at the freezing temperature.
!
!  SA_bulk =  bulk Absolute Salinity of the seawater and ice mixture
!                                                                  [ g/kg ]
!  h_bulk  =  bulk enthalpy of the seawater and ice mixture        [ J/kg ]
!  p       =  sea pressure                                         [ dbar ]
!             ( i.e. absolute pressure - 10.1325 dbar )
!
!  SA_final    =  Absolute Salinity of the seawater in the final state,
!                 whether or not any ice is present.               [ g/kg ]
!  CT_final    =  Conservative Temperature of the seawater in the the final
!                 state, whether or not any ice is present.       [ deg C ]
!  w_Ih_final  =  mass fraction of ice in the final seawater-ice mixture.
!                 If this ice mass fraction is positive, the system is at
!                 thermodynamic equilibrium.  If this ice mass fraction is
!                 zero there is no ice in the final state which consists
!                 only of seawater which is warmer than the freezing
!                 temperature.                                   [unitless]
!--------------------------------------------------------------------------
*/
void
gsw_frazil_properties(double sa_bulk, double h_bulk, double p,
        double *sa_final, double *ct_final, double *w_ih_final)
{
        int     number_of_iterations;
        double  cp_ih, ctf_sa, ctf, dfunc_dw_ih, dfunc_dw_ih_mean_poly,
                func, func0, hf, h_hat_ct, h_hat_sa,
                h_ihf, sa, tf_sa, tf, w_ih_mean, w_ih_old, w_ih,
             /*
              ! Throughout this code seawater is taken to contain
              ! no dissolved air.
              */
                saturation_fraction = 0.0,
                num_f = 5.0e-2, num_f2 = 6.9e-7, num_p = 2.21;
        /*
        !---------------
        ! Finding func0
        !--------------
        */
        ctf = gsw_ct_freezing(sa_bulk,p,saturation_fraction);
        func0 = h_bulk - gsw_enthalpy_ct_exact(sa_bulk,ctf,p);
        /*
        !-----------------------------------------------------------------------
        ! When func0 is zero or positive we can immediately calculate the three
        ! outputs, as the bulk enthalpy, h_bulk, is too large to allow any ice
        ! at thermodynamic equilibrium. The result will be (warm) seawater with
        ! no frazil ice being present. The three outputs can be set and the rest
        ! of this code does not need to be performed.
        !-----------------------------------------------------------------------
        */
        if (func0 >= 0.0) {
            *sa_final = sa_bulk;
            *ct_final = gsw_ct_from_enthalpy_exact(sa_bulk,h_bulk,p);
            *w_ih_final = 0.0;
            return;
        }
        /*
        !-----------------------------------------------------------------------
        ! Begin to find the solution for those data points that have func0 < 0,
        ! implying that the output will be a positive ice mass fraction
        ! w_Ih_final.
        !
        ! Do a quasi-Newton step with a separate polynomial estimate of the
        ! derivative of func with respect to the ice mass fraction.  This
        ! section of the code delivers initial values of both w_Ih and SA to
        ! the rest of the more formal modified Newtons Method approach of
        ! McDougall and Wotherspoon (2014).
        !-----------------------------------------------------------------------
        */
        dfunc_dw_ih_mean_poly = 3.347814e+05
                                - num_f*func0*(1.0 + num_f2*func0) - num_p*p;
        w_ih = min(-func0/dfunc_dw_ih_mean_poly, 0.95);
        sa = sa_bulk/(1.0 - w_ih);
        if (sa < 0.0 || sa > 120.0) {
            *sa_final = GSW_INVALID_VALUE;
            *ct_final = *sa_final;
            *w_ih_final = *sa_final;
            return;
        }
        /*
        !-----------------------------------------------------------------------
        ! Calculating the estimate of the derivative of func, dfunc_dw_Ih, to be
        ! fed into the iterative Newton's Method.
        !-----------------------------------------------------------------------
        */
        ctf = gsw_ct_freezing(sa,p,saturation_fraction);
        hf = gsw_enthalpy_ct_exact(sa,ctf,p);
        tf = gsw_t_freezing(sa,p,saturation_fraction);
        h_ihf = gsw_enthalpy_ice(tf,p);
        cp_ih = gsw_cp_ice(tf,p);
        gsw_enthalpy_first_derivatives_ct_exact(sa,ctf,p,
                &h_hat_sa,&h_hat_ct);
        gsw_ct_freezing_first_derivatives(sa,p,saturation_fraction,&ctf_sa,
                                                                        NULL);
        gsw_t_freezing_first_derivatives(sa,p,saturation_fraction,&tf_sa,NULL);

        dfunc_dw_ih = hf - h_ihf
                - sa*(h_hat_sa + h_hat_ct*ctf_sa + w_ih*cp_ih*tf_sa/
                (1.0 - w_ih));
        /*
        !-----------------------------------------------------------------------
        ! Enter the main McDougall-Wotherspoon (2014) modified Newton-Raphson
        | loop
        !-----------------------------------------------------------------------
        */
        for (number_of_iterations = 1; number_of_iterations <= 3;
            number_of_iterations++) {

            if (number_of_iterations > 1) {
                /* on the first iteration these values are already known */
                ctf = gsw_ct_freezing(sa,p,saturation_fraction);
                hf = gsw_enthalpy_ct_exact(sa,ctf,p);
                tf = gsw_t_freezing(sa,p,saturation_fraction);
                h_ihf = gsw_enthalpy_ice(tf,p);
            }

            func = h_bulk - (1.0 - w_ih)*hf - w_ih*h_ihf;

            w_ih_old = w_ih;
            w_ih = w_ih_old - func/dfunc_dw_ih;
            w_ih_mean = 0.5*(w_ih + w_ih_old);

            if (w_ih_mean > 0.9) {
                /*This ensures that the mass fraction of ice never exceeds 0.9*/
                *sa_final = GSW_INVALID_VALUE;
                *ct_final = *sa_final;
                *w_ih_final = *sa_final;
                return;
            }

            sa = sa_bulk/(1.0 - w_ih_mean);
            ctf = gsw_ct_freezing(sa,p,saturation_fraction);
            hf = gsw_enthalpy_ct_exact(sa,ctf,p);
            tf = gsw_t_freezing(sa,p,saturation_fraction);
            h_ihf = gsw_enthalpy_ice(tf,p);
            cp_ih = gsw_cp_ice(tf,p);
            gsw_enthalpy_first_derivatives_ct_exact(sa,ctf,p,
                &h_hat_sa,&h_hat_ct);
            gsw_ct_freezing_first_derivatives(sa,p,saturation_fraction,&ctf_sa,
                                                                NULL);
            gsw_t_freezing_first_derivatives(sa,p,saturation_fraction,&tf_sa,
                                                                NULL);

            dfunc_dw_ih = hf - h_ihf - sa*(h_hat_sa + h_hat_ct*ctf_sa
                                           + w_ih_mean*cp_ih*tf_sa/
                                                (1.0 - w_ih_mean));

            w_ih = w_ih_old - func/dfunc_dw_ih;

            if (w_ih > 0.9) {
                /*This ensures that the mass fraction of ice never exceeds 0.9*/
                *sa_final = GSW_INVALID_VALUE;
                *ct_final = *sa_final;
                *w_ih_final = *sa_final;
                return;
            }

            sa = sa_bulk/(1.0 - w_ih);
        }

        *sa_final = sa;
        *ct_final = gsw_ct_freezing(sa,p,saturation_fraction);
        *w_ih_final = w_ih;

        if (*w_ih_final < 0.0) {
            /*
            ! This will only trap cases that are smaller than zero by just
            ! machine precision
            */
            *sa_final = sa_bulk;
            *ct_final = gsw_ct_from_enthalpy_exact(*sa_final,h_bulk,p);
            *w_ih_final = 0.0;
        }
}
/*
!==========================================================================
elemental subroutine gsw_frazil_properties_potential (sa_bulk, h_pot_bulk,&
                                         p, sa_final, ct_final, w_ih_final)
!==========================================================================
!
!  Calculates the mass fraction of ice (mass of ice divided by mass of ice
!  plus seawater), w_Ih_final, which results from given values of the bulk
!  Absolute Salinity, SA_bulk, bulk potential enthalpy, h_pot_bulk,
!  occurring at pressure p.  The final equilibrium values of Absolute
!  Salinity, SA_final, and Conservative Temperature, CT_final, of the
!  interstitial seawater phase are also returned.  This code assumes that
!  there is no dissolved air in the seawater (that is, saturation_fraction
!  is assumed to be zero throughout the code).
!
!  When the mass fraction w_Ih_final is calculated as being a positive
!  value, the seawater-ice mixture is at thermodynamic equlibrium.
!
!  This code returns w_Ih_final = 0 when the input bulk enthalpy, h_bulk,
!  is sufficiently large (i.e. sufficiently "warm") so that there is no ice
!  present in the final state.  In this case the final state consists of
!  only seawater rather than being an equlibrium mixture of seawater and
!  ice which occurs when w_Ih_final is positive.  Note that when
!  w_Ih_final = 0, the final seawater is not at the freezing temperature.
!
!  Note that this code uses the exact forms of CT_freezing and
!  pot_enthalpy_ice_freezing.
!
!  SA_bulk     =  bulk Absolute Salinity of the seawater and ice mixture
!                                                                  [ g/kg ]
!  h_pot_bulk  =  bulk potential enthalpy of the seawater and ice mixture
!                                                                  [ J/kg ]
!  p           =  sea pressure                                  [ dbar ]
!                  ( i.e. absolute pressure - 10.1325 dbar )
!
!  SA_final    =  Absolute Salinity of the seawater in the final state,
!                 whether or not any ice is present.               [ g/kg ]
!  CT_final    =  Conservative Temperature of the seawater in the the final
!                 state, whether or not any ice is present.       [ deg C ]
!  w_Ih_final  =  mass fraction of ice in the final seawater-ice mixture.
!                 If this ice mass fraction is positive, the system is at
!                 thermodynamic equilibrium.  If this ice mass fraction is
!                 zero there is no ice in the final state which consists
!                 only of seawater which is warmer than the freezing
!                 temperature.                                   [unitless]
!--------------------------------------------------------------------------
*/
void
gsw_frazil_properties_potential(double sa_bulk, double h_pot_bulk, double p,
        double *sa_final, double *ct_final, double *w_ih_final)
{
        GSW_TEOS10_CONSTANTS;
        int     iterations, max_iterations;
        double  ctf_sa, ctf, dfunc_dw_ih, dfunc_dw_ih_mean_poly,
                dpot_h_ihf_dsa, func, func0, h_pot_ihf, sa, w_ih_old, w_ih,
                x, xa, y, z;

        double  f01 = -9.041191886754806e-1,
                f02 =  4.169608567309818e-2,
                f03 = -9.325971761333677e-3,
                f04 =  4.699055851002199e-2,
                f05 = -3.086923404061666e-2,
                f06 =  1.057761186019000e-2,
                f07 = -7.349302346007727e-2,
                f08 =  1.444842576424337e-1,
                f09 = -1.408425967872030e-1,
                f10 =  1.070981398567760e-1,
                f11 = -1.768451760854797e-2,
                f12 = -4.013688314067293e-1,
                f13 =  7.209753205388577e-1,
                f14 = -1.807444462285120e-1,
                f15 =  1.362305015808993e-1,
                f16 = -9.500974920072897e-1,
                f17 =  1.192134856624248,
                f18 = -9.191161283559850e-2,
                f19 = -1.008594411490973,
                f20 =  8.020279271484482e-1,
                f21 = -3.930534388853466e-1,
                f22 = -2.026853316399942e-2,
                f23 = -2.722731069001690e-2,
                f24 =  5.032098120548072e-2,
                f25 = -2.354888890484222e-2,
                f26 = -2.454090179215001e-2,
                f27 =  4.125987229048937e-2,
                f28 = -3.533404753585094e-2,
                f29 =  3.766063025852511e-2,
                f30 = -3.358409746243470e-2,
                f31 = -2.242158862056258e-2,
                f32 =  2.102254738058931e-2,
                f33 = -3.048635435546108e-2,
                f34 = -1.996293091714222e-2,
                f35 =  2.577703068234217e-2,
                f36 = -1.292053030649309e-2,

                g01 =  3.332286683867741e5,
                g02 =  1.416532517833479e4,
                g03 = -1.021129089258645e4,
                g04 =  2.356370992641009e4,
                g05 = -8.483432350173174e3,
                g06 =  2.279927781684362e4,
                g07 =  1.506238790315354e4,
                g08 =  4.194030718568807e3,
                g09 = -3.146939594885272e5,
                g10 = -7.549939721380912e4,
                g11 =  2.790535212869292e6,
                g12 =  1.078851928118102e5,
                g13 = -1.062493860205067e7,
                g14 =  2.082909703458225e7,
                g15 = -2.046810820868635e7,
                g16 =  8.039606992745191e6,
                g17 = -2.023984705844567e4,
                g18 =  2.871769638352535e4,
                g19 = -1.444841553038544e4,
                g20 =  2.261532522236573e4,
                g21 = -2.090579366221046e4,
                g22 = -1.128417003723530e4,
                g23 =  3.222965226084112e3,
                g24 = -1.226388046175992e4,
                g25 =  1.506847628109789e4,
                g26 = -4.584670946447444e4,
                g27 =  1.596119496322347e4,
                g28 = -6.338852410446789e4,
                g29 =  8.951570926106525e4,

                saturation_fraction = 0.0;
        /*
        !-----------------------------------------------------------------------
        ! Finding func0.  This is the value of the function, func, that would
        ! result in the output w_Ih_final being exactly zero.
        !-----------------------------------------------------------------------
        */
        func0 = h_pot_bulk - gsw_cp0
                *gsw_ct_freezing(sa_bulk,p,saturation_fraction);
        /*
        !-----------------------------------------------------------------------
        ! Setting the three outputs for data points that have func0 non-negative
        !-----------------------------------------------------------------------
        */
        if (func0 >= 0.0) {
            /*
            ! When func0 is zero or positive then the final answer will contain
            ! no frazil ice; that is, it will be pure seawater that is warmer
            ! than the freezing temperature. If func0 >= 0 we do not need to go
            ! through the modified Newton-Raphson procedure and we can simply
            ! write down the answer, as in the following 4 lines of code.
            */
            *sa_final = sa_bulk;
            *ct_final = h_pot_bulk/gsw_cp0;
            *w_ih_final = 0.0;
            return;
        }
        /*
        !-----------------------------------------------------------------------
        !Begin finding the solution for data points that have func0 < 0, so that
        !the output will have a positive ice mass fraction w_Ih_final.
        !-----------------------------------------------------------------------
        */

        /*Evalaute a polynomial for w_Ih in terms of SA_bulk, func0 and p*/
        x = sa_bulk*1e-2;
        y = func0/3e5;
        z = p*1e-4;

        w_ih = y*(f01 + x*(f02 + x*(f03 + x*(f04 + x*(f05 + f06*x))))
             + y*(f07 + x*(f08 + x*(f09 + x*(f10 + f11*x))) + y*(f12 + x*(f13
             + x*(f14 + f15*x)) + y*(f16 + x*(f17 + f18*x) + y*(f19 + f20*x
             + f21*y)))) + z*(f22 + x*(f23 + x*(f24 + f25*x)) + y*(x*(f26
             + f27*x)
             + y*(f28 + f29*x + f30*y)) + z*(f31 + x*(f32 + f33*x) + y*(f34
             + f35*x + f36*y))));

        if (w_ih > 0.9) {
            /*
            ! The ice mass fraction out of this code is restricted to be
            !less than 0.9.
            */
            *sa_final = GSW_INVALID_VALUE;
            *ct_final = *sa_final;
            *w_ih_final = *sa_final;
            return;
        }

        /*
        !The initial guess at the absolute salinity of the interstitial seawater
        */
        sa = sa_bulk/(1.0 - w_ih);
        /*
        !-----------------------------------------------------------------------
        ! Doing a Newton step with a separate polynomial estimate of the mean
        ! derivative dfunc_dw_Ih_mean_poly.
        !-----------------------------------------------------------------------'       */
        ctf = gsw_ct_freezing(sa,p,saturation_fraction);
        h_pot_ihf = gsw_pot_enthalpy_ice_freezing(sa,p);
        func = h_pot_bulk - (1.0 - w_ih)*gsw_cp0*ctf - w_ih*h_pot_ihf;

        xa = sa*1e-2;

        dfunc_dw_ih_mean_poly = g01 + xa*(g02 + xa*(g03 + xa*(g04 + g05*xa)))
            + w_ih*(xa*(g06 + xa*(g07 + g08*xa)) + w_ih*(xa*(g09 + g10*xa)
            + w_ih*xa*(g11 + g12*xa + w_ih*(g13 + w_ih*(g14 + w_ih*(g15
            + g16*w_ih)))))) + z*(g17 + xa*(g18 + g19*xa) + w_ih*(g20
            + w_ih*(g21 + g22*w_ih) + xa*(g23 + g24*xa*w_ih))
            + z*(g25 + xa*(g26 + g27*xa) + w_ih*(g28 + g29*w_ih)));

        w_ih_old = w_ih;
        w_ih = w_ih_old - func/dfunc_dw_ih_mean_poly;

        sa = sa_bulk/(1.0 - w_ih);
        /*
        !-----------------------------------------------------------------------
        ! Calculating the estimate of the derivative of func, dfunc_dw_Ih, to be
        ! fed into Newton's Method.
        !-----------------------------------------------------------------------
        */
        ctf = gsw_ct_freezing(sa,p,saturation_fraction);

        h_pot_ihf = gsw_pot_enthalpy_ice_freezing(sa,p);

        gsw_ct_freezing_first_derivatives(sa,p,saturation_fraction,&ctf_sa,
                NULL);
        gsw_pot_enthalpy_ice_freezing_first_derivatives(sa,p,&dpot_h_ihf_dsa,
                NULL);

        dfunc_dw_ih = gsw_cp0*ctf - h_pot_ihf -
                         sa*(gsw_cp0*ctf_sa + w_ih*dpot_h_ihf_dsa/(1.0 - w_ih));

        if (w_ih >= 0.0 && w_ih <= 0.20 && sa > 15.0
            && sa < 60.0 && p <= 3000.0) {
            max_iterations = 1;
        } else if (w_ih >= 0.0 && w_ih <= 0.85 && sa > 0.0
            && sa < 120.0 && p <= 3500.0) {
            max_iterations = 2;
        } else {
            max_iterations = 3;
        }

        for (iterations = 1; iterations <= max_iterations; iterations++) {

            if (iterations > 1) {
                /*On the first iteration ctf and h_pot_ihf are both known*/
                ctf = gsw_ct_freezing(sa,p,saturation_fraction);
                h_pot_ihf = gsw_pot_enthalpy_ice_freezing(sa,p);
            }

            /*This is the function, func, whose zero we seek ...*/
            func = h_pot_bulk - (1.0 - w_ih)*gsw_cp0*ctf - w_ih*h_pot_ihf;

            w_ih_old = w_ih;
            w_ih = w_ih_old - func/dfunc_dw_ih;

            if (w_ih > 0.9) {
                *sa_final = GSW_INVALID_VALUE;
                *ct_final = *sa_final;
                *w_ih_final = *sa_final;
                return;
            }

            sa = sa_bulk/(1.0 - w_ih);

        }

        if (w_ih < 0.0) {
            *sa_final = sa_bulk;
            *ct_final = h_pot_bulk/gsw_cp0;
            *w_ih_final = 0.0;
        } else {
            *sa_final = sa;
            *ct_final = gsw_ct_freezing(sa,p,saturation_fraction);
            *w_ih_final = w_ih;
        }
}
/*
!==========================================================================
elemental subroutine gsw_frazil_properties_potential_poly (sa_bulk, &
                             h_pot_bulk, p, sa_final, ct_final, w_ih_final)
!==========================================================================
!
!  Calculates the mass fraction of ice (mass of ice divided by mass of ice
!  plus seawater), w_Ih_final, which results from given values of the bulk
!  Absolute Salinity, SA_bulk, bulk potential enthalpy, h_pot_bulk,
!  occurring at pressure p.  The final equilibrium values of Absolute
!  Salinity, SA_final, and Conservative Temperature, CT_final, of the
!  interstitial seawater phase are also returned.  This code assumes that
!  there is no dissolved air in the seawater (that is, saturation_fraction
!  is assumed to be zero throughout the code).
!
!  When the mass fraction w_Ih_final is calculated as being a positive
!  value, the seawater-ice mixture is at thermodynamic equlibrium.
!
!  This code returns w_Ih_final = 0 when the input bulk enthalpy, h_bulk,
!  is sufficiently large (i.e. sufficiently "warm") so that there is no ice
!  present in the final state.  In this case the final state consists of
!  only seawater rather than being an equlibrium mixture of seawater and
!  ice which occurs when w_Ih_final is positive.  Note that when
!  w_Ih_final = 0, the final seawater is not at the freezing temperature.
!
!  Note that this code uses the polynomial forms of CT_freezing and
!  pot_enthalpy_ice_freezing. This code is intended to be used in ocean
!  models where the model prognostic variables are SA_bulk and h_pot_bulk.
!
!  SA_bulk     =  bulk Absolute Salinity of the seawater and ice mixture
!                                                                  [ g/kg ]
!  h_pot_bulk  =  bulk potential enthalpy of the seawater and ice mixture
!                                                                  [ J/kg ]
!  p           =  sea pressure                                  [ dbar ]
!                  ( i.e. absolute pressure - 10.1325 dbar )
!
!  SA_final    =  Absolute Salinity of the seawater in the final state,
!                 whether or not any ice is present.               [ g/kg ]
!  CT_final    =  Conservative Temperature of the seawater in the the final
!                 state, whether or not any ice is present.       [ deg C ]
!  w_Ih_final  =  mass fraction of ice in the final seawater-ice mixture.
!                 If this ice mass fraction is positive, the system is at
!                 thermodynamic equilibrium.  If this ice mass fraction is
!                 zero there is no ice in the final state which consists
!                 only of seawater which is warmer than the freezing
!                 temperature.                                   [unitless]
!--------------------------------------------------------------------------
*/
void
gsw_frazil_properties_potential_poly(double sa_bulk, double h_pot_bulk,
        double p, double *sa_final, double *ct_final, double *w_ih_final)
{
        GSW_TEOS10_CONSTANTS;
        int     iterations, max_iterations;
        double  ctf_sa, ctf, dfunc_dw_ih, dfunc_dw_ih_mean_poly, dpot_h_ihf_dsa,
                func, func0, h_pot_ihf, sa, w_ih_old, w_ih, x, xa, y, z;

        double  f01 = -9.041191886754806e-1,
                f02 =  4.169608567309818e-2,
                f03 = -9.325971761333677e-3,
                f04 =  4.699055851002199e-2,
                f05 = -3.086923404061666e-2,
                f06 =  1.057761186019000e-2,
                f07 = -7.349302346007727e-2,
                f08 =  1.444842576424337e-1,
                f09 = -1.408425967872030e-1,
                f10 =  1.070981398567760e-1,
                f11 = -1.768451760854797e-2,
                f12 = -4.013688314067293e-1,
                f13 =  7.209753205388577e-1,
                f14 = -1.807444462285120e-1,
                f15 =  1.362305015808993e-1,
                f16 = -9.500974920072897e-1,
                f17 =  1.192134856624248,
                f18 = -9.191161283559850e-2,
                f19 = -1.008594411490973,
                f20 =  8.020279271484482e-1,
                f21 = -3.930534388853466e-1,
                f22 = -2.026853316399942e-2,
                f23 = -2.722731069001690e-2,
                f24 =  5.032098120548072e-2,
                f25 = -2.354888890484222e-2,
                f26 = -2.454090179215001e-2,
                f27 =  4.125987229048937e-2,
                f28 = -3.533404753585094e-2,
                f29 =  3.766063025852511e-2,
                f30 = -3.358409746243470e-2,
                f31 = -2.242158862056258e-2,
                f32 =  2.102254738058931e-2,
                f33 = -3.048635435546108e-2,
                f34 = -1.996293091714222e-2,
                f35 =  2.577703068234217e-2,
                f36 = -1.292053030649309e-2,

                g01 =  3.332286683867741e5,
                g02 =  1.416532517833479e4,
                g03 = -1.021129089258645e4,
                g04 =  2.356370992641009e4,
                g05 = -8.483432350173174e3,
                g06 =  2.279927781684362e4,
                g07 =  1.506238790315354e4,
                g08 =  4.194030718568807e3,
                g09 = -3.146939594885272e5,
                g10 = -7.549939721380912e4,
                g11 =  2.790535212869292e6,
                g12 =  1.078851928118102e5,
                g13 = -1.062493860205067e7,
                g14 =  2.082909703458225e7,
                g15 = -2.046810820868635e7,
                g16 =  8.039606992745191e6,
                g17 = -2.023984705844567e4,
                g18 =  2.871769638352535e4,
                g19 = -1.444841553038544e4,
                g20 =  2.261532522236573e4,
                g21 = -2.090579366221046e4,
                g22 = -1.128417003723530e4,
                g23 =  3.222965226084112e3,
                g24 = -1.226388046175992e4,
                g25 =  1.506847628109789e4,
                g26 = -4.584670946447444e4,
                g27 =  1.596119496322347e4,
                g28 = -6.338852410446789e4,
                g29 =  8.951570926106525e4,

                saturation_fraction = 0.0;
        /*
        !-----------------------------------------------------------------------
        ! Finding func0.  This is the value of the function, func, that would
        ! result in the output w_Ih_final being exactly zero.
        !-----------------------------------------------------------------------
        */
        func0 = h_pot_bulk - gsw_cp0
                *gsw_ct_freezing_poly(sa_bulk,p,saturation_fraction);
        /*
        !-----------------------------------------------------------------------
        ! Setting the three outputs for data points that have func0 non-negative
        !-----------------------------------------------------------------------
        */
        if (func0 >= 0.0) {
            /*
            ! When func0 is zero or positive then the final answer will contain
            ! no frazil ice; that is, it will be pure seawater that is warmer
            ! han the freezing temperature.  If func0 >= 0 we do not need to go
            ! through the modified Newton-Raphson procedure and we can simply
            ! write down the answer, as in the following 4 lines of code.
            */
            *sa_final = sa_bulk;
            *ct_final = h_pot_bulk/gsw_cp0;
            *w_ih_final = 0.0;
            return;
        }
        /*
        !-----------------------------------------------------------------------
        !Begin finding the solution for data points that have func0 < 0, so that
        !the output will have a positive ice mass fraction w_Ih_final.
        !-----------------------------------------------------------------------
        */

        /*Evalaute a polynomial for w_Ih in terms of SA_bulk, func0 and p*/
        x = sa_bulk*1e-2;
        y = func0/3e5;
        z = p*1e-4;

        w_ih = y*(f01 + x*(f02 + x*(f03 + x*(f04 + x*(f05 + f06*x))))
             + y*(f07 + x*(f08 + x*(f09 + x*(f10 + f11*x))) + y*(f12 + x*(f13
             + x*(f14 + f15*x)) + y*(f16 + x*(f17 + f18*x) + y*(f19 + f20*x
             + f21*y)))) + z*(f22 + x*(f23 + x*(f24 + f25*x)) + y*(x*(f26
             + f27*x)
             + y*(f28 + f29*x + f30*y)) + z*(f31 + x*(f32 + f33*x) + y*(f34
             + f35*x + f36*y))));

        if (w_ih > 0.9) {
            /*
            ! The ice mass fraction out of this code is restricted to be
            ! less than 0.9.
            */
            *sa_final = GSW_INVALID_VALUE;
            *ct_final = *sa_final;
            *w_ih_final = *sa_final;
            return;
        }
        /*
        ! The initial guess at the absolute salinity of the interstitial
        ! seawater
        */
        sa = sa_bulk/(1.0 - w_ih);
        /*
        !-----------------------------------------------------------------------
        ! Doing a Newton step with a separate polynomial estimate of the mean
        ! derivative dfunc_dw_Ih_mean_poly.
        !-----------------------------------------------------------------------
        */
        ctf = gsw_ct_freezing_poly(sa,p,saturation_fraction);
        h_pot_ihf = gsw_pot_enthalpy_ice_freezing_poly(sa,p);
        func = h_pot_bulk - (1.0 - w_ih)*gsw_cp0*ctf - w_ih*h_pot_ihf;

        xa = sa*1e-2;

        dfunc_dw_ih_mean_poly = g01 + xa*(g02 + xa*(g03 + xa*(g04 + g05*xa)))
            + w_ih*(xa*(g06 + xa*(g07 + g08*xa)) + w_ih*(xa*(g09 + g10*xa)
            + w_ih*xa*(g11 + g12*xa + w_ih*(g13 + w_ih*(g14 + w_ih*(g15
            + g16*w_ih)))))) + z*(g17 + xa*(g18 + g19*xa) + w_ih*(g20
            + w_ih*(g21 + g22*w_ih) + xa*(g23 + g24*xa*w_ih))
            + z*(g25 + xa*(g26 + g27*xa) + w_ih*(g28 + g29*w_ih)));

        w_ih_old = w_ih;
        w_ih = w_ih_old - func/dfunc_dw_ih_mean_poly;

        sa = sa_bulk/(1.0 - w_ih);
        /*
        !-----------------------------------------------------------------------
        ! Calculating the estimate of the derivative of func, dfunc_dw_Ih, to be
        ! fed into Newton's Method.
        !-----------------------------------------------------------------------
        */
        ctf = gsw_ct_freezing_poly(sa,p,saturation_fraction);

        h_pot_ihf = gsw_pot_enthalpy_ice_freezing_poly(sa,p);

        gsw_ct_freezing_first_derivatives_poly(sa,p,saturation_fraction,&ctf_sa,
                NULL);
        gsw_pot_enthalpy_ice_freezing_first_derivatives_poly(sa,p,
                &dpot_h_ihf_dsa, NULL);

        dfunc_dw_ih = gsw_cp0*ctf - h_pot_ihf -
                         sa*(gsw_cp0*ctf_sa + w_ih*dpot_h_ihf_dsa/(1.0 - w_ih));

        if (w_ih >= 0.0 && w_ih <= 0.20 && sa > 15.0
            && sa < 60.0 && p <= 3000.0) {
            max_iterations = 1;
        } else if (w_ih >= 0.0 && w_ih <= 0.85 && sa > 0.0
            && sa < 120.0 && p <= 3500.0) {
            max_iterations = 2;
        } else {
            max_iterations = 3;
        }

        for (iterations = 1; iterations <= max_iterations; iterations++) {

            if (iterations > 1) {
                /*On the first iteration ctf and h_pot_ihf are both known*/
                ctf = gsw_ct_freezing_poly(sa,p,saturation_fraction);
                h_pot_ihf = gsw_pot_enthalpy_ice_freezing_poly(sa,p);
            }

            /*This is the function, func, whose zero we seek ...*/
            func = h_pot_bulk - (1.0 - w_ih)*gsw_cp0*ctf - w_ih*h_pot_ihf;

            w_ih_old = w_ih;
            w_ih = w_ih_old - func/dfunc_dw_ih;

            if (w_ih > 0.9) {
                *sa_final = GSW_INVALID_VALUE;
                *ct_final = *sa_final;
                *w_ih_final = *sa_final;
                return;
            }

            sa = sa_bulk/(1.0 - w_ih);

        }

        if (w_ih < 0.0) {
            *sa_final = sa_bulk;
            *ct_final = h_pot_bulk/gsw_cp0;
            *w_ih_final = 0.0;
        } else {
            *sa_final = sa;
            *ct_final = gsw_ct_freezing_poly(sa,p,saturation_fraction);
            *w_ih_final = w_ih;
        }
}
/*
!==========================================================================
elemental subroutine gsw_frazil_ratios_adiabatic (sa, p, w_ih, &
                              dsa_dct_frazil, dsa_dp_frazil, dct_dp_frazil)
!==========================================================================
!
!  Calculates the ratios of SA, CT and P changes when frazil ice forms or
!  melts in response to an adiabatic change in pressure of a mixture of
!  seawater and frazil ice crystals.
!
!  Note that the first output, dSA_dCT_frazil, is dSA/dCT rather than
!  dCT/dSA.  This is done so that when SA = 0, the output, dSA/dCT, is zero
!  whereas dCT/dSA would then be infinite.
!
!  Also note that both dSA_dP_frazil and dCT_dP_frazil are the pressure
!  derivatives with the pressure measured in Pa not dbar.
!
!  SA  =  Absolute Salinity of seawater                            [ g/kg ]
!  p   =  sea pressure of seawater at which melting occurs         [ dbar ]
!         ( i.e. absolute pressure - 10.1325d0 dbar )
!  w_Ih  =  mass fraction of ice, that is the mass of ice divided by the
!           sum of the masses of ice and seawater.  That is, the mass of
!           ice divided by the mass of the final mixed fluid.
!           w_Ih must be between 0 and 1.                      [ unitless ]
!
!  dSA_dCT_frazil =  the ratio of the changes in Absolute Salinity
!                    to that of Conservative Temperature       [ g/(kg K) ]
!  dSA_dP_frazil  =  the ratio of the changes in Absolute Salinity
!                    to that of pressure (in Pa)              [ g/(kg Pa) ]
!  dCT_dP_frazil  =  the ratio of the changes in Conservative Temperature
!                    to that of pressure (in Pa)                   [ K/Pa ]
!--------------------------------------------------------------------------
*/
void
gsw_frazil_ratios_adiabatic (double sa, double p, double w_ih,
        double *dsa_dct_frazil, double *dsa_dp_frazil, double *dct_dp_frazil)
{
        double  bracket1, bracket2, cp_ih, gamma_ih, h, h_ih, part,
                rec_bracket3, tf, wcp, h_hat_sa, h_hat_ct, tf_sa, tf_p,
                ctf, ctf_sa, ctf_p;
        double  saturation_fraction = 0.0;

        ctf = gsw_ct_freezing(sa,p,saturation_fraction);
        tf = gsw_t_freezing(sa,p,saturation_fraction);
        h = gsw_enthalpy_ct_exact(sa,ctf,p);
        h_ih = gsw_enthalpy_ice(tf,p);
        cp_ih = gsw_cp_ice(tf,p);
        gamma_ih = gsw_adiabatic_lapse_rate_ice(tf,p);
        gsw_enthalpy_first_derivatives_ct_exact(sa,ctf,p,&h_hat_sa,&h_hat_ct);
        gsw_t_freezing_first_derivatives(sa,p,saturation_fraction,&tf_sa,&tf_p);
        gsw_ct_freezing_first_derivatives(sa,p,saturation_fraction,
                &ctf_sa,&ctf_p);

        wcp = cp_ih*w_ih/(1.0 - w_ih);
        part = (tf_p - gamma_ih)/ctf_p;

        bracket1 = h_hat_ct + wcp*part;
        bracket2 = h - h_ih - sa*(h_hat_sa + wcp*(tf_sa - part*ctf_sa));
        rec_bracket3 = 1.0/(h - h_ih - sa*(h_hat_sa + h_hat_ct*ctf_sa
                        + wcp*tf_sa));

        *dsa_dct_frazil = sa*(bracket1/bracket2);
        *dsa_dp_frazil = sa*ctf_p*bracket1*rec_bracket3;
        *dct_dp_frazil = ctf_p*bracket2*rec_bracket3;
}
/*
!==========================================================================
elemental subroutine gsw_frazil_ratios_adiabatic_poly (sa, p, w_ih, &
                              dsa_dct_frazil, dsa_dp_frazil, dct_dp_frazil)
!==========================================================================
!
!  Calculates the ratios of SA, CT and P changes when frazil ice forms or
!  melts in response to an adiabatic change in pressure of a mixture of
!  seawater and frazil ice crystals.
!
!  Note that the first output, dSA_dCT_frazil, is dSA/dCT rather than
!  dCT/dSA.  This is done so that when SA = 0, the output, dSA/dCT, is zero
!  whereas dCT/dSA would then be infinite.
!
!  Also note that both dSA_dP_frazil and dCT_dP_frazil are the pressure
!  derivatives with the pressure measured in Pa not dbar.
!
!  SA  =  Absolute Salinity of seawater                            [ g/kg ]
!  p   =  sea pressure of seawater at which melting occurs         [ dbar ]
!         ( i.e. absolute pressure - 10.1325d0 dbar )
!  w_Ih  =  mass fraction of ice, that is the mass of ice divided by the
!           sum of the masses of ice and seawater.  That is, the mass of
!           ice divided by the mass of the final mixed fluid.
!           w_Ih must be between 0 and 1.                      [ unitless ]
!
!  dSA_dCT_frazil =  the ratio of the changes in Absolute Salinity
!                    to that of Conservative Temperature       [ g/(kg K) ]
!  dSA_dP_frazil  =  the ratio of the changes in Absolute Salinity
!                    to that of pressure (in Pa)              [ g/(kg Pa) ]
!  dCT_dP_frazil  =  the ratio of the changes in Conservative Temperature
!                    to that of pressure (in Pa)                   [ K/Pa ]
!--------------------------------------------------------------------------
*/
void
gsw_frazil_ratios_adiabatic_poly(double sa, double p, double w_ih,
        double *dsa_dct_frazil, double *dsa_dp_frazil, double *dct_dp_frazil)
{
        double  bracket1, bracket2, cp_ih, gamma_ih, h, h_ih, part,
                rec_bracket3, tf, wcp, h_hat_sa, h_hat_ct, tf_sa, tf_p,
                ctf, ctf_sa, ctf_p;
        double  saturation_fraction = 0.0;

        tf = gsw_t_freezing_poly(sa,p,saturation_fraction);
        ctf = gsw_ct_freezing_poly(sa,p,saturation_fraction);
        h = gsw_enthalpy(sa,ctf,p);
        h_ih = gsw_enthalpy_ice(tf,p);
        cp_ih = gsw_cp_ice(tf,p);
        gamma_ih = gsw_adiabatic_lapse_rate_ice(tf,p);
        gsw_enthalpy_first_derivatives(sa,ctf,p,&h_hat_sa,&h_hat_ct);
        gsw_t_freezing_first_derivatives_poly(sa,p,saturation_fraction,
                &tf_sa,&tf_p);
        gsw_ct_freezing_first_derivatives_poly(sa,p,saturation_fraction,
                &ctf_sa,&ctf_p);

        wcp = cp_ih*w_ih/(1.0 - w_ih);
        part = (tf_p - gamma_ih)/ctf_p;

        bracket1 = h_hat_ct + wcp*part;
        bracket2 = h - h_ih - sa*(h_hat_sa + wcp*(tf_sa - part*ctf_sa));
        rec_bracket3 = 1.0/(h - h_ih - sa*(h_hat_sa + h_hat_ct*ctf_sa
                        + wcp*tf_sa));

        *dsa_dct_frazil = sa*(bracket1/bracket2);
        *dsa_dp_frazil = sa*ctf_p*bracket1*rec_bracket3;
        *dct_dp_frazil = ctf_p*bracket2*rec_bracket3;
}

/*

    ** The following function is buggy; users are advised to use
    ** gsw_geo_strf_dyn_height_1 instead.

!==========================================================================
pure function gsw_geo_strf_dyn_height (sa, ct, p, p_ref)
!==========================================================================
!
!  Calculates dynamic height anomaly as the integral of specific volume
!  anomaly from the pressure p of the bottle to the reference pressure
!  p_ref.
!
!  Hence, geo_strf_dyn_height is the dynamic height anomaly with respect
!  to a given reference pressure.  This is the geostrophic streamfunction
!  for the difference between the horizontal velocity at the pressure
!  concerned, p, and the horizontal velocity at p_ref.  Dynamic height
!  anomaly is the geostrophic streamfunction in an isobaric surface.  The
!  reference values used for the specific volume anomaly are
!  SSO = 35.16504 g/kg and CT = 0 deg C.  This function calculates
!  specific volume anomaly using the computationally efficient
!  expression for specific volume of Roquet et al. (2015).
!
!  This function evaluates the pressure integral of specific volume using
!  SA and CT interpolated with respect to pressure using the method of
!  Reiniger and Ross (1968).  It uses a weighted mean of (i) values
!  obtained from linear interpolation of the two nearest data points, and
!  (ii) a linear extrapolation of the pairs of data above and below.  This
!  "curve fitting" method resembles the use of cubic splines.
!
!  SA    =  Absolute Salinity                                      [ g/kg ]
!  CT    =  Conservative Temperature (ITS-90)                     [ deg C ]
!  p     =  sea pressure                                           [ dbar ]
!           ( i.e. absolute pressure - 10.1325 dbar )
!  p_ref =  reference pressure                                     [ dbar ]
!           ( i.e. reference absolute pressure - 10.1325 dbar )
!
!  geo_strf_dyn_height  =  dynamic height anomaly               [ m^2/s^2 ]
!   Note. If p_ref exceeds the pressure of the deepest bottle on a
!     vertical profile, the dynamic height anomaly for each bottle
!     on the whole vertical profile is returned as NaN.
!--------------------------------------------------------------------------
*/
static void p_sequence(double p1,double p2,double max_dp_i,double *pseq,
        int *nps);      /* forward reference */

double  *       /* Returns NULL on error, dyn_height if okay */
gsw_geo_strf_dyn_heightRR(double *sa, double *ct, double *p, double p_ref,
        int n_levels, double *dyn_height)
{
        GSW_TEOS10_CONSTANTS;
        int     m_levels = (n_levels <= 0) ? 1 : n_levels,
                p_cnt, top_pad, i, nz, ibottle, ipref, np_max, np, ibpr=0,
                *iidata;
        double  dp_min, dp_max, p_min, p_max, max_dp_i,
                *b, *b_av, *dp, *dp_i, *sa_i=NULL, *ct_i, *p_i=NULL,
                *geo_strf_dyn_height0;

/*
!--------------------------------------------------------------------------
!  This max_dp_i is the limit we choose for the evaluation of specific
!  volume in the pressure integration.  That is, the vertical integration
!  of specific volume with respect to pressure is performed with the pressure
!  increment being no more than max_dp_i (the default value being 1 dbar).
!--------------------------------------------------------------------------
*/
        max_dp_i = 1.0;

        if ((nz = m_levels) <= 1)
            return (NULL);

        dp = (double *) malloc(nz*sizeof (double));
        dp_min = 11000.0;
        dp_max = -11000.0;
        for (i=0; i<nz-1; i++) {
            if ((dp[i] = p[i+1] - p[i]) < dp_min)
                dp_min = dp[i];
            if (dp[i] > dp_max)
                dp_max = dp[i];
        }

        if (dp_min <= 0.0) {
            /* pressure must be monotonic */
            free(dp);
            return (NULL);
        }
        p_min = p[0];
        p_max = p[nz-1];

        if (p_ref > p_max) {
            /*the reference pressure p_ref is deeper than all bottles*/
            free(dp);
            return (NULL);
        }

        /* Determine if there is a "bottle" at exactly p_ref */
        ipref = -1;
        for (ibottle = 0; ibottle < nz; ibottle++) {
            if (p[ibottle] == p_ref) {
                ipref = ibottle;
                break;
            }
        }
        if ((dp_max <= max_dp_i) && (p[0] == 0.0) && (ipref >= 0)) {
            /*
            !vertical resolution is good (bottle gap is no larger than max_dp_i)
            ! & the vertical profile begins at the surface (i.e. at p = 0 dbar)
            ! & the profile contains a "bottle" at exactly p_ref.
            */
            b = (double *) malloc(3*nz*sizeof (double));
            b_av = b+nz; geo_strf_dyn_height0 = b_av+nz;
            for (i=0; i<nz; i++) {
                b[i] = gsw_specvol_anom_standard(sa[i],ct[i],p[i]);
                if (i > 0)
                    b_av[i-1] = 0.5*(b[i] + b[i-1]);
            }
            /*
            ! "geo_strf_dyn_height0" is the dynamic height anomaly with respect
            ! to p_ref = 0 (the surface).
            */
            geo_strf_dyn_height0[0] = 0.0;
            for (i=1; i<nz; i++)
                geo_strf_dyn_height0[i] = b_av[i]*dp[i]*db2pa;
            for (i=1; i<nz; i++) /* cumulative sum */
                geo_strf_dyn_height0[i] = geo_strf_dyn_height0[i-1]
                                          - geo_strf_dyn_height0[i];
            for (i=0; i<nz; i++)
                dyn_height[i] = geo_strf_dyn_height0[i]
                                - geo_strf_dyn_height0[ipref];
            free(b);
        } else {
        /*
        ! Test if there are vertical gaps between adjacent "bottles" which are
        ! greater than max_dp_i, and that there is a "bottle" exactly at the
        ! reference pressure.
        */
            iidata = (int *) malloc((nz+1)*sizeof (int));

            if ((dp_max <= max_dp_i) && (ipref >= 0)) {
            /*
            ! Vertical resolution is already good (no larger than max_dp_i), and
            ! there is a "bottle" at exactly p_ref.
            */
                sa_i = (double *) malloc(2*(nz+1)*sizeof (double));
                ct_i = sa_i+nz+1;
                p_i = (double *) malloc((nz+1)*sizeof (double));;

                if (p_min > 0.0) {
                /*
                ! resolution is fine and there is a bottle at p_ref, but
                ! there is not a bottle at p = 0. So add an extra bottle.
                */
                    for (i=0; i<nz; i++) {
                        sa_i[i+1]       = sa[i];
                        ct_i[i+1]       = ct[i];
                        p_i[i+1]        = p[i];
                    }
                    sa_i[0] = sa[0];
                    ct_i[0] = ct[0];
                    p_i[0] = 0.0;
                    ibpr = ipref+1;
                    p_cnt = nz+1;
                    for (i=0; i<p_cnt; i++)
                        iidata[i] = i;
                } else {
                /*
                ! resolution is fine, there is a bottle at p_ref, and
                ! there is a bottle at p = 0
                */
                    memmove(sa_i, sa, nz*sizeof (double));
                    memmove(ct_i, ct, nz*sizeof (double));
                    memmove(p_i, p, nz*sizeof (double));
                    ibpr = ipref;
                    for (i=0; i<nz; i++)
                        iidata[i] = i;
                    p_cnt = nz;
                }

            } else {
            /*
            ! interpolation is needed.
            */
                np_max = 2*rint(p[nz-1]/max_dp_i+0.5);
                p_i = (double *) malloc(np_max*sizeof (double));
                /* sa_i is allocated below, when its size is known */

                if (p_min > 0.0) {
                /*
                ! there is not a bottle at p = 0.
                */
                    if (p_ref < p_min) {
                    /*
                    ! p_ref is shallower than the minimum bottle pressure.
                    */
                        p_i[0] = 0.0;
                        p_sequence(p_i[0],p_ref,max_dp_i, p_i+1,&np);
                        ibpr = p_cnt = np;
                        p_cnt++;
                        p_sequence(p_ref,p_min,max_dp_i, p_i+p_cnt,&np);
                        p_cnt += np;
                        top_pad = p_cnt;
                    } else {
                    /*
                    ! p_ref is deeper than the minimum bottle pressure.
                    */
                        p_i[0] = 0.0;
                        p_i[1] = p_min;
                        top_pad = 2;
                        p_cnt = 2;
                    }
                } else {
                /*
                ! there is a bottle at p = 0.
                */
                    p_i[0] = p_min;
                    top_pad = 1;
                    p_cnt = 1;
                }

                for (ibottle=0; ibottle < nz-1; ibottle++) {

                    iidata[ibottle] = p_cnt-1;
                    if (p[ibottle] == p_ref) ibpr = p_cnt-1;

                    if (p[ibottle] < p_ref && p[ibottle+1] > p_ref) {
                    /*
                    ! ... reference pressure is spanned by bottle pairs -
                    ! need to include p_ref as an interpolated pressure.
                    */
                        p_sequence(p[ibottle],p_ref,max_dp_i, p_i+p_cnt,&np);
                        p_cnt += np;
                        ibpr = p_cnt-1;
                        p_sequence(p_ref,p[ibottle+1],max_dp_i,p_i+p_cnt,&np);
                        p_cnt += np;
                    } else {
                    /*
                    ! ... reference pressure is not spanned by bottle pairs.
                    */
                        p_sequence(p[ibottle],p[ibottle+1],max_dp_i,
                                p_i+p_cnt,&np);
                        p_cnt += np;
                    }

                }

                iidata[nz-1] = p_cnt-1;
                if (p[nz-1] == p_ref) ibpr = p_cnt-1;

                sa_i = (double *) malloc(2*p_cnt*sizeof (double));
                ct_i = sa_i+p_cnt;

                if (top_pad > 1) {
                    gsw_linear_interp_sa_ct(sa,ct,p,nz,
                        p_i,top_pad-1,sa_i,ct_i);
                }
                gsw_rr68_interp_sa_ct(sa,ct,p,nz,p_i+top_pad-1,p_cnt-top_pad+1,
                                      sa_i+top_pad-1,ct_i+top_pad-1);
            }

            b = (double *) malloc(4*p_cnt*sizeof (double));
            b_av = b+p_cnt; dp_i = b_av+p_cnt;
            geo_strf_dyn_height0 = dp_i+p_cnt;
            for (i=0; i<p_cnt; i++) {
                b[i] = gsw_specvol_anom_standard(sa_i[i],ct_i[i],p_i[i]);
                if (i > 0) {
                    dp_i[i-1] = p_i[i]-p_i[i-1];
                    b_av[i-1] = 0.5*(b[i] + b[i-1]);
                }
            }
            /*
            ! "geo_strf_dyn_height0" is the dynamic height anomaly with respect
            ! to p_ref = 0 (the surface).
            */
            geo_strf_dyn_height0[0] = 0.0;
            for (i=1; i<p_cnt; i++)
                geo_strf_dyn_height0[i] = b_av[i-1]*dp_i[i-1];
            for (i=1; i<p_cnt; i++) /* cumulative sum */
                geo_strf_dyn_height0[i] = geo_strf_dyn_height0[i-1]
                                          - geo_strf_dyn_height0[i];
            for (i=0; i<nz; i++)
                dyn_height[i] = (geo_strf_dyn_height0[iidata[i]]
                                - geo_strf_dyn_height0[ibpr])*db2pa;

            free(b);
            free(iidata);
            if (sa_i != NULL)
                free(sa_i);
            if (p_i != NULL)
                free(p_i);

        }
        free(dp);
        return (dyn_height);
}

static void
p_sequence(double p1, double p2, double max_dp_i, double *pseq, int *nps)
{
        double  dp, pstep;
        int             n, i;

        dp = p2 - p1;
        n = ceil(dp/max_dp_i);
        pstep = dp/n;

        if (nps != NULL) *nps = n;
        /*
        ! Generate the sequence ensuring that the value of p2 is exact to
        ! avoid round-off issues, ie. don't do "pseq = p1+pstep*(i+1)".
        */
        for (i=0; i<n; i++)
            pseq[i] = p2-pstep*(n-1-i);

}
/*
    ** This is a replacement for gsw_geo_strf_dyn_height, with a different
    ** signature and interpolation algorithms.

!==========================================================================
int   (returns nonzero on error, 0 if OK)
gsw_geo_strf_dyn_height_1(double *sa, double *ct, double *p, double p_ref,
    int nz, double *dyn_height, double max_dp_i, int interp_method)
!==========================================================================
!
!  Calculates dynamic height anomaly as the integral of specific volume
!  anomaly from the pressure p of the bottle to the reference pressure
!  p_ref.
!
!  Hence, geo_strf_dyn_height is the dynamic height anomaly with respect
!  to a given reference pressure.  This is the geostrophic streamfunction
!  for the difference between the horizontal velocity at the pressure
!  concerned, p, and the horizontal velocity at p_ref.  Dynamic height
!  anomaly is the geostrophic streamfunction in an isobaric surface.  The
!  reference values used for the specific volume anomaly are
!  SSO = 35.16504 g/kg and CT = 0 deg C.  This function calculates
!  specific volume anomaly using the computationally efficient
!  expression for specific volume of Roquet et al. (2015).
!
!  This function evaluates the pressure integral of specific volume using
!  SA and CT interpolated with respect to pressure. The interpolation method
!  may be chosen as linear or "PCHIP", piecewise cubic Hermite using a shape-
!  preserving algorithm for setting the derivatives.
!
!  SA    =  Absolute Salinity                                      [ g/kg ]
!  CT    =  Conservative Temperature (ITS-90)                     [ deg C ]
!  p     =  sea pressure  (increasing with index)                  [ dbar ]
!           ( i.e. absolute pressure - 10.1325 dbar )
!  nz    =  number of points in each array
!  p_ref =  reference pressure                                     [ dbar ]
!           ( i.e. reference absolute pressure - 10.1325 dbar )
!  geo_strf_dyn_height  =  dynamic height anomaly               [ m^2/s^2 ]
!  max_dp_i = maximum pressure difference between points for triggering
!              interpolation.
!  interp_method = 1 for linear, 2 for PCHIP
!
!   Note. If p_ref falls outside the range of a
!     vertical profile, the dynamic height anomaly for each bottle
!     on the whole vertical profile is returned as NaN.
!--------------------------------------------------------------------------
*/

/*
    Make a new grid based on an original monotonic array, typically P, such that
    on the new monotonic grid, p_i:

        1) The first element is p[0], the last is p[nz-1].
        2) Approximate integer multiples of dp are included.
        3) All original p points are included.
        4) The value p_ref is included.

    Arguments:
        p : the original grid array
        p_ref : scalar reference pressure
        nz : size of p
        dp : target p interval for the output
        p_i : an array to hold the regridded pressures
        ni_max : size of p_i
        p_indices : an array of size nz
        p_ref_ind_ptr : pointer to an integer

    The function fills as much of p_i as it needs.  It returns with
    the array of indices (p_indices) of the original p grid points in
    p_i, and with a pointer to the index of p_ref within p_i.

    Its return argument is the number of elements used in p_i, or -1 on error.
*/
static int refine_grid_for_dh(double *p, double p_ref, int nz,
    double dp,
    double *p_i, int ni_max,  /* size of p_i array; larger than needed */
    int *p_indices, int *p_ref_ind_ptr)
{
    int i, iuniform, iorig;
    double p_next;
    /* Don't add a new point if it is within p_tol of an original. */
    double p_tol = 0.001 * dp;

    p_i[0] = p[0];
    p_indices[0] = 0;
    *p_ref_ind_ptr = -1;  /* initialize to a flag value */
    if (p_ref <= p[0] + p_tol) {
        *p_ref_ind_ptr = 0;
    }
    for (i=1, iuniform=1, iorig=1; i<ni_max && iorig<nz; i++) {
        /* Candidate insertion based on uniform grid: */
        p_next = p[0] + dp * iuniform;

        /* See if we need to insert p_ref: */
        if (*p_ref_ind_ptr == -1 && p_ref <= p_next && p_ref <= p[iorig]) {
            p_i[i] = p_ref;
            *p_ref_ind_ptr = i;
            if (p_ref == p[iorig]) {
                p_indices[iorig] = i;
                iorig++;
            }
            if (p_ref > p_next - p_tol) {
                iuniform++;
            }
            continue;
        }

        /* We did not insert p_ref, so insert either p_next or p[iorig]. */
        if (p_next < p[iorig] - p_tol) {
            p_i[i] = p_next;
            iuniform++;
        }
        else {
            p_i[i] = p[iorig];
            p_indices[iorig] = i;
            /* Skip this p_next if it is close to the point we just added. */
            if (p_next < p[iorig] + p_tol) {
                iuniform++;
            }
            iorig++;
        }
    }

    if (i == ni_max) {
        return (-1);  /* error! */
    }
    return (i);  /* number of elements in p_i */
}

/*  Linearly interpolate to the grid made by define_grid_for_dh.
    We take advantage of what we know about the grids: they match
    at the end points, and both are monotonic.
*/

static int linear_interp_SA_CT_for_dh(double *sa, double *ct, double *p, int nz,
    double *p_i, int n_i,
    double *sa_i, double *ct_i)
{
    int i, ii;
    double pfac;

    sa_i[0] = sa[0];
    sa_i[n_i-1] = sa[nz-1];
    ct_i[0] = ct[0];
    ct_i[n_i-1] = ct[nz-1];
    i = 1;
    for (ii=1; ii<n_i-1; ii++) {
        /* Find the second point of the pair in the original grid that
           bracket the target.
        */
        while (p[i] < p_i[ii]) {
            i++;
            if (i == nz) {
                return -1;  /* error! */
            }
        }
        pfac = (p_i[ii] - p[i-1]) / (p[i] - p[i-1]);
        sa_i[ii] = sa[i-1] + pfac * (sa[i] - sa[i-1]);
        ct_i[ii] = ct[i-1] + pfac * (ct[i] - ct[i-1]);
    }
    return 0;
}


int  /* returns nonzero on error, 0 if OK */
gsw_geo_strf_dyn_height_1(double *sa, double *ct, double *p, double p_ref,
    int nz, double *dyn_height, double max_dp_i, int interp_method)
{
    GSW_TEOS10_CONSTANTS;
    int i, ipref,
        *p_indices, n_i, ni_max, err;
    double    dp_min, dp_max, p_min, p_max,
        *b, *b_av, *dp, *sa_i, *ct_i, *p_i,
        *dh_i;
    double dh_ref;

    if (nz < 2)
        return (1);

    dp = (double *)malloc((nz-1) * sizeof(double));
    dp_min = 11000.0;
    dp_max = -11000.0;
    for (i=0; i<nz-1; i++) {
        dp[i] = p[i+1] - p[i];
        if (dp[i] < dp_min) {
             dp_min = dp[i];
        }
        if (dp[i] > dp_max) {
            dp_max = dp[i];
        }
    }

    if (dp_min <= 0.0) {
        /* pressure must be monotonic */
        free(dp);
        return (2);
    }
    p_min = p[0];
    p_max = p[nz-1];

    if (p_ref > p_max || p_ref < p_min) {
        /* Reference pressure must be within the data range. */
        free(dp);
        return (3);
    }

    /* Determine if there is a sample at exactly p_ref */
    ipref = -1;
    for (i = 0; i < nz; i++) {
        if (p[i] == p_ref) {
            ipref = i;
            break;
        }
    }

    if ((dp_max <= max_dp_i) && (ipref >= 0)) {
        /*
        !vertical resolution is good (bottle gap is no larger than max_dp_i)
        ! & the profile contains a "bottle" at exactly p_ref.
         */
        b = (double *)malloc(nz*sizeof (double));
        b_av = (double *)malloc((nz-1) * sizeof(double));
        for (i=0; i<nz; i++) {
            b[i] = gsw_specvol_anom_standard(sa[i],ct[i],p[i]);
        }
        for (i=0; i<(nz-1); i++) {
            b_av[i] = 0.5*(b[i+1] + b[i]);
        }
        /* First calculate dynamic height relative to the first (shallowest)
           depth. */
        dyn_height[0] = 0.0;
        for (i=1; i<nz; i++) {
            dyn_height[i] = dyn_height[i-1] - b_av[i-1]*dp[i-1]*db2pa;
        }
        /* Then subtract out the value at the reference pressure. */
        dh_ref = dyn_height[ipref];
        for (i=0; i<nz; i++) {
            dyn_height[i] -= dh_ref;
        }
        free(b);
        free(b_av);
        free(dp);
        return (0);
    }

    /*
    If we got this far, then we need to interpolate: either or both of
    inserting a point for p_ref and subdividing the intervals to keep the max
    interval less than max_dp_i.
    */

    free(dp);  /* Need to recalculate, so free here and malloc when needed. */

    ni_max = nz + (int) ceil((p[nz-1] - p[0]) / max_dp_i) + 2;
    /* Maximum possible size of new grid: Original grid size plus
       the number of dp intervals plus 1 for the p_ref,
       plus 1 so that we can know we exited the loop before we hit it.
    */

    p_i = (double *) malloc(ni_max * sizeof(double));
    p_indices = (int *) malloc(nz * sizeof(int));

    n_i = refine_grid_for_dh(p, p_ref, nz, max_dp_i,
                             p_i, ni_max,
                             p_indices, &ipref);
    /* Reminder: if successful, this allocated p_i and p_indices. */
    if (n_i == -1) {
        free(p_i);
        free(p_indices);
        return (4);
    }

    ct_i = (double *)malloc(n_i * sizeof(double));
    sa_i = (double *)malloc(n_i * sizeof(double));

    if (interp_method == INTERP_METHOD_LINEAR) {
        err = linear_interp_SA_CT_for_dh(sa, ct, p, nz,
                                         p_i, n_i,
                                         sa_i, ct_i);
        if (err) err = 5;
    }
    else if (interp_method == INTERP_METHOD_PCHIP) {
        err = gsw_util_pchip_interp(p, sa, nz, p_i, sa_i, n_i);
        err = err || gsw_util_pchip_interp(p, ct, nz, p_i, ct_i, n_i);
        if (err) err = 6;
    }
    else {
        err = 7;
    }
    if (err) {
        free(p_i);
        free(p_indices);
        free(ct_i);
        free(sa_i);
        return (err);
    }

    dh_i = (double *)malloc(n_i * sizeof(double));
    dp = (double *)malloc((n_i-1) * sizeof(double));
    b = (double *)malloc(n_i*sizeof (double));
    b_av = (double *)malloc((n_i-1) * sizeof(double));

    for (i=0; i<n_i; i++) {
        b[i] = gsw_specvol_anom_standard(sa_i[i], ct_i[i], p_i[i]);
    }
    free(ct_i);
    free(sa_i);

    for (i=0; i<(n_i-1); i++) {
        b_av[i] = 0.5*(b[i+1] + b[i]);
        dp[i] = p_i[i+1] - p_i[i];
    }
    free(p_i);
    /* First calculate dynamic height relative to the first (shallowest)
       depth. */
    dh_i[0] = 0.0;
    for (i=1; i<n_i; i++) {
        dh_i[i] = dh_i[i-1] - b_av[i-1]*dp[i-1]*db2pa;
    }
    free(b);
    free(b_av);
    free(dp);

    dh_ref = dh_i[ipref];

    for (i=0; i<nz; i++) {
        dyn_height[i] = dh_i[p_indices[i]] - dh_ref;
    }
    free(p_indices);
    free(dh_i);

    return 0;
}

/* Until additional major coding is done, we can get closer to the v3_06_11
   check values by using the pchip option in the alternative calculation
   above, which is also the version we presently use in GSW-Python.
*/
 /* returns NULL on error, dyn_height if OK */
double *
gsw_geo_strf_dyn_height(double *sa, double *ct, double *p, double p_ref,
    int nz, double *dyn_height)
{
    int err;
    err = gsw_geo_strf_dyn_height_1(sa, ct, p, p_ref, nz, dyn_height, 1.0, 2);
    if (err == 0) {
        return dyn_height;
    }
    return NULL;

}


/*
!==========================================================================
pure subroutine gsw_geo_strf_dyn_height_pc (sa, ct, delta_p, &
                                            geo_strf_dyn_height_pc, p_mid)
!==========================================================================
!
!  Calculates dynamic height anomaly as the integral of specific volume
!  anomaly from the the sea surface pressure (0 Pa) to the pressure p.
!  This function, gsw_geo_strf_dyn_height_pc, is to used when the
!  Absolute Salinity and Conservative Temperature are piecewise constant in
!  the vertical over successive pressure intervals of delta_p (such as in
!  a forward "z-coordinate" ocean model).  "geo_strf_dyn_height_pc" is
!  the dynamic height anomaly with respect to the sea surface.  That is,
!  "geo_strf_dyn_height_pc" is the geostrophic streamfunction for the
!  difference between the horizontal velocity at the pressure concerned, p,
!  and the horizontal velocity at the sea surface.  Dynamic height anomaly
!  is the geostrophic streamfunction in an isobaric surface.  The reference
!  values used for the specific volume anomaly are SA = SSO = 35.16504 g/kg
!  and CT = 0 deg C.  The output values of geo_strf_dyn_height_pc are
!  given at the mid-point pressures, p_mid, of each layer in which SA and
!  CT are vertically piecewice constant (pc).  This function calculates
!  enthalpy using the computationally-efficient 75-term expression for
!  specific volume of Roquet et al., (2015).
!
!  SA       =  Absolute Salinity                                   [ g/kg ]
!  CT       =  Conservative Temperature (ITS-90)                  [ deg C ]
!  delta_p  =  difference in sea pressure between the deep and     [ dbar ]
!              shallow extents of each layer in which SA and CT
!              are vertically constant. delta_p must be positive.
!
!  Note. sea pressure is absolute pressure minus 10.1325 dbar.
!
!  geo_strf_dyn_height_pc =  dynamic height anomaly             [ m^2/s^2 ]
!  p_mid                  =  mid-point pressure in each layer      [ dbar ]
!--------------------------------------------------------------------------
*/
double *
gsw_geo_strf_dyn_height_pc(double *sa, double *ct, double *delta_p, int n_levels,
        double *geo_strf_dyn_height_pc, double *p_mid)
{
        int     i, np;
        double  *delta_h, delta_h_half, dyn_height_deep=0.0,
                *p_deep, *p_shallow;

        for (i=0; i<n_levels; i++)
            if (delta_p[i] < 0.0)
                return (NULL);

        np = n_levels;
        delta_h = (double *) malloc(3*np*sizeof (double));
        p_deep = delta_h+np; p_shallow = p_deep+np;

        for (i=0; i<np; i++) {
            p_deep[i] = (i==0)? delta_p[0] : p_deep[i-1] + delta_p[i];
            p_shallow[i] = p_deep[i] - delta_p[i];
            delta_h[i] = gsw_enthalpy_diff(sa[i],ct[i],p_shallow[i],p_deep[i]);
        }

        for (i=0; i<np; i++) {
            dyn_height_deep = dyn_height_deep - delta_h[i];
                /* This is Phi minus Phi_0 of Eqn. (3.32.2) of IOC et al. (2010).*/
            p_mid[i] = 0.5*(p_shallow[i]  + p_deep[i]);
            delta_h_half = gsw_enthalpy_diff(sa[i],ct[i],p_mid[i],p_deep[i]);

            geo_strf_dyn_height_pc[i] = gsw_enthalpy_sso_0(p_mid[i]) +
                                   dyn_height_deep + delta_h_half;
        }
        free(delta_h);
        return (geo_strf_dyn_height_pc);
}
/*
!==========================================================================
function gsw_gibbs(ns,nt,np,sa,t,p)
!==========================================================================
!
! seawater specific Gibbs free energy and derivatives up to order 2
!
! ns     : order of s derivative
! nt     : order of t derivative
! np     : order of p derivative
! sa     : Absolute Salinity                               [g/kg]
! t      : temperature                                     [deg C]
! p      : sea pressure                                    [dbar]
!                                                               -1
! gsw_gibbs  : specific Gibbs energy or its derivative     [J kg  ]
*/
double
gsw_gibbs(int ns, int nt, int np, double sa, double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        double  x2, x, y, z, g03, g08, return_value = 0.0;

        x2      = gsw_sfac*sa;
        x       = sqrt(x2);
        y       = t*0.025;
        z       = p*1e-4;

        if (ns == 0  && nt == 0  && np == 0) {
            g03 = 101.342743139674 + z*(100015.695367145 +
                z*(-2544.5765420363 + z*(284.517778446287 +
                z*(-33.3146754253611 + (4.20263108803084 -
                   0.546428511471039*z)*z)))) +
                y*(5.90578347909402 + z*(-270.983805184062 +
                z*(776.153611613101 + z*(-196.51255088122 +
                   (28.9796526294175 - 2.13290083518327*z)*z))) +
                y*(-12357.785933039 + z*(1455.0364540468 +
                z*(-756.558385769359 + z*(273.479662323528 +
                   z*(-55.5604063817218 + 4.34420671917197*z)))) +
                y*(736.741204151612 + z*(-672.50778314507 +
                z*(499.360390819152 + z*(-239.545330654412 +
                   (48.8012518593872 - 1.66307106208905*z)*z))) +
                y*(-148.185936433658 + z*(397.968445406972 +
                z*(-301.815380621876 + (152.196371733841 -
                   26.3748377232802*z)*z)) +
                y*(58.0259125842571 + z*(-194.618310617595 +
                z*(120.520654902025 + z*(-55.2723052340152 +
                   6.48190668077221*z))) +
                y*(-18.9843846514172 + y*(3.05081646487967 -
                   9.63108119393062*z) +
                z*(63.5113936641785 + z*(-22.2897317140459 +
                   8.17060541818112*z))))))));

            g08 = x2*(1416.27648484197 + z*(-3310.49154044839 +
                z*(384.794152978599 + z*(-96.5324320107458 +
                   (15.8408172766824 - 2.62480156590992*z)*z))) +
                x*(-2432.14662381794 + x*(2025.80115603697 +
                y*(543.835333000098 + y*(-68.5572509204491 +
                y*(49.3667694856254 + y*(-17.1397577419788 +
                   2.49697009569508*y))) - 22.6683558512829*z) +
                x*(-1091.66841042967 - 196.028306689776*y +
                x*(374.60123787784 - 48.5891069025409*x +
                   36.7571622995805*y) + 36.0284195611086*z) +
                z*(-54.7919133532887 + (-4.08193978912261 -
                   30.1755111971161*z)*z)) +
                z*(199.459603073901 + z*(-52.2940909281335 +
                   (68.0444942726459 - 3.41251932441282*z)*z)) +
                y*(-493.407510141682 + z*(-175.292041186547 +
                   (83.1923927801819 - 29.483064349429*z)*z) +
                y*(-43.0664675978042 + z*(383.058066002476 +
                   z*(-54.1917262517112 + 25.6398487389914*z)) +
                y*(-10.0227370861875 - 460.319931801257*z +
                   y*(0.875600661808945 + 234.565187611355*z))))) +
                y*(168.072408311545 + z*(729.116529735046 +
                z*(-343.956902961561 + z*(124.687671116248 +
                   z*(-31.656964386073 + 7.04658803315449*z)))) +
                y*(880.031352997204 + y*(-225.267649263401 +
                y*(91.4260447751259 + y*(-21.6603240875311 +
                   2.13016970847183*y) +
                z*(-297.728741987187 + (74.726141138756 -
                   36.4872919001588*z)*z)) +
                z*(694.244814133268 + z*(-204.889641964903 +
                   (113.561697840594 - 11.1282734326413*z)*z))) +
                z*(-860.764303783977 + z*(337.409530269367 +
                z*(-178.314556207638 + (44.2040358308 -
                   7.92001547211682*z)*z))))));

            if (sa > 0.0)
                g08     = g08 + x2*(5812.81456626732 +
                          851.226734946706*y)*log(x);

            return_value        = g03 + g08;

        } else if (ns == 1  && nt == 0  && np == 0) {

            g08 = 8645.36753595126 + z*(-6620.98308089678 +
                z*(769.588305957198 + z*(-193.0648640214916 +
                   (31.6816345533648 - 5.24960313181984*z)*z))) +
                x*(-7296.43987145382 + x*(8103.20462414788 +
                y*(2175.341332000392 + y*(-274.2290036817964 +
                y*(197.4670779425016 + y*(-68.5590309679152 +
                   9.98788038278032*y))) - 90.6734234051316*z) +
                x*(-5458.34205214835 - 980.14153344888*y +
                x*(2247.60742726704 - 340.1237483177863*x +
                   220.542973797483*y) + 180.142097805543*z) +
                z*(-219.1676534131548 + (-16.32775915649044 -
                   120.7020447884644*z)*z)) +
                z*(598.378809221703 + z*(-156.8822727844005 +
                   (204.1334828179377 - 10.23755797323846*z)*z)) +
                y*(-1480.222530425046 + z*(-525.876123559641 +
                   (249.57717834054571 - 88.449193048287*z)*z) +
                y*(-129.1994027934126 + z*(1149.174198007428 +
                   z*(-162.5751787551336 + 76.9195462169742*z)) +
                y*(-30.0682112585625 - 1380.9597954037708*z +
                   y*(2.626801985426835 + 703.695562834065*z))))) +
                y*(1187.3715515697959 + z*(1458.233059470092 +
                z*(-687.913805923122 + z*(249.375342232496 +
                   z*(-63.313928772146 + 14.09317606630898*z)))) +
                y*(1760.062705994408 + y*(-450.535298526802 +
                y*(182.8520895502518 + y*(-43.3206481750622 +
                   4.26033941694366*y) +
                z*(-595.457483974374 + (149.452282277512 -
                   72.9745838003176*z)*z)) +
                z*(1388.489628266536 + z*(-409.779283929806 +
                   (227.123395681188 - 22.2565468652826*z)*z))) +
                z*(-1721.528607567954 + z*(674.819060538734 +
                z*(-356.629112415276 + (88.4080716616 -
                   15.84003094423364*z)*z)))));

            if (sa > 0.0)
                g08     = g08 + (11625.62913253464 + 1702.453469893412*y)*
                          log(x);
            else
                g08 = 0.0;

            return_value        = 0.5*gsw_sfac*g08;

        } else if (ns == 0  && nt == 1  && np == 0) {

            g03 = 5.90578347909402 + z*(-270.983805184062 +
                z*(776.153611613101 + z*(-196.51255088122 +
                   (28.9796526294175 - 2.13290083518327*z)*z))) +
                y*(-24715.571866078 + z*(2910.0729080936 +
                z*(-1513.116771538718 + z*(546.959324647056 +
                   z*(-111.1208127634436 + 8.68841343834394*z)))) +
                y*(2210.2236124548363 + z*(-2017.52334943521 +
                z*(1498.081172457456 + z*(-718.6359919632359 +
                   (146.4037555781616 - 4.9892131862671505*z)*z))) +
                y*(-592.743745734632 + z*(1591.873781627888 +
                z*(-1207.261522487504 + (608.785486935364 -
                   105.4993508931208*z)*z)) +
                y*(290.12956292128547 + z*(-973.091553087975 +
                z*(602.603274510125 + z*(-276.361526170076 +
                   32.40953340386105*z))) +
                y*(-113.90630790850321 + y*(21.35571525415769 -
                   67.41756835751434*z) +
                z*(381.06836198507096 + z*(-133.7383902842754 +
                   49.023632509086724*z)))))));

            g08 = x2*(168.072408311545 + z*(729.116529735046 +
                z*(-343.956902961561 + z*(124.687671116248 +
                   z*(-31.656964386073 + 7.04658803315449*z)))) +
                x*(-493.407510141682 + x*(543.835333000098 +
                   x*(-196.028306689776 + 36.7571622995805*x) +
                y*(-137.1145018408982 + y*(148.10030845687618 +
                   y*(-68.5590309679152 + 12.4848504784754*y))) -
                   22.6683558512829*z) + z*(-175.292041186547 +
                   (83.1923927801819 - 29.483064349429*z)*z) +
                y*(-86.1329351956084 + z*(766.116132004952 +
                   z*(-108.3834525034224 + 51.2796974779828*z)) +
                y*(-30.0682112585625 - 1380.9597954037708*z +
                   y*(3.50240264723578 + 938.26075044542*z)))) +
                y*(1760.062705994408 + y*(-675.802947790203 +
                y*(365.7041791005036 + y*(-108.30162043765552 +
                   12.78101825083098*y) +
                z*(-1190.914967948748 + (298.904564555024 -
                   145.9491676006352*z)*z)) +
                z*(2082.7344423998043 + z*(-614.668925894709 +
                   (340.685093521782 - 33.3848202979239*z)*z))) +
                z*(-1721.528607567954 + z*(674.819060538734 +
                z*(-356.629112415276 + (88.4080716616 -
                   15.84003094423364*z)*z)))));

            if (sa > 0.0)
                g08     = g08 + 851.226734946706*x2*log(x);

            return_value        = (g03 + g08)*0.025;

        } else if (ns == 0  && nt == 0  && np == 1) {

            g03 = 100015.695367145 + z*(-5089.1530840726 +
                z*(853.5533353388611 + z*(-133.2587017014444 +
                   (21.0131554401542 - 3.278571068826234*z)*z))) +
                y*(-270.983805184062 + z*(1552.307223226202 +
                z*(-589.53765264366 + (115.91861051767 -
                   10.664504175916349*z)*z)) +
                y*(1455.0364540468 + z*(-1513.116771538718 +
                z*(820.438986970584 + z*(-222.2416255268872 +
                   21.72103359585985*z))) +
                y*(-672.50778314507 + z*(998.720781638304 +
                z*(-718.6359919632359 + (195.2050074375488 -
                   8.31535531044525*z)*z)) +
                y*(397.968445406972 + z*(-603.630761243752 +
                   (456.589115201523 - 105.4993508931208*z)*z) +
                y*(-194.618310617595 + y*(63.5113936641785 -
                   9.63108119393062*y +
                z*(-44.5794634280918 + 24.511816254543362*z)) +
                z*(241.04130980405 + z*(-165.8169157020456 +
                25.92762672308884*z)))))));

            g08 = x2*(-3310.49154044839 + z*(769.588305957198 +
                z*(-289.5972960322374 + (63.3632691067296 -
                   13.1240078295496*z)*z)) +
                x*(199.459603073901 + x*(-54.7919133532887 +
                   36.0284195611086*x - 22.6683558512829*y +
                (-8.16387957824522 - 90.52653359134831*z)*z) +
                z*(-104.588181856267 + (204.1334828179377 -
                   13.65007729765128*z)*z) +
                y*(-175.292041186547 + (166.3847855603638 -
                   88.449193048287*z)*z +
                y*(383.058066002476 + y*(-460.319931801257 +
                   234.565187611355*y) +
                z*(-108.3834525034224 + 76.9195462169742*z)))) +
                y*(729.116529735046 + z*(-687.913805923122 +
                z*(374.063013348744 + z*(-126.627857544292 +
                   35.23294016577245*z))) +
                y*(-860.764303783977 + y*(694.244814133268 +
                y*(-297.728741987187 + (149.452282277512 -
                   109.46187570047641*z)*z) +
                z*(-409.779283929806 + (340.685093521782 -
                   44.5130937305652*z)*z)) +
                z*(674.819060538734 + z*(-534.943668622914 +
                   (176.8161433232 - 39.600077360584095*z)*z)))));

            return_value        = (g03 + g08)*1.0e-8;

        } else if (ns == 0  && nt == 2  && np == 0) {

            g03 = -24715.571866078 + z*(2910.0729080936 + z*
                (-1513.116771538718 + z*(546.959324647056 +
                 z*(-111.1208127634436 + 8.68841343834394*z)))) +
                y*(4420.4472249096725 + z*(-4035.04669887042 +
                z*(2996.162344914912 + z*(-1437.2719839264719 +
                   (292.8075111563232 - 9.978426372534301*z)*z))) +
                y*(-1778.231237203896 + z*(4775.621344883664 +
                z*(-3621.784567462512 + (1826.356460806092 -
                   316.49805267936244*z)*z)) +
                y*(1160.5182516851419 + z*(-3892.3662123519 +
                z*(2410.4130980405 + z*(-1105.446104680304 +
                   129.6381336154442*z))) +
                y*(-569.531539542516 + y*(128.13429152494615 -
                   404.50541014508605*z) +
                z*(1905.341809925355 + z*(-668.691951421377 +
                   245.11816254543362*z))))));

            g08 = x2*(1760.062705994408 + x*(-86.1329351956084 +
                x*(-137.1145018408982 + y*(296.20061691375236 +
                   y*(-205.67709290374563 + 49.9394019139016*y))) +
                z*(766.116132004952 + z*(-108.3834525034224 +
                   51.2796974779828*z)) +
                y*(-60.136422517125 - 2761.9195908075417*z +
                   y*(10.50720794170734 + 2814.78225133626*z))) +
                y*(-1351.605895580406 + y*(1097.1125373015109 +
                   y*(-433.20648175062206 + 63.905091254154904*y) +
                z*(-3572.7449038462437 + (896.713693665072 -
                   437.84750280190565*z)*z)) +
                z*(4165.4688847996085 + z*(-1229.337851789418 +
                   (681.370187043564 - 66.7696405958478*z)*z))) +
                z*(-1721.528607567954 + z*(674.819060538734 +
                z*(-356.629112415276 + (88.4080716616 -
                   15.84003094423364*z)*z))));

            return_value        = (g03 + g08)*0.000625;

        } else if (ns == 1  && nt == 0  && np == 1) {

            g08 =     -6620.98308089678 + z*(1539.176611914396 +
                z*(-579.1945920644748 + (126.7265382134592 -
                   26.2480156590992*z)*z)) +
                x*(598.378809221703 + x*(-219.1676534131548 +
                   180.142097805543*x - 90.6734234051316*y +
                (-32.65551831298088 - 362.10613436539325*z)*z) +
                z*(-313.764545568801 + (612.4004484538132 -
                   40.95023189295384*z)*z) +
                y*(-525.876123559641 + (499.15435668109143 -
                   265.347579144861*z)*z +
                y*(1149.174198007428 + y*(-1380.9597954037708 +
                   703.695562834065*y) +
                z*(-325.1503575102672 + 230.7586386509226*z)))) +
                y*(1458.233059470092 + z*(-1375.827611846244 +
                z*(748.126026697488 + z*(-253.255715088584 +
                   70.4658803315449*z))) +
                y*(-1721.528607567954 + y*(1388.489628266536 +
                y*(-595.457483974374 + (298.904564555024 -
                   218.92375140095282*z)*z) +
                z*(-819.558567859612 + (681.370187043564 -
                   89.0261874611304*z)*z)) +
                z*(1349.638121077468 + z*(-1069.887337245828 +
                   (353.6322866464 - 79.20015472116819*z)*z))));

            return_value        = g08*gsw_sfac*0.5e-8;

        } else if (ns == 0  && nt == 1  && np == 1) {

            g03 = -270.983805184062 + z*(1552.307223226202 +
                z*(-589.53765264366 + (115.91861051767 -
                   10.664504175916349*z)*z)) +
                y*(2910.0729080936 + z*(-3026.233543077436 +
                z*(1640.877973941168 + z*(-444.4832510537744 +
                   43.4420671917197*z))) +
                y*(-2017.52334943521 + z*(2996.162344914912 +
                z*(-2155.907975889708 + (585.6150223126464 -
                   24.946065931335752*z)*z)) +
                y*(1591.873781627888 + z*(-2414.523044975008 +
                   (1826.356460806092 - 421.9974035724832*z)*z) +
                y*(-973.091553087975 + z*(1205.20654902025 +
                   z*(-829.084578510228 + 129.6381336154442*z)) +
                y*(381.06836198507096 - 67.41756835751434*y +
                   z*(-267.4767805685508 + 147.07089752726017*z))))));

            g08 = x2*(729.116529735046 + z*(-687.913805923122 +
                z*(374.063013348744 + z*(-126.627857544292 +
                   35.23294016577245*z))) +
                x*(-175.292041186547 - 22.6683558512829*x +
                   (166.3847855603638 - 88.449193048287*z)*z +
                y*(766.116132004952 + y*(-1380.9597954037708 +
                   938.26075044542*y) +
                z*(-216.7669050068448 + 153.8390924339484*z))) +
                y*(-1721.528607567954 + y*(2082.7344423998043 +
                y*(-1190.914967948748 + (597.809129110048 -
                   437.84750280190565*z)*z) +
                z*(-1229.337851789418 + (1022.055280565346 -
                   133.5392811916956*z)*z)) +
                z*(1349.638121077468 + z*(-1069.887337245828 +
                   (353.6322866464 - 79.20015472116819*z)*z))));

            return_value        = (g03 + g08)*2.5e-10;

        } else if (ns == 1  && nt == 1  && np == 0) {

            g08 = 1187.3715515697959 + z*(1458.233059470092 +
                z*(-687.913805923122 + z*(249.375342232496 +
                z*(-63.313928772146 + 14.09317606630898*z)))) +
                x*(-1480.222530425046 + x*(2175.341332000392 +
                x*(-980.14153344888 + 220.542973797483*x) +
                y*(-548.4580073635929 + y*(592.4012338275047 +
                y*(-274.2361238716608 + 49.9394019139016*y))) -
                90.6734234051316*z) +
                z*(-525.876123559641 + (249.57717834054571 -
                88.449193048287*z)*z) +
                y*(-258.3988055868252 + z*(2298.348396014856 +
                z*(-325.1503575102672 + 153.8390924339484*z)) +
                y*(-90.2046337756875 - 4142.8793862113125*z +
                y*(10.50720794170734 + 2814.78225133626*z)))) +
                y*(3520.125411988816 + y*(-1351.605895580406 +
                y*(731.4083582010072 + y*(-216.60324087531103 +
                25.56203650166196*y) +
                z*(-2381.829935897496 + (597.809129110048 -
                291.8983352012704*z)*z)) +
                z*(4165.4688847996085 + z*(-1229.337851789418 +
                (681.370187043564 - 66.7696405958478*z)*z))) +
                z*(-3443.057215135908 + z*(1349.638121077468 +
                z*(-713.258224830552 + (176.8161433232 -
                31.68006188846728*z)*z))));

            if (sa > 0.0)
                g08 = g08 + 1702.453469893412*log(x);

            return_value        = 0.5*gsw_sfac*0.025*g08;

        } else if (ns == 2  && nt == 0  && np == 0) {

            g08 = 2.0*(8103.20462414788 +
                y*(2175.341332000392 + y*(-274.2290036817964 +
                y*(197.4670779425016 + y*(-68.5590309679152 +
                9.98788038278032*y))) - 90.6734234051316*z) +
                1.5*x*(-5458.34205214835 - 980.14153344888*y +
                (4.0/3.0)*x*(2247.60742726704 - 340.1237483177863*1.25*x +
                220.542973797483*y) + 180.142097805543*z) +
                z*(-219.1676534131548 + (-16.32775915649044 -
                120.7020447884644*z)*z));

            if (x > 0.0) {
                g08 += (-7296.43987145382 + z*(598.378809221703 +
                    z*(-156.8822727844005 + (204.1334828179377 -
                    10.23755797323846*z)*z)) +
                    y*(-1480.222530425046 + z*(-525.876123559641 +
                    (249.57717834054571 - 88.449193048287*z)*z) +
                    y*(-129.1994027934126 + z*(1149.174198007428 +
                    z*(-162.5751787551336 + 76.9195462169742*z)) +
                    y*(-30.0682112585625 - 1380.9597954037708*z +
                    y*(2.626801985426835 + 703.695562834065*z)))))/x +
                    (11625.62913253464 + 1702.453469893412*y)/x2;
            } else
                g08 = 0.0;

            return_value = 0.25*gsw_sfac*gsw_sfac*g08;

        } else if (ns == 0  && nt == 0  && np == 2) {

            g03 = -5089.1530840726 + z*(1707.1066706777221 +
                z*(-399.7761051043332 + (84.0526217606168 -
                   16.39285534413117*z)*z)) +
                y*(1552.307223226202 + z*(-1179.07530528732 +
                   (347.75583155301 - 42.658016703665396*z)*z) +
                y*(-1513.116771538718 + z*(1640.877973941168 +
                   z*(-666.7248765806615 + 86.8841343834394*z)) +
                y*(998.720781638304 + z*(-1437.2719839264719 +
                   (585.6150223126464 - 33.261421241781*z)*z) +
                y*(-603.630761243752 + (913.178230403046 -
                   316.49805267936244*z)*z +
                y*(241.04130980405 + y*(-44.5794634280918 +
                   49.023632509086724*z) +
                z*(-331.6338314040912 + 77.78288016926652*z))))));

            g08 = x2*(769.588305957198 + z*(-579.1945920644748 +
                     (190.08980732018878 - 52.4960313181984*z)*z) +
                x*(-104.588181856267 + x*(-8.16387957824522 -
                   181.05306718269662*z) +
                (408.2669656358754 - 40.95023189295384*z)*z +
                y*(166.3847855603638 - 176.898386096574*z +
                   y*(-108.3834525034224 + 153.8390924339484*z))) +
                y*(-687.913805923122 + z*(748.126026697488 +
                   z*(-379.883572632876 + 140.9317606630898*z)) +
                y*(674.819060538734 + z*(-1069.887337245828 +
                   (530.4484299696 - 158.40030944233638*z)*z) +
                y*(-409.779283929806 + y*(149.452282277512 -
                   218.92375140095282*z) +
                (681.370187043564 - 133.5392811916956*z)*z))));

            return_value        = (g03 + g08)*1e-16 ;

        } else
            return_value        = GSW_INVALID_VALUE;

        return (return_value);
}
/*
! =========================================================================
elemental function gsw_gibbs_ice (nt, np, t, p)
! =========================================================================
!
!  Ice specific Gibbs energy and derivatives up to order 2.
!
!  nt  =  order of t derivative                      [ integers 0, 1 or 2 ]
!  np  =  order of p derivative                      [ integers 0, 1 or 2 ]
!  t   =  in-situ temperature (ITS-90)                            [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!
!  gibbs_ice = Specific Gibbs energy of ice or its derivatives.
!            The Gibbs energy (when nt = np = 0) has units of:     [ J/kg ]
!            The temperature derivatives are output in units of:
!                                                      [ (J/kg) (K)^(-nt) ]
!            The pressure derivatives are output in units of:
!                                                     [ (J/kg) (Pa)^(-np) ]
!            The mixed derivatives are output in units of:
!                                           [ (J/kg) (K)^(-nt) (Pa)^(-np) ]
!  Note. The derivatives are taken with respect to pressure in Pa, not
!    withstanding that the pressure input into this routine is in dbar.
!--------------------------------------------------------------------------
*/
double
gsw_gibbs_ice (int nt, int np, double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_GIBBS_ICE_COEFFICIENTS;
        double  dzi, g0, g0p, g0pp, sqrec_pt;
        DCOMPLEX        r2, r2p, r2pp, g, sqtau_t1, sqtau_t2, tau,
                        tau_t1, tau_t2;
        double  s0 = -3.32733756492168e3;

        tau = (t + gsw_t0)*rec_tt;

        dzi = db2pa*p*rec_pt;

        if (nt == 0 && np == 0) {

            tau_t1 = tau/t1;
            sqtau_t1 = tau_t1*tau_t1;
            tau_t2 = tau/t2;
            sqtau_t2 = tau_t2*tau_t2;

            g0 = g00 + dzi*(g01 + dzi*(g02 + dzi*(g03 + g04*dzi)));

            r2 = r20 + dzi*(r21 + r22*dzi);

            g = r1*(tau*log((1.0 + tau_t1)/(1.0 - tau_t1))
                + t1*(log(1.0 - sqtau_t1) - sqtau_t1))
                + r2*(tau*log((1.0 + tau_t2)/(1.0 - tau_t2))
                + t2*(log(1.0 - sqtau_t2) - sqtau_t2));

            return real(g0 - tt*(s0*tau - real(g)));  // EF: bug in original.

        } else if (nt == 1 && np == 0) {

            tau_t1 = tau/t1;
            tau_t2 = tau/t2;

            r2 = r20 + dzi*(r21 + r22*dzi);

            g = r1*(log((1.0 + tau_t1)/(1.0 - tau_t1)) - 2.0*tau_t1)
                + r2*(log((1.0 + tau_t2)/(1.0 - tau_t2)) - 2.0*tau_t2);

            return (-s0 + real(g));

        } else if (nt == 0 && np == 1) {

            tau_t2 = tau/t2;
            sqtau_t2 = tau_t2*tau_t2;

            g0p = rec_pt*(g01 + dzi*(2.0*g02 + dzi*(3.0*g03 + 4.0*g04*dzi)));

            r2p = rec_pt*(r21 + 2.0*r22*dzi);

            g = r2p*(tau*log((1.0 + tau_t2)/(1.0 - tau_t2))
                + t2*(log(1.0 - sqtau_t2) - sqtau_t2));

            return (g0p + tt*real(g));

        } else if (nt == 1 && np == 1) {

            tau_t2 = tau/t2;

            r2p = rec_pt*(r21 + 2.0*r22*dzi) ;

            g = r2p*(log((1.0 + tau_t2)/(1.0 - tau_t2)) - 2.0*tau_t2);

            return (real(g));

        } else if (nt == 2 && np == 0) {

            r2 = r20 + dzi*(r21 + r22*dzi);

            g = r1*(1.0/(t1 - tau) + 1.0/(t1 + tau) - 2.0/t1)
                + r2*(1.0/(t2 - tau) + 1.0/(t2 + tau) - 2.0/t2);

            return (rec_tt*real(g));

        } else if (nt == 0 && np == 2) {

            sqrec_pt = rec_pt*rec_pt;

            tau_t2 = tau/t2;
            sqtau_t2 = tau_t2*tau_t2;

            g0pp = sqrec_pt*(2.0*g02 + dzi*(6.0*g03 + 12.0*g04*dzi));

            r2pp = 2.0*r22*sqrec_pt;

            g = r2pp*(tau*log((1.0 + tau_t2)/(1.0 - tau_t2))
                + t2*(log(1.0 - sqtau_t2) - sqtau_t2));

           return (g0pp + tt*real(g));

        } else
           return (GSW_INVALID_VALUE);
}
/*
! =========================================================================
elemental function gsw_gibbs_ice_part_t (t, p)
! =========================================================================
!
!  part of the the first temperature derivative of Gibbs energy of ice
!  that is the output is gibbs_ice(1,0,t,p) + S0
!
!  t   =  in-situ temperature (ITS-90)                            [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!
!  gibbs_ice_part_t = part of temperature derivative       [ J kg^-1 K^-1 ]
!--------------------------------------------------------------------------
*/
double
gsw_gibbs_ice_part_t(double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_GIBBS_ICE_COEFFICIENTS;
        double  dzi, tau;
        DCOMPLEX        g, tau_t1, tau_t2, r2;

        tau = (t + gsw_t0)*rec_tt;

        dzi = db2pa*p*rec_pt;

        tau_t1 = tau/t1;
        tau_t2 = tau/t2;

        r2 = r20 + dzi*(r21 + r22*dzi);

        g = r1*(log((1.0 + tau_t1)/(1.0 - tau_t1)) - 2.0*tau_t1)
            + r2*(log((1.0 + tau_t2)/(1.0 - tau_t2)) - 2.0*tau_t2);

        return (real(g));
}
/*
! =========================================================================
elemental function gsw_gibbs_ice_pt0 (pt0)
! =========================================================================
!
!  Part of the the first temperature derivative of Gibbs energy of ice
!  that is the output is "gibbs_ice(1,0,pt0,0) + s0"
!
!  pt0  =  potential temperature with reference sea pressure of zero dbar
!                                                                 [ deg C ]
!
!  gsw_gibbs_ice_pt0 = part of temperature derivative     [ J kg^-1 K^-1 ]
!--------------------------------------------------------------------------
*/
double
gsw_gibbs_ice_pt0(double pt0)
{
        GSW_TEOS10_CONSTANTS;
        GSW_GIBBS_ICE_COEFFICIENTS;
        double  tau;
        DCOMPLEX        g, tau_t1, tau_t2;

        tau = (pt0 + gsw_t0)*rec_tt;

        tau_t1 = tau/t1;
        tau_t2 = tau/t2;

        g = r1*(log((1.0 + tau_t1)/(1.0 - tau_t1)) - 2.0*tau_t1)
            + r20*(log((1.0 + tau_t2)/(1.0 - tau_t2)) - 2.0*tau_t2);

        return (real(g));
}
/*
! =========================================================================
elemental function gsw_gibbs_ice_pt0_pt0 (pt0)
! =========================================================================
!
!  The second temperature derivative of Gibbs energy of ice at the
!  potential temperature with reference sea pressure of zero dbar.  That is
!  the output is gibbs_ice(2,0,pt0,0).
!
!  pt0  =  potential temperature with reference sea pressure of zero dbar
!                                                                 [ deg C ]
!
!  gsw_gibbs_ice_pt0_pt0 = temperature second derivative at pt0
!--------------------------------------------------------------------------
*/
double
gsw_gibbs_ice_pt0_pt0(double pt0)
{
        GSW_TEOS10_CONSTANTS;
        GSW_GIBBS_ICE_COEFFICIENTS;
        double  tau;
        DCOMPLEX        g;

        tau = (pt0 + gsw_t0)*rec_tt;

        g = r1*(1.0/(t1 - tau) + 1.0/(t1 + tau) - 2.0/t1)
            + r20*(1.0/(t2 - tau) + 1.0/(t2 + tau) - 2.0/t2);

        return (rec_tt*real(g));
}
/*
!==========================================================================
function gsw_gibbs_pt0_pt0(sa,pt0)
!==========================================================================

! gibbs_tt at (sa,pt,0)
!
! sa     : Absolute Salinity                            [g/kg]
! pt0    : potential temperature                        [deg C]
!
! gibbs_pt0_pt0 : gibbs_tt at (sa,pt,0)
*/
double
gsw_gibbs_pt0_pt0(double sa, double pt0)
{
        GSW_TEOS10_CONSTANTS;
        double  x2, x, y, g03, g08;

        x2      = gsw_sfac*sa;
        x       = sqrt(x2);
        y       = pt0*0.025;

        g03     = -24715.571866078 +
                y*(4420.4472249096725 +
                y*(-1778.231237203896 +
                y*(1160.5182516851419 +
                y*(-569.531539542516 + y*128.13429152494615))));

        g08     = x2*(1760.062705994408 + x*(-86.1329351956084 +
                x*(-137.1145018408982 + y*(296.20061691375236 +
                y*(-205.67709290374563 + 49.9394019139016*y))) +
                y*(-60.136422517125 + y*10.50720794170734)) +
                y*(-1351.605895580406 + y*(1097.1125373015109 +
                y*(-433.20648175062206 + 63.905091254154904*y))));

        return ((g03 + g08)*0.000625);
}
/*
!==========================================================================
function gsw_grav(lat,p)
!==========================================================================

! Calculates acceleration due to gravity as a function of latitude and as
!  a function of pressure in the ocean.
!
! lat  =  latitude in decimal degrees north                [ -90 ... +90 ]
! p    =  sea pressure                                     [ dbar ]
!
! grav : grav  =  gravitational acceleration               [ m s^-2 ]
*/
double
gsw_grav(double lat, double p)
{
        GSW_TEOS10_CONSTANTS;
        double  x, sin2, gs, z;

        x       = sin(lat*deg2rad);  /* convert to radians */
        sin2    = x*x;
        gs      = 9.780327*(1.0 + (5.2792e-3 + (2.32e-5*sin2))*sin2);

        z       = gsw_z_from_p(p,lat, 0, 0);

        return (gs*(1.0 - gamma*z));    /* z is the height corresponding to p.
                                           Note. In the ocean z is negative. */
}
/*
!==========================================================================
elemental function gsw_helmholtz_energy_ice (t, p)
!==========================================================================
!
!  Calculates the Helmholtz energy of ice.
!
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  Helmholtz_energy_ice  =  Helmholtz energy of ice                [ J/kg ]
!--------------------------------------------------------------------------
*/
double
gsw_helmholtz_energy_ice(double t, double p)
{
        GSW_TEOS10_CONSTANTS;

        return (gsw_gibbs_ice(0,0,t,p)
                           - (db2pa*p + gsw_p0)*gsw_gibbs_ice(0,1,t,p));
}
/*
!==========================================================================
function  gsw_hill_ratio_at_sp2(t)
!==========================================================================

!  Calculates the Hill ratio, which is the adjustment needed to apply for
!  Practical Salinities smaller than 2.  This ratio is defined at a
!  Practical Salinity = 2 and in-situ temperature, t using PSS-78. The Hill
!  ratio is the ratio of 2 to the output of the Hill et al. (1986) formula
!  for Practical Salinity at the conductivity ratio, Rt, at which Practical
!  Salinity on the PSS-78 scale is exactly 2.
!
!  t                 : in-situ temperature (ITS-90)              [deg C]
!  hill_ratio_at_sp2 : Hill ratio                                [dimensionless]
*/
double
gsw_hill_ratio_at_sp2(double t)
{
        GSW_SP_COEFFICIENTS;
        double  g0 = 2.641463563366498e-1, g1 = 2.007883247811176e-4,
                g2 = -4.107694432853053e-6, g3 = 8.401670882091225e-8,
                g4 = -1.711392021989210e-9, g5 = 3.374193893377380e-11,
                g6 = -5.923731174730784e-13, g7 = 8.057771569962299e-15,
                g8 = -7.054313817447962e-17, g9 = 2.859992717347235e-19,
                sp2 = 2.0;
        double  t68, ft68, rtx0, dsp_drtx, sp_est, rtx, rtxm, x, part1, part2;
        double  sqrty, sp_hill_raw_at_sp2;

        t68     = t*1.00024;
        ft68    = (t68 - 15.0)/(1.0 + k*(t68 - 15.0));

    /*!------------------------------------------------------------------------
    **! Find the initial estimates of Rtx (Rtx0) and of the derivative dSP_dRtx
    **! at SP = 2.
    **!------------------------------------------------------------------------
    */
        rtx0    = g0 + t68*(g1 + t68*(g2 + t68*(g3 + t68*(g4 + t68*(g5
                + t68*(g6 + t68*(g7 + t68*(g8 + t68*g9))))))));

        dsp_drtx= a1 + (2*a2 + (3*a3 + (4*a4 + 5*a5*rtx0)*rtx0)*rtx0)*rtx0 +
                ft68*(b1 + (2*b2 + (3*b3 + (4*b4 + 5*b5*rtx0)*rtx0)*rtx0)*rtx0);

    /*!-------------------------------------------------------------------------
    **! Begin a single modified Newton-Raphson iteration to find Rt at SP = 2.
    **!-------------------------------------------------------------------------
    */
        sp_est  = a0 + (a1 + (a2 + (a3 + (a4 + a5*rtx0)*rtx0)*rtx0)*rtx0)*rtx0
                + ft68*(b0 + (b1 + (b2+ (b3 + (b4 + b5*rtx0)*rtx0)*rtx0)*
                  rtx0)*rtx0);
        rtx     = rtx0 - (sp_est - sp2)/dsp_drtx;
        rtxm    = 0.5*(rtx + rtx0);
        dsp_drtx= a1 + (2*a2 + (3*a3 + (4*a4 + 5*a5*rtxm)*rtxm)*rtxm)*rtxm
                + ft68*(b1 + (2*b2 + (3*b3 + (4*b4 + 5*b5*rtxm)*
                                                rtxm)*rtxm)*rtxm);
        rtx     = rtx0 - (sp_est - sp2)/dsp_drtx;
    /*
    **! This is the end of one full iteration of the modified Newton-Raphson
    **! iterative equation solver. The error in Rtx at this point is equivalent
    **! to an error in SP of 9e-16 psu.
    */

        x       = 400.0*rtx*rtx;
        sqrty   = 10.0*rtx;
        part1   = 1.0 + x*(1.5 + x);
        part2   = 1.0 + sqrty*(1.0 + sqrty*(1.0 + sqrty));
        sp_hill_raw_at_sp2 = sp2 - a0/part1 - b0*ft68/part2;

        return (2.0/sp_hill_raw_at_sp2);
}
/*
!==========================================================================
elemental subroutine gsw_ice_fraction_to_freeze_seawater (sa, ct, p, &
                                          t_ih, sa_freeze, ct_freeze, w_ih)
!==========================================================================
!
!  Calculates the mass fraction of ice (mass of ice divided by mass of ice
!  plus seawater), which, when melted into seawater having (SA,CT,p) causes
!  the final dilute seawater to be at the freezing temperature.  The other
!  outputs are the Absolute Salinity and Conservative Temperature of the
!  final diluted seawater.
!
!  SA   =  Absolute Salinity of seawater                           [ g/kg ]
!  CT   =  Conservative Temperature of seawater (ITS-90)          [ deg C ]
!  p    =  sea pressure                                            [ dbar ]
!            ( i.e. absolute pressure - 10.1325d0 dbar )
!  t_Ih =  in-situ temperature of the ice at pressure p (ITS-90)  [ deg C ]
!
!  SA_freeze = Absolute Salinity of seawater after the mass fraction of
!              ice, ice_fraction, at temperature t_Ih has melted into the
!              original seawater, and the final mixture is at the freezing
!              temperature of seawater.                            [ g/kg ]
!
!  CT_freeze = Conservative Temperature of seawater after the mass
!              fraction, w_Ih, of ice at temperature t_Ih has melted into
!              the original seawater, and the final mixture is at the
!              freezing temperature of seawater.                  [ deg C ]
!
!  w_Ih      = mass fraction of ice, having in-situ temperature t_Ih,
!              which, when melted into seawater at (SA,CT,p) leads to the
!              final diluted seawater being at the freezing temperature.
!              This output must be between 0 and 1.              [unitless]
!--------------------------------------------------------------------------
*/
void
gsw_ice_fraction_to_freeze_seawater(double sa, double ct, double p, double t_ih,
        double *sa_freeze, double *ct_freeze, double *w_ih)
{
        int     no_iter;
        double  ctf, ctf_mean, ctf_old, ctf_plus1, ctf_zero,
                dfunc_dsaf, func, func_plus1, func_zero, h, h_ih,
                saf, saf_mean, saf_old, tf, h_hat_sa, h_hat_ct, ctf_sa;
        double  sa0 = 0.0, saturation_fraction = 0.0;

        ctf = gsw_ct_freezing(sa,p,saturation_fraction);
        if (ct < ctf) {
            /*The seawater ct input is below the freezing temp*/
            *sa_freeze = GSW_INVALID_VALUE;
            *ct_freeze = *sa_freeze;
            *w_ih = *sa_freeze;
            return;
        }

        tf = gsw_t_freezing(sa0,p,saturation_fraction);
        if (t_ih > tf) {
            /*The input, t_Ih, exceeds the freezing temperature at sa = 0*/
            *sa_freeze = GSW_INVALID_VALUE;
            *ct_freeze = *sa_freeze;
            *w_ih = *sa_freeze;
            return;
        }

        h = gsw_enthalpy_ct_exact(sa,ct,p);
        h_ih = gsw_enthalpy_ice(t_ih,p);

        ctf_zero = gsw_ct_freezing(sa0,p,saturation_fraction);
        func_zero = sa*(gsw_enthalpy_ct_exact(sa0,ctf_zero,p) - h_ih);

        ctf_plus1 = gsw_ct_freezing(sa+1.0,p,saturation_fraction);
        func_plus1 = sa*(gsw_enthalpy_ct_exact(sa+1.0,ctf_plus1,p) - h)
                        - (h - h_ih);

        saf = -(sa+1.0)*func_zero/(func_plus1 - func_zero);   /*initial guess*/
        ctf = gsw_ct_freezing(saf,p,saturation_fraction);
        gsw_enthalpy_first_derivatives_ct_exact(saf,ctf,p,&h_hat_sa,&h_hat_ct);
        gsw_ct_freezing_first_derivatives(saf,p,1.0,&ctf_sa,NULL);

        dfunc_dsaf = sa*(h_hat_sa + h_hat_ct*ctf_sa) - (h - h_ih);

        for (no_iter = 1; no_iter <= 2; no_iter++) {
            saf_old = saf;
            ctf_old = ctf;
            func = sa*(gsw_enthalpy_ct_exact(saf_old,ctf_old,p) - h)
                   - (saf_old - sa)*(h - h_ih);
            saf = saf_old - func/dfunc_dsaf;
            saf_mean = 0.5*(saf + saf_old);
            ctf_mean = gsw_ct_freezing(saf_mean,p,saturation_fraction);
            gsw_enthalpy_first_derivatives_ct_exact(saf_mean,ctf_mean,p,
                        &h_hat_sa, &h_hat_ct);
            gsw_ct_freezing_first_derivatives(saf_mean,p,saturation_fraction,
                        &ctf_sa, NULL);
            dfunc_dsaf = sa*(h_hat_sa + h_hat_ct*ctf_sa) - (h - h_ih);
            saf = saf_old - func/dfunc_dsaf;
            ctf = gsw_ct_freezing(saf,p,saturation_fraction);
        }
        /*
        ! After these 2 iterations of this modified Newton-Raphson method, the
        ! error in SA_freeze is less than 1.3d0x10^-13 g/kg, in CT_freeze is
        ! less than ! 4x10^-13 deg C and in w_Ih is less than 3.8d0x10^-15
        ! which represent machine precision for these calculations.
        */

        *sa_freeze = saf;
        *ct_freeze = ctf;
        *w_ih = (h - gsw_enthalpy_ct_exact(*sa_freeze,*ct_freeze,p))/(h - h_ih);
}
/*
!==========================================================================
function gsw_internal_energy(sa,ct,p)
!==========================================================================

!  Calculates internal energy of seawater.
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature (ITS-90)               [deg C]
! p      : sea pressure                                    [dbar]
!
! internal_energy  :  internal_energy of seawater          [J/kg]
*/
double
gsw_internal_energy(double sa, double ct, double p)
{
        GSW_TEOS10_CONSTANTS;

        return (gsw_enthalpy(sa,ct,p) - (gsw_p0 + db2pa*p)
                *gsw_specvol(sa,ct,p));
}
/*
!==========================================================================
elemental function gsw_internal_energy_ice (t, p)
!==========================================================================
!
!  Calculates the specific internal energy of ice.
!
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  internal_energy_ice  =  specific internal energy (u)              [J/kg]
!--------------------------------------------------------------------------
*/
double
gsw_internal_energy_ice(double t, double p)
{
        GSW_TEOS10_CONSTANTS;

        return (gsw_gibbs_ice(0,0,t,p)
                          - (gsw_t0 + t)*gsw_gibbs_ice(1,0,t,p)
                          - (db2pa*p + gsw_p0)*gsw_gibbs_ice(0,1,t,p));
}
/*
!==========================================================================
subroutine gsw_ipv_vs_fnsquared_ratio(sa,ct,p,pref,nz,ipv_vs_fnsquared_ratio,p_mid)
!==========================================================================

!  Calculates the ratio of the vertical gradient of potential density to
!  the vertical gradient of locally-referenced potential density.  This
!  ratio is also the ratio of the planetary Isopycnal Potential Vorticity
!  (IPV) to f times N^2, hence the name for this variable,
!  IPV_vs_fNsquared_ratio (see Eqn. (3.20.5) of IOC et al. (2010)).
!  The reference sea pressure, p_ref, of the potential density surface must
!  have a constant value.
!
!  IPV_vs_fNsquared_ratio is evaluated at the mid pressure between the
!  individual data points in the vertical.

! sa      : Absolute Salinity         (a profile (length nz))     [g/kg]
! ct      : Conservative Temperature  (a profile (length nz))     [deg C]
! p       : sea pressure              (a profile (length nz))     [dbar]
! p_ref   : reference sea pressure of the potential density surface
!        ( i.e. absolute reference pressure - 10.1325 dbar )      [dbar]
! nz      : number of bottles
! IPV_vs_fNsquared_ratio
!         : The ratio of the vertical gradient of potential density
!           referenced to p_ref, to the vertical gradient of locally-
!           referenced potential density.  It is output on the same
!           vertical (M-1)xN grid as p_mid.
!           IPV_vs_fNsquared_ratio is dimensionless.          [ unitless ]
! p_mid   : Mid pressure between p grid  (length nz-1)           [dbar]
*/
void
gsw_ipv_vs_fnsquared_ratio(double *sa, double *ct, double *p, double p_ref,
        int nz, double *ipv_vs_fnsquared_ratio, double *p_mid)
{
        int     k;
        double  dsa, sa_mid, dct, ct_mid;
        double  alpha_mid, beta_mid;
        double  alpha_pref, beta_pref, numerator, denominator;

        if (nz < 2) {
            *p_mid = *ipv_vs_fnsquared_ratio = GSW_INVALID_VALUE;
            return;
        }
        for (k = 0; k < nz-1; k++) {
            dsa         = (sa[k] - sa[k+1]);
            dct         = (ct[k] - ct[k+1]);
            sa_mid      = 0.5*(sa[k] + sa[k+1]);
            ct_mid      = 0.5*(ct[k] + ct[k+1]);
            p_mid[k]    = 0.5*(p[k] + p[k+1]);

            alpha_mid = gsw_alpha(sa_mid,ct_mid,p_mid[k]);
            beta_mid = gsw_beta(sa_mid,ct_mid,p_mid[k]);
            alpha_pref = gsw_alpha(sa_mid,ct_mid,p_ref);
            beta_pref = gsw_beta(sa_mid,ct_mid,p_ref);

            numerator = dct*alpha_pref - dsa*beta_pref;
            denominator = dct*alpha_mid - dsa*beta_mid;

            if (denominator == 0.0)
                ipv_vs_fnsquared_ratio[k] = GSW_INVALID_VALUE;
            else
                ipv_vs_fnsquared_ratio[k] = numerator/denominator;
        }
}
/*
!==========================================================================
function gsw_kappa(sa,ct,p)
!==========================================================================

!  Calculates isentropic compressibility of seawater.  This function
!  has inputs of Absolute Salinity and Conservative Temperature.  This
!  function uses the computationally-efficient expression for
!  specific volume in terms of SA, CT and p (Roquet et al., 2014).
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature (ITS-90)               [deg C]
! p      : sea pressure                                    [dbar]
!
! kappa  :  isentropic compressibility                     [1.0/Pa]
*/
double
gsw_kappa(double sa, double ct, double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  v, v_p, xs, ys, z;

        xs      = sqrt(gsw_sfac*sa + offset);
        ys      = ct*0.025;
        z       = p*1e-4;

        v       = v000
    + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050
    + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 + xs*(v140
    + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220 + xs*(v230 + v240*xs)))
    + ys*(v300 + xs*(v310 + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410
    + v420*xs) + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011
    + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111
    + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211 + xs*(v221
    + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs) + ys*(v401 + v411*xs
    + v501*ys)))) + z*(v002 + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs)))
    + ys*(v102 + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212
    + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013
    + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004 + v014*xs + v104*ys
    + z*(v005 + v006*z)))));

        v_p     = c000
    + xs*(c100 + xs*(c200 + xs*(c300 + xs*(c400 + c500*xs))))
    + ys*(c010 + xs*(c110 + xs*(c210 + xs*(c310 + c410*xs))) + ys*(c020
    + xs*(c120 + xs*(c220 + c320*xs)) + ys*(c030 + xs*(c130 + c230*xs)
    + ys*(c040 + c140*xs + c050*ys)))) + z*(c001 + xs*(c101 + xs*(c201
    + xs*(c301 + c401*xs))) + ys*(c011 + xs*(c111 + xs*(c211 + c311*xs))
    + ys*(c021 + xs*(c121 + c221*xs) + ys*(c031 + c131*xs + c041*ys)))
    + z*( c002 + xs*(c102 + c202*xs) + ys*(c012 + c112*xs + c022*ys)
    + z*(c003 + c103*xs + c013*ys + z*(c004 + c005*z))));

        return (-1e-8*v_p/v);
}
/*
!==========================================================================
elemental function gsw_kappa_const_t_ice (t, p)
!==========================================================================
!
!  Calculates isothermal compressibility of ice.
!  Note. This is the compressibility of ice AT CONSTANT IN-SITU
!    TEMPERATURE
!
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  kappa_const_t_ice  =  isothermal compressibility                [ 1/Pa ]
!   Note. The output units are 1/Pa not 1/dbar.
!--------------------------------------------------------------------------
*/
double
gsw_kappa_const_t_ice(double t, double p)
{
        return (-gsw_gibbs_ice(0,2,t,p)/gsw_gibbs_ice(0,1,t,p));
}
/*
!==========================================================================
elemental function gsw_kappa_ice (t, p)
!==========================================================================
!
!  Calculates the isentropic compressibility of ice.
!
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  kappa_ice  =  isentropic compressibility                        [ 1/Pa ]
!   Note. The output units are 1/Pa not 1/dbar.
!--------------------------------------------------------------------------
*/
double
gsw_kappa_ice(double t, double p)
{
        double  gi_tp, gi_tt;

        gi_tt = gsw_gibbs_ice(2,0,t,p);
        gi_tp = gsw_gibbs_ice(1,1,t,p);

        return ((gi_tp*gi_tp - gi_tt*gsw_gibbs_ice(0,2,t,p))/
                  (gsw_gibbs_ice(0,1,t,p)*gi_tt));
}
/*
!==========================================================================
function gsw_kappa_t_exact(sa,t,p)
!==========================================================================

! isentropic compressibility of seawater
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
!
! gsw_kappa_t_exact : isentropic compressibility           [1/Pa]
*/
double
gsw_kappa_t_exact(double sa, double t, double p)
{
        int     n0=0, n1=1, n2=2;
        double  g_tt, g_tp;

        g_tt    = gsw_gibbs(n0,n2,n0,sa,t,p);
        g_tp    = gsw_gibbs(n0,n1,n1,sa,t,p);

        return ((g_tp*g_tp - g_tt*gsw_gibbs(n0,n0,n2,sa,t,p)) /
                (gsw_gibbs(n0,n0,n1,sa,t,p)*g_tt));
}
/*
!==========================================================================
function gsw_latentheat_evap_ct(sa,ct)
!==========================================================================

! Calculates latent heat, or enthalpy, of evaporation.
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature                        [deg C]
!
! latentheat_evaporation : latent heat of evaporation      [J/kg]
*/
double
gsw_latentheat_evap_ct(double sa, double ct)
{
        GSW_TEOS10_CONSTANTS;
        double  c0  =  2.499065844825125e6, c1  = -1.544590633515099e-1,
                c2  = -9.096800915831875e4, c3  =  1.665513670736000e2,
                c4  =  4.589984751248335e1, c5  =  1.894281502222415e1,
                c6  =  1.192559661490269e3, c7  = -6.631757848479068e3,
                c8  = -1.104989199195898e2, c9  = -1.207006482532330e3,
                c10 = -3.148710097513822e3, c11 =  7.437431482069087e2,
                c12 =  2.519335841663499e3, c13 =  1.186568375570869e1,
                c14 =  5.731307337366114e2, c15 =  1.213387273240204e3,
                c16 =  1.062383995581363e3, c17 = -6.399956483223386e2,
                c18 = -1.541083032068263e3, c19 =  8.460780175632090e1,
                c20 = -3.233571307223379e2, c21 = -2.031538422351553e2,
                c22 =  4.351585544019463e1, c23 = -8.062279018001309e2,
                c24 =  7.510134932437941e2, c25 =  1.797443329095446e2,
                c26 = -2.389853928747630e1, c27 =  1.021046205356775e2;

        double  x, y;

        x       = sqrt(gsw_sfac*sa);
        y       = ct/40.0;

        return (c0 + x*(c1 + c4*y + x*(c3
                   + y*(c7 + c12*y) + x*(c6 + y*(c11 + y*(c17 + c24*y))
                   + x*(c10 + y*(c16 + c23*y) + x*(c15 + c22*y + c21*x)))))
                   + y*(c2 + y*(c5 + c8*x + y*(c9 + x*(c13 + c18*x)
                   + y*(c14 + x*(c19 + c25*x) + y*(c20 + c26*x + c27*y))))));
}
/*
!==========================================================================
function gsw_latentheat_evap_t(sa,t)
!==========================================================================
!
! Calculates latent heat, or enthalpy, of evaporation.
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
!
! gsw_latentheat_evap_t : latent heat of evaporation       [J/kg]
*/
double
gsw_latentheat_evap_t(double sa, double t)
{

        double  ct = gsw_ct_from_pt(sa,t);

        return (gsw_latentheat_evap_ct(sa,ct));
}
/*
!--------------------------------------------------------------------------
! isobaric melting enthalpy and isobaric evaporation enthalpy
!--------------------------------------------------------------------------

!==========================================================================
function gsw_latentheat_melting(sa,p)
!==========================================================================

! Calculates latent heat, or enthalpy, of melting.
!
! sa     : Absolute Salinity                               [g/kg]
! p      : sea pressure                                    [dbar]
!
! latentheat_melting : latent heat of melting              [kg/m^3]
*/
double
gsw_latentheat_melting(double sa, double p)
{
        GSW_TEOS10_CONSTANTS;
        double  tf = gsw_t_freezing(sa,p,0.0);

        return (1000.0*(gsw_chem_potential_water_t_exact(sa,tf,p)
           - (gsw_t0 + tf)*gsw_t_deriv_chem_potential_water_t_exact(sa,tf,p))
           - gsw_enthalpy_ice(tf,p));
}
/*
!==========================================================================
pure subroutine gsw_linear_interp_sa_ct (sa, ct, p, p_i, sa_i, ct_i)
!==========================================================================
! This function interpolates the cast with respect to the interpolating
! variable p. This function finds the values of SA, CT at p_i on this cast.
!
! VERSION NUMBER: 3.05 (27th January 2015)
!
! This function was adapted from Matlab's interp1q.
!==========================================================================
*/
void
gsw_linear_interp_sa_ct(double *sa, double *ct, double *p, int np,
        double *p_i, int npi, double *sa_i, double *ct_i)
{
        char    *in_rng;
        int     *j, *k, *r, *jrev, *ki, imax_p, imin_p, i, n, m, ii;
        double  *xi, *xxi, u, max_p, min_p;

        min_p = max_p = p[0];
        imin_p = imax_p = 0;
        for (i=1; i<np; i++) {
            if (p[i] < min_p) {
                min_p = p[i];
                imin_p = i;
            } else if (p[i] > max_p) {
                max_p = p[i];
                imax_p = i;
            }
        }
        in_rng = (char *) malloc(npi*sizeof (char));
        memset(in_rng, 0, npi*sizeof (char));
        for (i=n=0; i<npi; i++) {
            if (p_i[i] <= min_p) {
                sa_i[i] = sa[imin_p];
                ct_i[i] = ct[imin_p];
            } else if (p_i[i] >= max_p) {
                sa_i[i] = sa[imax_p];
                ct_i[i] = ct[imax_p];
            } else {
                in_rng[i] = 1;
                n++;
            }
        }
        if (n==0)
            return;

        xi =(double *) malloc(n*sizeof (double));
        k  = (int *) malloc(3*n*sizeof (int)); ki = k+n; r = ki+n;
        m  = np + n;
        xxi = (double *) malloc(m*sizeof (double));
        j = (int *) malloc(2*m*sizeof (int)); jrev = j+m;

        ii = 0;
        for (i = 0; i<npi; i++) {
            if (in_rng[i]) {
                xi[ii] = p_i[i];
                ki[ii] = i;
                ii++;
            }
        }
        free(in_rng);
    /*
    **  Note that the following operations on the index
    **  vectors jrev and r depend on the sort utility
    **  gsw_util_sort_real() consistently ordering the
    **  sorting indexes either in ascending or descending
    **  sequence for replicate values in the real vector.
    */
        gsw_util_sort_real(xi, n, k);
        for (i = 0; i<np; i++)
            xxi[i] = p[i];
        for (i = 0; i<n; i++)
            xxi[np+i] = xi[k[i]];
        gsw_util_sort_real(xxi, m, j);

        for (i = 0; i<m; i++)
            jrev[j[i]] = i;
        for (i = 0; i<n; i++)
            r[k[i]] = jrev[np+i]-i-1;

        for (i = 0; i<n; i++) {
            u = (xi[i]-p[r[i]])/(p[r[i]+1]-p[r[i]]);
            sa_i[ki[i]] = sa[r[i]] + (sa[r[i]+1]-sa[r[i]])*u;
            ct_i[ki[i]] = ct[r[i]] + (ct[r[i]+1]-ct[r[i]])*u;
        }
        free(j); free(xxi); free(k); free(xi);
}
/*
!==========================================================================
elemental function gsw_melting_ice_equilibrium_sa_ct_ratio (sa, p)
!==========================================================================
!
!  Calculates the ratio of SA to CT changes when ice melts into seawater
!  with both the seawater and the seaice temperatures being almost equal to
!  the equilibrium freezing temperature.  It is assumed that a small mass
!  of ice melts into an infinite mass of seawater.  If indeed the
!  temperature of the seawater and the ice were both equal to the freezing
!  temperature, then no melting or freezing would occur an imbalance
!  between these three temperatures is needed for freezing or melting to
!  occur (the three temperatures being (1) the seawater temperature,
!  (2) the ice temperature, and (3) the freezing temperature.
!
!  The output, melting_ice_equilibrium_SA_CT_ratio, is dSA/dCT rather than
!  dCT/dSA.  This is done so that when SA = 0, the output, dSA/dCT is zero
!  whereas dCT/dSA would be infinite.
!
!  SA  =  Absolute Salinity of seawater                            [ g/kg ]
!  p   =  sea pressure at which the melting occurs                 [ dbar ]
!         ( i.e. absolute pressure - 10.1325d0 dbar )
!
!  melting_ice_equilibrium_SA_CT_ratio = the ratio dSA/dCT of SA to CT
!                                changes when ice melts into seawater, with
!                                the seawater and seaice being close to the
!                                freezing temperature.         [ g/(kg K) ]
!--------------------------------------------------------------------------
*/
double
gsw_melting_ice_equilibrium_sa_ct_ratio(double sa, double p)
{
        double  ctf, h, h_ih, t_seaice, h_hat_sa, h_hat_ct;
        double  saturation_fraction = 0.0;

        ctf = gsw_ct_freezing(sa,p,saturation_fraction);
        t_seaice = gsw_t_freezing(sa,p,saturation_fraction);

        h = gsw_enthalpy_ct_exact(sa,ctf,p);
        h_ih = gsw_enthalpy_ice(t_seaice,p);
        gsw_enthalpy_first_derivatives_ct_exact(sa,ctf,p,&h_hat_sa,&h_hat_ct);
              /*note that h_hat_ct is equal to cp0*(273.15 + t)/(273.15 + pt0)*/

        return (sa*h_hat_ct/(h - h_ih - sa*h_hat_sa));
}
/*
!==========================================================================
elemental function gsw_melting_ice_equilibrium_sa_ct_ratio_poly (sa, p)
!==========================================================================
!
!  Calculates the ratio of SA to CT changes when ice melts into seawater
!  with both the seawater and the seaice temperatures being almost equal to
!  the equilibrium freezing temperature.  It is assumed that a small mass
!  of ice melts into an infinite mass of seawater.  If indeed the
!  temperature of the seawater and the ice were both equal to the freezing
!  temperature, then no melting or freezing would occur an imbalance
!  between these three temperatures is needed for freezing or melting to
!  occur (the three temperatures being (1) the seawater temperature,
!  (2) the ice temperature, and (3) the freezing temperature.
!
!  The output, melting_ice_equilibrium_SA_CT_ratio, is dSA/dCT rather than
!  dCT/dSA.  This is done so that when SA = 0, the output, dSA/dCT is zero
!  whereas dCT/dSA would be infinite.
!
!  SA  =  Absolute Salinity of seawater                            [ g/kg ]
!  p   =  sea pressure at which the melting occurs                 [ dbar ]
!         ( i.e. absolute pressure - 10.1325d0 dbar )
!
!  melting_ice_equilibrium_SA_CT_ratio = the ratio dSA/dCT of SA to CT
!                                changes when ice melts into seawater, with
!                                the seawater and seaice being close to the
!                                freezing temperature.         [ g/(kg K) ]
!--------------------------------------------------------------------------
*/
double
gsw_melting_ice_equilibrium_sa_ct_ratio_poly(double sa, double p)
{
        double  ctf, h, h_ih, t_seaice, h_hat_sa, h_hat_ct;
        double  saturation_fraction = 0.0;

        ctf = gsw_ct_freezing_poly(sa,p,saturation_fraction);
        t_seaice = gsw_t_freezing_poly(sa,p,saturation_fraction);

        h = gsw_enthalpy(sa,ctf,p);
        h_ih = gsw_enthalpy_ice(t_seaice,p);
        gsw_enthalpy_first_derivatives(sa,ctf,p,&h_hat_sa,&h_hat_ct);
              /*note that h_hat_ct is equal to cp0*(273.15 + t)/(273.15 + pt0)*/

        return (sa*h_hat_ct / (h - h_ih - sa*h_hat_sa));
}
/*
!==========================================================================
elemental subroutine gsw_melting_ice_into_seawater (sa, ct, p, w_ih, t_ih,&
                                            sa_final, ct_final, w_ih_final)
!==========================================================================
!
!  Calculates the final Absolute Salinity, final Conservative Temperature
!  and final ice mass fraction that results when a given mass fraction of
!  ice melts and is mixed into seawater whose properties are (SA,CT,p).
!  This code takes the seawater to contain no dissolved air.
!
!  When the mass fraction w_Ih_final is calculated as being a positive
!  value, the seawater-ice mixture is at thermodynamic equlibrium.
!
!  This code returns w_Ih_final = 0 when the input bulk enthalpy, h_bulk,
!  is sufficiently large (i.e. sufficiently "warm") so that there is no ice
!  present in the final state.  In this case the final state consists of
!  only seawater rather than being an equlibrium mixture of seawater and
!  ice which occurs when w_Ih_final is positive.  Note that when
!  w_Ih_final = 0, the final seawater is not at the freezing temperature.
!
!  SA   =  Absolute Salinity of seawater                           [ g/kg ]
!  CT   =  Conservative Temperature of seawater (ITS-90)          [ deg C ]
!  p    =  sea pressure at which the melting occurs                [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  w_Ih =  mass fraction of ice, that is the mass of ice divided by the
!          sum of the masses of ice and seawater.  That is, the mass of
!          ice divided by the mass of the final mixed fluid.
!          w_Ih must be between 0 and 1.                       [ unitless ]
!  t_Ih =  the in-situ temperature of the ice (ITS-90)            [ deg C ]
!
!  SA_final    =  Absolute Salinity of the seawater in the final state,
!                 whether or not any ice is present.               [ g/kg ]
!  CT_final    =  Conservative Temperature of the seawater in the the final
!                 state, whether or not any ice is present.       [ deg C ]
!  w_Ih_final  =  mass fraction of ice in the final seawater-ice mixture.
!                 If this ice mass fraction is positive, the system is at
!                 thermodynamic equilibrium.  If this ice mass fraction is
!                 zero there is no ice in the final state which consists
!                 only of seawater which is warmer than the freezing
!                 temperature.                                   [unitless]
!--------------------------------------------------------------------------
*/
void
gsw_melting_ice_into_seawater(double sa, double ct, double p, double w_ih,
        double t_ih, double *sa_final, double *ct_final, double *w_ih_final)
{
        double  ctf, h_bulk, sa_bulk, tf_ih;
        double  saturation_fraction = 0.0;

        ctf = gsw_ct_freezing(sa,p,saturation_fraction);
        if (ct < ctf) {
            /*The seawater ct input is below the freezing temp*/
            *sa_final = GSW_INVALID_VALUE;
            *ct_final = *sa_final;
            *w_ih_final = *sa_final;
            return;
        }

        tf_ih = gsw_t_freezing(0.0,p,saturation_fraction) - 1e-6;
        if (t_ih > tf_ih) {
            /*
            ! t_ih input exceeds the freezing temp.
            ! The 1e-6 C buffer in the allowable
            ! t_Ih is to ensure that there is some ice Ih in the sea ice.
            */
            *sa_final = GSW_INVALID_VALUE;
            *ct_final = *sa_final;
            *w_ih_final = *sa_final;
            return;
        }

        sa_bulk = (1.0 - w_ih)*sa;
        h_bulk = (1.0 - w_ih)*gsw_enthalpy_ct_exact(sa,ct,p)
                          + w_ih*gsw_enthalpy_ice(t_ih,p);
        gsw_frazil_properties(sa_bulk,h_bulk,p,sa_final,ct_final,w_ih_final);
        if (*sa_final > GSW_ERROR_LIMIT) {
            *sa_final = GSW_INVALID_VALUE;
            *ct_final = *sa_final;
            *w_ih_final = *sa_final;
            return;
        }
}
/*
!==========================================================================
elemental function gsw_melting_ice_sa_ct_ratio (sa, ct, p, t_ih)
!==========================================================================
!
!  Calculates the ratio of SA to CT changes when ice melts into seawater.
!  It is assumed that a small mass of ice melts into an infinite mass of
!  seawater.  Because of the infinite mass of seawater, the ice will always
!  melt.
!
!  The output, melting_seaice_SA_CT_ratio, is dSA/dCT rather than dCT/dSA.
!  This is done so that when SA = 0, the output, dSA/dCT is zero whereas
!  dCT/dSA would be infinite.
!
!  SA   =  Absolute Salinity of seawater                           [ g/kg ]
!  CT   =  Conservative Temperature of seawater (ITS-90)          [ deg C ]
!  p    =  sea pressure at which the melting occurs                [ dbar ]
!         ( i.e. absolute pressure - 10.1325d0 dbar )
!  t_Ih =  the in-situ temperature of the ice (ITS-90)            [ deg C ]
!
!  melting_ice_SA_CT_ratio = the ratio of SA to CT changes when ice melts
!                            into a large mass of seawater
!                                                          [ g kg^-1 K^-1 ]
!--------------------------------------------------------------------------
*/
double
gsw_melting_ice_sa_ct_ratio(double sa, double ct, double p, double t_ih)
{
        double  ctf, h, h_ih, tf, h_hat_sa, h_hat_ct;
        double  saturation_fraction = 0.0;

        ctf = gsw_ct_freezing(sa,p,saturation_fraction);
        if (ct < ctf) {
            /*the seawater ct input is below the freezing temperature*/
            return (GSW_INVALID_VALUE);
        }

        tf = gsw_t_freezing(0.0,p,saturation_fraction);
        if (t_ih > tf) {
            /*t_ih exceeds the freezing temperature at sa = 0*/
            return (GSW_INVALID_VALUE);
        }

        h = gsw_enthalpy_ct_exact(sa,ct,p);
        h_ih = gsw_enthalpy_ice(t_ih,p);
        gsw_enthalpy_first_derivatives_ct_exact(sa,ct,p,&h_hat_sa,&h_hat_ct);
            /*Note that h_hat_CT is equal to cp0*(273.15 + t)/(273.15 + pt0)*/

        return (sa*h_hat_ct/(h - h_ih - sa*h_hat_sa));
}
/*
!==========================================================================
elemental function gsw_melting_ice_sa_ct_ratio_poly (sa, ct, p, t_ih)
!==========================================================================
!
!  Calculates the ratio of SA to CT changes when ice melts into seawater.
!  It is assumed that a small mass of ice melts into an infinite mass of
!  seawater.  Because of the infinite mass of seawater, the ice will always
!  melt.
!
!  The output, melting_seaice_SA_CT_ratio, is dSA/dCT rather than dCT/dSA.
!  This is done so that when SA = 0, the output, dSA/dCT is zero whereas
!  dCT/dSA would be infinite.
!
!  SA   =  Absolute Salinity of seawater                           [ g/kg ]
!  CT   =  Conservative Temperature of seawater (ITS-90)          [ deg C ]
!  p    =  sea pressure at which the melting occurs                [ dbar ]
!         ( i.e. absolute pressure - 10.1325d0 dbar )
!  t_Ih =  the in-situ temperature of the ice (ITS-90)            [ deg C ]
!
!  melting_ice_SA_CT_ratio = the ratio of SA to CT changes when ice melts
!                            into a large mass of seawater
!                                                          [ g kg^-1 K^-1 ]
!--------------------------------------------------------------------------
*/
double
gsw_melting_ice_sa_ct_ratio_poly(double sa, double ct, double p, double t_ih)
{
        double  ctf, h, h_ih, tf, h_hat_sa, h_hat_ct;
        double  saturation_fraction = 0.0;

        ctf = gsw_ct_freezing_poly(sa,p,saturation_fraction);
        if (ct < ctf) {
            /*the seawater ct input is below the freezing temperature*/
            return (GSW_INVALID_VALUE);
        }

        tf = gsw_t_freezing_poly(0.0,p,saturation_fraction);
        if (t_ih > tf) {
            /*t_ih exceeds the freezing temperature at sa = 0*/
            return (GSW_INVALID_VALUE);
        }

        h = gsw_enthalpy(sa,ct,p);
        h_ih = gsw_enthalpy_ice(t_ih,p);
        gsw_enthalpy_first_derivatives(sa,ct,p,&h_hat_sa,&h_hat_ct);
            /*Note that h_hat_CT is equal to cp0*(273.15 + t)/(273.15 + pt0)*/

        return (sa*h_hat_ct/(h - h_ih - sa*h_hat_sa));
}
/*
!==========================================================================
elemental function gsw_melting_seaice_equilibrium_sa_ct_ratio (sa, p)
!==========================================================================
!
!  Calculates the ratio of SA to CT changes when sea ice melts into
!  seawater with both the seawater and the sea ice temperatures being
!  almost equal to the equilibrium freezing temperature.  It is assumed
!  that a small mass of seaice melts into an infinite mass of seawater.  If
!  indeed the temperature of the seawater and the sea ice were both equal
!  to the freezing temperature, then no melting or freezing would occur; an
!  imbalance between these three temperatures is needed for freezing or
!  melting to occur (the three temperatures being (1) the seawater
!  temperature, (2) the sea ice temperature, and (3) the freezing
!  temperature.
!
!  Note that the output of this function, dSA/dCT is independent of the
!  sea ice salinity, SA_seaice.  That is, the output applies equally to
!  pure ice Ih and to sea ice with seaice salinity, SA_seaice.  This result
!  is proven in the manuscript, McDougall et al. (2013).
!
!  The output, melting_seaice_equilibrium_SA_CT_ratio, is dSA/dCT rather
!  than dCT/dSA.  This is done so that when SA = 0, the output, dSA/dCT is
!  zero whereas dCT/dSA would be infinite.
!
!  SA  =  Absolute Salinity of seawater                            [ g/kg ]
!  p   =  sea pressure at which the melting occurs                 [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  melting_seaice_equilibrium_SA_CT_ratio = the ratio dSA/dCT of SA to CT
!                            changes when sea ice melts into seawater, with
!                            the seawater and sea ice being close to the
!                            freezing temperature.             [ g/(kg K) ]
!--------------------------------------------------------------------------
*/
double
gsw_melting_seaice_equilibrium_sa_ct_ratio(double sa, double p)
{
        double  ctf, h, h_ih, t_seaice, h_hat_sa, h_hat_ct;
        double  saturation_fraction = 0.0;

        ctf = gsw_ct_freezing(sa,p,saturation_fraction);
        t_seaice = gsw_t_freezing(sa,p,saturation_fraction);

        h = gsw_enthalpy_ct_exact(sa,ctf,p);
        h_ih = gsw_enthalpy_ice(t_seaice,p);
        gsw_enthalpy_first_derivatives_ct_exact(sa,ctf,p,&h_hat_sa,&h_hat_ct);

        return (sa*h_hat_ct / (h - h_ih - sa*h_hat_sa));
}
/*
!==========================================================================
elemental function gsw_melting_seaice_equilibrium_sa_ct_ratio_poly (sa, p)
!==========================================================================
!
!  Calculates the ratio of SA to CT changes when sea ice melts into
!  seawater with both the seawater and the sea ice temperatures being
!  almost equal to the equilibrium freezing temperature.  It is assumed
!  that a small mass of seaice melts into an infinite mass of seawater.  If
!  indeed the temperature of the seawater and the sea ice were both equal
!  to the freezing temperature, then no melting or freezing would occur; an
!  imbalance between these three temperatures is needed for freezing or
!  melting to occur (the three temperatures being (1) the seawater
!  temperature, (2) the sea ice temperature, and (3) the freezing
!  temperature.
!
!  Note that the output of this function, dSA/dCT is independent of the
!  sea ice salinity, SA_seaice.  That is, the output applies equally to
!  pure ice Ih and to sea ice with seaice salinity, SA_seaice.  This result
!  is proven in the manuscript, McDougall et al. (2013).
!
!  The output, melting_seaice_equilibrium_SA_CT_ratio, is dSA/dCT rather
!  than dCT/dSA.  This is done so that when SA = 0, the output, dSA/dCT is
!  zero whereas dCT/dSA would be infinite.
!
!  SA  =  Absolute Salinity of seawater                            [ g/kg ]
!  p   =  sea pressure at which the melting occurs                 [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  melting_seaice_equilibrium_SA_CT_ratio = the ratio dSA/dCT of SA to CT
!                            changes when sea ice melts into seawater, with
!                            the seawater and sea ice being close to the
!                            freezing temperature.             [ g/(kg K) ]
!--------------------------------------------------------------------------
*/
double
gsw_melting_seaice_equilibrium_sa_ct_ratio_poly(double sa, double p)
{
        double  ctf, h, h_ih, t_seaice, h_hat_sa, h_hat_ct;
        double  saturation_fraction = 0.0;

        ctf = gsw_ct_freezing_poly(sa,p,saturation_fraction);
        t_seaice = gsw_t_freezing_poly(sa,p,saturation_fraction);

        h = gsw_enthalpy(sa,ctf,p);
        h_ih = gsw_enthalpy_ice(t_seaice,p);
        gsw_enthalpy_first_derivatives(sa,ctf,p,&h_hat_sa,&h_hat_ct);

        return (sa*h_hat_ct / (h - h_ih - sa*h_hat_sa));
}
/*
!==========================================================================
elemental subroutine gsw_melting_seaice_into_seawater (sa, ct, p, &
                         w_seaice, sa_seaice, t_seaice, sa_final, ct_final)
!==========================================================================
!
!  Calculates the Absolute Salinity and Conservative Temperature that
!  results when a given mass of sea ice (or ice) melts and is mixed into a
!  known mass of seawater (whose properties are (SA,CT,p)).
!
!  If the ice contains no salt (e.g. if it is of glacial origin), then the
!  input 'SA_seaice' should be set to zero.
!
!  Ice formed at the sea surface (sea ice) typically contains between 2 g/kg
!  and 12 g/kg of salt (defined as the mass of salt divided by the mass of
!  ice Ih plus brine) and this programme returns NaN's if the input
!  SA_seaice is greater than 15 g/kg.  If the SA_seaice input is not zero,
!  usually this would imply that the pressure p should be zero, as sea ice
!  only occurs near the sea surface.  The code does not impose that p = 0
!  if SA_seaice is non-zero.  Rather, this is left to the user.
!
!  The Absolute Salinity, SA_brine, of the brine trapped in little pockets
!  in the sea ice, is in thermodynamic equilibrium with the ice Ih that
!  surrounds these pockets.  As the sea ice temperature, t_seaice, may be
!  less than the freezing temperature, SA_brine is usually greater than the
!  Absolute Salinity of the seawater at the time and place when and where
!  the sea ice was formed.  So usually SA_brine will be larger than SA.
!
!  SA  =  Absolute Salinity of seawater                            [ g/kg ]
!  CT  =  Conservative Temperature of seawater (ITS-90)           [ deg C ]
!  p   =  sea pressure at which the melting occurs                 [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  w_seaice  =  mass fraction of sea ice, that is the mass of sea ice
!               divided by the sum of the masses of sea ice and seawater.
!               That is, the mass of sea ice divided by the mass of the
!               final mixed fluid.  w_seaice must be between 0 and 1.
!                                                              [ unitless ]
!  SA_seaice =  Absolute Salinity of sea ice, that is, the mass fraction of
!               salt in sea ice, expressed in g of salt per kg of sea ice.
!                                                                  [ g/kg ]
!  t_seaice  =  the in-situ temperature of the sea ice (or ice) (ITS-90)
!                                                                 [ deg C ]
!
!  SA_final  =  Absolute Salinity of the mixture of the melted sea ice
!               (or ice) and the original seawater                  [ g/kg ]
!  CT_final  =  Conservative Temperature of the mixture of the melted
!               sea ice (or ice) and the original seawater         [ deg C ]
!--------------------------------------------------------------------------
*/
void
gsw_melting_seaice_into_seawater(double sa, double ct, double p,
        double w_seaice, double sa_seaice, double t_seaice,
        double *sa_final, double *ct_final)
{
        double  ctf, h, h_brine, h_final, h_ih, sa_brine, tf_sa_seaice;
        double  saturation_fraction = 0.0;

        ctf = gsw_ct_freezing(sa,p,saturation_fraction);
        if (ct < ctf) {
            /*The seawater ct input is below the freezing temp*/
            *sa_final = GSW_INVALID_VALUE;
            *ct_final = *sa_final;
            return;
        }

        tf_sa_seaice = gsw_t_freezing(sa_seaice,p,saturation_fraction)
                        - 1e-6;
        if (t_seaice > tf_sa_seaice) {
        /*
        ! The 1e-6 C buffer in the allowable t_seaice is to ensure that there is
        ! some ice Ih in the sea ice. Without this buffer, that is if t_seaice
        ! is allowed to be exactly equal to tf_sa_seaice, the seaice is
        ! actually 100% brine at Absolute Salinity of SA_seaice.
        */
            *sa_final = GSW_INVALID_VALUE;
            *ct_final = *sa_final;
            return;
        }

        sa_brine = gsw_sa_freezing_from_t(t_seaice,p,saturation_fraction);
        if (sa_brine >= GSW_ERROR_LIMIT) {
            *sa_final = GSW_INVALID_VALUE;
            *ct_final = *sa_final;
            return;
        }
        h_brine = gsw_enthalpy_t_exact(sa_brine,t_seaice,p);

        h = gsw_enthalpy_ct_exact(sa,ct,p);
        h_ih = gsw_enthalpy_ice(t_seaice,p);

        h_final = h - w_seaice*(h - h_ih - (h_brine - h_ih)*sa_seaice/sa_brine);

        *sa_final = sa - w_seaice*(sa - sa_seaice);
        /*
        !ctf = gsw_ct_freezing(sa_final,p,saturation_fraction)
        !
        !if (h_final .lt. gsw_enthalpy_ct_exact(sa_final,ctf,p)) then
        !    ! Melting this much seaice is not possible as it would result in
        !    ! frozen seawater
        !    sa_final = gsw_error_code(4,func_name)
        !    ct_final = sa_final
        !    return
        !end if
        */
        *ct_final = gsw_ct_from_enthalpy_exact(*sa_final,h_final,p);
        if (*ct_final > GSW_ERROR_LIMIT) {
            *sa_final = *ct_final;
            return;
        }
}
/*
!==========================================================================
elemental function gsw_melting_seaice_sa_ct_ratio (sa, ct, p, sa_seaice, &
                                                   t_seaice)
!==========================================================================
!
! Calculates the ratio of SA to CT changes when sea ice melts into seawater.
! It is assumed that a small mass of sea ice melts into an infinite mass of
! seawater.  Because of the infinite mass of seawater, the sea ice will
! always melt.
!
! Ice formed at the sea surface (sea ice) typically contains between 2 g/kg
! and 12 g/kg of salt (defined as the mass of salt divided by the mass of
! ice Ih plus brine) and this programme returns NaN's if the input
! SA_seaice is greater than 15 g/kg.  If the SA_seaice input is not zero,
! usually this would imply that the pressure p should be zero, as sea ice
! only occurs near the sea surface.  The code does not impose that p = 0 if
! SA_seaice is non-zero.  Rather, this is left to the user.
!
! The Absolute Salinity, SA_brine, of the brine trapped in little pockets
! in the sea ice, is in thermodynamic equilibrium with the ice Ih that
! surrounds these pockets.  As the seaice temperature, t_seaice, may be
! less than the freezing temperature, SA_brine is usually greater than the
! Absolute Salinity of the seawater at the time and place when and where
! the sea ice was formed.  So usually SA_brine will be larger than SA.
!
! The output, melting_seaice_SA_CT_ratio, is dSA/dCT rather than dCT/dSA.
! This is done so that when (SA - seaice_SA) = 0, the output, dSA/dCT is
! zero whereas dCT/dSA would be infinite.
!
!  SA  =  Absolute Salinity of seawater                            [ g/kg ]
!  CT  =  Conservative Temperature of seawater (ITS-90)           [ deg C ]
!  p   =  sea pressure at which the melting occurs                 [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  SA_seaice  =  Absolute Salinity of sea ice, that is, the mass fraction
!                of salt in sea ice expressed in g of salt per kg of
!                sea ice                                           [ g/kg ]
!  t_seaice = the in-situ temperature of the sea ice (ITS-90)     [ deg C ]
!
!  melting_seaice_SA_CT_ratio = the ratio dSA/dCT of SA to CT changes when
!                sea ice melts into a large mass of seawater   [ g/(kg K) ]
!--------------------------------------------------------------------------
*/
double
gsw_melting_seaice_sa_ct_ratio(double sa, double ct, double p,
        double sa_seaice, double t_seaice)
{
        double  ctf, delsa, h, h_brine, h_ih, sa_brine,
                tf_sa_seaice, h_hat_sa, h_hat_ct;
        double  saturation_fraction = 0.0;

        if (sa_seaice < 0.0 || sa_seaice > 15.0) {
            return (GSW_INVALID_VALUE);
        }

        ctf = gsw_ct_freezing(sa,p,saturation_fraction);
        if (ct < ctf) {    /*the seawater ct input is below the freezing temp*/
            return (GSW_INVALID_VALUE);
        }
        tf_sa_seaice = gsw_t_freezing(sa_seaice,p,saturation_fraction) - 1e-6;
        if (t_seaice > tf_sa_seaice) {   /*t_seaice exceeds the freezing sa*/
            return (GSW_INVALID_VALUE);
        }
        /*
        !-----------------------------------------------------------------------
        !The 1e-6 C buffer in the allowable t_seaice is to ensure that there is
        !some ice Ih in the sea ice.  Without this buffer, that is if t_seaice
        !is allowed to be exactly equal to tf_sa_seaice, the sea ice is actually
        !100% brine at Absolute Salinity of SA_seaice.
        !-----------------------------------------------------------------------
        */
        h = gsw_enthalpy_ct_exact(sa,ct,p);
        h_ih = gsw_enthalpy_ice(t_seaice,p);
        gsw_enthalpy_first_derivatives_ct_exact(sa,ct,p,&h_hat_sa,&h_hat_ct);

        sa_brine = gsw_sa_freezing_from_t(t_seaice,p,saturation_fraction);
        if (sa_brine > GSW_ERROR_LIMIT) {
            return (GSW_INVALID_VALUE);
        }
        h_brine = gsw_enthalpy_t_exact(sa_brine,t_seaice,p);
        delsa = sa - sa_seaice;

        return (h_hat_ct*delsa /
             (h - h_ih - delsa*h_hat_sa - sa_seaice*(h_brine - h_ih)/sa_brine));
}
/*
!==========================================================================
elemental function gsw_melting_seaice_sa_ct_ratio_poly (sa, ct, p, &
                                                       sa_seaice, t_seaice)
!==========================================================================
!
! Calculates the ratio of SA to CT changes when sea ice melts into seawater.
! It is assumed that a small mass of sea ice melts into an infinite mass of
! seawater.  Because of the infinite mass of seawater, the sea ice will
! always melt.
!
! Ice formed at the sea surface (sea ice) typically contains between 2 g/kg
! and 12 g/kg of salt (defined as the mass of salt divided by the mass of
! ice Ih plus brine) and this programme returns NaN's if the input
! SA_seaice is greater than 15 g/kg.  If the SA_seaice input is not zero,
! usually this would imply that the pressure p should be zero, as sea ice
! only occurs near the sea surface.  The code does not impose that p = 0 if
! SA_seaice is non-zero.  Rather, this is left to the user.
!
! The Absolute Salinity, SA_brine, of the brine trapped in little pockets
! in the sea ice, is in thermodynamic equilibrium with the ice Ih that
! surrounds these pockets.  As the seaice temperature, t_seaice, may be
! less than the freezing temperature, SA_brine is usually greater than the
! Absolute Salinity of the seawater at the time and place when and where
! the sea ice was formed.  So usually SA_brine will be larger than SA.
!
! The output, melting_seaice_SA_CT_ratio, is dSA/dCT rather than dCT/dSA.
! This is done so that when (SA - seaice_SA) = 0, the output, dSA/dCT is
! zero whereas dCT/dSA would be infinite.
!
!  SA  =  Absolute Salinity of seawater                            [ g/kg ]
!  CT  =  Conservative Temperature of seawater (ITS-90)           [ deg C ]
!  p   =  sea pressure at which the melting occurs                 [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  SA_seaice  =  Absolute Salinity of sea ice, that is, the mass fraction
!                of salt in sea ice expressed in g of salt per kg of
!                sea ice                                           [ g/kg ]
!  t_seaice = the in-situ temperature of the sea ice (ITS-90)     [ deg C ]
!
!  melting_seaice_SA_CT_ratio = the ratio dSA/dCT of SA to CT changes when
!                sea ice melts into a large mass of seawater   [ g/(kg K) ]
!--------------------------------------------------------------------------
*/
double
gsw_melting_seaice_sa_ct_ratio_poly(double sa, double ct, double p,
        double sa_seaice, double t_seaice)
{
        double  ctf, delsa, h, h_brine, h_ih, sa_brine, ct_brine,
                tf_sa_seaice, h_hat_sa, h_hat_ct;
        double  saturation_fraction = 0.0;

        if (sa_seaice < 0.0 || sa_seaice > 15.0) {
            return (GSW_INVALID_VALUE);
        }

        ctf = gsw_ct_freezing_poly(sa,p,saturation_fraction);
        if (ct < ctf) { /*the seawater ct input is below the freezing temp*/
            return (GSW_INVALID_VALUE);
        }

        tf_sa_seaice = gsw_t_freezing_poly(sa_seaice,p,saturation_fraction)
                        - 1e-6;
        if (t_seaice > tf_sa_seaice) {   /*t_seaice exceeds the freezing sa*/
            return (GSW_INVALID_VALUE);
        }
        /*
        !-----------------------------------------------------------------------
        !The 1e-6 C buffer in the allowable t_seaice is to ensure that there is
        !some ice Ih in the sea ice.  Without this buffer, that is if t_seaice
        !is allowed to be exactly equal to tf_sa_seaice, the sea ice is actually
        !100% brine at Absolute Salinity of SA_seaice.
        !-----------------------------------------------------------------------
        */
        h = gsw_enthalpy(sa,ct,p);
        h_ih = gsw_enthalpy_ice(t_seaice,p);
        gsw_enthalpy_first_derivatives(sa,ct,p,&h_hat_sa,&h_hat_ct);

        sa_brine = gsw_sa_freezing_from_t_poly(t_seaice,p,saturation_fraction);
        if (sa_brine > GSW_ERROR_LIMIT) {
            return (GSW_INVALID_VALUE);
        }
        ct_brine = gsw_ct_from_t(sa_brine,t_seaice,p);
        h_brine = gsw_enthalpy(sa_brine,ct_brine,p);
        delsa = sa - sa_seaice;

        return (h_hat_ct*delsa /
             (h - h_ih - delsa*h_hat_sa - sa_seaice*(h_brine - h_ih)/sa_brine));
}
/*
!--------------------------------------------------------------------------
! water column properties, based on the 48-term expression for density
!--------------------------------------------------------------------------

!==========================================================================
subroutine gsw_nsquared(sa,ct,p,lat,nz,n2,p_mid)
!==========================================================================

!  Calculates the buoyancy frequency squared (N^2)(i.e. the Brunt-Vaisala
!  frequency squared) at the mid pressure from the equation,
!
!
!           2      2             beta.d(SA) - alpha.d(CT)
!         N   =  g  .rho_local. -------------------------
!                                          dP
!
!  The pressure increment, dP, in the above formula is in Pa, so that it is
!  10^4 times the pressure increment dp in dbar.
!
! sa     : Absolute Salinity         (a profile (length nz))     [g/kg]
! ct     : Conservative Temperature  (a profile (length nz))     [deg C]
! p      : sea pressure              (a profile (length nz))     [dbar]
! lat    : latitude                  (a profile (length nz))     [deg N]
! nz     : number of levels in the profile
! n2     : Brunt-Vaisala Frequency squared  (length nz-1)        [s^-2]
! p_mid  : Mid pressure between p grid      (length nz-1)        [dbar]
*/
void
gsw_nsquared(double *sa, double *ct, double *p, double *lat, int nz,
        double *n2, double *p_mid)
{
        GSW_TEOS10_CONSTANTS;
        int     k;
        double  p_grav, n_grav, grav_local, dsa, sa_mid, dct, ct_mid,
                dp, rho_mid, alpha_mid, beta_mid;

        if (nz < 2)
            return;
        p_grav  = gsw_grav(lat[0],p[0]);
        for (k = 0; k < nz-1; k++) {
            n_grav      = gsw_grav(lat[k+1],p[k+1]);
            grav_local  = 0.5*(p_grav + n_grav);
            dsa         = (sa[k+1] - sa[k]);
            sa_mid      = 0.5*(sa[k] + sa[k+1]);
            dct         = (ct[k+1] - ct[k]);
            ct_mid      = 0.5*(ct[k] + ct[k+1]);
            dp          = (p[k+1] - p[k]);
            p_mid[k]    = 0.5*(p[k] + p[k+1]);
            rho_mid     = gsw_rho(sa_mid,ct_mid,p_mid[k]);
            alpha_mid   = gsw_alpha(sa_mid,ct_mid,p_mid[k]);
            beta_mid    = gsw_beta(sa_mid,ct_mid,p_mid[k]);

            n2[k]       = (grav_local*grav_local)*(rho_mid/(db2pa*dp))*
                          (beta_mid*dsa - alpha_mid*dct);
            p_grav      = n_grav;
        }
}
/*
!==========================================================================
function gsw_o2sol(sa, ct, p, lon, lat)
!==========================================================================
!
! Calculates the oxygen concentration expected at equilibrium with air at
! an Absolute Pressure of 101325 Pa (sea pressure of 0 dbar) including
! saturated water vapor.  This function uses the solubility coefficients
! derived from the data of Benson and Krause (1984), as fitted by Garcia
! and Gordon (1992, 1993).
!
! Note that this algorithm has not been approved by IOC and is not work
! from SCOR/IAPSO Working Group 127. It is included in the GSW
! Oceanographic Toolbox as it seems to be oceanographic best practice.
!
! SA  :  Absolute Salinity of seawater                           [ g/kg ]
! CT  :  Conservative Temperature of seawater (ITS-90)           [ deg C ]
! p   :  sea pressure at which the melting occurs                [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
! lat : latitude                                                 [deg]
! lon : longitude                                                [deg]
!
! gsw_o2sol : olubility of oxygen in micro-moles per kg          [umol/kg]
*/
double
gsw_o2sol(double sa, double ct, double p, double lon, double lat)
{
    GSW_TEOS10_CONSTANTS;
    double sp, pt, pt68, x, y, o2sol,
           a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, c0;

    sp = gsw_sp_from_sa(sa, p, lon, lat);
    x = sp;
    pt = gsw_pt_from_ct(sa, ct);

    pt68 = pt*1.00024;

    y = log((298.15 - pt68)/(gsw_t0 + pt68));

    a0 =  5.80871;
    a1 =  3.20291;
    a2 =  4.17887;
    a3 =  5.10006;
    a4 = -9.86643e-2;
    a5 =  3.80369;
    b0 = -7.01577e-3;
    b1 = -7.70028e-3;
    b2 = -1.13864e-2;
    b3 = -9.51519e-3;
    c0 = -2.75915e-7;

    o2sol = exp(a0 + y*(a1 + y*(a2 + y*(a3 + y*(a4 + a5*y))))
                  + x*(b0 + y*(b1 + y*(b2 + b3*y)) + c0*x));

    return o2sol;

}
/*
!==========================================================================
function gsw_o2sol_sp_pt(sp, pt)
!==========================================================================
!
! Calculates the oxygen concentration expected at equilibrium with air at
! an Absolute Pressure of 101325 Pa (sea pressure of 0 dbar) including
! saturated water vapor.  This function uses the solubility coefficients
! derived from the data of Benson and Krause (1984), as fitted by Garcia
! and Gordon (1992, 1993).
!
! Note that this algorithm has not been approved by IOC and is not work
! from SCOR/IAPSO Working Group 127. It is included in the GSW
! Oceanographic Toolbox as it seems to be oceanographic best practice.
!
! SP  :  Practical Salinity  (PSS-78)                         [ unitless ]
! pt  :  potential temperature (ITS-90) referenced               [ dbar ]
!         to one standard atmosphere (0 dbar).
!
! gsw_o2sol_sp_pt : olubility of oxygen in micro-moles per kg     [umol/kg]
*/
double
gsw_o2sol_sp_pt(double sp, double pt)
{
    GSW_TEOS10_CONSTANTS;
    double pt68, x, y, o2sol,
           a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, c0;

    x = sp;

    pt68 = pt*1.00024;

    y = log((298.15 - pt68)/(gsw_t0 + pt68));

    a0 =  5.80871;
    a1 =  3.20291;
    a2 =  4.17887;
    a3 =  5.10006;
    a4 = -9.86643e-2;
    a5 =  3.80369;
    b0 = -7.01577e-3;
    b1 = -7.70028e-3;
    b2 = -1.13864e-2;
    b3 = -9.51519e-3;
    c0 = -2.75915e-7;

    o2sol = exp(a0 + y*(a1 + y*(a2 + y*(a3 + y*(a4 + a5*y))))
                  + x*(b0 + y*(b1 + y*(b2 + b3*y)) + c0*x));

    return o2sol;

}
/*
!==========================================================================
function gsw_p_from_z(z,lat, geo_strf_dyn_height, sea_surface_geopotential)
!==========================================================================

! Calculates the pressure p from height z
!
! z      : height                                          [m]
! lat    : latitude                                        [deg]
! geo_strf_dyn_height : dynamic height anomaly             [m^2/s^2]
!    Note that the reference pressure, p_ref, of geo_strf_dyn_height must
!     be zero (0) dbar.
! sea_surface_geopotential : geopotential at zero sea pressure  [m^2/s^2]
!
! gsw_p_from_z : pressure                                  [dbar]
*/
double
gsw_p_from_z(double z, double lat, double geo_strf_dyn_height,
             double sea_surface_geopotential)
{
    GSW_TEOS10_CONSTANTS;
    double sinlat, sin2, gs, c1, p, df_dp, f, p_old, p_mid;

    if (z > 5) return GSW_INVALID_VALUE;

    sinlat = sin(lat*deg2rad);
    sin2 = sinlat*sinlat;
    gs = 9.780327*(1.0 + (5.2792e-3 + (2.32e-5*sin2))*sin2);

    /* get the first estimate of p from Saunders (1981) */
    c1 =  5.25e-3*sin2 + 5.92e-3;
    p  = -2.0*z/((1-c1) + sqrt((1-c1)*(1-c1) + 8.84e-6*z)) ;
    /* end of the first estimate from Saunders (1981) */

    df_dp = db2pa*gsw_specvol_sso_0(p); /* initial value of the derivative of f */

    f = gsw_enthalpy_sso_0(p) + gs*(z - 0.5*gamma*(z*z))
        - (geo_strf_dyn_height + sea_surface_geopotential);
    p_old = p;
    p = p_old - f/df_dp;
    p_mid = 0.5*(p + p_old);
    df_dp = db2pa*gsw_specvol_sso_0(p_mid);
    p = p_old - f/df_dp;

    return p;
}
/*
!==========================================================================
elemental function gsw_pot_enthalpy_from_pt_ice (pt0_ice)
!==========================================================================
!
!  Calculates the potential enthalpy of ice from potential temperature of
!  ice (whose reference sea pressure is zero dbar).
!
!  pt0_ice  =  potential temperature of ice (ITS-90)              [ deg C ]
!
!  gsw_pot_enthalpy_ice  =  potential enthalpy of ice              [ J/kg ]
!--------------------------------------------------------------------------
*/
double
gsw_pot_enthalpy_from_pt_ice(double pt0_ice)
{
        GSW_TEOS10_CONSTANTS;
        GSW_GIBBS_ICE_COEFFICIENTS;
        double  tau;
        DCOMPLEX        h0_part, sqtau_t1, sqtau_t2;

        tau = (pt0_ice + gsw_t0)*rec_tt;

        sqtau_t1 = (tau/t1)*(tau/t1);
        sqtau_t2 = (tau/t2)*(tau/t2);

        h0_part = r1*t1*(log(1.0 - sqtau_t1) + sqtau_t1)
                  + r20*t2*(log(1.0 - sqtau_t2) + sqtau_t2);

        return (g00 + tt*real(h0_part));
}
/*
!==========================================================================
elemental function gsw_pot_enthalpy_from_pt_ice_poly (pt0_ice)
!==========================================================================
!
!  Calculates the potential enthalpy of ice from potential temperature of
!  ice (whose reference sea pressure is zero dbar).  This is a
!  compuationally efficient polynomial fit to the potential enthalpy of
!  ice.
!
!  pt0_ice  =  potential temperature of ice (ITS-90)              [ deg C ]
!
!  pot_enthalpy_ice  =  potential enthalpy of ice                  [ J/kg ]
!--------------------------------------------------------------------------
*/
double
gsw_pot_enthalpy_from_pt_ice_poly(double pt0_ice)
{
        int     iteration;
        double  df_dt, f, pot_enthalpy_ice,
                pot_enthalpy_ice_mid, pot_enthalpy_ice_old,
                p0 = -3.333601570157700e5,
                p1 =  2.096693916810367e3,
                p2 =  3.687110754043292,
                p3 =  4.559401565980682e-4,
                p4 = -2.516011957758120e-6,
                p5 = -1.040364574632784e-8,
                p6 = -1.701786588412454e-10,
                p7 = -7.667191301635057e-13;

        /*initial estimate of the potential enthalpy.*/
        pot_enthalpy_ice = p0 + pt0_ice*(p1 + pt0_ice*(p2 + pt0_ice*(p3
                           + pt0_ice*(p4 + pt0_ice*(p5 + pt0_ice*(p6
                           + pt0_ice*p7))))));

        df_dt = gsw_pt_from_pot_enthalpy_ice_poly_dh(pot_enthalpy_ice);

        for (iteration = 1; iteration <= 5; iteration++) {
            pot_enthalpy_ice_old = pot_enthalpy_ice;
            f = gsw_pt_from_pot_enthalpy_ice_poly(pot_enthalpy_ice_old)
                        - pt0_ice;
            pot_enthalpy_ice = pot_enthalpy_ice_old - f/df_dt;
            pot_enthalpy_ice_mid = 0.5*(pot_enthalpy_ice+pot_enthalpy_ice_old);
            df_dt = gsw_pt_from_pot_enthalpy_ice_poly_dh(pot_enthalpy_ice_mid);
            pot_enthalpy_ice = pot_enthalpy_ice_old - f/df_dt;
        }
        /*
        ! The error of this fit ranges between -6e-3 and 6e-3 J/kg over the
        | potential temperature range of -100 to 2 deg C, or the potential
        | enthalpy range of -5.7 x 10^5 to -3.3 x 10^5 J/kg.
        */
        return (pot_enthalpy_ice);
}
/*
!==========================================================================
elemental function gsw_pot_enthalpy_ice_freezing (sa, p)
!==========================================================================
!
!  Calculates the potential enthalpy of ice at which seawater freezes.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  pot_enthalpy_ice_freezing = potential enthalpy of ice at freezing
!                              of seawater                        [ deg C ]
!--------------------------------------------------------------------------
*/
double
gsw_pot_enthalpy_ice_freezing(double sa, double p)
{
        double  pt0_ice, t_freezing;

        t_freezing = gsw_t_freezing(sa,p,0.0) ;

        pt0_ice = gsw_pt0_from_t_ice(t_freezing,p);

        return (gsw_pot_enthalpy_from_pt_ice(pt0_ice));
}
/*
!==========================================================================
elemental subroutine gsw_pot_enthalpy_ice_freezing_first_derivatives (sa, &
              p, pot_enthalpy_ice_freezing_sa, pot_enthalpy_ice_freezing_p)
!==========================================================================
!
!  Calculates the first derivatives of the potential enthalpy of ice at
!  which seawater freezes, with respect to Absolute Salinity SA and
!  pressure P (in Pa).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  pot_enthalpy_ice_freezing_SA = the derivative of the potential enthalpy
!                  of ice at freezing (ITS-90) with respect to Absolute
!                  salinity at fixed pressure  [ K/(g/kg) ] i.e. [ K kg/g ]
!
!  pot_enthalpy_ice_freezing_P  = the derivative of the potential enthalpy
!                  of ice at freezing (ITS-90) with respect to pressure
!                  (in Pa) at fixed Absolute Salinity              [ K/Pa ]
!--------------------------------------------------------------------------
*/
void
gsw_pot_enthalpy_ice_freezing_first_derivatives(double sa, double p,
    double *pot_enthalpy_ice_freezing_sa, double *pot_enthalpy_ice_freezing_p)
{
        GSW_TEOS10_CONSTANTS;
        double  cp_ihf, pt_icef, ratio_temp, tf, tf_p, tf_sa;
        double  saturation_fraction = 0.0;

        tf = gsw_t_freezing(sa,p,saturation_fraction);
        pt_icef = gsw_pt0_from_t_ice(tf,p);
        ratio_temp = (gsw_t0 + pt_icef)/(gsw_t0 + tf);

        cp_ihf = gsw_cp_ice(tf,p);

        if ((pot_enthalpy_ice_freezing_sa != NULL) &&
            (pot_enthalpy_ice_freezing_p != NULL)) {
            gsw_t_freezing_first_derivatives(sa,p,saturation_fraction,
                        &tf_sa,&tf_p);
        } else if (pot_enthalpy_ice_freezing_sa != NULL) {
            gsw_t_freezing_first_derivatives(sa,p,saturation_fraction,
                                                  &tf_sa, NULL);
        } else if (pot_enthalpy_ice_freezing_p != NULL) {
            gsw_t_freezing_first_derivatives(sa,p,saturation_fraction,
                                                  NULL,&tf_p);
        }

        if (pot_enthalpy_ice_freezing_sa != NULL)
            *pot_enthalpy_ice_freezing_sa = ratio_temp*cp_ihf*tf_sa;

        if (pot_enthalpy_ice_freezing_p != NULL)
            *pot_enthalpy_ice_freezing_p = ratio_temp*cp_ihf*tf_p
                              - (gsw_t0 + pt_icef)*gsw_gibbs_ice(1,1,tf,p);
}
/*
!==========================================================================
elemental subroutine gsw_pot_enthalpy_ice_freezing_first_derivatives_poly(&
         sa, p, pot_enthalpy_ice_freezing_sa, pot_enthalpy_ice_freezing_p)
!==========================================================================
!
!  Calculates the first derivatives of the potential enthalpy of ice Ih at
!  which ice melts into seawater with Absolute Salinity SA and at pressure
!  p.  This code uses the comptationally efficient polynomial fit of the
!  freezing potential enthalpy of ice Ih (McDougall et al., 2015).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  pot_enthalpy_ice_freezing_SA = the derivative of the potential enthalpy
!                of ice at freezing (ITS-90) with respect to Absolute
!                salinity at fixed pressure  [ (J/kg)/(g/kg) ] i.e. [ J/g ]
!
!  pot_enthalpy_ice_freezing_P  = the derivative of the potential enthalpy
!                of ice at freezing (ITS-90) with respect to pressure
!                (in Pa) at fixed Absolute Salinity           [ (J/kg)/Pa ]
!--------------------------------------------------------------------------
*/
void
gsw_pot_enthalpy_ice_freezing_first_derivatives_poly(double sa, double p,
    double *pot_enthalpy_ice_freezing_sa, double *pot_enthalpy_ice_freezing_p)
{
        double  p_r, sa_r, x,
                d1 =  -1.249490228128056e4,
                d2 =   1.336783910789822e4,
                d3 =  -4.811989517774642e4,
                d4 =   8.044864276240987e4,
                d5 =  -7.124452125071862e4,
                d6 =   2.280706828014839e4,
                d7 =   0.315423710959628e3,
                d8 =  -3.592775732074710e2,
                d9 =   1.644828513129230e3,
                d10 = -4.809640968940840e3,
                d11 =  2.901071777977272e3,
                d12 = -9.218459682855746e2,
                d13 =  0.379377450285737e3,
                d14 = -2.672164989849465e3,
                d15 =  5.044317489422632e3,
                d16 = -2.631711865886377e3,
                d17 = -0.160245473297112e3,
                d18 =  4.029061696035465e2,
                d19 = -3.682950019675760e2,

                f1 =  -2.034535061416256e4,
                f2 =   0.315423710959628e3,
                f3 =  -0.239518382138314e3,
                f4 =   0.822414256564615e3,
                f5 =  -1.923856387576336e3,
                f6 =   0.967023925992424e3,
                f7 =  -0.263384562367307e3,
                f8 =  -5.051613740291480e3,
                f9 =   7.587549005714740e2,
                f10 = -3.562886653132620e3,
                f11 =  5.044317489422632e3,
                f12 = -2.105369492709102e3,
                f13 =  6.387082316647800e2,
                f14 = -4.807364198913360e2,
                f15 =  8.058123392070929e2,
                f16 = -5.524425029513641e2;

        sa_r = sa*1e-2;
        x = sqrt(sa_r);
        p_r = p*1e-4;

        if (pot_enthalpy_ice_freezing_sa != NULL)
            *pot_enthalpy_ice_freezing_sa =
                (d1 + x*(d2  + x*(d3  + x*(d4  + x*(d5  + d6*x))))
               + p_r*(d7 + x*(d8 + x*(d9 + x*(d10 + x*(d11 + d12*x))))
               + p_r*(d13 + x*(d14 + x*(d15 + d16*x))
               + p_r*(d17 + x*(d18 + d19*x)))))*1e-2;

        if (pot_enthalpy_ice_freezing_p != NULL)
            *pot_enthalpy_ice_freezing_p =
                (f1 + sa_r*(f2 + x*(f3 + x*(f4 + x*(f5 + x*(f6 + f7*x)))))
               + p_r*(f8 + sa_r*(f9 + x*(f10 + x*(f11 + f12*x)))
               + p_r*(f13 + sa_r*(f14 + x*(f15 + f16*x)))))*1e-8;
}
/*
!==========================================================================
elemental function gsw_pot_enthalpy_ice_freezing_poly (sa, p)
!==========================================================================
!
!  Calculates the potential enthalpy of ice at which seawater freezes.
!  The error of this fit ranges between -2.5 and 1 J/kg with an rms of
!  1.07, between SA of 0 and 120 g/kg and p between 0 and 10,000 dbar (the
!  error in the fit is between -0.7 and 0.7 with an rms of
!  0.3, between SA of 0 and 120 g/kg and p between 0 and 5,000 dbar) when
!  compared with the potential enthalpy calculated from the exact in-situ
!  freezing temperature which is found by a Newton-Raphson iteration of the
!  equality of the chemical potentials of water in seawater and in ice.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  pot_enthalpy_ice_freezing = potential enthalpy of ice at freezing
!                              of seawater                         [ J/kg ]
!--------------------------------------------------------------------------
*/
double
gsw_pot_enthalpy_ice_freezing_poly(double sa, double p)
{
        double  p_r, sa_r, x,
                c0  = -3.333548730778702e5,
                c1  = -1.249490228128056e4,
                c2  =  0.891189273859881e4,
                c3  = -2.405994758887321e4,
                c4  =  3.217945710496395e4,
                c5  = -2.374817375023954e4,
                c6  =  0.651630522289954e4,
                c7  = -2.034535061416256e4,
                c8  = -0.252580687014574e4,
                c9  =  0.021290274388826e4,
                c10 =  0.315423710959628e3,
                c11 = -0.239518382138314e3,
                c12 =  0.379377450285737e3,
                c13 =  0.822414256564615e3,
                c14 = -1.781443326566310e3,
                c15 = -0.160245473297112e3,
                c16 = -1.923856387576336e3,
                c17 =  2.522158744711316e3,
                c18 =  0.268604113069031e3,
                c19 =  0.967023925992424e3,
                c20 = -1.052684746354551e3,
                c21 = -0.184147500983788e3,
                c22 = -0.263384562367307e3;

        sa_r = sa*1e-2;
        x = sqrt(sa_r);
        p_r = p*1e-4;

        return (c0 + sa_r*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + c6*x)))))
            + p_r*(c7 + p_r*(c8 + c9*p_r)) + sa_r*p_r*(c10 + p_r*(c12
            + p_r*(c15 + c21*sa_r)) + sa_r*(c13 + c17*p_r + c19*sa_r)
            + x*(c11 + p_r*(c14 + c18*p_r) + sa_r*(c16 + c20*p_r + c22*sa_r))));
}
/*
!==========================================================================
function gsw_pot_rho_t_exact(sa,t,p,p_ref)
!==========================================================================

! Calculates the potential density of seawater
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
! p_ref  : reference sea pressure                          [dbar]
!
! gsw_pot_rho_t_exact : potential density                  [kg/m^3]
*/
double
gsw_pot_rho_t_exact(double sa, double t, double p, double p_ref)
{
        double  pt = gsw_pt_from_t(sa,t,p,p_ref);

        return (gsw_rho_t_exact(sa,pt,p_ref));
}
/*
!==========================================================================
elemental function gsw_pressure_coefficient_ice (t, p)
!==========================================================================
!
!  Calculates pressure coefficient of ice.
!
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  pressure_coefficient_ice  =  pressure coefficient of ice          [Pa/K]
!   Note. The output units are Pa/K NOT dbar/K.
!--------------------------------------------------------------------------
*/
double
gsw_pressure_coefficient_ice(double t, double p)
{
        return (-gsw_gibbs_ice(1,1,t,p)/gsw_gibbs_ice(0,2,t,p));
}
/*
!==========================================================================
elemental function gsw_pressure_freezing_ct (sa, ct, saturation_fraction)
!==========================================================================
!
!  Calculates the pressure (in dbar) of seawater at the freezing
!  temperature.  That is, the output is the pressure at which seawater,
!  with Absolute Salinity SA, Conservative Temperature CT, and with
!  saturation_fraction of dissolved air, freezes.  If the input values are
!  such that there is no value of pressure in the range between 0 dbar and
!  10,000 dbar for which seawater is at the freezing temperature, the
!  output, pressure_freezing, is put equal to NaN.
!
!  SA  =  Absolute Salinity of seawater                            [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  saturation_fraction = the saturation fraction of dissolved air in
!                        seawater
!
!  pressure_freezing = sea pressure at which the seawater freezes  [ dbar ]
!        ( i.e. absolute pressure - 10.1325 dbar )
!--------------------------------------------------------------------------
*/
double
gsw_pressure_freezing_ct(double sa, double ct, double saturation_fraction)
{
        GSW_TEOS10_CONSTANTS;
        int     i_iter, number_of_iterations = 3;
        double  ct_freezing_p0, ct_freezing_p10000, dctf_dp, f,
                pf, pfm, pf_old, ctfreezing_p;

        /*
        ! rec_Pa2dbar is to have dCTf_dp in units of K/dbar rather than K/Pa

        ! Find CT > CT_freezing_p0.  If this is the case, the input CT value
        ! represent seawater that will not be frozen at any positive p.
        */
        ct_freezing_p0 = gsw_ct_freezing(sa,0.0,saturation_fraction);
        if (ct > ct_freezing_p0) {
            return (GSW_INVALID_VALUE);
        }
        /*
        ! Find CT < CT_freezing_p10000.  If this is the case, the input CT value
        ! represent seawater that is frozen even at p = 10,000 dbar.
        */
        ct_freezing_p10000 = gsw_ct_freezing(sa,1e4,saturation_fraction);
        if (ct < ct_freezing_p10000) {
            return (GSW_INVALID_VALUE);
        }
        /*
        ! This is the initial (linear) guess of the freezing pressure, in dbar.
        */
        pf = rec_pa2db*(ct_freezing_p0 - ct)/
                        (ct_freezing_p0 - ct_freezing_p10000);

        gsw_ct_freezing_first_derivatives(sa,pf,saturation_fraction,
                                               NULL,&ctfreezing_p);
        dctf_dp = rec_pa2db*ctfreezing_p;
            /*
            !  this dctf_dp is the initial value of the partial derivative of
            !  ct_freezing with respect to pressure (in dbar) at fixed sa,
            !  assuming that the saturation_fraction is zero.
            */
        for (i_iter = 1; i_iter <= number_of_iterations; i_iter++) {
            pf_old = pf;
            f = gsw_ct_freezing(sa,pf_old,saturation_fraction) - ct;
            pf = pf_old - f/dctf_dp;
            pfm = 0.5*(pf + pf_old);
            gsw_ct_freezing_first_derivatives(sa,pfm,saturation_fraction,
                                                   NULL,&ctfreezing_p);
            dctf_dp = rec_pa2db*ctfreezing_p;
            pf = pf_old - f/dctf_dp;
        }

        if (gsw_sa_p_inrange(sa,pf))
            return (pf);
        return (GSW_INVALID_VALUE);
}
/*
!==========================================================================
elemental function gsw_pt0_cold_ice_poly (pot_enthalpy_ice)
!==========================================================================
!
!  Calculates an initial estimate of pt0_ice when it is less than about
!  -100 deg C.
!
!  pot_enthalpy_ice  =  potential enthalpy of ice                  [ J/kg ]
!
!  pt0_cold_ice_poly  =  initial estimate of potential temperature
!                        of very cold ice in dgress C (not K)     [ deg C ]
!--------------------------------------------------------------------------
*/
double
gsw_pt0_cold_ice_poly(double pot_enthalpy_ice)
{
        GSW_TEOS10_CONSTANTS;
        double  log_abs_theta0, log_h_diff,
                /*h00 = gsw_enthalpy_ice(-gsw_t0,0)*/
                h00 = -6.320202333358860e5,

                s0 =  1.493103204647916,
                s1 =  2.372788609320607e-1,
                s2 = -2.014996002119374e-3,
                s3 =  2.640600197732682e-6,
                s4 =  3.134706016844293e-5,
                s5 =  2.733592344937913e-6,
                s6 =  4.726828010223258e-8,
                s7 = -2.735193883189589e-9,
                s8 = -8.547714991377670e-11;

        log_h_diff = log(pot_enthalpy_ice - h00);

        log_abs_theta0 = s0 + log_h_diff*(s1 + log_h_diff*(s2 + log_h_diff*(s3
                        + log_h_diff*(s4 + log_h_diff*(s5 + log_h_diff*(s6
                        + log_h_diff*(s7 + log_h_diff*s8)))))));

        return (exp(log_abs_theta0) - gsw_t0);
}
/*
!==========================================================================
function gsw_pt0_from_t(sa,t,p)
!==========================================================================

! Calculates potential temperature with reference pressure, p_ref = 0 dbar.
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
!
! gsw_pt0_from_t : potential temperature, p_ref = 0        [deg C]
*/
double
gsw_pt0_from_t(double sa, double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        int     no_iter;
        double  pt0, pt0_old, dentropy, dentropy_dt;
        double  s1, true_entropy_part, pt0m;

        s1      = sa/gsw_ups;

        pt0     = t+p*( 8.65483913395442e-6  -
                  s1 *  1.41636299744881e-6  -
                   p *  7.38286467135737e-9  +
                   t *(-8.38241357039698e-6  +
                  s1 *  2.83933368585534e-8  +
                   t *  1.77803965218656e-8  +
                   p *  1.71155619208233e-10));

        dentropy_dt     = gsw_cp0/((gsw_t0+pt0)*(1.0-0.05*(1.0-sa/gsw_sso)));

        true_entropy_part = gsw_entropy_part(sa,t,p);

        for (no_iter=1; no_iter <= 2; no_iter++) {
            pt0_old     = pt0;
            dentropy    = gsw_entropy_part_zerop(sa,pt0_old) -
                          true_entropy_part;
            pt0         = pt0_old - dentropy/dentropy_dt;
            pt0m        = 0.5*(pt0 + pt0_old);
            dentropy_dt = -gsw_gibbs_pt0_pt0(sa,pt0m);
            pt0         = pt0_old - dentropy/dentropy_dt;
        }
        return (pt0);
}
/*
! =========================================================================
elemental function gsw_pt0_from_t_ice (t, p)
! =========================================================================
!
!  Calculates potential temperature of ice Ih with a reference pressure of
!  0 dbar, from in-situ temperature, t.
!
!  t   =  in-situ temperature  (ITS-90)                           [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  pt0_ice  =  potential temperature of ice Ih with reference pressure of
!              zero dbar (ITS-90)                                 [ deg C ]
!--------------------------------------------------------------------------
*/
double
gsw_pt0_from_t_ice(double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        int     number_of_iterations;
        double  dentropy, dentropy_dt, pt0_ice,
                pt0_ice_old, ptm_ice, true_entropy,
                /*This is the starting polynomial for pt0 of ice Ih.*/
                s1 = -2.256611570832386e-4,
                s2 = -6.045305921314694e-7,
                s3 =  5.546699019612661e-9,
                s4 =  1.795030639186685e-11,
                s5 =  1.292346094030742e-9,

                p1 = -2.259745637898635e-4,
                p2 =  1.486236778150360e-9,
                p3 =  6.257869607978536e-12,
                p4 = -5.253795281359302e-7,
                p5 =  6.752596995671330e-9,
                p6 =  2.082992190070936e-11,

                q1 = -5.849191185294459e-15,
                q2 =  9.330347971181604e-11,
                q3 =  3.415888886921213e-13,
                q4 =  1.064901553161811e-12,
                q5 = -1.454060359158787e-10,
                q6 = -5.323461372791532e-13;

        true_entropy = -gsw_gibbs_ice_part_t(t,p);

        if (t < -45.0 && t > -273.0) {

            pt0_ice = t + p*(p1 + p*(p2 + p3*t) + t*(p4 + t*(p5 + p6*t)));

            if (pt0_ice < -gsw_t0) pt0_ice = -gsw_t0;
            /*
            ! we add 0.05d0 to the initial estimate of pt0_ice at
            ! temps less than -273 to ensure that it is never less than -273.15.
            */
            if (pt0_ice < -273.0) pt0_ice = pt0_ice + 0.05;

            dentropy_dt = -gsw_gibbs_ice_pt0_pt0(pt0_ice);

            for (number_of_iterations = 1; number_of_iterations <= 3;
                number_of_iterations++) {
                pt0_ice_old = pt0_ice;
                dentropy = -gsw_gibbs_ice_pt0(pt0_ice_old) - true_entropy;
                pt0_ice = pt0_ice_old - dentropy/dentropy_dt;
                ptm_ice = 0.5*(pt0_ice + pt0_ice_old);
                dentropy_dt = -gsw_gibbs_ice_pt0_pt0(ptm_ice);
                pt0_ice = pt0_ice_old - dentropy/dentropy_dt;
            }

        } else {

            pt0_ice = t + p*(s1 + t*(s2 + t*(s3 + t*s4)) + s5*p);
            dentropy_dt = -gsw_gibbs_ice_pt0_pt0(pt0_ice);

            pt0_ice_old = pt0_ice;
            dentropy = -gsw_gibbs_ice_pt0(pt0_ice_old) - true_entropy;

            pt0_ice = pt0_ice_old - dentropy/dentropy_dt;
            ptm_ice = 0.5*(pt0_ice + pt0_ice_old);
            dentropy_dt = -gsw_gibbs_ice_pt0_pt0(ptm_ice);
            pt0_ice = pt0_ice_old - dentropy/dentropy_dt;

        }

        if (pt0_ice < -273.0) {

            pt0_ice = t + p*(q1 + p*(q2 + q3*t) + t*(q4 + t*(q5 + q6*t)));
            /*
            ! add 0.01d0 to the initial estimate of pt_ice used in the
            ! derivative to ensure that it is never less than -273.15d0
            ! because the derivative approaches zero at absolute zero.
            */
            dentropy_dt = -gsw_gibbs_ice_pt0_pt0(pt0_ice+0.01);

            for (number_of_iterations = 1; number_of_iterations <= 3;
                number_of_iterations++) {
                pt0_ice_old = pt0_ice;
                dentropy = -gsw_gibbs_ice_pt0(pt0_ice_old) - true_entropy;
                pt0_ice = pt0_ice_old - dentropy/dentropy_dt;
                ptm_ice = 0.5*(pt0_ice + pt0_ice_old);
                /*
                ! add 0.01d0 to the estimate of ptm_ice for temperatures less
                | than -273 to ensure that they are never less than -273.15d0
                ! because the derivative approaches zero at absolute zero and
                ! the addition of 0.01d0 degrees c ensures that when we divide
                ! by the derivatve in the modified newton routine the function
                ! does not blow up.
                */
                ptm_ice = ptm_ice + 0.01;
                dentropy_dt = -gsw_gibbs_ice_pt0_pt0(ptm_ice);
                pt0_ice = pt0_ice_old - dentropy/dentropy_dt;
            }

        }
        /*
        ! For temperatures less than -273.1 degsC the maximum error is less
        ! than 2x10^-7 degsC. For temperatures between -273.1 and 273 the
        ! maximum error is less than 8x10^-8 degsC, and for temperatures
        ! greater than -273 degsC the ! maximum error is 1.5x10^-12 degsC.
        ! These errors are over the whole ocean depths with p varying between
        ! 0 and 10,000 dbar, while the in-situ temperature varied independently
        ! between -273.15 and +2 degsC.
        */

        return (pt0_ice);
}
/*
!==========================================================================
elemental subroutine gsw_pt_first_derivatives (sa, ct, pt_sa, pt_ct)
! =========================================================================
!
!  Calculates the following two partial derivatives of potential temperature
!  (the regular potential temperature whose reference sea pressure is 0 dbar)
!  (1) pt_SA, the derivative with respect to Absolute Salinity at
!       constant Conservative Temperature, and
!  (2) pt_CT, the derivative with respect to Conservative Temperature at
!       constant Absolute Salinity.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!
!  pt_SA =  The derivative of potential temperature with respect to
!           Absolute Salinity at constant Conservative Temperature.
!                                                               [ K/(g/kg)]
!  pt_CT =  The derivative of potential temperature with respect to
!           Conservative Temperature at constant Absolute Salinity.
!           pt_CT is dimensionless.                            [ unitless ]
!--------------------------------------------------------------------------
*/
void
gsw_pt_first_derivatives (double sa, double ct, double *pt_sa, double *pt_ct)
{
        GSW_TEOS10_CONSTANTS;
        double  abs_pt, ct_pt, ct_sa, pt, pr0 = 0.0;
        int     n0=0, n1=1, n2=2;

        pt = gsw_pt_from_ct(sa,ct);
        abs_pt = (gsw_t0 + pt);

        ct_pt = -(abs_pt*gsw_gibbs(n0,n2,n0,sa,pt,pr0))/gsw_cp0;

        if (pt_sa != NULL) {

            ct_sa = (gsw_gibbs(n1,n0,n0,sa,pt,pr0) -
                abs_pt*gsw_gibbs(n1,n1,n0,sa,pt,pr0))/gsw_cp0;

            *pt_sa = -ct_sa/ct_pt;

        }

        if (pt_ct != NULL)
            *pt_ct = 1.0/ct_pt;
}
/*
!==========================================================================
function gsw_pt_from_ct(sa,ct)
!==========================================================================

! potential temperature of seawater from conservative temperature
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature                        [deg C]
! p      : sea pressure                                    [dbar]
!
! gsw_pt_from_ct : potential temperature with              [deg C]
!                  reference pressure of  0 dbar
*/
double
gsw_pt_from_ct(double sa, double ct)
{
        GSW_TEOS10_CONSTANTS;
        double  a5ct, b3ct, ct_factor, pt_num, pt_recden, ct_diff;
        double  pt, pt_old, ptm, dpt_dct, s1;
        double  a0      = -1.446013646344788e-2,
                a1      = -3.305308995852924e-3,
                a2      =  1.062415929128982e-4,
                a3      =  9.477566673794488e-1,
                a4      =  2.166591947736613e-3,
                a5      =  3.828842955039902e-3,
                b0      =  1.000000000000000e0,
                b1      =  6.506097115635800e-4,
                b2      =  3.830289486850898e-3,
                b3      =  1.247811760368034e-6;

        s1      = sa/gsw_ups;

        a5ct    = a5*ct;
        b3ct    = b3*ct;

        ct_factor       = (a3 + a4*s1 + a5ct);
        pt_num          = a0 + s1*(a1 + a2*s1) + ct*ct_factor;
        pt_recden       = 1.0/(b0 + b1*s1 + ct*(b2 + b3ct));
        pt              = pt_num*pt_recden;

        dpt_dct = pt_recden*(ct_factor + a5ct - (b2 + b3ct + b3ct)*pt);

    /*
    **  Start the 1.5 iterations through the modified Newton-Raphson
    **  iterative method.
    */

        ct_diff = gsw_ct_from_pt(sa,pt) - ct;
        pt_old  = pt;
        pt      = pt_old - ct_diff*dpt_dct;
        ptm     = 0.5*(pt + pt_old);

        dpt_dct = -gsw_cp0/((ptm + gsw_t0)*gsw_gibbs_pt0_pt0(sa,ptm));

        pt      = pt_old - ct_diff*dpt_dct;
        ct_diff = gsw_ct_from_pt(sa,pt) - ct;
        pt_old  = pt;
        return (pt_old - ct_diff*dpt_dct);
}
/*
! =========================================================================
elemental function gsw_pt_from_entropy (sa, entropy)
! =========================================================================
!
!  Calculates potential temperature with reference pressure p_ref = 0 dbar
!  and with entropy as an input variable.
!
!  SA       =  Absolute Salinity                                   [ g/kg ]
!  entropy  =  specific entropy                                   [ deg C ]
!
!  pt   =  potential temperature                                  [ deg C ]
!          with reference sea pressure (p_ref) = 0 dbar.
!  Note. The reference sea pressure of the output, pt, is zero dbar.
!--------------------------------------------------------------------------
*/
double
gsw_pt_from_entropy(double sa, double entropy)
{
        GSW_TEOS10_CONSTANTS;
        int     number_of_iterations;
        double  c, dentropy, dentropy_dt, ent_sa, part1, part2, pt, ptm,
                pt_old;

        /*Find the initial value of pt*/
        part1 = 1.0 - sa/gsw_sso;
        part2 = 1.0 - 0.05*part1;
        ent_sa = (gsw_cp0/gsw_t0)*part1*(1.0 - 1.01*part1);
        c = (entropy - ent_sa)*(part2/gsw_cp0);
        pt = gsw_t0*(exp(c) - 1.0);
        dentropy_dt = gsw_cp0/((gsw_t0 + pt)*part2);

        for (number_of_iterations = 1; number_of_iterations <= 2;
            number_of_iterations++) {
            pt_old = pt;
            dentropy = gsw_entropy_from_pt(sa,pt_old) - entropy;
            pt = pt_old - dentropy/dentropy_dt;
            ptm = 0.5*(pt + pt_old);
            dentropy_dt = -gsw_gibbs_pt0_pt0(sa,ptm);
            pt = pt_old - dentropy/dentropy_dt;
        }
        /*
        ! Maximum error of 2.2x10^-6 degrees C for one iteration.
        ! Maximum error is 1.4x10^-14 degrees C for two iterations
        ! (two iterations is the default, "for Number_of_iterations = 1:2").
        */
        return (pt);
}
/*
!==========================================================================
elemental function gsw_pt_from_pot_enthalpy_ice (pot_enthalpy_ice)
!==========================================================================
!
!  Calculates the potential temperature of ice from the potential enthalpy
!  of ice.  The reference sea pressure of both the potential temperature
!  and the potential enthalpy is zero dbar.
!
!  pot_enthalpy_ice  =  potential enthalpy of ice                  [ J/kg ]
!
!  pt0_ice  =  potential temperature of ice (ITS-90)              [ deg C ]
!--------------------------------------------------------------------------
*/
double
gsw_pt_from_pot_enthalpy_ice(double pot_enthalpy_ice)
{
        int     iteration;
        double  df_dt, f, mod_pot_enthalpy_ice, pt0_cold_ice, recip_df_dt,
                pt0_cold_ice_old, pt0_ice, pt0_ice_old, ptm_cold_ice, ptm_ice;
        double  h00 = -6.320202333358860e5, /*gsw_enthalpy_ice(-gsw_t0,0)*/
                p0 = 0.0;

        mod_pot_enthalpy_ice = max(pot_enthalpy_ice,h00);

        if (mod_pot_enthalpy_ice >= -5.1e5) {
        /*
        ! For input potential enthalpies greater than -5.1e-5, the above part of
        ! the code gives the output potential temperature of ice accurate to
        | 1e-13 degrees C.
        */
            pt0_ice = gsw_pt_from_pot_enthalpy_ice_poly(mod_pot_enthalpy_ice);
        /*
        ! The variable "df_dt" below is the derivative of the above polynomial
        ! with respect to pot_enthalpy_ice.  This is the initial value of the
        ! derivative of the function f.
        */
            recip_df_dt =
                gsw_pt_from_pot_enthalpy_ice_poly_dh(mod_pot_enthalpy_ice);

            pt0_ice_old = pt0_ice;
            f = gsw_pot_enthalpy_from_pt_ice(pt0_ice_old)
                        - mod_pot_enthalpy_ice;
            pt0_ice = pt0_ice_old - f*recip_df_dt;
            ptm_ice = 0.5*(pt0_ice + pt0_ice_old);
            recip_df_dt = 1.0/gsw_cp_ice(ptm_ice,p0);
            pt0_ice = pt0_ice_old - f*recip_df_dt;

        } else {
        /*
        ! For  pot_enthalpy_ice < -5.1e5 (or pt0_ice less than about -100 deg c)
        ! these temperatures are less than those found in nature on planet earth
        */
            pt0_cold_ice = gsw_pt0_cold_ice_poly(mod_pot_enthalpy_ice);

            df_dt = gsw_cp_ice(pt0_cold_ice+0.02,p0);
            /*
            ! the heat capacity, cp, is
            ! evaluated at 0.02 c greater than usual in order to avoid stability
            ! issues and to ensure convergence near zero absolute temperature.
            */
            for (iteration = 1; iteration <= 6; iteration++) {
                pt0_cold_ice_old = pt0_cold_ice;
                f = gsw_pot_enthalpy_from_pt_ice(pt0_cold_ice_old)
                        - mod_pot_enthalpy_ice;
                pt0_cold_ice = pt0_cold_ice_old - f/df_dt;
                ptm_cold_ice = 0.5*(pt0_cold_ice + pt0_cold_ice_old);
                df_dt = gsw_cp_ice(ptm_cold_ice+0.02,p0);
                /*note the extra 0.02 c here as well*/
                pt0_cold_ice = pt0_cold_ice_old - f/df_dt;
            }
            pt0_ice = pt0_cold_ice;
        }
/*
!The potential temperature has a maximum error as listed in the table below.
!
!  potential temperature error (deg C)  |  @ potential temperature (deg C)
!--------------------------------------|--------------------------------
!                0.012                 |     -273.15 to -273.12
!              4 x 10^-4               |     -232.12 to -273.0
!             2.5 x 10^-6              |          -273
!              7 x 10^-9               |          -272
!            3.7 x 10^-10              |          -270
!              6 x 10^-11              |          -268
!             2.5 x 10^11              |          -266
!             3 x 10^-12               |          -260
!             7 x 10^-13               |          -250
!            2.2 x 10^-13              |          -220
!            1.7 x 10^-13              |         >= -160
!
! Note.  The above errors in each temperature range are machine precissions
! for this calculation.
*/

        return (pt0_ice);
}
/*
!==========================================================================
elemental function gsw_pt_from_pot_enthalpy_ice_poly (pot_enthalpy_ice)
!==========================================================================
!
!  Calculates the potential temperature of ice (whose reference sea
!  pressure is zero dbar) from the potential enthalpy of ice.  This is a
!  compuationally efficient polynomial fit to the potential enthalpy of
!  ice.
!
!  pot_enthalpy_ice  =  potential enthalpy of ice                  [ J/kg ]
!
!  pt0_ice  =  potential temperature of ice (ITS-90)              [ deg C ]
!--------------------------------------------------------------------------
*/
double
gsw_pt_from_pot_enthalpy_ice_poly(double pot_enthalpy_ice)
{
        double  q0 = 2.533588268773218e2,
                q1 = 2.594351081876611e-3,
                q2 = 1.765077810213815e-8,
                q3 = 7.768070564290540e-14,
                q4 = 2.034842254277530e-19,
                q5 = 3.220014531712841e-25,
                q6 = 2.845172809636068e-31,
                q7 = 1.094005878892950e-37;
/*
! The error of this fit ranges between -5e-5 and 2e-4 deg C over the potential
! temperature range of -100 to 2 deg C, or the potential enthalpy range of
! -5.7 x 10^5 to -3.3 x 10^5 J/kg.
*/
        return (q0
         + pot_enthalpy_ice*(q1 + pot_enthalpy_ice*(q2 + pot_enthalpy_ice*(q3
         + pot_enthalpy_ice*(q4 + pot_enthalpy_ice*(q5 + pot_enthalpy_ice*(q6
         + pot_enthalpy_ice*q7)))))));
}
/*
!==========================================================================
elemental function gsw_pt_from_pot_enthalpy_ice_poly_dh (pot_enthalpy_ice)
!==========================================================================
!
!  Calculates the derivative of potential temperature of ice with respect
!  to potential enthalpy.  This is based on the compuationally-efficient
!  polynomial fit to the potential enthalpy of ice.
!
!  pot_enthalpy_ice  =  potential enthalpy of ice                  [ J/kg ]
!
!  dpt0_ice_dh  =  derivative of potential temperature of ice
!                  with respect to potential enthalpy             [ deg C ]
!--------------------------------------------------------------------------
*/
double
gsw_pt_from_pot_enthalpy_ice_poly_dh(double pot_enthalpy_ice)
{
        double  q1 = 2.594351081876611e-3,
                p2 = 3.530155620427630e-8,
                p3 = 2.330421169287162e-13,
                p4 = 8.139369017110120e-19,
                p5 = 1.610007265856420e-24,
                p6 = 1.707103685781641e-30,
                p7 = 7.658041152250651e-37;

        return (q1 + pot_enthalpy_ice*(p2 + pot_enthalpy_ice*(p3
            + pot_enthalpy_ice*(p4 + pot_enthalpy_ice*(p5 + pot_enthalpy_ice*(p6
            + pot_enthalpy_ice*p7))))));
}
/*
!==========================================================================
function gsw_pt_from_t(sa,t,p,p_ref)
!==========================================================================

! Calculates potential temperature of seawater from in-situ temperature
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
! p_ref  : reference sea pressure                          [dbar]
!
! gsw_pt_from_t : potential temperature                    [deg C]
*/
double
gsw_pt_from_t(double sa, double t, double p, double p_ref)
{
        GSW_TEOS10_CONSTANTS;
        int     n0=0, n2=2, no_iter;
        double  s1, pt, ptm, pt_old, dentropy, dentropy_dt,
                true_entropy_part;

        s1      = sa/gsw_ups;
        pt      = t+(p-p_ref)*( 8.65483913395442e-6  -
                          s1 *  1.41636299744881e-6  -
                   (p+p_ref) *  7.38286467135737e-9  +
                          t  *(-8.38241357039698e-6  +
                          s1 *  2.83933368585534e-8  +
                          t  *  1.77803965218656e-8  +
                   (p+p_ref) *  1.71155619208233e-10));

        dentropy_dt     = gsw_cp0/((gsw_t0 + pt)*(1.0-0.05*(1.0 - sa/gsw_sso)));
        true_entropy_part       = gsw_entropy_part(sa,t,p);
        for (no_iter=1; no_iter <= 2; no_iter++) {
            pt_old      = pt;
            dentropy    = gsw_entropy_part(sa,pt_old,p_ref) - true_entropy_part;
            pt          = pt_old - dentropy/dentropy_dt;
            ptm         = 0.5*(pt + pt_old);
            dentropy_dt = -gsw_gibbs(n0,n2,n0,sa,ptm,p_ref);
            pt          = pt_old - dentropy/dentropy_dt;
        }
        return (pt);
}
/*
! =========================================================================
elemental function gsw_pt_from_t_ice (t, p, p_ref)
! =========================================================================
!
!  Calculates potential temperature of ice Ih with the general reference
!  pressure, p_ref, from in-situ temperature, t.
!
!  A faster gsw routine exists if p_ref is indeed zero dbar.  This routine
!  is "gsw_pt0_from_t_ice(t,p)".
!
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  p_ref  =  reference pressure                                    [ dbar ]
!--------------------------------------------------------------------------
*/
double
gsw_pt_from_t_ice(double t, double p, double p_ref)
{
        GSW_TEOS10_CONSTANTS;
        int     number_of_iterations;
        double  dentropy, dentropy_dt, dp,
                pt_ice, pt_ice_old, ptm_ice, true_entropy,

                p1 = -2.259745637898635e-4,
                p2 =  1.486236778150360e-9,
                p3 =  6.257869607978536e-12,
                p4 = -5.253795281359302e-7,
                p5 =  6.752596995671330e-9,
                p6 =  2.082992190070936e-11,

                q1 = -5.849191185294459e-15,
                q2 =  9.330347971181604e-11,
                q3 =  3.415888886921213e-13,
                q4 =  1.064901553161811e-12,
                q5 = -1.454060359158787e-10,
                q6 = -5.323461372791532e-13;
                /*This is the starting polynomial for pt of ice Ih.*/

        dp = p - p_ref;

        pt_ice = t + dp*(p1 + (p + p_ref)*(p2 + p3*t) + t*(p4 + t*(p5 + p6*t)));

        if (pt_ice < -gsw_t0) pt_ice = -gsw_t0;

        if (pt_ice < -273.0) pt_ice = pt_ice + 0.05;
        /*
        ! we add 0.05 to the initial estimate of pt_ice at temps less than
        ! -273 to ensure that it is never less than -273.15.
        */
        dentropy_dt = -gsw_gibbs_ice(2,0,pt_ice,p_ref);

        true_entropy = -gsw_gibbs_ice_part_t(t,p);

        for (number_of_iterations = 1; number_of_iterations <= 3;
            number_of_iterations++) {
            pt_ice_old = pt_ice;
            dentropy = -gsw_gibbs_ice_part_t(pt_ice_old,p_ref) - true_entropy;
            pt_ice = pt_ice_old - dentropy/dentropy_dt;
            ptm_ice = 0.5*(pt_ice + pt_ice_old);
            dentropy_dt = -gsw_gibbs_ice(2,0,ptm_ice,p_ref);
            pt_ice = pt_ice_old - dentropy/dentropy_dt;
        }

        if (pt_ice < -273.0) {

            pt_ice = t + (p - p_ref)*(q1 + (p + p_ref)*(q2 + q3*t)
                   + t*(q4 + t*(q5 + q6*t)));

            dentropy_dt = -gsw_gibbs_ice(2,0,pt_ice+0.01,p_ref);
        /*
        ! we add 0.01 to the initial estimate of pt_ice used in the derivative
        ! to ensure that it is never less than -273.15 because the derivative
        ! approaches zero at absolute zero.
        */
            for (number_of_iterations = 1; number_of_iterations <= 3;
                number_of_iterations++) {
                pt_ice_old = pt_ice;
                dentropy = -gsw_gibbs_ice_part_t(pt_ice_old,p_ref)
                                - true_entropy;
                pt_ice = pt_ice_old - dentropy/dentropy_dt;
                ptm_ice = 0.5*(pt_ice + pt_ice_old);
                ptm_ice = ptm_ice + 0.01;
                dentropy_dt = -gsw_gibbs_ice(2,0,ptm_ice,p_ref);
                pt_ice = pt_ice_old - dentropy/dentropy_dt;
            }

        }
/*
! For temperatures less than -273.1 degsC the maximum error is less than
! 2x10^-7 degsC. For temperatures between -273.1 and 273 the maximum error
! is less than 8x10^-8 degsC, and for temperatures greater than -273 degsC the
! maximum error is 1.5x10^-12 degsC.  These errors are over the whole
! ocean depths with both p and pref varying independently between 0 and
! 10,000 dbar, while the in-situ temperature varied independently between
! -273.15 and +2 degsC.
*/
        return (pt_ice);
}
/*
!==========================================================================
elemental subroutine gsw_pt_second_derivatives (sa, ct, pt_sa_sa, &
                                                pt_sa_ct, pt_ct_ct)
! =========================================================================
!
!  Calculates the following three second-order derivatives of potential
!  temperature (the regular potential temperature which has a reference
!  sea pressure of 0 dbar),
!   (1) pt_SA_SA, the second derivative with respect to Absolute Salinity
!       at constant Conservative Temperature,
!   (2) pt_SA_CT, the derivative with respect to Conservative Temperature
!       and Absolute Salinity, and
!   (3) pt_CT_CT, the second derivative with respect to Conservative
!       Temperature at constant Absolute Salinity.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!
!  pt_SA_SA  =  The second derivative of potential temperature (the
!               regular potential temperature which has reference sea
!               pressure of 0 dbar) with respect to Absolute Salinity
!               at constant Conservative Temperature.
!               pt_SA_SA has units of:                     [ K/((g/kg)^2) ]
!  pt_SA_CT  =  The derivative of potential temperature with respect
!               to Absolute Salinity and Conservative Temperature.
!               pt_SA_CT has units of:                         [ 1/(g/kg) ]
!  pt_CT_CT  =  The second derivative of potential temperature (the
!               regular one with p_ref = 0 dbar) with respect to
!               Conservative Temperature at constant SA.
!               pt_CT_CT has units of:                              [ 1/K ]
!--------------------------------------------------------------------------
*/
void
gsw_pt_second_derivatives (double sa, double ct, double *pt_sa_sa,
        double *pt_sa_ct, double *pt_ct_ct)
{
        double  ct_l, ct_u, pt_ct_l, pt_ct_u, pt_sa_l, pt_sa_u, sa_l, sa_u,
                delta, dct = 1e-2, dsa = 1e-3;

        if (pt_sa_sa != NULL) {

            sa_u = sa + dsa;
            sa_l = sa - dsa;
            if (sa_l < 0.0) {
                sa_l = 0.0;
                delta = sa_u;
            } else {
                delta = 2 * dsa;
            }

            gsw_pt_first_derivatives(sa_l,ct,&pt_sa_l,NULL);
            gsw_pt_first_derivatives(sa_u,ct,&pt_sa_u,NULL);

            *pt_sa_sa = (pt_sa_u - pt_sa_l)/delta;

        }

        if (pt_sa_ct != NULL || pt_ct_ct != NULL) {

            ct_l = ct - dct;
            ct_u = ct + dct;
            delta = 2 * dct;

            if ((pt_sa_ct != NULL) && (pt_ct_ct != NULL)) {

                gsw_pt_first_derivatives(sa,ct_l,&pt_sa_l,&pt_ct_l);
                gsw_pt_first_derivatives(sa,ct_u,&pt_sa_u,&pt_ct_u);

                *pt_sa_ct = (pt_sa_u - pt_sa_l)/delta;
                *pt_ct_ct = (pt_ct_u - pt_ct_l)/delta;

            } else if ((pt_sa_ct != NULL) && (pt_ct_ct == NULL)) {

                gsw_pt_first_derivatives(sa,ct_l,&pt_sa_l,NULL);
                gsw_pt_first_derivatives(sa,ct_u,&pt_sa_u,NULL);

                *pt_sa_ct = (pt_sa_u - pt_sa_l)/delta;

            } else if ((pt_sa_ct == NULL) && (pt_ct_ct != NULL)) {

                gsw_pt_first_derivatives(sa,ct_l,NULL,&pt_ct_l);
                gsw_pt_first_derivatives(sa,ct_u,NULL,&pt_ct_u);

                *pt_ct_ct = (pt_ct_u - pt_ct_l)/delta;
            }
        }
}
/*
!--------------------------------------------------------------------------
! density and enthalpy, based on the 48-term expression for density
!--------------------------------------------------------------------------

!==========================================================================
function gsw_rho(sa,ct,p)
!==========================================================================

!  Calculates in-situ density from Absolute Salinity and Conservative
!  Temperature, using the computationally-efficient expression for
!  specific volume in terms of SA, CT and p (Roquet et al., 2014).
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature (ITS-90)               [deg C]
! p      : sea pressure                                    [dbar]
!          ( i.e. absolute pressure - 10.1325 dbar )
!
! rho    : in-situ density                                 [kg/m]
*/
double
gsw_rho(double sa, double ct, double p)
{
        return (1.0/gsw_specvol(sa,ct,p));
}
/*
!==========================================================================
elemental subroutine gsw_rho_alpha_beta (sa, ct, p, rho, alpha, beta)
!==========================================================================
!
!  Calculates in-situ density, the appropriate thermal expansion coefficient
!  and the appropriate saline contraction coefficient of seawater from
!  Absolute Salinity and Conservative Temperature.  This function uses the
!  computationally-efficient expression for specific volume in terms of
!  SA, CT and p (Roquet et al., 2014).
!
!  Note that potential density (pot_rho) with respect to reference pressure
!  p_ref is obtained by calling this function with the pressure argument
!  being p_ref as in [pot_rho, ~, ~] = gsw_rho_alpha_beta(SA,CT,p_ref).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  rho    =  in-situ density                                       [ kg/m ]
!  alpha  =  thermal expansion coefficient                          [ 1/K ]
!            with respect to Conservative Temperature
!  beta   =  saline (i.e. haline) contraction                      [ kg/g ]
!            coefficient at constant Conservative Temperature
!--------------------------------------------------------------------------
*/
void
gsw_rho_alpha_beta (double sa, double ct, double p, double *rho, double *alpha,
                        double *beta)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  v, v_ct_part, v_sa_part, xs, ys, z;

        xs = sqrt(gsw_sfac*sa + offset);
        ys = ct*0.025;
        z = p*1e-4;

        v = v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050
    + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 + xs*(v140
    + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220 + xs*(v230 + v240*xs)))
    + ys*(v300 + xs*(v310 + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410
    + v420*xs) + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011
    + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111
    + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211 + xs*(v221
    + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs) + ys*(v401 + v411*xs
    + v501*ys)))) + z*(v002 + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs)))
    + ys*(v102 + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212
    + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013
    + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004 + v014*xs + v104*ys
    + z*(v005 + v006*z)))));

        if (rho != NULL)
            *rho = 1.0/v;

        if (alpha != NULL) {

            v_ct_part = a000 + xs*(a100 + xs*(a200 + xs*(a300
                + xs*(a400 + a500*xs))))
                + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 + a410*xs)))
                + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs)) + ys*(a030
                + xs*(a130 + a230*xs) + ys*(a040 + a140*xs + a050*ys ))))
                + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 + a401*xs)))
                + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) + ys*(a021
                + xs*(a121 + a221*xs) + ys*(a031 + a131*xs + a041*ys)))
                + z*(a002 + xs*(a102 + xs*(a202 + a302*xs)) + ys*(a012
                + xs*(a112 + a212*xs) + ys*(a022 + a122*xs + a032*ys))
                + z*(a003 + a103*xs + a013*ys + a004*z)));

            *alpha = 0.025*v_ct_part/v;

        }

        if (beta != NULL) {

            v_sa_part = b000 + xs*(b100 + xs*(b200 + xs*(b300
                + xs*(b400 + b500*xs))))
                + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 + b410*xs)))
                + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs)) + ys*(b030
                + xs*(b130 + b230*xs) + ys*(b040 + b140*xs + b050*ys))))
                + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 + b401*xs)))
                + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) + ys*(b021
                + xs*(b121 + b221*xs) + ys*(b031 + b131*xs + b041*ys)))
                + z*(b002 + xs*(b102 + xs*(b202 + b302*xs))+ ys*(b012
                + xs*(b112 + b212*xs) + ys*(b022 + b122*xs + b032*ys))
                + z*(b003 +  b103*xs + b013*ys + b004*z)));

            *beta = -v_sa_part*0.5*gsw_sfac/(v*xs);

        }
}
/*
!==========================================================================
subroutine gsw_rho_first_derivatives(sa, ct, p, drho_dsa, drho_dct, drho_dp)
!==========================================================================

!  Calculates the three (3) partial derivatives of in situ density with
!  respect to Absolute Salinity, Conservative Temperature and pressure.
!  Note that the pressure derivative is done with respect to pressure in
!  Pa, not dbar.  This function uses the computationally-efficient expression
!  for specific volume in terms of SA, CT and p (Roquet et al., 2014).
!
! sa        : Absolute Salinity                               [g/kg]
! ct        : Conservative Temperature                        [deg C]
! p         : sea pressure                                    [dbar]
! drho_dsa  : partial derivatives of density                  [kg^2/(g m^3)]
!             with respect to Absolute Salinity
! drho_dct  : partial derivatives of density                  [kg/(K m^3)]
!             with respect to Conservative Temperature
! drho_dp   : partial derivatives of density                  [kg/(Pa m^3)]
!             with respect to pressure in Pa
*/
void
gsw_rho_first_derivatives(double sa, double ct, double p,
        double *drho_dsa, double *drho_dct, double *drho_dp)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  rho2, v_ct, v_p, v_sa, xs, ys, z, v;

        xs      = sqrt(gsw_sfac*sa + offset);
        ys      = ct*0.025;
        z       = p*1e-4;

        v = v000
    + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050
    + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 + xs*(v140
    + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220 + xs*(v230 + v240*xs)))
    + ys*(v300 + xs*(v310 + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410
    + v420*xs) + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011
    + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111
    + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211 + xs*(v221
    + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs) + ys*(v401 + v411*xs
    + v501*ys)))) + z*(v002 + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs)))
    + ys*(v102 + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212
    + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013
    + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004 + v014*xs + v104*ys
    + z*(v005 + v006*z)))));

        rho2 = pow(1.0/v, 2.0);

        if (drho_dsa != NULL) {

            v_sa = b000
           + xs*(b100 + xs*(b200 + xs*(b300 + xs*(b400 + b500*xs))))
           + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 + b410*xs)))
           + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs)) + ys*(b030
           + xs*(b130 + b230*xs) + ys*(b040 + b140*xs + b050*ys))))
           + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 + b401*xs)))
           + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) + ys*(b021
           + xs*(b121 + b221*xs) + ys*(b031 + b131*xs + b041*ys)))
           + z*(b002 + xs*(b102 + xs*(b202 + b302*xs))+ ys*(b012
           + xs*(b112 + b212*xs) + ys*(b022 + b122*xs + b032*ys))
           + z*(b003 +  b103*xs + b013*ys + b004*z)));

            *drho_dsa = -rho2*0.5*gsw_sfac*v_sa/xs;
        }

        if (drho_dct != NULL) {

            v_ct = a000
           + xs*(a100 + xs*(a200 + xs*(a300 + xs*(a400 + a500*xs))))
           + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 + a410*xs)))
           + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs)) + ys*(a030
           + xs*(a130 + a230*xs) + ys*(a040 + a140*xs + a050*ys ))))
           + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 + a401*xs)))
           + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) + ys*(a021
           + xs*(a121 + a221*xs) + ys*(a031 + a131*xs + a041*ys)))
           + z*(a002 + xs*(a102 + xs*(a202 + a302*xs)) + ys*(a012
           + xs*(a112 + a212*xs) + ys*(a022 + a122*xs + a032*ys))
           + z*(a003 + a103*xs + a013*ys + a004*z)));

            *drho_dct = -rho2*0.025*v_ct;
        }

        if (drho_dp != NULL) {

            v_p = c000
          + xs*(c100 + xs*(c200 + xs*(c300 + xs*(c400 + c500*xs))))
          + ys*(c010 + xs*(c110 + xs*(c210 + xs*(c310 + c410*xs))) + ys*(c020
          + xs*(c120 + xs*(c220 + c320*xs)) + ys*(c030 + xs*(c130 + c230*xs)
          + ys*(c040 + c140*xs + c050*ys)))) + z*(c001 + xs*(c101 + xs*(c201
          + xs*(c301 + c401*xs))) + ys*(c011 + xs*(c111 + xs*(c211 + c311*xs))
          + ys*(c021 + xs*(c121 + c221*xs) + ys*(c031 + c131*xs + c041*ys)))
          + z*(c002 + xs*(c102 + c202*xs) + ys*(c012 + c112*xs + c022*ys)
          + z*(c003 + c103*xs + c013*ys + z*(c004 + c005*z))));

            *drho_dp = 1e-4*pa2db*-rho2*v_p;
        }

        return;
}
/*
!==========================================================================
elemental subroutine gsw_rho_first_derivatives_wrt_enthalpy (sa, ct, p, &
                                                             rho_sa, rho_h)
! =========================================================================
!
!  Calculates two first-order derivatives of specific volume (v).
!  Note that this function uses the using the computationally-efficient
!  expression for specific volume (Roquet et al., 2014).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  rho_SA =  The first derivative of rho with respect to
!              Absolute Salinity at constant CT & p.    [ J/(kg (g/kg)^2) ]
!  rho_h  =  The first derivative of rho with respect to
!              SA and CT at constant p.                  [ J/(kg K(g/kg)) ]
!--------------------------------------------------------------------------
*/
void
gsw_rho_first_derivatives_wrt_enthalpy (double sa, double ct, double p,
        double *rho_sa, double *rho_h)
{
        double  rec_v2, v_h=0.0, v_sa;

        if ((rho_sa != NULL) && (rho_h != NULL)) {

            gsw_specvol_first_derivatives_wrt_enthalpy(sa,ct,p,&v_sa,&v_h);

        } else if (rho_sa != NULL) {

            gsw_specvol_first_derivatives_wrt_enthalpy(sa,ct,p,&v_sa,NULL);

        } else if (rho_h != NULL) {

            gsw_specvol_first_derivatives_wrt_enthalpy(sa,ct,p,NULL,&v_h);

        }

        rec_v2 = pow(1.0/gsw_specvol(sa,ct,p), 2);

        if (rho_sa != NULL) *rho_sa = -v_sa*rec_v2;

        if (rho_h != NULL) *rho_h = -v_h*rec_v2;
}
/*
!==========================================================================
elemental function gsw_rho_ice (t, p)
!==========================================================================
!
!  Calculates in-situ density of ice from in-situ temperature and pressure.
!  Note that the output, rho_ice, is density, not density anomaly;  that
!  is, 1000 kg/m^3 is not subtracted from it.
!
!  t   =  in-situ temperature (ITS-90)                            [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  rho_ice  =  in-situ density of ice (not density anomaly)      [ kg/m^3 ]
!--------------------------------------------------------------------------
*/
double
gsw_rho_ice(double t, double p)
{
        return (1.0/gsw_gibbs_ice(0,1,t,p));
}
/*
!==========================================================================
elemental subroutine gsw_rho_second_derivatives (sa, ct, p, rho_sa_sa, &
                                  rho_sa_ct, rho_ct_ct, rho_sa_p, rho_ct_p)
!==========================================================================
!
!  Calculates five second-order derivatives of rho. Note that this function
!  uses the computationally-efficient expression for specific
!  volume (Roquet et al., 2014).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  rho_SA_SA = The second-order derivative of rho with respect to
!              Absolute Salinity at constant CT & p.    [ J/(kg (g/kg)^2) ]
!  rho_SA_CT = The second-order derivative of rho with respect to
!              SA and CT at constant p.                  [ J/(kg K(g/kg)) ]
!  rho_CT_CT = The second-order derivative of rho with respect to CT at
!              constant SA & p
!  rho_SA_P  = The second-order derivative with respect to SA & P at
!              constant CT.
!  rho_CT_P  = The second-order derivative with respect to CT & P at
!              constant SA.
!--------------------------------------------------------------------------
*/
void
gsw_rho_second_derivatives(double sa, double ct, double p, double *rho_sa_sa,
        double *rho_sa_ct, double *rho_ct_ct, double *rho_sa_p,
        double *rho_ct_p)
{
        double  rec_v, rec_v2, rec_v3, v_ct, v_ct_ct, v_ct_p, v_p, v_sa,
                v_sa_ct, v_sa_p, v_sa_sa;

        gsw_specvol_first_derivatives(sa,ct,p,&v_sa,&v_ct,&v_p);
        gsw_specvol_second_derivatives(sa,ct,p,&v_sa_sa,&v_sa_ct,&v_ct_ct,
                                    &v_sa_p,&v_ct_p);

        rec_v = 1.0/gsw_specvol(sa,ct,p);
        rec_v2 = pow(rec_v, 2);
        rec_v3 = rec_v2*rec_v;

        if (rho_sa_sa != NULL)
            *rho_sa_sa = -v_sa_sa*rec_v2 + 2.0*v_sa*v_sa*rec_v3;

        if (rho_sa_ct != NULL)
            *rho_sa_ct = -v_sa_ct*rec_v2 + 2.0*v_sa*v_ct*rec_v3;

        if (rho_ct_ct != NULL)
            *rho_ct_ct = -v_ct_ct*rec_v2 + 2.0*v_ct*v_ct*rec_v3;

        if (rho_sa_p != NULL)
            *rho_sa_p = -v_sa_p*rec_v2 + 2.0*v_sa*v_p*rec_v3;

        if (rho_ct_p != NULL)
            *rho_ct_p = -v_ct_p*rec_v2 + 2.0*v_ct*v_p*rec_v3;
}
/*
!==========================================================================
elemental subroutine gsw_rho_second_derivatives_wrt_enthalpy (sa, ct, p, &
                                              rho_sa_sa, rho_sa_h, rho_h_h)
! =========================================================================
!
!  Calculates three second-order derivatives of rho with respect to enthalpy.
!  Note that this function uses the using the computationally-efficient
!  expression for specific volume (Roquet et al., 2014).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  rho_SA_SA = The second-order derivative of rho with respect to
!              Absolute Salinity at constant h & p.     [ J/(kg (g/kg)^2) ]
!  rho_SA_h  = The second-order derivative of rho with respect to
!              SA and h at constant p.                   [ J/(kg K(g/kg)) ]
!  rho_h_h   = The second-order derivative of rho with respect to h at
!              constant SA & p
!--------------------------------------------------------------------------
*/
void
gsw_rho_second_derivatives_wrt_enthalpy(double sa, double ct, double p,
        double *rho_sa_sa, double *rho_sa_h, double *rho_h_h)
{
        double  rec_v, rec_v2, rec_v3, v_h, v_h_h, v_sa, v_sa_h, v_sa_sa,
                *pv_sa, *pv_h, *pv_sa_sa, *pv_sa_h, *pv_h_h;

        pv_sa   = ((rho_sa_sa != NULL) || (rho_sa_h != NULL)) ? &v_sa : NULL;
        pv_h    = ((rho_sa_h != NULL) || (rho_h_h != NULL)) ?  &v_h : NULL;

        gsw_specvol_first_derivatives_wrt_enthalpy(sa,ct,p,pv_sa,pv_h);

        pv_sa_sa = ((rho_sa_sa != NULL)) ? &v_sa_sa : NULL;
        pv_sa_h  = ((rho_sa_h != NULL)) ? &v_sa_h : NULL;
        pv_h_h   = ((rho_h_h != NULL)) ? &v_h_h : NULL;

        gsw_specvol_second_derivatives_wrt_enthalpy(sa,ct,p,
                                                pv_sa_sa,pv_sa_h,pv_h_h);

        rec_v = 1.0/gsw_specvol(sa,ct,p);
        rec_v2 = rec_v*rec_v;
        rec_v3 = rec_v2*rec_v;

        if (rho_sa_sa != NULL)
            *rho_sa_sa = -v_sa_sa*rec_v2 + 2.0*v_sa*v_sa*rec_v3;

        if (rho_sa_h != NULL)
            *rho_sa_h = -v_sa_h*rec_v2 + 2.0*v_sa*v_h*rec_v3;

        if (rho_h_h != NULL)
            *rho_h_h = -v_h_h*rec_v2 + 2.0*v_h*v_h*rec_v3;
}
/*
!==========================================================================
function gsw_rho_t_exact(sa,t,p)
!==========================================================================

! Calculates in-situ density of seawater from Absolute Salinity and
! in-situ temperature.
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
!
! gsw_rho_t_exact : in-situ density                        [kg/m^3]
*/
double
gsw_rho_t_exact(double sa, double t, double p)
{
        int     n0=0, n1=1;

        return (1.0/gsw_gibbs(n0,n0,n1,sa,t,p));
}
/*
!==========================================================================
pure subroutine gsw_rr68_interp_sa_ct (sa, ct, p, p_i, sa_i, ct_i)
!==========================================================================
!
!  Interpolate Absolute Salinity and Conservative Temperature values to
!  arbitrary pressures using the Reiniger and Ross (1968) interpolation
!  scheme.
!  Note that this interpolation scheme requires at least four observed
!  bottles on the cast.
!
!  SA   =  Absolute Salinity                                  [ g/kg ]
!  CT   =  Conservative Temperature (ITS-90)                 [ deg C ]
!  p    =  sea pressure                                       [ dbar ]
!           ( i.e. absolute pressure - 10.1325 dbar )
!  p_i  =  pressures to interpolate to.
!
!  SA_i = interpolated SA values at pressures p_i.
!  CT_i = interpolated CT values at pressures p_i.
!--------------------------------------------------------------------------
*/
static void rr68_interp_section(int sectnum, double *sa, double *ct, double *p,
        int mp, int nsect, double *ip_sect,
        int *ip_isect, double *p_i, double *sa_i, double *ct_i);
/* forward reference */

void
gsw_rr68_interp_sa_ct(double *sa, double *ct, double *p, int mp, double *p_i,
        int mp_i, double *sa_i, double *ct_i)
{
        int     i, j, nshallow, ncentral, ndeep,
                *ip, *ip_i, *ip_ishallow, *ip_icentral, *ip_ideep;
        char    *shallow, *central, *deep;
        double  *ip_shallow, *ip_central, *ip_deep, *dp, *p_ii;

        if (mp < 4) {
            /* need at least four bottles to perform this interpolation */
            ct_i[0] = sa_i[0] = GSW_INVALID_VALUE;
            return;
        }

        dp = (double *) malloc(mp*sizeof (double));
        for (i=1; i<mp; i++) {
            if ((dp[i-1] = (p[i] - p[i-1])) <= 0.0) {
                free(dp);
                ct_i[0] = sa_i[0] = GSW_INVALID_VALUE;
                return;
            }
        }

        shallow = (char *) malloc(3*mp_i*sizeof (char));
        central = shallow+mp_i; deep = central+mp_i;
        nshallow=ncentral=ndeep=0;
        memset(shallow, 0, 3*mp_i*sizeof (char));
        for (i=0; i<mp_i; i++) {
            if (p_i[i] >= p[0] && p_i[i] <= p[1]) {
                nshallow++;
                shallow[i] = 1;
            }
            if (p_i[i] >= p[1] && p_i[i] <= p[mp-2]) {
                ncentral++;
                central[i] = 1;
            }
            if (p_i[i] >= p[mp-2] && p_i[i] <= p[mp-1]) {
                ndeep++;
                deep[i] = 1;
            }
        }

        if ((nshallow == 0) || (ncentral == 0) || (ndeep == 0)) {
            free(shallow); free(dp);
            ct_i[0] = sa_i[0] = GSW_INVALID_VALUE;
            return;
        }

        ip = (int *) malloc((mp+mp_i)*sizeof (int)); ip_i = ip+mp;
        for (i=0; i<mp; i++)
            ip[i] = i;
        for (i=0; i<mp_i; i++)
            ip_i[i] = i;

        ip_ishallow = (int *) malloc((nshallow+ncentral+ndeep)*sizeof (int));
        ip_icentral = ip_ishallow+nshallow; ip_ideep = ip_icentral+ncentral;
        ip_shallow = (double *) malloc(2*(nshallow+ncentral+ndeep)*sizeof (double));
        ip_central = ip_shallow+nshallow; ip_deep = ip_central+ncentral;
        p_ii = ip_deep+ndeep;
        /*
        ! Calculate the 2 outer extrapolated values and the inner
        ! interpolated values
        */
        for (i=j=0; i<mp_i; i++) {
            if (central[i]) {
                ip_icentral[j] = ip_i[i];
                j++;
            }
        }
        for (i=0; i<ncentral; i++)
            p_ii[i] = p_i[ip_icentral[i]];
        gsw_util_interp1q_int(mp,p,ip,ncentral,p_ii,ip_central);
        rr68_interp_section(0,sa,ct,p,mp,ncentral,ip_central,ip_icentral,
                                p_i,sa_i,ct_i);

        for (i=j=0; i<mp_i; i++) {
            if (shallow[i]) {
                ip_ishallow[j] = ip_i[i];
                j++;
            }
        }
        for (i=0; i<nshallow; i++)
            p_ii[i] = p_i[ip_ishallow[i]];
        gsw_util_interp1q_int(mp,p,ip,nshallow,p_ii,ip_shallow);
        rr68_interp_section(-1,sa,ct,p,mp,nshallow,ip_shallow,ip_ishallow,
                                p_i,sa_i,ct_i);

        for (i=j=0; i<mp_i; i++) {
            if (deep[i]) {
                ip_ideep[j] = ip_i[i];
                j++;
            }
        }
        for (i=0; i<ndeep; i++)
            p_ii[i] = p_i[ip_ideep[i]];
        gsw_util_interp1q_int(mp,p,ip,ndeep,p_ii,ip_deep);
        rr68_interp_section(1,sa,ct,p,mp,ndeep,ip_deep,ip_ideep,p_i,sa_i,ct_i);

        /*
        ! Insert any observed bottles that are at the required interpolated
        ! pressures
        */
        for (i=0; i<mp_i; i++) {
            for (j=0; j<mp; j++) {
                if (p_i[i] == p[j]) {
                    sa_i[i] = sa[j];
                    ct_i[i] = ct[j];
                }
            }
        }
        free(ip_shallow); free(ip_ishallow); free(ip); free(shallow); free(dp);
}

/*
pure subroutine rr68_interp_section (sectnum, ip_sect, ip_isect, sa_i, ct_i)
*/
static void
rr68_interp_section(int sectnum, double *sa, double *ct, double *p, int mp,
        int nsect, double *ip_sect, int *ip_isect, double *p_i, double *sa_i,
        double *ct_i)
{
        int     i, *ip_1, *ip_2, *ip_3, *ip_4;
        double  m, *ct_12, *ct_13, *ct_23, *ct_34, ctp1,
                ctp2, *ct_ref, ctref_denom,
                ct_ref_minus_ctp1, ct_ref_minus_ctp2,
                ctref_num,
                gamma1_23, gamma1_24, gamma2_31,
                gamma2_34, gamma2_41, gamma3_12,
                gamma3_42, gamma4_12, gamma4_23,
                *sa_12, *sa_13, *sa_23, *sa_34, sap1,
                sap2, *sa_ref, saref_denom,
                sa_ref_minus_sap1, sa_ref_minus_sap2,
                saref_num, *p_ii;

        ip_1 = (int *) malloc(4*nsect*sizeof (int)); ip_2 = ip_1+nsect;
        ip_3 = ip_2+nsect; ip_4 = ip_3+nsect;

        ct_12 = (double *) malloc(12*nsect*sizeof (double));
        sa_12   = ct_12 +  1*nsect;
        sa_13   = ct_12 +  2*nsect;
        sa_23   = ct_12 +  3*nsect;
        sa_34   = ct_12 +  4*nsect;
        sa_ref  = ct_12 +  5*nsect;
        ct_13   = ct_12 +  6*nsect;
        ct_23   = ct_12 +  7*nsect;
        ct_34   = ct_12 +  8*nsect;
        ct_ref  = ct_12 +  9*nsect;
        p_ii    = ct_12 + 10*nsect;

        if (sectnum < 0) {       /* shallow */
            for (i=0; i<nsect; i++) {
                ip_1[i] = floor(ip_sect[i]);
                ip_2[i] = ceil(ip_sect[i]);
                if (ip_1[i] == ip_2[i])
                    ip_2[i] = ip_1[i] + 1;
                ip_3[i] = ip_2[i] + 1;
                ip_4[i] = ip_3[i] + 1;
            }
        } else if (sectnum == 0) {  /* central */
            for (i=0; i<nsect; i++) {
                ip_2[i] = floor(ip_sect[i]);
                ip_3[i] = ceil(ip_sect[i]);
                if (ip_2[i] == ip_3[i])
                    ip_2[i] = ip_3[i] - 1;
                ip_1[i] = ip_2[i] - 1;
                if (ip_1[i] < 0) {
                    ip_1[i] = 0;
                    ip_2[i] = 1;
                    ip_3[i] = 2;
                }
                ip_4[i] = ip_3[i] + 1;
            }
        } else if (sectnum > 0) {  /* deep */
            for (i=0; i<nsect; i++) {
                ip_1[i] = ceil(ip_sect[i]);
                ip_2[i] = floor(ip_sect[i]);
                if (ip_1[i] == ip_2[i])
                    ip_2[i] = ip_1[i] - 1;
                ip_3[i] = ip_2[i] - 1;
                ip_4[i] = ip_3[i] - 1;
            }
        }

        for (i=0; i<nsect; i++)
            p_ii[i] = p_i[ip_isect[i]];

        /* eqn (3d) */
        for (i=0; i<nsect; i++) {
            sa_34[i] = sa[ip_3[i]] + ((sa[ip_4[i]] - sa[ip_3[i]])
                *(p_ii[i] - p[ip_3[i]])/ (p[ip_4[i]] - p[ip_3[i]]));
            ct_34[i] = ct[ip_3[i]] + ((ct[ip_4[i]] - ct[ip_3[i]])
                *(p_ii[i] - p[ip_3[i]])/ (p[ip_4[i]] - p[ip_3[i]]));
        }
        /*
        ! Construct the Reiniger & Ross reference curve equation.
        ! m = the power variable
        */
        m = 1.7;

        if (sectnum == 0) {

            gsw_linear_interp_sa_ct(sa,ct,p,mp,p_ii,nsect,sa_23,ct_23);

            /* eqn (3a) */
            for (i=0; i<nsect; i++) {
                sa_12[i] = sa[ip_1[i]] + ((sa[ip_2[i]] - sa[ip_1[i]])
                        *(p_ii[i] - p[ip_1[i]])/ (p[ip_2[i]] - p[ip_1[i]]));
                ct_12[i] = ct[ip_1[i]] + ((ct[ip_2[i]] - ct[ip_1[i]])
                        *(p_ii[i] - p[ip_1[i]])/ (p[ip_2[i]] - p[ip_1[i]]));

                saref_num = (pow(fabs(sa_23[i]-sa_34[i]),m))*sa_12[i]
                          + (pow(fabs(sa_12[i]-sa_23[i]),m))*sa_34[i];
                ctref_num = (pow(fabs(ct_23[i]-ct_34[i]),m))*ct_12[i]
                          + (pow(fabs(ct_12[i]-ct_23[i]),m))*ct_34[i];

                saref_denom = pow(fabs(sa_23[i]-sa_34[i]),m)
                            + pow(fabs(sa_12[i]-sa_23[i]),m);
                ctref_denom = pow(fabs(ct_23[i]-ct_34[i]),m)
                            + pow(fabs(ct_12[i]-ct_23[i]),m);

                if (saref_denom == 0.0) {
                    sa_23[i] = sa_23[i] + 1.0e-6;
                    saref_num = (pow(fabs(sa_23[i]-sa_34[i]),m))*sa_12[i]
                              + (pow(fabs(sa_12[i]-sa_23[i]),m))*sa_34[i];
                    saref_denom = pow(fabs(sa_23[i]-sa_34[i]),m)
                                + pow(fabs(sa_12[i]-sa_23[i]),m);
                }
                if (ctref_denom == 0.0) {
                    ct_23[i] = ct_23[i] + 1.0e-6;
                    ctref_num = (pow(fabs(ct_23[i]-ct_34[i]),m))*ct_12[i]
                              + (pow(fabs(ct_12[i]-ct_23[i]),m))*ct_34[i];
                    ctref_denom = pow(fabs(ct_23[i]-ct_34[i]),m)
                                + pow(fabs(ct_12[i]-ct_23[i]),m);
                }

                sa_ref[i] = 0.5*(sa_23[i] + (saref_num/saref_denom));
                ct_ref[i] = 0.5*(ct_23[i] + (ctref_num/ctref_denom));
            }

        } else {

            gsw_linear_interp_sa_ct(sa,ct,p,mp,p_ii,nsect,sa_12,ct_12);

            for (i=0; i<nsect; i++) {
                sa_13[i] = sa[ip_1[i]] + ((sa[ip_3[i]] - sa[ip_1[i]])
                        *(p_ii[i] - p[ip_1[i]])/ (p[ip_3[i]] - p[ip_1[i]]));
                ct_13[i] = ct[ip_1[i]] + ((ct[ip_3[i]] - ct[ip_1[i]])
                        *(p_ii[i] - p[ip_1[i]])/ (p[ip_3[i]] - p[ip_1[i]]));

                sa_23[i] = sa[ip_2[i]] + ((sa[ip_3[i]] - sa[ip_2[i]])
                        *(p_ii[i] - p[ip_2[i]])/ (p[ip_3[i]] - p[ip_2[i]]));
                ct_23[i] = ct[ip_2[i]] + ((ct[ip_3[i]] - ct[ip_2[i]])
                        *(p_ii[i] - p[ip_2[i]])/ (p[ip_3[i]] - p[ip_2[i]]));

                /* eqn (3a') */
                saref_num = (pow(fabs(sa_12[i]-sa_23[i]),m))*sa_34[i]
                          + (pow(fabs(sa_12[i]-sa_13[i]),m))*sa_23[i];
                ctref_num = (pow(fabs(ct_12[i]-ct_23[i]),m))*ct_34[i]
                          + (pow(fabs(ct_12[i]-ct_13[i]),m))*ct_23[i];

                saref_denom = pow(fabs(sa_12[i]-sa_23[i]),m)
                            + pow(fabs(sa_12[i]-sa_13[i]),m);
                ctref_denom = pow(fabs(ct_12[i]-ct_23[i]),m)
                            + pow(fabs(ct_12[i]-ct_13[i]),m);

                if (saref_denom == 0.0) {
                    sa_23[i] = sa_23[i] + 1.0e-6;
                    saref_num = (pow(fabs(sa_12[i]-sa_23[i]),m))*sa_34[i]
                              + (pow(fabs(sa_12[i]-sa_13[i]),m))*sa_23[i];
                    saref_denom = pow(fabs(sa_12[i]-sa_23[i]),m)
                                + pow(fabs(sa_12[i]-sa_13[i]),m);
                }
                if (ctref_denom == 0.0) {
                    ct_23[i] = ct_23[i] + 1.0e-6;
                    ctref_num = (pow(fabs(ct_12[i]-ct_23[i]),m))*ct_34[i]
                              + (pow(fabs(ct_12[i]-ct_13[i]),m))*ct_23[i];
                    ctref_denom = pow(fabs(ct_12[i]-ct_23[i]),m)
                                + pow(fabs(ct_12[i]-ct_13[i]),m);
                }

                sa_ref[i] = 0.5*(sa_12[i] + (saref_num/saref_denom));
                ct_ref[i] = 0.5*(ct_12[i] + (ctref_num/ctref_denom));
            }
        }

        for (i=0; i<nsect; i++) {
            /* eqn (3c) */
            gamma1_23 = ((p_ii[i] - p[ip_2[i]])*(p_ii[i] - p[ip_3[i]]))/
                        ((p[ip_1[i]] - p[ip_2[i]])*(p[ip_1[i]] - p[ip_3[i]]));
            gamma2_31 = ((p_ii[i] - p[ip_3[i]])*(p_ii[i] - p[ip_1[i]]))/
                        ((p[ip_2[i]] - p[ip_3[i]])*(p[ip_2[i]] - p[ip_1[i]]));
            gamma3_12 = ((p_ii[i] - p[ip_1[i]])*(p_ii[i] - p[ip_2[i]]))/
                        ((p[ip_3[i]] - p[ip_1[i]])*(p[ip_3[i]] - p[ip_2[i]]));

            if (sectnum == 0) {
                gamma2_34 = ((p_ii[i] - p[ip_3[i]])*(p_ii[i] - p[ip_4[i]]))/
                        ((p[ip_2[i]] - p[ip_3[i]])*(p[ip_2[i]] - p[ip_4[i]]));
                gamma3_42 = ((p_ii[i] - p[ip_4[i]])*(p_ii[i] - p[ip_2[i]]))/
                        ((p[ip_3[i]] - p[ip_4[i]])*(p[ip_3[i]] - p[ip_2[i]]));
                gamma4_23 = ((p_ii[i] - p[ip_2[i]])*(p_ii[i] - p[ip_3[i]]))/
                        ((p[ip_4[i]] - p[ip_2[i]])*(p[ip_4[i]] - p[ip_3[i]]));
            } else {
                gamma1_24 = ((p_ii[i] - p[ip_2[i]])*(p_ii[i] - p[ip_4[i]]))/
                        ((p[ip_1[i]] - p[ip_2[i]])*(p[ip_1[i]] - p[ip_4[i]]));
                gamma2_41 = ((p_ii[i] - p[ip_4[i]])*(p_ii[i] - p[ip_1[i]]))/
                        ((p[ip_2[i]] - p[ip_4[i]])*(p[ip_2[i]] - p[ip_1[i]]));
                gamma4_12 = ((p_ii[i] - p[ip_1[i]])*(p_ii[i] - p[ip_2[i]]))/
                        ((p[ip_4[i]] - p[ip_1[i]])*(p[ip_4[i]] - p[ip_2[i]]));
            }

            /* eqn (3b/3b') */
            sap1 = gamma1_23*sa[ip_1[i]] + gamma2_31*sa[ip_2[i]]
                        + gamma3_12*sa[ip_3[i]];
            ctp1 = gamma1_23*ct[ip_1[i]] + gamma2_31*ct[ip_2[i]]
                        + gamma3_12*ct[ip_3[i]];
            if (sectnum == 0) {
                sap2 = gamma2_34*sa[ip_2[i]] + gamma3_42*sa[ip_3[i]]
                        + gamma4_23*sa[ip_4[i]];
                ctp2 = gamma2_34*ct[ip_2[i]] + gamma3_42*ct[ip_3[i]]
                        + gamma4_23*ct[ip_4[i]];
            } else {
                sap2 = gamma1_24*sa[ip_1[i]] + gamma2_41*sa[ip_2[i]]
                        + gamma4_12*sa[ip_4[i]];
                ctp2 = gamma1_24*ct[ip_1[i]] + gamma2_41*ct[ip_2[i]]
                        + gamma4_12*ct[ip_4[i]];
            }

            /* eqn (3) */
            sa_ref_minus_sap1 = fabs(sa_ref[i] - sap1);
            sa_ref_minus_sap2 = fabs(sa_ref[i] - sap2);
            if (sa_ref_minus_sap1 == 0.0 && sa_ref_minus_sap2 == 0.0) {
                sa_ref[i] = sa_ref[i] + 1.0e-6;
                sa_ref_minus_sap1 = fabs(sa_ref[i] - sap1);
                sa_ref_minus_sap2 = fabs(sa_ref[i] - sap2);
            }

            ct_ref_minus_ctp1 = fabs(ct_ref[i] - ctp1);
            ct_ref_minus_ctp2 = fabs(ct_ref[i] - ctp2);
            if (ct_ref_minus_ctp1 == 0.0 && ct_ref_minus_ctp2 == 0.0) {
                ct_ref[i] = ct_ref[i] + 1.0e-6;
                ct_ref_minus_ctp1 = fabs(ct_ref[i] - ctp1);
                ct_ref_minus_ctp2 = fabs(ct_ref[i] - ctp2);
            }

            sa_i[ip_isect[i]] = (sa_ref_minus_sap1*sap2+sa_ref_minus_sap2*sap1)/
                                (sa_ref_minus_sap1 + sa_ref_minus_sap2);
            ct_i[ip_isect[i]] = (ct_ref_minus_ctp1*ctp2+ct_ref_minus_ctp2*ctp1)/
                                (ct_ref_minus_ctp1 + ct_ref_minus_ctp2);
        }
        free(ct_12); free(ip_1);
}
/*
!==========================================================================
elemental function gsw_sa_freezing_estimate (p, saturation_fraction, ct, t)
!==========================================================================
!
! Form an estimate of SA from a polynomial in CT and p
!
!--------------------------------------------------------------------------
*/
double
gsw_sa_freezing_estimate(double p, double saturation_fraction, double *ct,
        double *t)
{
        GSW_TEOS10_CONSTANTS;
        double  ctx, ctsat, sa,
                /*note that aa = 0.502500117621d0/35.16504*/
                aa = 0.014289763856964,
                bb = 0.057000649899720,

                p0  =  2.570124672768757e-1,
                p1  = -1.917742353032266e1,
                p2  = -1.413382858617969e-2,
                p3  = -5.427484830917552e-1,
                p4  = -4.126621135193472e-4,
                p5  = -4.176407833276121e-7,
                p6  =  4.688217641883641e-5,
                p7  = -3.039808885885726e-8,
                p8  = -4.990118091261456e-11,
                p9  = -9.733920711119464e-9,
                p10 = -7.723324202726337e-12,
                p11 =  7.121854166249257e-16,
                p12 =  1.256474634100811e-12,
                p13 =  2.105103897918125e-15,
                p14 =  8.663811778227171e-19;

        /*A very rough estimate of sa to get the saturated ct*/
        if (ct != NULL) {
            sa = max(-(*ct + 9e-4*p)/0.06, 0.0);
            ctx = *ct;
        } else if (t != NULL) {
            sa = max(-(*t + 9e-4*p)/0.06, 0.0);
            ctx = gsw_ct_from_t(sa,*t,p);
        } else {
            return (0.0);
        }
        /*
        ! CTsat is the estimated value of CT if the seawater were saturated with
        ! dissolved air, recognizing that it actually has the air fraction
        ! saturation_fraction; see McDougall, Barker and Feistel, 2014).
        */
        ctsat = ctx - (1.0-saturation_fraction)*
                (1e-3)*(2.4-aa*sa)*(1.0+bb*(1.0-sa/gsw_sso));

        return (p0 + p*(p2 + p4*ctsat + p*(p5 + ctsat*(p7 + p9*ctsat)
            + p*(p8  + ctsat*(p10 + p12*ctsat) + p*(p11 + p13*ctsat + p14*p))))
            + ctsat*(p1 + ctsat*(p3 + p6*p)));
}
/*
!==========================================================================
elemental function gsw_sa_freezing_from_ct (ct, p, saturation_fraction)
!==========================================================================
!
!  Calculates the Absolute Salinity of seawater at the freezing temperature.
!  That is, the output is the Absolute Salinity of seawater, with
!  Conservative Temperature CT, pressure p and the fraction
!  saturation_fraction of dissolved air, that is in equilibrium
!  with ice at the same in situ temperature and pressure.  If the input
!  values are such that there is no positive value of Absolute Salinity for
!  which seawater is frozen, the output is made a NaN.
!
!  CT  =  Conservative Temperature of seawater (ITS-90)           [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  saturation_fraction  =  the saturation fraction of dissolved air in
!                          seawater
!
!  sa_freezing_from_ct  =  Absolute Salinity of seawater when it freezes,
!                 for given input values of its Conservative Temperature,
!                 pressure and air saturation fraction.            [ g/kg ]
!--------------------------------------------------------------------------
*/
double
gsw_sa_freezing_from_ct(double ct, double p, double saturation_fraction)
{
        int     i_iter, number_of_iterations = 3;
        double  ct_freezing_zero_sa, f, ctfreezing_sa, sa, sa_mean, sa_old;
        /*
        ! This is the band of sa within +- 2.5 g/kg of sa = 0, which we treat
        ! differently in calculating the initial values of both SA and dCT_dSA.
        */
        double  sa_cut_off = 2.5;
        /*
        ! Find CT > CT_freezing_zero_SA.  If this is the case, the input values
        ! represent seawater that is not frozen (for any positive SA).
        */
        ct_freezing_zero_sa = gsw_ct_freezing(0.0,p,saturation_fraction);
        if (ct > ct_freezing_zero_sa)
            return (GSW_INVALID_VALUE);

        /*Form the first estimate of SA from a polynomial in CT and p*/
        sa = gsw_sa_freezing_estimate(p,saturation_fraction,&ct,NULL);
        if (sa < -sa_cut_off)
            return (GSW_INVALID_VALUE);
        /*
        ! Form the first estimate of CTfreezing_SA,
        ! the derivative of CT_freezing with respect to SA at fixed p.
        */
        sa = max(sa,0.0);
        gsw_ct_freezing_first_derivatives(sa,p,saturation_fraction,
                                               &ctfreezing_sa, NULL);
        /*
        ! For -SA_cut_off < SA < SA_cut_off, replace the above estimate of SA
        ! with one based on (CT_freezing_zero_SA - CT).
        */
        if (fabs(sa) < sa_cut_off)
            sa = (ct - ct_freezing_zero_sa)/ctfreezing_sa;
        /*
        !-----------------------------------------------------------------------
        ! Begin the modified Newton-Raphson method to solve
        ! f = (CT_freezing - CT) = 0 for SA.
        !-----------------------------------------------------------------------
        */
        for (i_iter = 1; i_iter <= number_of_iterations; i_iter++) {
            sa_old = sa;
            f = gsw_ct_freezing(sa,p,saturation_fraction) - ct;
            sa = sa_old - f/ctfreezing_sa;
            sa_mean = 0.5*(sa + sa_old);
            gsw_ct_freezing_first_derivatives(sa_mean,p,saturation_fraction,
                                        &ctfreezing_sa, NULL);
            sa = sa_old - f/ctfreezing_sa;
        }

        if (gsw_sa_p_inrange(sa,p))
            return (sa);
        return (GSW_INVALID_VALUE);
}
/*
!==========================================================================
elemental function gsw_sa_freezing_from_ct_poly (ct, p, saturation_fraction)
!==========================================================================
!
!  Calculates the Absolute Salinity of seawater at the freezing temperature.
!  That is, the output is the Absolute Salinity of seawater, with the
!  fraction saturation_fraction of dissolved air, that is in equilibrium
!  with ice at Conservative Temperature CT and pressure p.  If the input
!  values are such that there is no positive value of Absolute Salinity for
!  which seawater is frozen, the output is put equal to Nan.
!
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  saturation_fraction  =  the saturation fraction of dissolved air in
!                          seawater
!
!  sa_freezing_from_ct  =  Absolute Salinity of seawater when it freezes,
!                 for given input values of Conservative Temperature
!                 pressure and air saturation fraction.            [ g/kg ]
!--------------------------------------------------------------------------
*/
double
gsw_sa_freezing_from_ct_poly(double ct, double p, double saturation_fraction)
{
        int     i_iter, number_of_iterations = 2;
        double  ct_freezing, ct_freezing_zero_sa, dct_dsa, sa, sa_old, sa_mean;
        /*
        ! This is the band of sa within +- 2.5 g/kg of sa = 0, which we treat
        ! differently in calculating the initial values of both SA and dCT_dSA.
        */
        double  sa_cut_off = 2.5;
        /*
        ! Find CT > CT_freezing_zero_SA.  If this is the case, the input values
        ! represent seawater that is not frozen (at any positive SA).
        */
        ct_freezing_zero_sa = gsw_ct_freezing_poly(0.0,p,saturation_fraction);
        if (ct > ct_freezing_zero_sa)
            return (GSW_INVALID_VALUE);

        /*Form the first estimate of SA from a polynomial in CT and p */
        sa = gsw_sa_freezing_estimate(p,saturation_fraction,&ct,NULL);
        if (sa < -sa_cut_off)
            return (GSW_INVALID_VALUE);
        /*
        ! Form the first estimate of dCT_dSA, the derivative of CT with respect
        ! to SA at fixed p.
        */
        sa = max(sa,0.0);
        gsw_ct_freezing_first_derivatives_poly(sa,p,saturation_fraction,
                                                    &dct_dsa, NULL);
        /*
        ! For -SA_cut_off < SA < SA_cut_off, replace the above estimate of SA
        ! with one based on (CT_freezing_zero_SA - CT).
        */
        if (fabs(sa) < sa_cut_off)
            sa = (ct - ct_freezing_zero_sa)/dct_dsa;
        /*
        !-----------------------------------------------------------------------
        ! Begin the modified Newton-Raphson method to solve the root of
        ! CT_freezing = CT for SA.
        !-----------------------------------------------------------------------
        */
        for (i_iter = 1; i_iter <= number_of_iterations; i_iter++) {
            sa_old = sa;
            ct_freezing = gsw_ct_freezing_poly(sa_old,p,saturation_fraction);
            sa = sa_old - (ct_freezing - ct)/dct_dsa;
            sa_mean = 0.5*(sa + sa_old);
            gsw_ct_freezing_first_derivatives_poly(sa_mean,p,
                                saturation_fraction, &dct_dsa, NULL);
            sa = sa_old - (ct_freezing - ct)/dct_dsa;
        }

        if (gsw_sa_p_inrange(sa,p))
            return (sa);
        return (GSW_INVALID_VALUE);
}
/*
!==========================================================================
elemental function gsw_sa_freezing_from_t (t, p, saturation_fraction)
!==========================================================================
!
!  Calculates the Absolute Salinity of seawater at the freezing temperature.
!  That is, the output is the Absolute Salinity of seawater, with the
!  fraction saturation_fraction of dissolved air, that is in equilibrium
!  with ice at in-situ temperature t and pressure p.  If the input values
!  are such that there is no positive value of Absolute Salinity for which
!  seawater is frozen, the output is set to NaN.
!
!  t  =  in-situ Temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!        ( i.e. absolute pressure - 10.1325 dbar )
!  saturation_fraction = the saturation fraction of dissolved air in
!                        seawater
!  (i.e., saturation_fraction must be between 0 and 1, and the default
!    is 1, completely saturated)
!
!  sa_freezing_from_t  =  Absolute Salinity of seawater when it freezes, for
!                given input values of in situ temperature, pressure and
!                air saturation fraction.                          [ g/kg ]
!---------------------------------------------------------------------------
*/
double
gsw_sa_freezing_from_t(double t, double p, double saturation_fraction)
{
        int     i_iter, number_of_iterations = 2;
        double  f, sa, sa_mean, sa_old, t_freezing_zero_sa, tfreezing_sa;
        /*
        ! This is the band of sa within +- 2.5 g/kg of sa = 0, which we treat
        ! differently in calculating the initial values of both SA and dCT_dSA.
        */
        double  sa_cut_off = 2.5;
        /*
        ! Find t > t_freezing_zero_SA.  If this is the case, the input values
        ! represent seawater that is not frozen (at any positive SA).
        */
        t_freezing_zero_sa = gsw_t_freezing(0.0,p,saturation_fraction);
        if (t > t_freezing_zero_sa)
            return (GSW_INVALID_VALUE);
        /*
        ! This is the initial guess of SA using a purpose-built
        ! polynomial in CT and p
        */
        sa = gsw_sa_freezing_estimate(p,saturation_fraction,NULL,&t);
        if (sa < -sa_cut_off)
            return (GSW_INVALID_VALUE);
        /*
        ! Form the first estimate of tfreezing_SA, the derivative of
        ! CT_freezing with respect to SA at fixed p.
        */
        sa = max(sa,0.0);
        gsw_t_freezing_first_derivatives(sa,p,saturation_fraction,
                                              &tfreezing_sa,NULL);
        /*
        ! For -SA_cut_off < SA < SA_cut_off, replace the above estimate of SA
        ! with one based on (t_freezing_zero_SA - t).
        */
        if (fabs(sa) < sa_cut_off)
            sa = (t - t_freezing_zero_sa)/tfreezing_sa;
        /*
        !-----------------------------------------------------------------------
        ! Begin the modified Newton-Raphson method to find the root of
        ! f = (t_freezing - t) = 0 for SA.
        !-----------------------------------------------------------------------
        */
        for (i_iter = 1; i_iter <= number_of_iterations; i_iter++) {
            sa_old = sa;
            f = gsw_t_freezing(sa_old,p,saturation_fraction) - t;
            sa = sa_old - f/tfreezing_sa;
            sa_mean = 0.5*(sa + sa_old);
            gsw_t_freezing_first_derivatives(sa_mean,p,saturation_fraction,
                                                  &tfreezing_sa, NULL);
            sa = sa_old - f/tfreezing_sa;
        }

        if (gsw_sa_p_inrange(sa,p))
            return (sa);
        return (GSW_INVALID_VALUE);
}
/*
!==========================================================================
elemental function gsw_sa_freezing_from_t_poly (t, p, saturation_fraction)
!==========================================================================
!
!  Calculates the Absolute Salinity of seawater at the freezing temperature.
!  That is, the output is the Absolute Salinity of seawater, with the
!  fraction saturation_fraction of dissolved air, that is in equilibrium
!  with ice at in-situ temperature t and pressure p.  If the input values
!  are such that there is no positive value of Absolute Salinity for which
!  seawater is frozen, the output is put equal to Nan.
!
!  t  =  in-situ Temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!        ( i.e. absolute pressure - 10.1325 dbar )
!  saturation_fraction = the saturation fraction of dissolved air in
!                        seawater
!
!  sa_freezing_from_t_poly  =  Absolute Salinity of seawater when it freezes,
!                for given input values of in situ temperature, pressure and
!                air saturation fraction.                          [ g/kg ]
!--------------------------------------------------------------------------
*/
double
gsw_sa_freezing_from_t_poly(double t, double p, double saturation_fraction)
{
        int     i_iter, number_of_iterations = 5;
        double  dt_dsa, sa, sa_old, sa_mean, t_freezing, t_freezing_zero_sa;
        /*
        ! This is the band of sa within +- 2.5 g/kg of sa = 0, which we treat
        ! differently in calculating the initial values of both SA and dCT_dSA.
        */
        double  sa_cut_off = 2.5;
        /*
        ! Find t > t_freezing_zero_SA.  If this is the case, the input values
        ! represent seawater that is not frozen (at any positive SA).
        */
        t_freezing_zero_sa = gsw_t_freezing_poly(0.0,p,saturation_fraction);
        if (t > t_freezing_zero_sa)
            return (GSW_INVALID_VALUE);
        /*
        ! This is the initial guess of SA using a purpose-built
        ! polynomial in CT and p
        */
        sa = gsw_sa_freezing_estimate(p,saturation_fraction,NULL,&t);
        if (sa < -sa_cut_off)
            return (GSW_INVALID_VALUE);
        /*
        ! Form the first estimate of dt_dSA, the derivative of t with respect
        ! to SA at fixed p.
        */
        sa = max(sa,0.0);
        gsw_t_freezing_first_derivatives_poly(sa,p,saturation_fraction,
                                                   &dt_dsa, NULL);
        /*
        ! For -SA_cut_off < SA < SA_cut_off, replace the above estimate of SA
        ! with one based on (t_freezing_zero_SA - t).
        */
        if (fabs(sa) < sa_cut_off)
            sa = (t - t_freezing_zero_sa)/dt_dsa;
        /*
        !-----------------------------------------------------------------------
        ! Begin the modified Newton-Raphson method to find the root of
        ! t_freezing = t for SA.
        !-----------------------------------------------------------------------
        */
        for (i_iter = 1; i_iter <= number_of_iterations; i_iter++) {
            sa_old = sa;
            t_freezing = gsw_t_freezing_poly(sa_old,p,saturation_fraction);
            sa = sa_old - (t_freezing - t)/dt_dsa;
            sa_mean = 0.5*(sa + sa_old);
            gsw_t_freezing_first_derivatives_poly(sa_mean,p,saturation_fraction,
                                                       &dt_dsa, NULL);
            sa = sa_old - (t_freezing - t)/dt_dsa;
        }

        if (gsw_sa_p_inrange(sa,p))
            return (sa);
        return (GSW_INVALID_VALUE);
}
/*
!==========================================================================
function gsw_sa_from_rho(rho,ct,p)
!==========================================================================

!  Calculates the Absolute Salinity of a seawater sample, for given values
!  of its density, Conservative Temperature and sea pressure (in dbar).
!
!  rho =  density of a seawater sample (e.g. 1026 kg/m^3).       [ kg/m^3 ]
!   Note. This input has not had 1000 kg/m^3 subtracted from it.
!     That is, it is 'density', not 'density anomaly'.
!  ct  =  Conservative Temperature (ITS-90)                      [ deg C ]
!  p   =  sea pressure                                           [ dbar ]
!
!  sa  =  Absolute Salinity                                      [g/kg]
*/
double
gsw_sa_from_rho(double rho, double ct, double p)
{
        int     no_iter;

        double  sa, v_lab, v_0, v_50, v_sa, sa_old, delta_v, sa_mean;

        v_lab   = 1.0/rho;
        v_0     = gsw_specvol(0.0,ct,p);
        v_50    = gsw_specvol(50.0,ct,p);

        sa      = 50.0*(v_lab - v_0)/(v_50 - v_0);
        if (sa < 0.0 || sa > 50.0)
            return (GSW_INVALID_VALUE);

        v_sa    = (v_50 - v_0)/50.0;

        for (no_iter=1; no_iter <= 2; no_iter++) {
            sa_old      = sa;
            delta_v     = gsw_specvol(sa_old,ct,p) - v_lab;
            sa          = sa_old - delta_v/v_sa;
            sa_mean     = 0.5*(sa + sa_old);
            gsw_specvol_first_derivatives(sa_mean,ct,p,&v_sa,NULL,NULL);
            sa          = sa_old - delta_v/v_sa;
            if (sa < 0.0 || sa > 50.0)
                return (GSW_INVALID_VALUE);
        }
        return (sa);
}
/*
!==========================================================================
function gsw_sa_from_sp(sp,p,lon,lat)
!==========================================================================

! Calculates Absolute Salinity, SA, from Practical Salinity, SP
!
! sp     : Practical Salinity                              [unitless]
! p      : sea pressure                                    [dbar]
! lon    : longitude                                       [DEG E]
! lat    : latitude                                        [DEG N]
!
! gsw_sa_from_sp   : Absolute Salinity                     [g/kg]
*/
double
gsw_sa_from_sp(double sp, double p, double lon, double lat)
{
        GSW_TEOS10_CONSTANTS;
        double  saar, gsw_sa_baltic;

        gsw_sa_baltic   = gsw_sa_from_sp_baltic(sp,lon,lat);
        if (gsw_sa_baltic < GSW_ERROR_LIMIT)
            return (gsw_sa_baltic);
        saar            = gsw_saar(p,lon,lat);
        if (saar == GSW_INVALID_VALUE)
            return (saar);
        return (gsw_ups*sp*(1.e0 + saar));
}
/*
!==========================================================================
function gsw_sa_from_sp_baltic(sp,lon,lat)
!==========================================================================

! For the Baltic Sea, calculates Absolute Salinity with a value
! computed analytically from Practical Salinity
!
! sp     : Practical Salinity                              [unitless]
! lon    : longitude                                       [deg E]
! lat    : latitude                                        [deg N]
!
! sa_from_sp_baltic : Absolute Salinity                    [g/kg]
*/
double
gsw_sa_from_sp_baltic(double sp, double lon, double lat)
{
        GSW_TEOS10_CONSTANTS;
        GSW_BALTIC_DATA;
        double  xx_left, xx_right, return_value;

        lon = fmod(lon, 360.0);
        if (lon < 0.0)
            lon += 360.0;

        if (xb_left[1] < lon  && lon < xb_right[0]  && yb_left[0] < lat  &&
            lat < yb_left[2]) {

            xx_left     = gsw_util_xinterp1(yb_left, xb_left, 3, lat);

            xx_right    = gsw_util_xinterp1(yb_right, xb_right, 2, lat);

            if (xx_left <= lon  && lon <= xx_right)
                return_value    =((gsw_sso - 0.087)/35.0)*sp + 0.087;
            else
                return_value    = GSW_INVALID_VALUE;
        } else
            return_value        = GSW_INVALID_VALUE;

        return (return_value);
}
/*
!==========================================================================
function gsw_sa_from_sstar(sstar,p,lon,lat)
!==========================================================================

! Calculates Absolute Salinity, SA, from Preformed Salinity, Sstar.
!
! Sstar  : Preformed Salinity                              [g/kg]
! p      : sea pressure                                    [dbar]
! lon   : longitude                                       [deg E]
! lat    : latitude                                        [deg N]
!
! gsw_sa_from_sstar   : Absolute Salinity                  [g/kg]
*/
double
gsw_sa_from_sstar(double sstar, double p, double lon, double lat)
{
        double  saar;

        saar    = gsw_saar(p,lon,lat);
        if (saar == GSW_INVALID_VALUE)
            return (saar);
    /*
    **! In the Baltic Sea, Sstar = SA, and note that gsw_saar returns zero
    **! for SAAR in the Baltic.
    */
        return (sstar*(1e0 + saar)/(1e0 - 0.35e0*saar));
}
/*
!==========================================================================
elemental function gsw_sa_p_inrange (sa, p)
!==========================================================================
!
!  Check for any values that are out of the TEOS-10 range ...
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!---------------------------------------------------------------------------
*/
int
gsw_sa_p_inrange(double sa, double p)
{
        if (p > 10000.0 || sa > 120.0 ||
            (p + sa*71.428571428571402) > 13571.42857142857)
            return (0);
        return (1);
}
/*
!==========================================================================
elemental subroutine gsw_seaice_fraction_to_freeze_seawater (sa, ct, p, &
                       sa_seaice, t_seaice, sa_freeze, ct_freeze, w_seaice)
!==========================================================================
!
!  Calculates the mass fraction of sea ice (mass of sea ice divided by mass
!  of sea ice plus seawater), which, when melted into seawater having the
!  properties (SA,CT,p) causes the final seawater to be at the freezing
!  temperature.  The other outputs are the Absolute Salinity and
!  Conservative Temperature of the final seawater.
!
!  SA        =  Absolute Salinity of seawater                      [ g/kg ]
!  CT        =  Conservative Temperature of seawater (ITS-90)     [ deg C ]
!  p         =  sea pressure                                       [ dbar ]
!            ( i.e. absolute pressure - 10.1325 dbar )
!  SA_seaice =  Absolute Salinity of sea ice, that is, the mass fraction of
!               salt in sea ice, expressed in g of salt per kg of sea ice.
!                                                                  [ g/kg ]
!  t_seaice  =  in-situ temperature of the sea ice at pressure p (ITS-90)
!                                                                 [ deg C ]
!
!  SA_freeze  =  Absolute Salinity of seawater after the mass fraction of
!                sea ice, w_seaice, at temperature t_seaice has melted into
!                the original seawater, and the final mixture is at the
!                freezing temperature of seawater.                 [ g/kg ]
!
!  CT_freeze  =  Conservative Temperature of seawater after the mass
!                fraction, w_seaice, of sea ice at temperature t_seaice has
!                melted into the original seawater, and the final mixture
!                is at the freezing temperature of seawater.      [ deg C ]
!
!  w_seaice   =  mass fraction of sea ice, at SA_seaice and t_seaice,
!                which, when melted into seawater at (SA,CT,p) leads to the
!                final mixed seawater being at the freezing temperature.
!                This output is between 0 and 1.                 [unitless]
!--------------------------------------------------------------------------
*/
void
gsw_seaice_fraction_to_freeze_seawater(double sa, double ct, double p,
        double sa_seaice, double t_seaice, double *sa_freeze, double *ct_freeze,
        double *w_seaice)
{
        int     number_of_iterations;
        double  ctf, ctf_mean, ctf_old, ctf_plus1, ctf_zero,
                dfunc_dsaf, func, func_plus1, func_zero, h, h_brine,
                h_ih, sa_freezing, saf, saf_mean, saf_old,
                salt_ratio, tf_sa_seaice, h_hat_sa, h_hat_ct, ctf_sa,
                sa0 = 0.0, saturation_fraction = 0.0;

        ctf = gsw_ct_freezing(sa,p,saturation_fraction);
        if (ct < ctf) {
            /*The seawater ct input is below the freezing temp*/
            *sa_freeze = *ct_freeze = *w_seaice = GSW_INVALID_VALUE;
            return;
        }

        tf_sa_seaice = gsw_t_freezing(sa_seaice,p,saturation_fraction)
                                                - 1e-6;
        if (t_seaice > tf_sa_seaice) {
        /*
        ! The 1e-6 C buffer in the allowable t_seaice is to ensure that there is
        ! some ice Ih in the sea ice.   Without this buffer, that is if t_seaice
        ! is allowed to be exactly equal to tf_sa_seaice, the sea ice is
        ! actually 100% brine at Absolute Salinity of SA_seaice.
        */
            *sa_freeze = *ct_freeze = *w_seaice = GSW_INVALID_VALUE;
            return;
        }

        sa_freezing = gsw_sa_freezing_from_t(t_seaice,p,saturation_fraction);
        if (sa_freezing > GSW_ERROR_LIMIT) {
            *sa_freeze = *ct_freeze = *w_seaice = GSW_INVALID_VALUE;
            return;
        }
        h_brine = gsw_enthalpy_t_exact(sa_freezing,t_seaice,p);
        salt_ratio = sa_seaice/sa_freezing;

        h = gsw_enthalpy_ct_exact(sa,ct,p);
        h_ih = gsw_enthalpy_ice(t_seaice,p);

        ctf_plus1 = gsw_ct_freezing(sa+1.0,p,saturation_fraction);
        func_plus1 = (sa - sa_seaice)
                        *(gsw_enthalpy_ct_exact(sa+1.0,ctf_plus1,p)
                        - h) - (h - h_ih) + salt_ratio*(h_brine - h_ih);

        ctf_zero = gsw_ct_freezing(sa0,p,saturation_fraction);
        func_zero = (sa - sa_seaice)
                        *(gsw_enthalpy_ct_exact(sa0,ctf_zero,p) - h)
                        + sa*((h - h_ih) - salt_ratio*(h_brine - h_ih));

        saf = -(sa+1.0)*func_zero/(func_plus1 - func_zero);
                /*initial guess of saf*/
        ctf = gsw_ct_freezing(saf,p,saturation_fraction);
        gsw_enthalpy_first_derivatives_ct_exact(saf,ctf,p,&h_hat_sa,&h_hat_ct);
        gsw_ct_freezing_first_derivatives(saf,p,saturation_fraction,
                                               &ctf_sa, NULL);

        dfunc_dsaf = (sa - sa_seaice)*(h_hat_sa + h_hat_ct*ctf_sa)
                        - (h - h_ih) + salt_ratio*(h_brine - h_ih);

        for (number_of_iterations = 1; number_of_iterations <= 4;
            number_of_iterations++) {
            saf_old = saf;
            ctf_old = ctf;
            func = (sa - sa_seaice)
                *(gsw_enthalpy_ct_exact(saf_old,ctf_old,p) - h)
                - (saf_old - sa)*((h - h_ih) - salt_ratio*(h_brine - h_ih));
            saf = saf_old - func/dfunc_dsaf;
            saf_mean = 0.5*(saf + saf_old);
            ctf_mean = gsw_ct_freezing(saf_mean,p,saturation_fraction);
            gsw_enthalpy_first_derivatives_ct_exact(saf_mean,ctf_mean,p,
                                                         &h_hat_sa,&h_hat_ct);
            gsw_ct_freezing_first_derivatives(saf_mean,p,saturation_fraction,
                                                   &ctf_sa, NULL);
            dfunc_dsaf = (sa - sa_seaice)*(h_hat_sa + h_hat_ct*ctf_sa)
                        - (h - h_ih) + salt_ratio*(h_brine - h_ih);
            saf = saf_old - func/dfunc_dsaf;
            ctf = gsw_ct_freezing(saf,p,saturation_fraction);
        }
/*
! After these 4 iterations of this modified Newton-Raphson method, the
! errors in SA_freeze is less than 1.5x10^-12 g/kg, in CT_freeze is less than
! 2x10^-13 deg C and in w_seaice is less than 2.8x10^-13 which represent machine
! precision for these calculations.
*/
        *sa_freeze = saf;
        *ct_freeze = ctf;
        *w_seaice = (h - gsw_enthalpy_ct_exact(*sa_freeze,*ct_freeze,p)) /
                                   (h - h_ih - salt_ratio*(h_brine - h_ih));
        return;
}
/*
!==========================================================================
function gsw_sigma0(sa,ct)
!==========================================================================

!  Calculates potential density anomaly with reference pressure of 0 dbar,
!  this being this particular potential density minus 1000 kg/m^3.  This
!  function has inputs of Absolute Salinity and Conservative Temperature.
!  This function uses the computationally-efficient 48-term expression for
!  density in terms of SA, CT and p (IOC et al., 2010).
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature                        [deg C]
!
! gsw_sigma0  : potential density anomaly with reference pressure of 0
!                                                      (48 term equation)
*/
double
gsw_sigma0(double sa, double ct)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  vp0, xs, ys;

        xs      = sqrt(gsw_sfac*sa + offset);
        ys      = ct*0.025;

        vp0     = v000
    + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050
    + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 + xs*(v140
    + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220 + xs*(v230 + v240*xs)))
    + ys*(v300 + xs*(v310 + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410
    + v420*xs) + ys*(v500 + v510*xs + v600*ys)))));

        return (1.0/vp0 - 1000.0);
}
/*
!==========================================================================
function gsw_sigma1(sa,ct)
!==========================================================================

!  Calculates potential density anomaly with reference pressure of 1000 dbar,
!  this being this particular potential density minus 1000 kg/m^3.  This
!  function has inputs of Absolute Salinity and Conservative Temperature.
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature                        [deg C]
!
! sigma1 : potential density anomaly with reference pressure of 1000
*/
double
gsw_sigma1(double sa, double ct)
{
        return (gsw_rho(sa,ct,1000.0) - 1000.0);
}
/*
!==========================================================================
function gsw_sigma2(sa,ct)
!==========================================================================

!  Calculates potential density anomaly with reference pressure of 2000 dbar,
!  this being this particular potential density minus 1000 kg/m^3.  This
!  function has inputs of Absolute Salinity and Conservative Temperature.
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature                        [deg C]
!
! sigma2 : potential density anomaly with reference pressure of 2000
*/
double
gsw_sigma2(double sa, double ct)
{
        return (gsw_rho(sa,ct,2000.0) - 1000.0);
}
/*
!==========================================================================
function gsw_sigma3(sa,ct)
!==========================================================================

!  Calculates potential density anomaly with reference pressure of 3000 dbar,
!  this being this particular potential density minus 1000 kg/m^3.  This
!  function has inputs of Absolute Salinity and Conservative Temperature.
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature                        [deg C]
!
! sigma3 : potential density anomaly with reference pressure of 3000
*/
double
gsw_sigma3(double sa, double ct)
{
        return (gsw_rho(sa,ct,3000.0) - 1000.0);
}
/*
!==========================================================================
function gsw_sigma4(sa,ct)
!==========================================================================

!  Calculates potential density anomaly with reference pressure of 4000 dbar,
!  this being this particular potential density minus 1000 kg/m^3.  This
!  function has inputs of Absolute Salinity and Conservative Temperature.
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature                        [deg C]
!
! sigma4  : potential density anomaly with reference pressure of 4000
*/
double
gsw_sigma4(double sa, double ct)
{
        return (gsw_rho(sa,ct,4000.0) - 1000.0);
}
/*
!==========================================================================
function gsw_sound_speed(sa,ct,p)
!==========================================================================

!  Calculates the speed of sound in seawater.  This function has inputs of
!  Absolute Salinity and Conservative Temperature.  This function uses the
!  computationally-efficient expression for specific volume in terms of SA,
!  CT and p (Roquet et al., 2014).
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature (ITS-90)               [deg C]
! p      : sea pressure                                    [dbar]
!
! sound_speed  : speed of sound in seawater                [m/s]
*/
double
gsw_sound_speed(double sa, double ct, double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  v, v_p, xs, ys, z;

        xs      = sqrt(gsw_sfac*sa + offset);
        ys      = ct*0.025;
        z       = p*1e-4;

        v       = v000
    + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050
    + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 + xs*(v140
    + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220 + xs*(v230 + v240*xs)))
    + ys*(v300 + xs*(v310 + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410
    + v420*xs) + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011
    + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111
    + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211 + xs*(v221
    + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs) + ys*(v401 + v411*xs
    + v501*ys)))) + z*(v002 + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs)))
    + ys*(v102 + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212
    + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013
    + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004 + v014*xs + v104*ys
    + z*(v005 + v006*z)))));

        v_p     = c000
    + xs*(c100 + xs*(c200 + xs*(c300 + xs*(c400 + c500*xs))))
    + ys*(c010 + xs*(c110 + xs*(c210 + xs*(c310 + c410*xs))) + ys*(c020
    + xs*(c120 + xs*(c220 + c320*xs)) + ys*(c030 + xs*(c130 + c230*xs)
    + ys*(c040 + c140*xs + c050*ys)))) + z*(c001 + xs*(c101 + xs*(c201
    + xs*(c301 + c401*xs))) + ys*(c011 + xs*(c111 + xs*(c211 + c311*xs))
    + ys*(c021 + xs*(c121 + c221*xs) + ys*(c031 + c131*xs + c041*ys)))
    + z*( c002 + xs*(c102 + c202*xs) + ys*(c012 + c112*xs + c022*ys)
    + z*(c003 + c103*xs + c013*ys + z*(c004 + c005*z))));

        return (10000.0*sqrt(-v*v/v_p));
}
/*
!==========================================================================
elemental function gsw_sound_speed_ice (t, p)
!==========================================================================
!
!  Calculates the compression speed of sound in ice.
!
!  t   =  in-situ temperature (ITS-90)                            [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  sound_speed_ice  =  compression speed of sound in ice            [ m/s ]
!--------------------------------------------------------------------------
*/
double
gsw_sound_speed_ice(double t, double p)
{
        double  gi_tp, gi_tt;

        gi_tt = gsw_gibbs_ice(2,0,t,p);
        gi_tp = gsw_gibbs_ice(1,1,t,p);

        return (gsw_gibbs_ice(0,1,t,p) *
               sqrt(gi_tt/(gi_tp*gi_tp - gi_tt*gsw_gibbs_ice(0,2,t,p))));
}
/*
!==========================================================================
function gsw_sound_speed_t_exact(sa,t,p)
!==========================================================================

! Calculates the speed of sound in seawater
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
!
! gsw_sound_speed_t_exact : sound speed                    [m/s]
*/
double
gsw_sound_speed_t_exact(double sa, double t, double p)
{
        int     n0=0, n1=1, n2=2;
        double  g_tt, g_tp;

        g_tt    = gsw_gibbs(n0,n2,n0,sa,t,p);
        g_tp    = gsw_gibbs(n0,n1,n1,sa,t,p);

        return (gsw_gibbs(n0,n0,n1,sa,t,p) *
                sqrt(g_tt/(g_tp*g_tp - g_tt*gsw_gibbs(n0,n0,n2,sa,t,p))));
}
/*
!--------------------------------------------------------------------------
! Practical Salinity (SP), PSS-78
!--------------------------------------------------------------------------

!==========================================================================
function gsw_sp_from_c(c,t,p)
!==========================================================================

!  Calculates Practical Salinity, SP, from conductivity, C, primarily using
!  the PSS-78 algorithm.  Note that the PSS-78 algorithm for Practical
!  Salinity is only valid in the range 2 < SP < 42.  If the PSS-78
!  algorithm produces a Practical Salinity that is less than 2 then the
!  Practical Salinity is recalculated with a modified form of the Hill et
!  al. (1986) formula.  The modification of the Hill et al. (1986)
!  expression is to ensure that it is exactly consistent with PSS-78
!  at SP = 2.  Note that the input values of conductivity need to be in
!  units of mS/cm (not S/m).
!
! c      : conductivity                                     [ mS/cm ]
! t      : in-situ temperature [ITS-90]                     [deg C]
! p      : sea pressure                                     [dbar]
!
! sp     : Practical Salinity                               [unitless]
*/
double
gsw_sp_from_c(double c, double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SP_COEFFICIENTS;
        double  sp, t68, ft68, r, rt_lc, rp, rt, rtx,
                hill_ratio, x, sqrty, part1, part2,
                sp_hill_raw;

        t68     = t*1.00024e0;
        ft68    = (t68 - 15e0)/(1e0 + k*(t68 - 15e0));
    /*
     ! The dimensionless conductivity ratio, R, is the conductivity input, C,
     ! divided by the present estimate of C(SP=35, t_68=15, p=0) which is
     ! 42.9140 mS/cm (=4.29140 S/m), (Culkin and Smith, 1980).
    */

        r = c/gsw_c3515;        /* 0.023302418791070513 = 1./42.9140 */

        /*rt_lc corresponds to rt as defined in the UNESCO 44 (1983) routines.*/
        rt_lc   = c0 + (c1 + (c2 + (c3 + c4*t68)*t68)*t68)*t68;
        rp      = 1e0 + (p*(e1 + e2*p + e3*p*p))/(1e0 + d1*t68 + d2*t68*t68 +
                  (d3 + d4*t68)*r);
        rt      = r/(rp*rt_lc);

        if (rt < 0.0) {
            return (GSW_INVALID_VALUE);
        }

        rtx     = sqrt(rt);

        sp      = a0 + (a1 + (a2 + (a3 + (a4 + a5*rtx)*rtx)*rtx)*rtx)*rtx +
                  ft68*(b0 + (b1 + (b2 + (b3 +
                        (b4 + b5*rtx)*rtx)*rtx)*rtx)*rtx);
    /*
     ! The following section of the code is designed for SP < 2 based on the
     ! Hill et al. (1986) algorithm.  This algorithm is adjusted so that it is
     ! exactly equal to the PSS-78 algorithm at SP = 2.
    */

        if (sp < 2) {
            hill_ratio  = gsw_hill_ratio_at_sp2(t);
            x           = 400e0*rt;
            sqrty       = 10e0*rtx;
            part1       = 1e0 + x*(1.5e0 + x);
            part2       = 1e0 + sqrty*(1e0 + sqrty*(1e0 + sqrty));
            sp_hill_raw = sp - a0/part1 - b0*ft68/part2;
            sp          = hill_ratio*sp_hill_raw;
        }

    /* This line ensures that SP is non-negative. */
        if (sp < 0.0) {
            sp  = GSW_INVALID_VALUE;
        }

        return (sp);
}
/*
!==========================================================================
function gsw_sp_from_sa(sa,p,lon,lat)
!==========================================================================

! Calculates Practical salinity, sp, from Absolute salinity, sa
!
! sa     : Absolute Salinity                               [g/kg]
! p      : sea pressure                                    [dbar]
! lon    : longitude                                       [DEG E]
! lat    : latitude                                        [DEG N]
!
! gsw_sp_from_sa      : Practical Salinity                 [unitless]
*/
double
gsw_sp_from_sa(double sa, double p, double lon, double lat)
{
        GSW_TEOS10_CONSTANTS;
        double  saar, gsw_sp_baltic;

        gsw_sp_baltic   = gsw_sp_from_sa_baltic(sa,lon,lat);
        if (gsw_sp_baltic < GSW_ERROR_LIMIT)
            return (gsw_sp_baltic);
        saar    = gsw_saar(p,lon,lat);
        if (saar == GSW_INVALID_VALUE)
            return (saar);
        return ((sa/gsw_ups)/(1e0 + saar));
}
/*
!==========================================================================
function gsw_sp_from_sa_baltic(sa,lon,lat)
!==========================================================================

! For the Baltic Sea, calculates Practical Salinity with a value
! computed analytically from Absolute Salinity
!
! sa     : Absolute Salinity                               [g/kg]
! lon    : longitude                                       [deg E]
! lat    : latitude                                        [deg N]
!
! gsw_sp_from_sa_baltic  : Practical Salinity              [unitless]
*/
double
gsw_sp_from_sa_baltic(double sa, double lon, double lat)
{
        GSW_TEOS10_CONSTANTS;
        GSW_BALTIC_DATA;
        double  xx_left, xx_right, return_value;

        lon = fmod(lon, 360.0);
        if (lon < 0.0)
            lon += 360.0;

        if (xb_left[1] < lon  && lon < xb_right[0]  && yb_left[0] < lat  &&
            lat < yb_left[2]) {

            xx_left     = gsw_util_xinterp1(yb_left, xb_left, 3, lat);

            xx_right    = gsw_util_xinterp1(yb_right, xb_right, 2, lat);

            if (xx_left <= lon  && lon <= xx_right)
                return_value    = (35.0/(gsw_sso - 0.087))*(sa - 0.087);
            else
                return_value    = GSW_INVALID_VALUE;
        } else
            return_value        = GSW_INVALID_VALUE;

        return (return_value);
}
/*
!==========================================================================
function gsw_sp_from_sk(sk)
!==========================================================================

! Calculates Practical Salinity, SP, from SK
!
!  SK    : Knudsen Salinity                        [parts per thousand, ppt]
!
! gsw_sp_from_sk  : Practical Salinity                              [unitless]
*/
double
gsw_sp_from_sk(double sk)
{
        GSW_TEOS10_CONSTANTS;
        double  gsw_sp_from_sk_value;


        gsw_sp_from_sk_value = (sk - 0.03e0)*(gsw_soncl/1.805e0);

        if (gsw_sp_from_sk_value < 0e0)
            gsw_sp_from_sk_value = GSW_INVALID_VALUE;

        return (gsw_sp_from_sk_value);
}
/*
!==========================================================================
function gsw_sp_from_sr(sr)
!==========================================================================

! Calculates Practical Salinity, sp, from Reference Salinity, sr.
!
! sr     : Reference Salinity                              [g/kg]
!
! gsw_sp_from_sr  : Practical Salinity                     [unitless]
*/
double
gsw_sp_from_sr(double sr)
{
        GSW_TEOS10_CONSTANTS;

        return(sr/gsw_ups);
}
/*
!==========================================================================
function gsw_sp_from_sstar(sstar,p,lon,lat)
!==========================================================================

! Calculates Practical Salinity, SP, from Preformed Salinity, Sstar.
!
! sstar  : Preformed Salinity                              [g/kg]
! p      : sea pressure                                    [dbar]
! lon   : longitude                                       [deg E]
! lat    : latitude                                        [deg N]
!
! gsw_sp_from_Sstar : Preformed Salinity                   [g/kg]
*/
double
gsw_sp_from_sstar(double sstar, double p, double lon, double lat)
{
        GSW_TEOS10_CONSTANTS;
        double  saar, sp_baltic;

    /*
    **! In the Baltic Sea, SA = Sstar.
    */
        sp_baltic       = gsw_sp_from_sa_baltic(sstar,lon,lat);
        if (sp_baltic < GSW_ERROR_LIMIT)
            return (sp_baltic);
        saar    = gsw_saar(p,lon,lat);
        if (saar == GSW_INVALID_VALUE)
            return (saar);
        return ((sstar/gsw_ups)/(1.0 - 0.35e0*saar));
}
/*
!==========================================================================
function gsw_sp_salinometer(rt,t)
!==========================================================================
!  Calculates Practical Salinity SP from a salinometer, primarily using the
!  PSS-78 algorithm.  Note that the PSS-78 algorithm for Practical Salinity
!  is only valid in the range 2 < SP < 42.  If the PSS-78 algorithm
!  produces a Practical Salinity that is less than 2 then the Practical
!  Salinity is recalculated with a modified form of the Hill et al. (1986)
!  formula.  The modification of the Hill et al. (1986) expression is to
!  ensure that it is exactly consistent with PSS-78 at SP = 2.
!
!  A laboratory salinometer has the ratio of conductivities, Rt, as an
!  output, and the present function uses this conductivity ratio and the
!  temperature t of the salinometer bath as the two input variables.
!
!  rt  = C(SP,t_68,0)/C(SP=35,t_68,0)                          [ unitless ]
!  t   = temperature of the bath of the salinometer,
!        measured on the ITS-90 scale (ITS-90)                 [ deg C ]
!
!  gsw_sp_salinometer = Practical Salinity on the PSS-78 scale [ unitless ]
*/
double
gsw_sp_salinometer(double rt, double t)
{
  GSW_SP_COEFFICIENTS;
  double t68, ft68, rtx, sp, hill_ratio,
         x, sqrty, part1, part2, sp_hill_raw;

  if (rt < 0){
    return NAN;
  }

  t68 = t*1.00024;
  ft68 = (t68 - 15)/(1 + k*(t68 - 15));

  rtx = sqrt(rt);

  sp = a0 + (a1 + (a2 + (a3 + (a4 + a5 * rtx) * rtx) * rtx) * rtx) * rtx
    + ft68 * (b0 + (b1 + (b2+ (b3 + (b4 + b5 * rtx) * rtx) * rtx) * rtx) * rtx);

    /*
     ! The following section of the code is designed for SP < 2 based on the
     ! Hill et al. (1986) algorithm.  This algorithm is adjusted so that it is
     ! exactly equal to the PSS-78 algorithm at SP = 2.
    */

   if (sp < 2) {
       hill_ratio  = gsw_hill_ratio_at_sp2(t);
       x           = 400e0*rt;
       sqrty       = 10e0*rtx;
       part1       = 1e0 + x*(1.5e0 + x);
       part2       = 1e0 + sqrty*(1e0 + sqrty*(1e0 + sqrty));
       sp_hill_raw = sp - a0/part1 - b0*ft68/part2;
       sp          = hill_ratio*sp_hill_raw;
   }

  return sp;

}
/*
!==========================================================================
function gsw_specvol(sa,ct,p)
!==========================================================================
!
!  Calculates specific volume from Absolute Salinity, Conservative
!  Temperature and pressure, using the computationally-efficient
!  polynomial expression for specific volume (Roquet et al., 2014).
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature (ITS-90)               [deg C]
! p      : sea pressure                                    [dbar]
!          ( i.e. absolute pressure - 10.1325 dbar )
!
! specvol: specific volume                                 [m^3/kg]
*/
double
gsw_specvol(double sa, double ct, double p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  xs, ys, z, value;

        xs      = sqrt(gsw_sfac*sa + offset);
        ys      = ct*0.025;
        z       = p*1e-4;

        value = v000
    + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050
    + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 + xs*(v140
    + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220 + xs*(v230 + v240*xs)))
    + ys*(v300 + xs*(v310 + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410
    + v420*xs) + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011
    + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111
    + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211 + xs*(v221
    + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs) + ys*(v401 + v411*xs
    + v501*ys)))) + z*(v002 + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs)))
    + ys*(v102 + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212
    + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013
    + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004 + v014*xs + v104*ys
    + z*(v005 + v006*z)))));

        return (value);
}
/*
!==========================================================================
elemental subroutine gsw_specvol_alpha_beta (sa, ct, p, specvol, alpha, &
                                             beta)
!==========================================================================
!
!  Calculates specific volume, the appropriate thermal expansion coefficient
!  and the appropriate saline contraction coefficient of seawater from
!  Absolute Salinity and Conservative Temperature.  This function uses the
!  computationally-efficient expression for specific volume in terms of
!  SA, CT and p (Roquet et al., 2014).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  specvol =  specific volume                                      [ m/kg ]
!  alpha   =  thermal expansion coefficient                         [ 1/K ]
!             with respect to Conservative Temperature
!  beta    =  saline (i.e. haline) contraction                     [ kg/g ]
!             coefficient at constant Conservative Temperature
!--------------------------------------------------------------------------
*/
void
gsw_specvol_alpha_beta(double sa, double ct, double p, double *specvol,
                        double *alpha, double *beta)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  v, v_ct, v_sa_part, xs, ys, z;

        xs = sqrt(gsw_sfac*sa + offset);
        ys = ct*0.025;
        z = p*1e-4;

        v = v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050
    + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 + xs*(v140
    + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220 + xs*(v230 + v240*xs)))
    + ys*(v300 + xs*(v310 + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410
    + v420*xs) + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011
    + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111
    + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211 + xs*(v221
    + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs) + ys*(v401 + v411*xs
    + v501*ys)))) + z*(v002 + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs)))
    + ys*(v102 + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212
    + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013
    + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004 + v014*xs + v104*ys
    + z*(v005 + v006*z)))));

        if (specvol != NULL)
            *specvol = v;

        if (alpha != NULL) {

            v_ct = a000 + xs*(a100 + xs*(a200 + xs*(a300
             + xs*(a400 + a500*xs))))
             + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 + a410*xs)))
             + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs)) + ys*(a030
             + xs*(a130 + a230*xs) + ys*(a040 + a140*xs + a050*ys ))))
             + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 + a401*xs)))
             + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) + ys*(a021
             + xs*(a121 + a221*xs) + ys*(a031 + a131*xs + a041*ys)))
             + z*(a002 + xs*(a102 + xs*(a202 + a302*xs)) + ys*(a012
             + xs*(a112 + a212*xs) + ys*(a022 + a122*xs + a032*ys))
             + z*(a003 + a103*xs + a013*ys + a004*z)));

            *alpha = 0.025*v_ct/v;

        }

        if (beta != NULL) {

            v_sa_part = b000 + xs*(b100 + xs*(b200 + xs*(b300
           + xs*(b400 + b500*xs))))
           + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 + b410*xs)))
           + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs)) + ys*(b030
           + xs*(b130 + b230*xs) + ys*(b040 + b140*xs + b050*ys))))
           + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 + b401*xs)))
           + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) + ys*(b021
           + xs*(b121 + b221*xs) + ys*(b031 + b131*xs + b041*ys)))
           + z*(b002 + xs*(b102 + xs*(b202 + b302*xs))+ ys*(b012
           + xs*(b112 + b212*xs) + ys*(b022 + b122*xs + b032*ys))
           + z*(b003 +  b103*xs + b013*ys + b004*z)));

            *beta = -v_sa_part*0.5*gsw_sfac/(v*xs);

        }
}
/*
!==========================================================================
function gsw_specvol_anom_standard(sa,ct,p)
!==========================================================================
!
!  Calculates specific volume anomaly of seawater.
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature (ITS-90)               [deg C]
! p      : sea pressure                                    [dbar]
!
! specvol_anom  :  specific volume anomaly of seawater
*/
double
gsw_specvol_anom_standard(double sa, double ct, double p)
{
        return (gsw_specvol(sa,ct,p) - gsw_specvol_sso_0(p));
}
/*
!==========================================================================
elemental subroutine gsw_specvol_first_derivatives (sa, ct, p, v_sa, v_ct, &
                                                    v_p, iflag)
! =========================================================================
!
!  Calculates three first-order derivatives of specific volume (v).
!  Note that this function uses the computationally-efficient
!  expression for specific volume (Roquet et al., 2014).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  v_SA  =  The first derivative of specific volume with respect to
!           Absolute Salinity at constant CT & p.       [ J/(kg (g/kg)^2) ]
!  v_CT  =  The first derivative of specific volume with respect to
!           CT at constant SA and p.                     [ J/(kg K(g/kg)) ]
!  v_P   =  The first derivative of specific volume with respect to
!           P at constant SA and CT.                         [ J/(kg K^2) ]
!--------------------------------------------------------------------------
*/
void
gsw_specvol_first_derivatives (double sa, double ct, double p,
                                double *v_sa, double *v_ct, double *v_p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  v_ct_part, v_p_part, v_sa_part, xs, ys, z;

        xs = sqrt(gsw_sfac*sa + offset);
        ys = ct*0.025;
        z = p*1e-4;

        if (v_sa != NULL) {

            v_sa_part = b000 + xs*(b100 + xs*(b200 + xs*(b300
           + xs*(b400 + b500*xs))))
           + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 + b410*xs)))
           + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs)) + ys*(b030
           + xs*(b130 + b230*xs) + ys*(b040 + b140*xs + b050*ys))))
           + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 + b401*xs)))
           + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) + ys*(b021
           + xs*(b121 + b221*xs) + ys*(b031 + b131*xs + b041*ys)))
           + z*(b002 + xs*(b102 + xs*(b202 + b302*xs))+ ys*(b012
           + xs*(b112 + b212*xs) + ys*(b022 + b122*xs + b032*ys))
           + z*(b003 +  b103*xs + b013*ys + b004*z)));

            *v_sa = 0.5*gsw_sfac*v_sa_part/xs;

        }

        if (v_ct != NULL) {

            v_ct_part = a000 + xs*(a100 + xs*(a200 + xs*(a300
             + xs*(a400 + a500*xs))))
             + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 + a410*xs)))
             + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs)) + ys*(a030
             + xs*(a130 + a230*xs) + ys*(a040 + a140*xs + a050*ys ))))
             + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 + a401*xs)))
             + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) + ys*(a021
             + xs*(a121 + a221*xs) + ys*(a031 + a131*xs + a041*ys)))
             + z*(a002 + xs*(a102 + xs*(a202 + a302*xs)) + ys*(a012
             + xs*(a112 + a212*xs) + ys*(a022 + a122*xs + a032*ys))
             + z*(a003 + a103*xs + a013*ys + a004*z))) ;

            *v_ct = 0.025*v_ct_part;

        }

        if (v_p != NULL) {

            v_p_part = c000 + xs*(c100 + xs*(c200 + xs*(c300
        + xs*(c400 + c500*xs))))
        + ys*(c010 + xs*(c110 + xs*(c210 + xs*(c310 + c410*xs))) + ys*(c020
        + xs*(c120 + xs*(c220 + c320*xs)) + ys*(c030 + xs*(c130 + c230*xs)
        + ys*(c040 + c140*xs + c050*ys)))) + z*(c001 + xs*(c101 + xs*(c201
        + xs*(c301 + c401*xs))) + ys*(c011 + xs*(c111 + xs*(c211 + c311*xs))
        + ys*(c021 + xs*(c121 + c221*xs) + ys*(c031 + c131*xs + c041*ys)))
        + z*( c002 + xs*(c102 + c202*xs) + ys*(c012 + c112*xs + c022*ys)
        + z*(c003 + c103*xs + c013*ys + z*(c004 + c005*z))));

            *v_p = 1e-8*v_p_part;

        }
}
/*
!==========================================================================
elemental subroutine gsw_specvol_first_derivatives_wrt_enthalpy (sa, ct, &
                                                       p, v_sa, v_h, iflag)
! =========================================================================
!
!  Calculates two first-order derivatives of specific volume (v).
!  Note that this function uses the using the computationally-efficient
!  expression for specific volume (Roquet et al., 2014).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  v_SA  =  The first derivative of specific volume with respect to
!              Absolute Salinity at constant CT & p.    [ J/(kg (g/kg)^2) ]
!  v_h  =  The first derivative of specific volume with respect to
!              SA and CT at constant p.                  [ J/(kg K(g/kg)) ]
!--------------------------------------------------------------------------
*/
void
gsw_specvol_first_derivatives_wrt_enthalpy(double sa, double ct, double p,
        double *v_sa, double *v_h)
{
        double  h_ct=1.0, h_sa, rec_h_ct, vct_ct, vct_sa;

        if (v_sa != NULL) {

            gsw_specvol_first_derivatives(sa,ct,p,&vct_sa,&vct_ct,NULL);
            gsw_enthalpy_first_derivatives(sa,ct,p,&h_sa,&h_ct);

        } else if (v_h != NULL) {

            gsw_specvol_first_derivatives(sa,ct,p,NULL,&vct_ct,NULL);
            gsw_enthalpy_first_derivatives(sa,ct,p,NULL,&h_ct);

        }

        rec_h_ct = 1.0/h_ct;

        if (v_sa != NULL)
            *v_sa = vct_sa - (vct_ct*h_sa)*rec_h_ct;

        if (v_h != NULL)
            *v_h = vct_ct*rec_h_ct;

        return;
}
/*
!==========================================================================
elemental function gsw_specvol_ice (t, p)
!==========================================================================
!
!  Calculates the specific volume of ice.
!
!  t  =  in-situ temperature (ITS-90)                             [ deg C ]
!  p  =  sea pressure                                              [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  specvol_ice  =  specific volume                               [ m^3/kg ]
!--------------------------------------------------------------------------
*/
double
gsw_specvol_ice(double t, double p)
{
        return (gsw_gibbs_ice(0,1,t,p));
}
/*
!==========================================================================
elemental subroutine gsw_specvol_second_derivatives (sa, ct, p, v_sa_sa, &
                                   v_sa_ct, v_ct_ct, v_sa_p, v_ct_p, iflag)
! =========================================================================
!
!  Calculates five second-order derivatives of specific volume (v).
!  Note that this function uses the computationally-efficient
!  expression for specific volume (Roquet et al., 2014).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  v_SA_SA  =  The second derivative of specific volume with respect to
!              Absolute Salinity at constant CT & p.    [ J/(kg (g/kg)^2) ]
!  v_SA_CT  =  The second derivative of specific volume with respect to
!              SA and CT at constant p.                  [ J/(kg K(g/kg)) ]
!  v_CT_CT  =  The second derivative of specific volume with respect to
!              CT at constant SA and p.                      [ J/(kg K^2) ]
!  v_SA_P  =  The second derivative of specific volume with respect to
!              SA and P at constant CT.                  [ J/(kg K(g/kg)) ]
!  v_CT_P  =  The second derivative of specific volume with respect to
!              CT and P at constant SA.                  [ J/(kg K(g/kg)) ]
!--------------------------------------------------------------------------
*/
void
gsw_specvol_second_derivatives (double sa, double ct, double p,
        double *v_sa_sa, double *v_sa_ct, double *v_ct_ct, double *v_sa_p,
        double *v_ct_p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_SPECVOL_COEFFICIENTS;
        double  v_ct_ct_part, v_ct_p_part, v_sa_ct_part, v_sa_p_part,
                v_sa_sa_part, xs, xs2, ys, z;

        xs2 = gsw_sfac*sa + offset;
        xs = sqrt(xs2);
        ys = ct*0.025;
        z = p*1e-4;

        if (v_sa_sa != NULL) {

            v_sa_sa_part = (-b000 + xs2*(b200 + xs*(2.0*b300 + xs*(3.0*b400
                + 4.0*b500*xs))) + ys*(-b010 + xs2*(b210 + xs*(2.0*b310
                + 3.0*b410*xs)) + ys*(-b020 + xs2*(b220 + 2.0*b320*xs)
                + ys*(-b030 + b230*xs2 + ys*(-b040 - b050*ys)))) + z*(-b001
                + xs2*(b201 + xs*(2.0*b301 + 3.0*b401*xs)) + ys*(-b011
                + xs2*(b211 + 2.0*b311*xs) + ys*(-b021 + b221*xs2
                + ys*(-b031 - b041*ys))) + z*(-b002 + xs2*(b202 + 2.0*b302*xs)
                + ys*(-b012 + b212*xs2 + ys*(-b022 - b032*ys)) + z*(-b003
                - b013*ys - b004*z))))/xs2;

            *v_sa_sa = 0.25*gsw_sfac*gsw_sfac*v_sa_sa_part/xs;

        }

        if (v_sa_ct != NULL) {

            v_sa_ct_part = (b010 + xs*(b110 + xs*(b210 + xs*(b310 + b410*xs)))
                + ys*(2.0*(b020 + xs*(b120 + xs*(b220 + b320*xs)))
                + ys*(3.0*(b030 + xs*(b130 + b230*xs)) + ys*(4.0*(b040
                + b140*xs) + 5.0*b050*ys))) + z*(b011 + xs*(b111 + xs*(b211
                + b311*xs)) + ys*(2.0*(b021 + xs*(b121 + b221*xs))
                + ys*(3.0*(b031 + b131*xs) + 4.0*b041*ys)) + z*(b012
                + xs*(b112 + b212*xs) + ys*(2.0*(b022 + b122*xs)
                + 3.0*b032*ys) + b013*z)))/xs;

            *v_sa_ct = 0.025*0.5*gsw_sfac*v_sa_ct_part;

        }

        if (v_ct_ct != NULL) {

            v_ct_ct_part = a010 + xs*(a110 + xs*(a210 + xs*(a310 + a410*xs)))
                + ys*(2.0*(a020 + xs*(a120 + xs*(a220 + a320*xs)))
                + ys*(3.0*(a030 + xs*(a130 + a230*xs)) + ys*(4.0*(a040
                + a140*xs) + 5.0*a050*ys))) + z*( a011 + xs*(a111 + xs*(a211
                + a311*xs)) + ys*(2.0*(a021 + xs*(a121 + a221*xs))
                + ys*(3.0*(a031 + a131*xs) + 4.0*a041*ys)) + z*(a012
                + xs*(a112 + a212*xs) + ys*(2.0*(a022 + a122*xs)
                + 3.0*a032*ys) + a013*z));

            *v_ct_ct = 0.025*0.025*v_ct_ct_part;

        }

        if (v_sa_p != NULL) {

            v_sa_p_part = b001 + xs*(b101 + xs*(b201 + xs*(b301
                + b401*xs))) + ys*(b011 + xs*(b111 + xs*(b211
                + b311*xs)) + ys*(b021 + xs*(b121 + b221*xs)
                + ys*(b031 + b131*xs + b041*ys))) + z*(2.0*(b002 + xs*(b102
                + xs*(b202 + b302*xs)) + ys*(b012 + xs*(b112
                + b212*xs) + ys*(b022
                + b122*xs + b032*ys))) + z*(3.0*(b003 + b103*xs + b013*ys)
                + 4.0*b004*z));

            *v_sa_p = 1e-8*0.5*gsw_sfac*v_sa_p_part/xs;

        }

        if (v_ct_p != NULL) {

            v_ct_p_part = a001 + xs*(a101 + xs*(a201 + xs*(a301
                + a401*xs))) + ys*(a011
                + xs*(a111 + xs*(a211 + a311*xs)) + ys*(a021
                + xs*(a121 + a221*xs)
                + ys*(a031 + a131*xs + a041*ys))) + z*(2.0*(a002 + xs*(a102
                + xs*(a202 + a302*xs)) + ys*(a012 + xs*(a112 + a212*xs)
                + ys*(a022 + a122*xs + a032*ys))) + z*(3.0*(a003
                + a103*xs + a013*ys) + 4.0*a004*z));

            *v_ct_p = 1e-8*0.025*v_ct_p_part;

        }
}
/*
!==========================================================================
elemental subroutine gsw_specvol_second_derivatives_wrt_enthalpy (sa, ct, &
                                          p, v_sa_sa, v_sa_h, v_h_h, iflag)
! =========================================================================
!
!  Calculates three first-order derivatives of specific volume (v) with
!  respect to enthalpy. Note that this function uses the using the
!  computationally-efficient expression for specific volume
!  (Roquet et al., 2014).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  v_SA_SA = The second-order derivative of specific volume with respect to
!            Absolute Salinity at constant h & p.       [ J/(kg (g/kg)^2) ]
!  v_SA_h  = The second-order derivative of specific volume with respect to
!            SA and h at constant p.                     [ J/(kg K(g/kg)) ]
!  v_h_h   = The second-order derivative with respect to h at
!            constant SA & p.
!--------------------------------------------------------------------------
*/
void
gsw_specvol_second_derivatives_wrt_enthalpy (double sa, double ct, double p,
        double *v_sa_sa, double *v_sa_h, double *v_h_h)
{
        double  h_ct, h_ct_ct, h_sa, h_sa_ct, h_sa_sa, rec_h_ct, v_h_h_part,
                rec_h_ct2, v_ct, vct_ct_ct, vct_sa_ct, vct_sa_sa, v_sa_h_part;

        gsw_specvol_first_derivatives(sa,ct,p,NULL, &v_ct, NULL);

        if ((v_sa_sa != NULL) || (v_sa_h != NULL))
           gsw_enthalpy_first_derivatives(sa,ct,p,&h_sa,&h_ct);
        else
           gsw_enthalpy_first_derivatives(sa,ct,p,NULL,&h_ct);

        if (v_sa_sa != NULL)
           gsw_specvol_second_derivatives(sa,ct,p,&vct_sa_sa,&vct_sa_ct,
                                                &vct_ct_ct, NULL, NULL);
        else if (v_sa_h != NULL)
           gsw_specvol_second_derivatives(sa,ct,p,NULL,&vct_sa_ct,&vct_ct_ct,
                                                                NULL, NULL);
        else
           gsw_specvol_second_derivatives(sa,ct,p,NULL,NULL,&vct_ct_ct,
                                                                NULL, NULL);

        if (v_sa_sa != NULL)
           gsw_enthalpy_second_derivatives(sa,ct,p,&h_sa_sa,&h_sa_ct,&h_ct_ct);
        else if (v_sa_h != NULL)
           gsw_enthalpy_second_derivatives(sa,ct,p,NULL,&h_sa_ct,&h_ct_ct);
        else
           gsw_enthalpy_second_derivatives(sa,ct,p,NULL,NULL,&h_ct_ct);

        rec_h_ct = 1.0/h_ct;
        rec_h_ct2 = rec_h_ct*rec_h_ct;

        v_h_h_part = (vct_ct_ct*h_ct - h_ct_ct*v_ct)*(rec_h_ct2*rec_h_ct);

        if (v_h_h != NULL) *v_h_h = v_h_h_part;

        if ((v_sa_sa != NULL) || (v_sa_h != NULL)) {

            v_sa_h_part = (vct_sa_ct*h_ct - v_ct*h_sa_ct)*rec_h_ct2
                        - h_sa*v_h_h_part;

            if (v_sa_h != NULL) *v_sa_h = v_sa_h_part;

            if (v_sa_sa != NULL)
                *v_sa_sa = vct_sa_sa - (h_ct*(vct_sa_ct*h_sa
                        + v_ct*h_sa_sa) - v_ct*h_sa*h_sa_ct)*rec_h_ct2
                        - h_sa*v_sa_h_part;
        }
}
/*
!==========================================================================
function gsw_specvol_sso_0(p)
!==========================================================================

!  This function calculates specific volume at the Standard Ocean Salinty,
!  SSO, and at a Conservative Temperature of zero degrees C, as a function
!  of pressure, p, in dbar, using a streamlined version of the CT version
!  of specific volume, that is, a streamlined version of the code
!  "gsw_specvol(SA,CT,p)".
!
! p      : sea pressure                                    [dbar]
!                                                            3   -1
! specvol_sso_0 : specvol(sso,0,p)                         [m  kg  ]
*/
double
gsw_specvol_sso_0(double p)
{
        GSW_SPECVOL_COEFFICIENTS;
        double  z, return_value;

        z = p*1.0e-4;

        return_value = 9.726613854843870e-04 + z*(-4.505913211160929e-05
                + z*(7.130728965927127e-06 + z*(-6.657179479768312e-07
                + z*(-2.994054447232880e-08 + z*(v005 + v006*z)))));
        return (return_value);
}
/*
!==========================================================================
function gsw_specvol_t_exact(sa,t,p)
!==========================================================================

! Calculates the specific volume of seawater
!
! sa     : Absolute Salinity                               [g/kg]
! t      : in-situ temperature                             [deg C]
! p      : sea pressure                                    [dbar]
!
! specvol_t_exact : specific volume                        [kg/m^3]
*/
double
gsw_specvol_t_exact(double sa, double t, double p)
{
        int     n0=0, n1=1;

        return (gsw_gibbs(n0,n0,n1,sa,t,p));
}
/*
!==========================================================================
function gsw_spiciness0(sa,ct)
!==========================================================================
!
!  Calculates spiciness from Absolute Salinity and Conservative
!  Temperature at a pressure of 0 dbar, as described by McDougall and
!  Krzysik (2015).  This routine is based on the computationally-efficient
!  expression for specific volume in terms of SA, CT and p (Roquet et al.,
!  2015).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                        [ deg C ]
!
!  spiciness0  =  spiciness referenced to a pressure of 0 dbar,
!                 i.e. the surface                                 [ kg/m^3 ]
!
*/

double
gsw_spiciness0(double sa, double ct)
{
        GSW_TEOS10_CONSTANTS;
        double  s01 = -9.22982898371678e1,      s02 = -1.35727873628866e1,
                s03 =  1.87353650994010e1,      s04 = -1.61360047373455e1,
                s05 =  3.76112762286425e1,      s06 = -4.27086671461257e1,
                s07 =  2.00820111041594e1,      s08 =  2.87969717584045e2,
                s09 =  1.13747111959674e1,      s10 =  6.07377192990680e1,
                s11 = -7.37514033570187e1,      s12 = -7.51171878953574e1,
                s13 =  1.63310989721504e2,      s14 = -8.83222751638095e1,
                s15 = -6.41725302237048e2,      s16 =  2.79732530789261e1,
                s17 = -2.49466901993728e2,      s18 =  3.26691295035416e2,
                s19 =  2.66389243708181e1,      s20 = -2.93170905757579e2,
                s21 =  1.76053907144524e2,      s22 =  8.27634318120224e2,
                s23 = -7.02156220126926e1,      s24 =  3.82973336590803e2,
                s25 = -5.06206828083959e2,      s26 =  6.69626565169529e1,
                s27 =  3.02851235050766e2,      s28 = -1.96345285604621e2,
                s29 = -5.74040806713526e2,      s30 =  7.03285905478333e1,
                s31 = -2.97870298879716e2,      s32 =  3.88340373735118e2,
                s33 = -8.29188936089122e1,      s34 = -1.87602137195354e2,
                s35 =  1.27096944425793e2,      s36 =  2.11671167892147e2,
                s37 = -3.15140919876285e1,      s38 =  1.16458864953602e2,
                s39 = -1.50029730802344e2,      s40 =  3.76293848660589e1,
                s41 =  6.47247424373200e1,      s42 = -4.47159994408867e1,
                s43 = -3.23533339449055e1,      s44 =  5.30648562097667,
                s45 = -1.82051249177948e1,      s46 =  2.33184351090495e1,
                s47 = -6.22909903460368,        s48 = -9.55975464301446,
                s49 =  6.61877073960113;
        double  xs, ys, spiciness0;

        xs      = sqrt(gsw_sfac*sa + offset);
        ys      = ct*0.025;

        spiciness0= s01+ys*(s02+ys*(s03+ys*(s04+ys*(s05+ys*(s06+s07*ys)))))
                +xs*(s08+ys*(s09+ys*(s10+ys*(s11+ys*(s12+ys*(s13+s14*ys)))))
                +xs*(s15+ys*(s16+ys*(s17+ys*(s18+ys*(s19+ys*(s20+s21*ys)))))
                +xs*(s22+ys*(s23+ys*(s24+ys*(s25+ys*(s26+ys*(s27+s28*ys)))))
                +xs*(s29+ys*(s30+ys*(s31+ys*(s32+ys*(s33+ys*(s34+s35*ys)))))
                +xs*(s36+ys*(s37+ys*(s38+ys*(s39+ys*(s40+ys*(s41+s42*ys)))))
                +xs*(s43+ys*(s44+ys*(s45+ys*(s46+ys*(s47+ys*(s48+s49*ys)))))
                ))))));
        return (spiciness0);
}
/*
!==========================================================================
function gsw_spiciness1(sa,ct)
!==========================================================================
!
!  Calculates spiciness from Absolute Salinity and Conservative
!  Temperature at a pressure of 1000 dbar, as described by McDougall and
!  Krzysik (2015).  This routine is based on the computationally-efficient
!  expression for specific volume in terms of SA, CT and p (Roquet et al.,
!  2015).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!
!  spiciness1  =  spiciness referenced to a pressure of 1000 dbar [ kg/m^3 ]
!
*/
double
gsw_spiciness1(double sa, double ct)
{
        GSW_TEOS10_CONSTANTS;
        double  s01 = -9.19874584868912e1,      s02 = -1.33517268529408e1,
                s03 =  2.18352211648107e1,      s04 = -2.01491744114173e1,
                s05 =  3.70004204355132e1,      s06 = -3.78831543226261e1,
                s07 =  1.76337834294554e1,      s08 =  2.87838842773396e2,
                s09 =  2.14531420554522e1,      s10 =  3.14679705198796e1,
                s11 = -4.04398864750692e1,      s12 = -7.70796428950487e1,
                s13 =  1.36783833820955e2,      s14 = -7.36834317044850e1,
                s15 = -6.41753415180701e2,      s16 =  1.33701981685590,
                s17 = -1.75289327948412e2,      s18 =  2.42666160657536e2,
                s19 =  3.17062400799114e1,      s20 = -2.28131490440865e2,
                s21 =  1.39564245068468e2,      s22 =  8.27747934506435e2,
                s23 = -3.50901590694775e1,      s24 =  2.87473907262029e2,
                s25 = -4.00227341144928e2,      s26 =  6.48307189919433e1,
                s27 =  2.16433334701578e2,      s28 = -1.48273032774305e2,
                s29 = -5.74545648799754e2,      s30 =  4.50446431127421e1,
                s31 = -2.30714981343772e2,      s32 =  3.15958389253065e2,
                s33 = -8.60635313930106e1,      s34 = -1.22978455069097e2,
                s35 =  9.18287282626261e1,      s36 =  2.12120473062203e2,
                s37 = -2.21528216973820e1,      s38 =  9.19013417923270e1,
                s39 = -1.24400776026014e2,      s40 =  4.08512871163839e1,
                s41 =  3.91127352213516e1,      s42 = -3.10508021853093e1,
                s43 = -3.24790035899152e1,      s44 =  3.91029016556786,
                s45 = -1.45362719385412e1,      s46 =  1.96136194246355e1,
                s47 = -7.06035474689088,        s48 = -5.36884688614009,
                s49 =  4.43247303092448;
        double  xs, ys, spiciness1;

        xs      = sqrt(gsw_sfac*sa + offset);
        ys      = ct*0.025;
        spiciness1= s01+ys*(s02+ys*(s03+ys*(s04+ys*(s05+ys*(s06+s07*ys)))))
                +xs*(s08+ys*(s09+ys*(s10+ys*(s11+ys*(s12+ys*(s13+s14*ys)))))
                +xs*(s15+ys*(s16+ys*(s17+ys*(s18+ys*(s19+ys*(s20+s21*ys)))))
                +xs*(s22+ys*(s23+ys*(s24+ys*(s25+ys*(s26+ys*(s27+s28*ys)))))
                +xs*(s29+ys*(s30+ys*(s31+ys*(s32+ys*(s33+ys*(s34+s35*ys)))))
                +xs*(s36+ys*(s37+ys*(s38+ys*(s39+ys*(s40+ys*(s41+s42*ys)))))
                +xs*(s43+ys*(s44+ys*(s45+ys*(s46+ys*(s47+ys*(s48+s49*ys)))))
                ))))));
        return (spiciness1);
}
/*
!==========================================================================
function gsw_spiciness2(sa,ct)
!==========================================================================
!
!  Calculates spiciness from Absolute Salinity and Conservative
!  Temperature at a pressure of 2000 dbar, as described by McDougall and
!  Krzysik (2015).  This routine is based on the computationally-efficient
!  expression for specific volume in terms of SA, CT and p (Roquet et al.,
!  2015).
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
!
!  spiciness2  =  spiciness referenced to a pressure of 2000 dbar [ kg/m^3 ]
!
*/
double
gsw_spiciness2(double sa, double ct)
{
        GSW_TEOS10_CONSTANTS;
        double  s01 = -9.17327320732265e1,      s02 = -1.31200235147912e1,
                s03 =  2.49574345782503e1,      s04 = -2.41678075247398e1,
                s05 =  3.61654631402053e1,      s06 = -3.22582164667710e1,
                s07 =  1.45092623982509e1,      s08 =  2.87776645983195e2,
                s09 =  3.13902307672447e1,      s10 =  1.69777467534459,
                s11 = -5.69630115740438,        s12 = -7.97586359017987e1,
                s13 =  1.07507460387751e2,      s14 = -5.58234404964787e1,
                s15 = -6.41708068766557e2,      s16 = -2.53494801286161e1,
                s17 = -9.86755437385364e1,      s18 =  1.52406930795842e2,
                s19 =  4.23888258264105e1,      s20 = -1.60118811141438e2,
                s21 =  9.67497898053989e1,      s22 =  8.27674355478637e2,
                s23 =  5.27561234412133e-1,     s24 =  1.87440206992396e2,
                s25 = -2.83295392345171e2,      s26 =  5.14485994597635e1,
                s27 =  1.29975755062696e2,      s28 = -9.36526588377456e1,
                s29 = -5.74911728972948e2,      s30 =  1.91175851862772e1,
                s31 = -1.59347231968841e2,      s32 =  2.33884725744938e2,
                s33 = -7.87744010546157e1,      s34 = -6.04757235443685e1,
                s35 =  5.27869695599657e1,      s36 =  2.12517758478878e2,
                s37 = -1.24351794740528e1,      s38 =  6.53904308937490e1,
                s39 = -9.44804080763788e1,      s40 =  3.93874257887364e1,
                s41 =  1.49425448888996e1,      s42 = -1.62350721656367e1,
                s43 = -3.25936844276669e1,      s44 =  2.44035700301595,
                s45 = -1.05079633683795e1,      s46 =  1.51515796259082e1,
                s47 = -7.06609886460683,        s48 = -1.48043337052968,
                s49 =  2.10066653978515;
        double  xs, ys, spiciness2;

        xs      = sqrt(gsw_sfac*sa + offset);
        ys      = ct*0.025;

        spiciness2= s01+ys*(s02+ys*(s03+ys*(s04+ys*(s05+ys*(s06+s07*ys)))))
                +xs*(s08+ys*(s09+ys*(s10+ys*(s11+ys*(s12+ys*(s13+s14*ys)))))
                +xs*(s15+ys*(s16+ys*(s17+ys*(s18+ys*(s19+ys*(s20+s21*ys)))))
                +xs*(s22+ys*(s23+ys*(s24+ys*(s25+ys*(s26+ys*(s27+s28*ys)))))
                +xs*(s29+ys*(s30+ys*(s31+ys*(s32+ys*(s33+ys*(s34+s35*ys)))))
                +xs*(s36+ys*(s37+ys*(s38+ys*(s39+ys*(s40+ys*(s41+s42*ys)))))
                +xs*(s43+ys*(s44+ys*(s45+ys*(s46+ys*(s47+ys*(s48+s49*ys)))))
                ))))));
        return (spiciness2);
}
/*
!==========================================================================
function gsw_sr_from_sp(sp)
!==========================================================================

! Calculates Reference Salinity, SR, from Practical Salinity, SP.
!
! sp     : Practical Salinity                              [unitless]
!
! gsw_sr_from_sp : Reference Salinity                      [g/kg]
*/
double
gsw_sr_from_sp(double sp)
{
        GSW_TEOS10_CONSTANTS;

        return (sp*gsw_ups);
}
/*
!==========================================================================
function gsw_sstar_from_sa(sa,p,lon,lat)
!==========================================================================

! Calculates Preformed Salinity, Sstar, from Absolute Salinity, SA.
!
! sa     : Absolute Salinity                               [g/kg]
! p      : sea pressure                                    [dbar]
! lon   : longitude                                       [deg E]
! lat    : latitude                                        [deg N]
!
! gsw_sstar_from_sa : Preformed Salinity                   [g/kg]
*/
double
gsw_sstar_from_sa(double sa, double p, double lon, double lat)
{
        double  saar;

        saar    = gsw_saar(p,lon,lat);
    /*
        ! In the Baltic Sea, Sstar = sa, and note that gsw_saar returns zero
        ! for saar in the Baltic.
    */
        if (saar == GSW_INVALID_VALUE)
            return (saar);
        return (sa*(1e0 - 0.35e0*saar)/(1e0 + saar));
}
/*
!==========================================================================
function gsw_sstar_from_sp(sp,p,lon,lat)
!==========================================================================

! Calculates Preformed Salinity, Sstar, from Practical Salinity, SP.
!
! sp     : Practical Salinity                              [unitless]
! p      : sea pressure                                    [dbar]
! lon    : longitude                                       [deg E]
! lat    : latitude                                        [deg N]
!
! gsw_sstar_from_sp  : Preformed Salinity                  [g/kg]
*/
double
gsw_sstar_from_sp(double sp, double p, double lon, double lat)
{
        GSW_TEOS10_CONSTANTS;
        double  saar, sstar_baltic;

    /*
        !In the Baltic Sea, Sstar = SA.
    */
        sstar_baltic    = gsw_sa_from_sp_baltic(sp,lon,lat);
        if (sstar_baltic < GSW_ERROR_LIMIT)
            return (sstar_baltic);
        saar            = gsw_saar(p,lon,lat);
        if (saar == GSW_INVALID_VALUE)
            return (saar);
        return (gsw_ups*sp*(1 - 0.35e0*saar));
}
/*
!==========================================================================
elemental function gsw_t_deriv_chem_potential_water_t_exact (sa, t, p)
!==========================================================================
!
!  Calculates the temperature derivative of the chemical potential of water
!  in seawater so that it is valid at exactly SA = 0.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  t   =  in-situ temperature (ITS-90)                            [ deg C ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!
!  chem_potential_water_dt  =  temperature derivative of the chemical
!                           potential of water in seawater  [ J g^-1 K^-1 ]
!--------------------------------------------------------------------------
*/
double
gsw_t_deriv_chem_potential_water_t_exact(double sa, double t, double p)
{
        GSW_TEOS10_CONSTANTS;
        double  g03_t, g08_sa_t, x, x2, y, z, g08_t, kg2g = 1e-3;
/*
! Note. The kg2g, a factor of 1e-3, is needed to convert the output of this
! function into units of J/g. See section (2.9) of the TEOS-10 Manual.
*/

        x2 = gsw_sfac*sa;
        x = sqrt(x2);
        y = t*0.025;
        z = p*rec_db2pa;
            /* the input pressure (p) is sea pressure in units of dbar. */

        g03_t = 5.90578347909402 + z*(-270.983805184062 +
        z*(776.153611613101 + z*(-196.51255088122 + (28.9796526294175 -
        2.13290083518327*z)*z))) +
        y*(-24715.571866078 + z*(2910.0729080936 +
        z*(-1513.116771538718 + z*(546.959324647056 +
        z*(-111.1208127634436 + 8.68841343834394*z)))) +
        y*(2210.2236124548363 + z*(-2017.52334943521 +
        z*(1498.081172457456 + z*(-718.6359919632359 +
        (146.4037555781616 - 4.9892131862671505*z)*z))) +
        y*(-592.743745734632 + z*(1591.873781627888 +
        z*(-1207.261522487504 + (608.785486935364 -
        105.4993508931208*z)*z)) +
        y*(290.12956292128547 + z*(-973.091553087975 +
        z*(602.603274510125 + z*(-276.361526170076 +
        32.40953340386105*z))) +
        y*(-113.90630790850321 + y*(21.35571525415769 -
        67.41756835751434*z) +
        z*(381.06836198507096 + z*(-133.7383902842754 +
        49.023632509086724*z)))))));

        g08_t = x2*(168.072408311545 +
        x*(-493.407510141682 + x*(543.835333000098 +
        x*(-196.028306689776 + 36.7571622995805*x) +
        y*(-137.1145018408982 + y*(148.10030845687618 +
        y*(-68.5590309679152 + 12.4848504784754*y))) -
        22.6683558512829*z) + z*(-175.292041186547 +
        (83.1923927801819 - 29.483064349429*z)*z) +
        y*(-86.1329351956084 + z*(766.116132004952 +
        z*(-108.3834525034224 + 51.2796974779828*z)) +
        y*(-30.0682112585625 - 1380.9597954037708*z +
        y*(3.50240264723578 + 938.26075044542*z)))));

        g08_sa_t = 1187.3715515697959 +
        x*(-1480.222530425046 + x*(2175.341332000392 +
        x*(-980.14153344888 + 220.542973797483*x) +
        y*(-548.4580073635929 + y*(592.4012338275047 +
        y*(-274.2361238716608 + 49.9394019139016*y))) -
        90.6734234051316*z) + z*(-525.876123559641 +
        (249.57717834054571 - 88.449193048287*z)*z) +
        y*(-258.3988055868252 + z*(2298.348396014856 +
        z*(-325.1503575102672 + 153.8390924339484*z)) +
        y*(-90.2046337756875 - 4142.8793862113125*z +
        y*(10.50720794170734 + 2814.78225133626*z))));

        return (kg2g*((g03_t + g08_t)*0.025 - 0.5*gsw_sfac*0.025*sa*g08_sa_t));
}
/*
!==========================================================================
function gsw_t_freezing(sa,p,saturation_fraction)
!==========================================================================

! Calculates the in-situ temperature at which seawater freezes
!
! sa     : Absolute Salinity                                 [g/kg]
! p      : sea pressure                                      [dbar]
!         ( i.e. absolute pressure - 10.1325 dbar )
! saturation_fraction : the saturation fraction of dissolved air
!                       in seawater
!
! t_freezing : in-situ temperature at which seawater freezes.[deg C]
*/
double
gsw_t_freezing(double sa, double p, double saturation_fraction)
{
        GSW_TEOS10_CONSTANTS;
        GSW_FREEZING_POLY_COEFFICIENTS;
        double sa_r, x, p_r;
        double  df_dt, tf, tfm, tf_old, f, return_value;

        /* The initial value of t_freezing_exact (for air-free seawater) */
        sa_r = sa*1e-2;
        x = sqrt(sa_r);
        p_r = p*1e-4;

        tf = t0
        + sa_r*(t1 + x*(t2 + x*(t3 + x*(t4 + x*(t5 + t6*x)))))
        + p_r*(t7 + p_r*(t8 + t9*p_r))
        + sa_r*p_r*(t10 + p_r*(t12 + p_r*(t15 + t21*sa_r))
        + sa_r*(t13 + t17*p_r + t19*sa_r)
        + x*(t11 + p_r*(t14 + t18*p_r) + sa_r*(t16 + t20*p_r
        + t22*sa_r)));

        /* Adjust for the effects of dissolved air */
        tf -= saturation_fraction*(1e-3)*(2.4 - sa/(2.0*gsw_sso));

        df_dt = 1e3*gsw_t_deriv_chem_potential_water_t_exact(sa,tf,p) -
                gsw_gibbs_ice(1,0,tf,p);
/*
! df_dt here is the initial value of the derivative of the function f whose
! zero (f = 0) we are finding (see Eqn. (3.33.2) of IOC et al (2010)).
*/

        tf_old = tf;
        f = 1e3*gsw_chem_potential_water_t_exact(sa,tf_old,p) -
                gsw_gibbs_ice(0,0,tf_old,p);
        tf = tf_old - f/df_dt;
        tfm = 0.5*(tf + tf_old);
        df_dt = 1e3*gsw_t_deriv_chem_potential_water_t_exact(sa,tfm,p) -
                gsw_gibbs_ice(1,0,tfm,p);
        tf = tf_old - f/df_dt;

        tf_old = tf;
        f = 1e3*gsw_chem_potential_water_t_exact(sa,tf_old,p) -
                gsw_gibbs_ice(0,0,tf_old,p);
        tf = tf_old - f/df_dt;

        /* Adjust for the effects of dissolved air */
        return_value = tf -
                saturation_fraction*(1e-3)*(2.4 - sa/(2.0*gsw_sso));
        return (return_value);
}
/*
!==========================================================================
elemental subroutine gsw_t_freezing_first_derivatives (sa, p, &
                            saturation_fraction, tfreezing_sa, tfreezing_p)
!==========================================================================
!
!  Calculates the first derivatives of the in-situ temperature at which
!  seawater freezes with respect to Absolute Salinity SA and pressure P (in
!  Pa).  These expressions come from differentiating the expression that
!  defines the freezing temperature, namely the equality between the
!  chemical potentials of water in seawater and in ice.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  saturation_fraction = the saturation fraction of dissolved air in
!                        seawater
!
!  tfreezing_SA = the derivative of the in-situ freezing temperature
!                 (ITS-90) with respect to Absolute Salinity at fixed
!                 pressure                     [ K/(g/kg) ] i.e. [ K kg/g ]
!
!  tfreezing_P  = the derivative of the in-situ freezing temperature
!                 (ITS-90) with respect to pressure (in Pa) at fixed
!                 Absolute Salinity                                [ K/Pa ]
!--------------------------------------------------------------------------
*/
void
gsw_t_freezing_first_derivatives(double sa, double p,
        double saturation_fraction, double *tfreezing_sa, double *tfreezing_p)
{
        GSW_TEOS10_CONSTANTS;
        double  rec_denom, tf, g_per_kg = 1000.0;

        tf = gsw_t_freezing(sa,p,saturation_fraction);
        rec_denom = 1.0/
                (g_per_kg*gsw_t_deriv_chem_potential_water_t_exact(sa,tf,p)
                + gsw_entropy_ice(tf,p));

        if (tfreezing_sa != NULL)
            *tfreezing_sa =
                       gsw_dilution_coefficient_t_exact(sa,tf,p)*rec_denom
                       + saturation_fraction*(1e-3)/(2.0*gsw_sso);

        if (tfreezing_p != NULL)
            *tfreezing_p =
                -(gsw_specvol_t_exact(sa,tf,p) - sa*gsw_gibbs(1,0,1,sa,tf,p)
                - gsw_specvol_ice(tf,p))*rec_denom;

        return;
}
/*
!==========================================================================
elemental subroutine gsw_t_freezing_first_derivatives_poly (sa, p, &
                            saturation_fraction, tfreezing_sa, tfreezing_p)
!==========================================================================
!
!  Calculates the first derivatives of the in-situ temperature at which
!  seawater freezes with respect to Absolute Salinity SA and pressure P (in
!  Pa).  These expressions come from differentiating the expression that
!  defines the freezing temperature, namely the equality between the
!  chemical potentials of water in seawater and in ice.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  saturation_fraction = the saturation fraction of dissolved air in
!                        seawater
!
!  tfreezing_SA = the derivative of the in-situ freezing temperature
!                 (ITS-90) with respect to Absolute Salinity at fixed
!                 pressure                     [ K/(g/kg) ] i.e. [ K kg/g ]
!
!  tfreezing_P  = the derivative of the in-situ freezing temperature
!                 (ITS-90) with respect to pressure (in Pa) at fixed
!                 Absolute Salinity                                [ K/Pa ]
!--------------------------------------------------------------------------
*/
void
gsw_t_freezing_first_derivatives_poly(double sa, double p,
        double saturation_fraction, double *tfreezing_sa, double *tfreezing_p)
{
        GSW_TEOS10_CONSTANTS;
        GSW_FREEZING_POLY_COEFFICIENTS;
        double  p_r, sa_r, x, c = 1e-3/(2.0*gsw_sso);

        sa_r = sa*1e-2;
        x = sqrt(sa_r);
        p_r = p*1e-4;

        if (tfreezing_sa != NULL)
            *tfreezing_sa =
            (t1 + x*(1.5*t2 + x*(2.0*t3 + x*(2.5*t4 + x*(3.0*t5
                + 3.5*t6*x)))) + p_r*(t10 + x*(1.5*t11 + x*(2.0*t13
                + x*(2.5*t16 + x*(3.0*t19 + 3.5*t22*x))))
                + p_r*(t12 + x*(1.5*t14 + x*(2.0*t17 + 2.5*t20*x))
                + p_r*(t15 + x*(1.5*t18 + 2.0*t21*x)))))*1e-2
                + saturation_fraction*c;

        if (tfreezing_p != NULL)
            *tfreezing_p =
            (t7 + sa_r*(t10 + x*(t11 + x*(t13 + x*(t16 + x*(t19 + t22*x)))))
                + p_r*(2.0*t8 + sa_r*(2.0*t12 + x*(2.0*t14 + x*(2.0*t17
                + 2.0*t20*x))) + p_r*(3.0*t9 + sa_r*(3.0*t15 + x*(3.0*t18
                + 3.0*t21*x)))))*1e-8;

        return;
}
/*
!==========================================================================
elemental function gsw_t_freezing_poly (sa, p, saturation_fraction)
!==========================================================================
!
!  Calculates the in-situ temperature at which seawater freezes from a
!  computationally efficient polynomial.
!
!  SA  =  Absolute Salinity                                        [ g/kg ]
!  p   =  sea pressure                                             [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!  saturation_fraction = the saturation fraction of dissolved air in
!                        seawater
!
!  t_freezing = in-situ temperature at which seawater freezes.    [ deg C ]
!               (ITS-90)
!--------------------------------------------------------------------------
*/
double
gsw_t_freezing_poly(double sa, double p, double saturation_fraction)
{
        GSW_TEOS10_CONSTANTS;
        double ctf, return_value;

    ctf = gsw_ct_freezing_poly(sa,p,saturation_fraction);
    return_value = gsw_t_from_ct(sa,ctf,p);
        return (return_value);
}
/*
!==========================================================================
function gsw_t_from_ct(sa,ct,p)
!==========================================================================

! Calculates in-situ temperature from Conservative Temperature of seawater
!
! sa      : Absolute Salinity                              [g/kg]
! ct      : Conservative Temperature                       [deg C]
!
! gsw_t_from_ct : in-situ temperature                      [deg C]
*/
double
gsw_t_from_ct(double sa, double ct, double p)
{
        double  pt0, p0=0.0;

        pt0     = gsw_pt_from_ct(sa,ct);
        return (gsw_pt_from_t(sa,pt0,p0,p));
}
/*
! =========================================================================
elemental function gsw_t_from_pt0_ice (pt0_ice, p)
! =========================================================================
!
!  Calculates in-situ temperature from the potential temperature of ice Ih
!  with reference pressure, p_ref, of 0 dbar (the surface), and the
!  in-situ pressure.
!
!  pt0_ice  =  potential temperature of ice Ih with reference pressure of
!              zero dbar (ITS-90)                                 [ deg C ]
!  p        =  sea pressure                                        [ dbar ]
!         ( i.e. absolute pressure - 10.1325 dbar )
!--------------------------------------------------------------------------
*/
double
gsw_t_from_pt0_ice(double pt0_ice, double p)
{
        double  p0 = 0.0;

        return (gsw_pt_from_t_ice(pt0_ice,p0,p));
}
/*
!==========================================================================
function gsw_thermobaric(sa,ct,p)
!==========================================================================

!  Calculates the thermobaric coefficient of seawater with respect to
!  Conservative Temperature.  This routine is based on the
!  computationally-efficient expression for specific volume in terms of
!  SA, CT and p (Roquet et al., 2014).
!
! sa     : Absolute Salinity                               [g/kg]
! ct     : Conservative Temperature (ITS-90)               [deg C]
! p      : sea pressure                                    [dbar]
!
! thermobaric  : thermobaric coefficient with              [1/(K Pa)]
!                    respect to Conservative Temperature (48 term equation)
*/
double
gsw_thermobaric(double sa, double ct, double p)
{
        double  v_ct, v_ct_p, v_sa, v_sa_p;

        gsw_specvol_first_derivatives(sa,ct,p,&v_sa,&v_ct,NULL);

        gsw_specvol_second_derivatives(sa,ct,p,NULL,NULL,NULL,&v_sa_p,&v_ct_p);

        return (gsw_rho(sa,ct,p)*(v_ct_p - (v_ct/v_sa)*v_sa_p));
}
/*
!==========================================================================
subroutine gsw_turner_rsubrho(sa,ct,p,nz,tu,rsubrho,p_mid)
!==========================================================================

!  Calculates the Turner angle and the Rsubrho as a function of pressure
!  down a vertical water column.  These quantities express the relative
!  contributions of the vertical gradients of Conservative Temperature
!  and Absolute Salinity to the vertical stability (the square of the
!  Brunt-Vaisala Frequency squared, N^2).  Tu and Rsubrho are evaluated at
!  the mid pressure between the individual data points in the vertical.
!
!  Note that in the double-diffusive literature, papers concerned with
!  the "diffusive" form of double-diffusive convection often define the
!  stability ratio as the reciprocal of what is defined here as the
!  stability ratio.

! sa      : Absolute Salinity         (a profile (length nz))     [g/kg]
! ct      : Conservative Temperature  (a profile (length nz))     [deg C]
! p       : sea pressure              (a profile (length nz))     [dbar]
! nz      : number of bottles
! tu      : Turner angle, on the same (nz-1) grid as p_mid.
!           Turner angle has units of:           [ degrees of rotation ]
! rsubrho : Stability Ratio, on the same (nz-1) grid as p_mid.
!           Rsubrho is dimensionless.                       [ unitless ]
! p_mid   : Mid pressure between p grid  (length nz-1)           [dbar]
*/
void
gsw_turner_rsubrho(double *sa, double *ct, double *p, int nz,
        double *tu, double *rsubrho, double *p_mid)
{
        GSW_TEOS10_CONSTANTS;
        int     k;
        double  dsa, sa_mid, dct, ct_mid, alpha_mid, beta_mid;

        if (nz < 2)
            return;

        for (k = 0; k < nz-1; k++) {
            dsa         = (sa[k] - sa[k+1]);
            sa_mid      = 0.5e0*(sa[k] + sa[k+1]);
            dct         = (ct[k] - ct[k+1]);
            ct_mid      = 0.5e0*(ct[k] + ct[k+1]);
            p_mid[k]    = 0.5e0*(p[k] + p[k+1]);
            gsw_specvol_alpha_beta(sa_mid,ct_mid,p_mid[k],NULL,&alpha_mid,
                                        &beta_mid);
            tu[k] = rad2deg*atan2((alpha_mid*dct + beta_mid*dsa),
                                (alpha_mid*dct - beta_mid*dsa));
            if (dsa == 0.0)
                rsubrho[k] = GSW_INVALID_VALUE;
            else
                rsubrho[k] = (alpha_mid*dct)/(beta_mid*dsa);
        }
}
/*
!==========================================================================
subroutine gsw_util_indx(x,n,z,k)
!==========================================================================

!  Finds the index of the value in a monotonically increasing array
!
!  x     :  array of monotonically increasing values
!  n     :  length of the array
!  z     :  value to be indexed
!
!  K      : index K - if X(K) <= Z < X(K+1), or
!  N-1                      - if Z = X(N)
!
*/
int
gsw_util_indx(double *x, int n, double z)
{
        int     k, ku, kl, km;

        if (z > x[0] && z < x[n-1]) {
            kl  = 0;
            ku  = n-1;
            while (ku-kl > 1) {
                km      = (ku+kl)>>1;
                if (z > x[km])
                    kl  = km;
                else
                    ku  = km;
            }
            k   = kl;
            if (z == x[k+1])
                k++;
        } else if (z <= x[0])
            k   = 0;
        else
            k   = n-2;

        return (k);
}
/*
!==========================================================================
pure function gsw_util_interp1q_int (x, iy, x_i) result(y_i)
!==========================================================================
! Returns the value of the 1-D function iy (integer) at the points of column
! vector x_i using linear interpolation. The vector x specifies the
! coordinates of the underlying interval.
!==========================================================================
*/
double *
gsw_util_interp1q_int(int nx, double *x, int *iy, int nxi, double *x_i,
        double *y_i)
{
        char    *in_rng;
        int     *j, *k, *r, *jrev, *ki, imax_x, imin_x, i, n, m, ii;
        double  *xi, *xxi, u, max_x, min_x;

        if (nx <= 0 || nxi <= 0)
            return (NULL);

        min_x = max_x = x[0];
        imin_x = imax_x = 0;
        for (i=0; i<nx; i++) {
            if (x[i] < min_x) {
                min_x = x[i];
                imin_x = i;
            } else if (x[i] > max_x) {
                max_x = x[i];
                imax_x = i;
            }
        }
        in_rng = (char *) malloc(nxi*sizeof (char));
        memset(in_rng, 0, nxi*sizeof (char));

        for (i=n=0; i<nxi; i++) {
            if (x_i[i] <= min_x) {
                y_i[i] = iy[imin_x];
            } else if (x_i[i] >= max_x) {
                y_i[i] = iy[imax_x];
            } else {
                in_rng[i] = 1;
                n++;
            }
        }
        if (n==0)
            return (y_i);

        xi = (double *) malloc(n*sizeof (double));
        k  = (int *) malloc(3*n*sizeof (int)); ki = k+n; r = ki+n;
        m  = nx + n;
        xxi = (double *) malloc(m*sizeof (double));
        j = (int *) malloc(2*m*sizeof (int)); jrev = j+m;

        ii = 0;
        for (i = 0; i<nxi; i++) {
            if (in_rng[i]) {
                xi[ii] = x_i[i];
                ki[ii] = i;
                ii++;
            }
        }
        free(in_rng);
    /*
    **  Note that the following operations on the index
    **  vectors jrev and r depend on the sort utility
    **  gsw_util_sort_real() consistently ordering the
    **  sorting indexes either in ascending or descending
    **  sequence for replicate values in the real vector.
    */
        gsw_util_sort_real(xi, n, k);
        for (i = 0; i<nx; i++)
            xxi[i] = x[i];
        for (i = 0; i<n; i++)
            xxi[nx+i] = xi[k[i]];
        gsw_util_sort_real(xxi, nx+n, j);

        for (i = 0; i<nx+n; i++)
            jrev[j[i]] = i;
        for (i = 0; i<n; i++)
            r[k[i]] = jrev[nx+i] - i-1;

        for (i = 0; i<n; i++) {
            u = (xi[i]-x[r[i]])/(x[r[i]+1]-x[r[i]]);
            y_i[ki[i]] = iy[r[i]] + (iy[r[i]+1]-iy[r[i]])*u;
        }
        free(j); free(xxi); free(k); free(xi);
        return (y_i);
}
/*
!==========================================================================
pure function gsw_util_linear_interp (x, y, x_i) result(y_i)
!==========================================================================
! Returns the values of the functions y{ny} at the points of column
! vector x_i using linear interpolation. The vector x specifies the
! coordinates of the underlying interval, and the matrix y specifies
| the function values at each x coordinate. Note that y has dimensions
| nx x ny and y_i has dimensions nxi x ny.
! This function was adapted from Matlab's interp1q.
!==========================================================================
*/
double *
gsw_util_linear_interp(int nx, double *x, int ny, double *y, int nxi,
        double *x_i, double *y_i)
{
        char    *in_rng;
        int     *j, *k, *r, *jrev, *ki, imax_x, imin_x, i, n, m, ii, jy,
                jy0, jyi0, r0;
        double  *xi, *xxi, u, max_x, min_x;

        if (nx <= 0 || nxi <= 0 || ny <= 0)
            return (NULL);

        min_x = max_x = x[0];
        imin_x = imax_x = 0;
        for (i=0; i<nx; i++) {
            if (x[i] < min_x) {
                min_x = x[i];
                imin_x = i;
            } else if (x[i] > max_x) {
                max_x = x[i];
                imax_x = i;
            }
        }
        in_rng = (char *) malloc(nxi*sizeof (char));
        memset(in_rng, 0, nxi*sizeof (char));

        for (i=n=0; i<nxi; i++) {
            if (x_i[i] <= min_x) {
                for (jy=jy0=jyi0=0; jy<ny; jy++, jy0+=nx, jyi0+=nxi)
                    y_i[jyi0+i] = y[jy0+imin_x];
            } else if (x_i[i] >= max_x) {
                for (jy=jy0=jyi0=0; jy<ny; jy++, jy0+=nx, jyi0+=nxi)
                    y_i[jyi0+i] = y[jy0+imax_x];
            } else {
                in_rng[i] = 1;
                n++;
            }
        }
        if (n==0)
            return (y_i);
        xi = (double *) malloc(n*sizeof (double));
        k  = (int *) malloc(3*n*sizeof (int)); ki = k+n; r = ki+n;
        m  = nx + n;
        xxi = (double *) malloc(m*sizeof (double));
        j = (int *) malloc(2*m*sizeof (int)); jrev = j+m;

        ii = 0;
        for (i = 0; i<nxi; i++) {
            if (in_rng[i]) {
                xi[ii] = x_i[i];
                ki[ii] = i;
                ii++;
            }
        }
        free(in_rng);
    /*
    **  This algorithm mimics the Matlab interp1q function.
    **
    **  An explanation of this algorithm:
    **  We have points we are interpolating from (x) and
    **  points that we are interpolating to (xi).  We
    **  sort the interpolating from points, concatenate
    **  them with the interpolating to points and sort the result.
    **  We then construct index r, the interpolation index in x for
    **  each point in xi.
    **
    **  Note that the following operations on the index
    **  vectors jrev and r depend on the sort utility
    **  gsw_util_sort_real() consistently ordering the
    **  sorting indexes either in ascending or descending
    **  sequence for replicate values in the real vector.
    */
        gsw_util_sort_real(xi, n, k);
        memmove(xxi, x, nx*sizeof (double));
        memmove(xxi+nx, xi, n*sizeof (double));
        gsw_util_sort_real(xxi, m, j);

        for (i = 0; i<m; i++)
            jrev[j[i]] = i;
        for (i = 0; i<n; i++)
            r[k[i]] = jrev[nx+i] - i - 1;
            /* this is now the interpolation index in x for a point in xi */

        for (jy=jy0=jyi0=0; jy < ny; jy++, jy0+=nx, jyi0+=nxi) {
            for (i = 0; i<n; i++) {
                u = (xi[i]-x[r[i]])/(x[r[i]+1]-x[r[i]]);
                r0 = jy0+r[i];
                y_i[jyi0+ki[i]] = y[r0] + (y[r0+1]-y[r0])*u;
            }
        }
        free(j); free(xxi); free(k); free(xi);
        return (y_i);
}
/*******************************************************************************
    Functions for pchip interpolation
    (Piecewise Cubic Hermite Interpolating Polynomial)
    based on
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html#scipy.interpolate.PchipInterpolator

    See references therein.
    This is a shape-preserving algorithm, in the sense that it does not
    overshoot the original points; extrema of the interpolated curve match
    the extrema of the original points.
*/

#define sgn(x) (((x) > 0) ? 1 : (((x) < 0) ? -1 : 0))

static double pchip_edge_case(double h0, double h1, double m0, double m1)
{
    double d;
    int mask, mask2;

    d = ((2*h0 + h1)*m0 - h0*m1) / (h0 + h1);
    mask = sgn(d) != sgn(m0);
    mask2 = (sgn(m0) != sgn(m1)) && (fabs(d) > 3.0*fabs(m0));
    if (mask)
    {
        return 0.0;
    }
    if (!mask && mask2)
    {
        return 3.0*m0;
    }
    return d;
}

/*
    Calculation of the derivatives is the key to the shape-preservation.
    There are other algorithms that could be used, but this appears to be
    the simplest, and adequate for our purposes.

    At minimal computational cost, we include here a check for increasing x.

    Returns 0 on success, 1 if x is not strictly increasing.
*/
static int pchip_derivs(double *x, double *y, int n,
                 double *d)
{
    double mm, mp;      /* slopes bracketing x */
    double hm, hp;      /* bracketing delta-x values */
    int smm, smp;       /* slope signs */
    double w1, w2;
    int i;

    if (n == 2)
    {
        d[0] = d[1] = (y[1] - y[0]) / (x[1] - x[0]);
        return 0;
    }

    hm = x[1] - x[0];
    hp = x[2] - x[1];
    mm = (y[1] - y[0]) / hm;
    mp = (y[2] - y[1]) / hp;
    d[0] = pchip_edge_case(hm, hp, mm, mp);
    smm = sgn(mm);
    smp = sgn(mp);

    for (i=1; i<n-1; i++)
    {
        if (hm <= 0)
        {
            return 1;
        }
        /* change of sign, or either slope is zero */
        if ((smm != smp) || mp == 0 || mm == 0)
        {
            d[i] = 0.0;
        }
        else
        {
            w1 = 2*hp + hm;
            w2 = hp + 2*hm;
            d[i] = (w1 + w2) / (w1/mm + w2/mp);
        }
        if (i < n-2)
        {
            hm = hp;
            hp = x[i+2] - x[i+1];
            mm = mp;
            mp = (y[i+2] - y[i+1]) / hp;
            smm = smp;
            smp = sgn(mp);
        }
    }
    if (hp <= 0)
    {
        return 1;
    }
    d[n-1] = pchip_edge_case(hp, hm, mp, mm);
    return 0;
}

/*************************************************************************
   Piecewise-Hermite algorithm from
   https://en.wikipedia.org/wiki/Cubic_Hermite_spline

   Extrapolation to points outside the range is done by setting those
   points to the corresponding end values.

   The input x must be monotonically increasing; the interpolation points,
   xi, may be in any order, but the algorithm will be faster if they are
   monotonic, increasing or decreasing.

   Returns 0 on success, 1 if it fails because there are fewer than 2 points,
   2 if it fails because x is not increasing.
   Consistent with other GSW-C code at present, the memory allocations
   are assumed to succeed.
*/
int gsw_util_pchip_interp(double *x, double *y, int n,
                          double *xi, double *yi, int ni)
{
    double *d;
    double t, tt, ttt, xx, dx;
    int i, j0, j1, err;
    double h00, h10, h01, h11;

    if (n<2)
    {
        return 1;
    }
    d = (double *)calloc(n, sizeof(double));
    err = pchip_derivs(x, y, n, d);
    if (err)
    {
                free(d);
        return 2;
    }

    j0 = 0;
    for (i=0; i<ni; i++)
    {
        xx = xi[i];
        /* Linear search is appropriate and probably optimal for the
           expected primary use case of interpolation to a finer grid.
           It is inefficient but still functional in the worst case of
           randomly distributed xi.
        */
        while (xx < x[j0] && j0 > 0)
        {
            j0--;
        }
        while (xx > x[j0+1] && j0 < n - 2)
        {
            j0++;
        }
        j1 = j0 + 1;
        if (xx >= x[j0] && xx <= x[j1])
        {
            dx = x[j1] - x[j0];
            t = (xx - x[j0]) / dx;
            tt = t * t;
            ttt = tt * t;
            /* Using intermediate variables for readability. */
            h00 = (2*ttt - 3*tt + 1);
            h10 =  (ttt - 2*tt + t);
            h01 = (-2*ttt + 3*tt);
            h11 = (ttt - tt);
            yi[i] = y[j0] * h00 + d[j0] * dx * h10 +
                    y[j1] * h01 + d[j1] * dx * h11;
        }
        else
        {
            /* extrapolate with constant end values */
            yi[i] = (xx < x[0]) ? y[0] : y[n-1];
        }
    }
    free(d);
    return 0;
}

/*
    End of the pchip interpolation.
    *******************************
*/
/*
pure function gsw_util_sort_real (rarray) result(iarray)
*/
typedef struct {
        double  d;
        int     i;
} DI;

/*
 * Rank two items, by value if possible,
 * and by inverse index, if the values are
 * equal.
 * FIXME: decide if index method matches docs.
 */
int
compareDI(const void *a, const void *b)
{
        DI      *A = (DI*)a;
        DI      *B = (DI*)b;
        if (A->d < B->d)
                return (-1);
        if (A->d > B->d)
                return (1);
        if (A->i < B->i)
                return (1);
        return (-1);
}

/*
**  Sort the double array rarray into ascending value sequence
**  returning an index array of the sorted result.  This function
**  is thread-safe.
*/
void
gsw_util_sort_real(double *rarray, int nx, int *iarray)
{
        int     i;
        DI* di = (DI*)malloc(nx*sizeof(DI));
        for (i=0; i<nx; i++) {
                di[i].d = rarray[i];
                di[i].i = i;
        }
        qsort(di, nx, sizeof(DI), compareDI);
        for (i=0; i<nx; i++)
                iarray[i] = di[i].i;
        free(di);
}
/*
!==========================================================================
function gsw_util_xinterp1(x,y,n,x0)
!==========================================================================

! Linearly interpolate a real array
!
! x      : y array (Must be monotonic)
! y      : y array
! n      : length of X and Y arrays
! x0     : value to be interpolated
!
! gsw_xinterp1 : Linearly interpolated value
*/
double
gsw_util_xinterp1(double *x, double *y, int n, double x0)
{
        int     k;
        double  r;

        k       = gsw_util_indx(x,n,x0);
        r       = (x0-x[k])/(x[k+1]-x[k]);
        return (y[k] + r*(y[k+1]-y[k]));
}
/*
!==========================================================================
function gsw_z_from_p(p,lat,geo_strf_dyn_height,sea_surface_geopotential)
!==========================================================================

! Calculates the height z from pressure p
!
! p      : sea pressure                                    [dbar]
! lat    : latitude                                        [deg]
! geo_strf_dyn_height : dynamic height anomaly             [m^2/s^2]
!    Note that the reference pressure, p_ref, of geo_strf_dyn_height must
!     be zero (0) dbar.
! sea_surface_geopotential : geopotential at zero sea pressure  [m^2/s^2]
!
! gsw_z_from_p : height                                    [m]
*/
double
gsw_z_from_p(double p, double lat, double geo_strf_dyn_height,
                double sea_surface_geopotential)
{
        GSW_TEOS10_CONSTANTS;
        double  x, sin2, b, c, a;

        x       = sin(lat*deg2rad);
        sin2    = x*x;
        b       = 9.780327*(1.0 + (5.2792e-3 + (2.32e-5*sin2))*sin2);
        a       = -0.5*gamma*b;
        c       = gsw_enthalpy_sso_0(p)
                  - (geo_strf_dyn_height + sea_surface_geopotential);

        return (-2.0*c/(b + sqrt(b*b - 4.0*a*c)));
}

/*
**  The End
**!==========================================================================
*/
