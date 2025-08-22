
"""
Auto-generated wrapper for C ufunc extension; do not edit!
"""

from . import _gsw_ufuncs
from ._utilities import masked_array_support


@masked_array_support
def adiabatic_lapse_rate_from_CT(SA, CT, p, **kwargs):
    """
    Calculates the adiabatic lapse rate of sea water from Conservative
    Temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    adiabatic_lapse_rate : array-like, K/Pa
        adiabatic lapse rate


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See Eqn. (2.22.1) of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.adiabatic_lapse_rate_from_ct(SA, CT, p, **kwargs)

@masked_array_support
def adiabatic_lapse_rate_ice(t, p, **kwargs):
    """
    Calculates the adiabatic lapse rate of ice.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    adiabatic_lapse_rate_ice : array-like, K/Pa
        adiabatic lapse rate


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.


    """
    return _gsw_ufuncs.adiabatic_lapse_rate_ice(t, p)
adiabatic_lapse_rate_ice.types = _gsw_ufuncs.adiabatic_lapse_rate_ice.types
adiabatic_lapse_rate_ice = match_args_return(adiabatic_lapse_rate_ice)

def alpha(SA, CT, p):
    """
    Calculates the thermal expansion coefficient of seawater with respect to
    Conservative Temperature using the computationally-efficient expression
    for specific volume in terms of SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    alpha : array-like, 1/K
        thermal expansion coefficient
        with respect to Conservative Temperature


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (2.18.3) of this TEOS-10 manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.alpha(SA, CT, p)
alpha.types = _gsw_ufuncs.alpha.types
alpha = match_args_return(alpha)

def alpha_on_beta(SA, CT, p):
    """
    Calculates alpha divided by beta, where alpha is the thermal expansion
    coefficient and beta is the saline contraction coefficient of seawater
    from Absolute Salinity and Conservative Temperature.  This function uses
    the computationally-efficient expression for specific volume in terms of
    SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    alpha_on_beta : array-like, g kg^-1 K^-1
        thermal expansion coefficient with respect to
        Conservative Temperature divided by the saline
        contraction coefficient at constant Conservative
        Temperature


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix A.20 and appendix K of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling.


    """
    return _gsw_ufuncs.alpha_on_beta(SA, CT, p)
alpha_on_beta.types = _gsw_ufuncs.alpha_on_beta.types
alpha_on_beta = match_args_return(alpha_on_beta)

def alpha_wrt_t_exact(SA, t, p):
    """
    Calculates the thermal expansion coefficient of seawater with respect to
    in-situ temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    alpha_wrt_t_exact : array-like, 1/K
        thermal expansion coefficient
        with respect to in-situ temperature


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See Eqn. (2.18.1) of this TEOS-10 manual.


    """
    return _gsw_ufuncs.alpha_wrt_t_exact(SA, t, p)
alpha_wrt_t_exact.types = _gsw_ufuncs.alpha_wrt_t_exact.types
alpha_wrt_t_exact = match_args_return(alpha_wrt_t_exact)

def alpha_wrt_t_ice(t, p):
    """
    Calculates the thermal expansion coefficient of ice with respect to
    in-situ temperature.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    alpha_wrt_t_ice : array-like, 1/K
        thermal expansion coefficient of ice with respect
        to in-situ temperature


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See Eqn. (2.18.1) of this TEOS-10 manual.


    """
    return _gsw_ufuncs.alpha_wrt_t_ice(t, p)
alpha_wrt_t_ice.types = _gsw_ufuncs.alpha_wrt_t_ice.types
alpha_wrt_t_ice = match_args_return(alpha_wrt_t_ice)

def beta(SA, CT, p):
    """
    Calculates the saline (i.e. haline) contraction coefficient of seawater
    at constant Conservative Temperature using the computationally-efficient
    75-term expression for specific volume in terms of SA, CT and p
    (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    beta : array-like, kg/g
        saline contraction coefficient
        at constant Conservative Temperature


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (2.19.3) of this TEOS-10 manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.beta(SA, CT, p)
beta.types = _gsw_ufuncs.beta.types
beta = match_args_return(beta)

def beta_const_t_exact(SA, t, p):
    """
    Calculates the saline (i.e. haline) contraction coefficient of seawater
    at constant in-situ temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    beta_const_t_exact : array-like, kg/g
        saline contraction coefficient
        at constant in-situ temperature


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (2.19.1) of this TEOS-10 manual.


    """
    return _gsw_ufuncs.beta_const_t_exact(SA, t, p)
beta_const_t_exact.types = _gsw_ufuncs.beta_const_t_exact.types
beta_const_t_exact = match_args_return(beta_const_t_exact)

def C_from_SP(SP, t, p):
    """
    Calculates conductivity, C, from (SP,t,p) using PSS-78 in the range
    2 < SP < 42.  If the input Practical Salinity is less than 2 then a
    modified form of the Hill et al. (1986) formula is used for Practical
    Salinity.  The modification of the Hill et al. (1986) expression is to
    ensure that it is exactly consistent with PSS-78 at SP = 2.

    Parameters
    ----------
    SP : array-like
        Practical Salinity (PSS-78), unitless
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    C : array-like, mS/cm
        conductivity


    Notes
    -----
    The conductivity ratio returned by this function is consistent with the
    input value of Practical Salinity, SP, to 2x10^-14 psu over the full
    range of input parameters (from pure fresh water up to SP = 42 psu).
    This error of 2x10^-14 psu is machine precision at typical seawater
    salinities.  This accuracy is achieved by having four different
    polynomials for the starting value of Rtx (the square root of Rt) in
    four different ranges of SP, and by using one and a half iterations of
    a computationally efficient modified Newton-Raphson technique (McDougall
    and Wotherspoon, 2013) to find the root of the equation.

    Note that strictly speaking PSS-78 (Unesco, 1983) defines Practical
    Salinity in terms of the conductivity ratio, R, without actually
    specifying the value of C(35,15,0) (which we currently take to be
    42.9140 mS/cm).


    References
    ----------
    Hill, K.D., T.M. Dauphinee and D.J. Woods, 1986: The extension of the
    Practical Salinity Scale 1978 to low salinities. IEEE J. Oceanic Eng.,
    OE-11, 1, 109 - 112.

    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix E of this TEOS-10 Manual.

    McDougall T. J. and S. J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.

    Unesco, 1983: Algorithms for computation of fundamental properties of
    seawater. Unesco Technical Papers in Marine Science, 44, 53 pp.


    """
    return _gsw_ufuncs.c_from_sp(SP, t, p)
C_from_SP.types = _gsw_ufuncs.c_from_sp.types
C_from_SP = match_args_return(C_from_SP)

def cabbeling(SA, CT, p):
    """
    Calculates the cabbeling coefficient of seawater with respect to
    Conservative Temperature.  This function uses the computationally-
    efficient expression for specific volume in terms of SA, CT and p
    (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    cabbeling : array-like, 1/K^2
        cabbeling coefficient with respect to
        Conservative Temperature.


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqns. (3.9.2) and (P.4) of this TEOS-10 manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.cabbeling(SA, CT, p)
cabbeling.types = _gsw_ufuncs.cabbeling.types
cabbeling = match_args_return(cabbeling)

def chem_potential_water_ice(t, p):
    """
    Calculates the chemical potential of water in ice from in-situ
    temperature and pressure.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    chem_potential_water_ice : array-like, J/kg
        chemical potential of ice


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.chem_potential_water_ice(t, p)
chem_potential_water_ice.types = _gsw_ufuncs.chem_potential_water_ice.types
chem_potential_water_ice = match_args_return(chem_potential_water_ice)

def chem_potential_water_t_exact(SA, t, p):
    """
    Calculates the chemical potential of water in seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    chem_potential_water_t_exact : array-like, J/g
        chemical potential of water in seawater


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.chem_potential_water_t_exact(SA, t, p)
chem_potential_water_t_exact.types = _gsw_ufuncs.chem_potential_water_t_exact.types
chem_potential_water_t_exact = match_args_return(chem_potential_water_t_exact)

def cp_ice(t, p):
    """
    Calculates the isobaric heat capacity of ice.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    cp_ice : array-like, J kg^-1 K^-1
        heat capacity of ice


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.cp_ice(t, p)
cp_ice.types = _gsw_ufuncs.cp_ice.types
cp_ice = match_args_return(cp_ice)

def cp_t_exact(SA, t, p):
    """
    Calculates the isobaric heat capacity of seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    cp_t_exact : array-like, J/(kg*K)
        heat capacity of seawater


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.cp_t_exact(SA, t, p)
cp_t_exact.types = _gsw_ufuncs.cp_t_exact.types
cp_t_exact = match_args_return(cp_t_exact)

def CT_first_derivatives(SA, pt):
    """
    Calculates the following two derivatives of Conservative Temperature
    (1) CT_SA, the derivative with respect to Absolute Salinity at
    constant potential temperature (with pr = 0 dbar), and
    2) CT_pt, the derivative with respect to potential temperature
    (the regular potential temperature which is referenced to 0 dbar)
    at constant Absolute Salinity.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    pt : array-like
        Potential temperature referenced to a sea pressure, degrees C

    Returns
    -------
    CT_SA : array-like, K/(g/kg)
        The derivative of Conservative Temperature with respect to
        Absolute Salinity at constant potential temperature
        (the regular potential temperature which has reference
        sea pressure of 0 dbar).
    CT_pt : array-like, unitless
        The derivative of Conservative Temperature with respect to
        potential temperature (the regular one with pr = 0 dbar)
        at constant SA. CT_pt is dimensionless.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See Eqns. (A.12.3a,b) and (A.15.8) of this TEOS-10 Manual.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.ct_first_derivatives(SA, pt)
CT_first_derivatives.types = _gsw_ufuncs.ct_first_derivatives.types
CT_first_derivatives = match_args_return(CT_first_derivatives)

def CT_first_derivatives_wrt_t_exact(SA, t, p):
    """
    Calculates the following three derivatives of Conservative Temperature.
    These derivatives are done with respect to in-situ temperature t (in the
    case of CT_T_wrt_t) or at constant in-situ tempertature (in the cases of
    CT_SA_wrt_t and CT_P_wrt_t).
    (1) CT_SA_wrt_t, the derivative of CT with respect to Absolute Salinity
    at constant t and p, and
    (2) CT_T_wrt_t, derivative of CT with respect to in-situ temperature t
    at constant SA and p.
    (3) CT_P_wrt_t, derivative of CT with respect to pressure P (in Pa) at
    constant SA and t.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    CT_SA_wrt_t : array-like, K kg/g
        The first derivative of Conservative Temperature with
        respect to Absolute Salinity at constant t and p.
        [ K/(g/kg)]  i.e.
    CT_T_wrt_t : array-like, unitless
        The first derivative of Conservative Temperature with
        respect to in-situ temperature, t, at constant SA and p.
    CT_P_wrt_t : array-like, K/Pa
        The first derivative of Conservative Temperature with
        respect to pressure P (in Pa) at constant SA and t.


    Notes
    -----
    This function uses the full Gibbs function. Note that this function
    avoids the NaN that would exist in CT_SA_wrt_t at SA = 0 if it were
    evaluated in the straightforward way from the derivatives of the Gibbs
    function function.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See Eqns. (A.15.3) and (A.15.8) of this TEOS-10 Manual for
    CT_T_wrt_t and CT_SA_wrt_t respectively.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.ct_first_derivatives_wrt_t_exact(SA, t, p)
CT_first_derivatives_wrt_t_exact.types = _gsw_ufuncs.ct_first_derivatives_wrt_t_exact.types
CT_first_derivatives_wrt_t_exact = match_args_return(CT_first_derivatives_wrt_t_exact)

def CT_freezing(SA, p, saturation_fraction):
    """
    Calculates the Conservative Temperature at which seawater freezes.  The
    Conservative Temperature freezing point is calculated from the exact
    in-situ freezing temperature which is found by a modified Newton-Raphson
    iteration (McDougall and Wotherspoon, 2014) of the equality of the
    chemical potentials of water in seawater and in ice.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    CT_freezing : array-like, deg C
        Conservative Temperature at freezing of seawater
        That is, the freezing temperature expressed in terms of
        Conservative Temperature (ITS-90).


    Notes
    -----
    An alternative GSW function, gsw_CT_freezing_poly, it is based on a
    computationally-efficient polynomial, and is accurate to within -5e-4 K
    and 6e-4 K, when compared with this function.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall, T.J., and S.J. Wotherspoon, 2014: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.ct_freezing(SA, p, saturation_fraction)
CT_freezing.types = _gsw_ufuncs.ct_freezing.types
CT_freezing = match_args_return(CT_freezing)

def CT_freezing_first_derivatives(SA, p, saturation_fraction):
    """
    Calculates the first derivatives of the Conservative Temperature at
    which seawater freezes, with respect to Absolute Salinity SA and
    pressure P (in Pa).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    CTfreezing_SA : array-like, K kg/g
        the derivative of the Conservative Temperature at
        freezing (ITS-90) with respect to Absolute Salinity at
        fixed pressure              [ K/(g/kg) ] i.e.
    CTfreezing_P : array-like, K/Pa
        the derivative of the Conservative Temperature at
        freezing (ITS-90) with respect to pressure (in Pa) at
        fixed Absolute Salinity


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.


    """
    return _gsw_ufuncs.ct_freezing_first_derivatives(SA, p, saturation_fraction)
CT_freezing_first_derivatives.types = _gsw_ufuncs.ct_freezing_first_derivatives.types
CT_freezing_first_derivatives = match_args_return(CT_freezing_first_derivatives)

def CT_freezing_first_derivatives_poly(SA, p, saturation_fraction):
    """
    Calculates the first derivatives of the Conservative Temperature at
    which seawater freezes, with respect to Absolute Salinity SA and
    pressure P (in Pa) of the comptationally efficient polynomial fit of the
    freezing temperature (McDougall et al., 2014).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    CTfreezing_SA : array-like, K kg/g
        the derivative of the Conservative Temperature at
        freezing (ITS-90) with respect to Absolute Salinity at
        fixed pressure              [ K/(g/kg) ] i.e.
    CTfreezing_P : array-like, K/Pa
        the derivative of the Conservative Temperature at
        freezing (ITS-90) with respect to pressure (in Pa) at
        fixed Absolute Salinity


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.


    """
    return _gsw_ufuncs.ct_freezing_first_derivatives_poly(SA, p, saturation_fraction)
CT_freezing_first_derivatives_poly.types = _gsw_ufuncs.ct_freezing_first_derivatives_poly.types
CT_freezing_first_derivatives_poly = match_args_return(CT_freezing_first_derivatives_poly)

def CT_freezing_poly(SA, p, saturation_fraction):
    """
    Calculates the Conservative Temperature at which seawater freezes.
    The error of this fit ranges between -5e-4 K and 6e-4 K when compared
    with the Conservative Temperature calculated from the exact in-situ
    freezing temperature which is found by a Newton-Raphson iteration of the
    equality of the chemical potentials of water in seawater and in ice.
    Note that the Conservative Temperature freezing temperature can be found
    by this exact method using the function gsw_CT_freezing.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    CT_freezing : array-like, deg C
        Conservative Temperature at freezing of seawater
        That is, the freezing temperature expressed in
        terms of Conservative Temperature (ITS-90).


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.


    """
    return _gsw_ufuncs.ct_freezing_poly(SA, p, saturation_fraction)
CT_freezing_poly.types = _gsw_ufuncs.ct_freezing_poly.types
CT_freezing_poly = match_args_return(CT_freezing_poly)

def CT_from_enthalpy(SA, h, p):
    """
    Calculates the Conservative Temperature of seawater, given the Absolute
    Salinity, specific enthalpy, h, and pressure p.  The specific enthalpy
    input is the one calculated from the computationally-efficient
    expression for specific volume in terms of SA, CT and p (Roquet et al.,
    2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    h : array-like
        Specific enthalpy, J/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    CT : array-like, deg C
        Conservative Temperature ( ITS-90)


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., 2003: Potential enthalpy: A conservative oceanic
    variable for evaluating heat content and heat fluxes. Journal of
    Physical Oceanography, 33, 945-963.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    McDougall, T.J., and S.J. Wotherspoon, 2014: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.ct_from_enthalpy(SA, h, p)
CT_from_enthalpy.types = _gsw_ufuncs.ct_from_enthalpy.types
CT_from_enthalpy = match_args_return(CT_from_enthalpy)

def CT_from_enthalpy_exact(SA, h, p):
    """
    Calculates the Conservative Temperature of seawater, given the Absolute
    Salinity, SA, specific enthalpy, h, and pressure p.  The specific
    enthalpy input is calculated from the full Gibbs function of seawater,
    gsw_enthalpy_t_exact.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    h : array-like
        Specific enthalpy, J/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    CT : array-like, deg C
        Conservative Temperature ( ITS-90)


    Notes
    -----
    Note that this function uses the full Gibbs function.  There is an
    alternative to calling this function, namely
    gsw_CT_from_enthalpy(SA,h,p), which uses the computationally
    efficient 75-term expression for specific volume in terms of SA, CT
    and p (Roquet et al., 2015).


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., 2003: Potential enthalpy: A conservative oceanic
    variable for evaluating heat content and heat fluxes. Journal of
    Physical Oceanography, 33, 945-963.

    McDougall T.J. and S.J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.ct_from_enthalpy_exact(SA, h, p)
CT_from_enthalpy_exact.types = _gsw_ufuncs.ct_from_enthalpy_exact.types
CT_from_enthalpy_exact = match_args_return(CT_from_enthalpy_exact)

def CT_from_entropy(SA, entropy):
    """
    Calculates Conservative Temperature with entropy as an input variable.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    entropy : array-like
        Specific entropy, J/(kg*K)

    Returns
    -------
    CT : array-like, deg C
        Conservative Temperature (ITS-90)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix  A.10 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.ct_from_entropy(SA, entropy)
CT_from_entropy.types = _gsw_ufuncs.ct_from_entropy.types
CT_from_entropy = match_args_return(CT_from_entropy)

def CT_from_pt(SA, pt):
    """
    Calculates Conservative Temperature of seawater from potential
    temperature (whose reference sea pressure is zero dbar).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    pt : array-like
        Potential temperature referenced to a sea pressure, degrees C

    Returns
    -------
    CT : array-like, deg C
        Conservative Temperature (ITS-90)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 3.3 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.ct_from_pt(SA, pt)
CT_from_pt.types = _gsw_ufuncs.ct_from_pt.types
CT_from_pt = match_args_return(CT_from_pt)

def CT_from_rho(rho, SA, p):
    """
    Calculates the Conservative Temperature of a seawater sample, for given
    values of its density, Absolute Salinity and sea pressure (in dbar),
    using the computationally-efficient expression for specific volume in
    terms of SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    rho : array-like
        Seawater density (not anomaly) in-situ, e.g., 1026 kg/m^3.
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    CT : array-like, deg C
        Conservative Temperature  (ITS-90)
    CT_multiple : array-like, deg C
        Conservative Temperature  (ITS-90)


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling, 90, pp. 29-43.


    """
    return _gsw_ufuncs.ct_from_rho(rho, SA, p)
CT_from_rho.types = _gsw_ufuncs.ct_from_rho.types
CT_from_rho = match_args_return(CT_from_rho)

def CT_from_t(SA, t, p):
    """
    Calculates Conservative Temperature of seawater from in-situ
    temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    CT : array-like, deg C
        Conservative Temperature (ITS-90)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 3.3 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.ct_from_t(SA, t, p)
CT_from_t.types = _gsw_ufuncs.ct_from_t.types
CT_from_t = match_args_return(CT_from_t)

def CT_maxdensity(SA, p):
    """
    Calculates the Conservative Temperature of maximum density of seawater.
    This function returns the Conservative temperature at which the density
    of seawater is a maximum, at given Absolute Salinity, SA, and sea
    pressure, p (in dbar).  This function uses the computationally-efficient
    expression for specific volume in terms of SA, CT and p
    (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    CT_maxdensity : array-like, deg C
        Conservative Temperature at which
        the density of seawater is a maximum for
        given Absolute Salinity and pressure.


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 3.42 of this TEOS-10 Manual.

    McDougall T. J. and S. J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.ct_maxdensity(SA, p)
CT_maxdensity.types = _gsw_ufuncs.ct_maxdensity.types
CT_maxdensity = match_args_return(CT_maxdensity)

def CT_second_derivatives(SA, pt):
    """
    Calculates the following three, second-order derivatives of Conservative
    Temperature
    (1) CT_SA_SA, the second derivative with respect to Absolute Salinity
    at constant potential temperature (with p_ref = 0 dbar),
    (2) CT_SA_pt, the derivative with respect to potential temperature
    (the regular potential temperature which is referenced to 0 dbar)
    and Absolute Salinity, and
    (3) CT_pt_pt, the second derivative with respect to potential
    temperature (the regular potential temperature which is referenced
    to 0 dbar) at constant Absolute Salinity.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    pt : array-like
        Potential temperature referenced to a sea pressure, degrees C

    Returns
    -------
    CT_SA_SA : array-like, K/((g/kg)^2)
        The second derivative of Conservative Temperature with
        respect to Absolute Salinity at constant potential
        temperature (the regular potential temperature which
        has reference sea pressure of 0 dbar).
    CT_SA_pt : array-like,
        The derivative of Conservative Temperature with
        respect to potential temperature (the regular one with
    p_ref : array-like, 1/(g/kg)
        0 dbar) and Absolute Salinity.
    CT_pt_pt : array-like,
        The second derivative of Conservative Temperature with
        respect to potential temperature (the regular one with
    p_ref : array-like, 1/K
        0 dbar) at constant SA.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See appendix A.12 of this TEOS-10 Manual.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.ct_second_derivatives(SA, pt)
CT_second_derivatives.types = _gsw_ufuncs.ct_second_derivatives.types
CT_second_derivatives = match_args_return(CT_second_derivatives)

def deltaSA_atlas(p, lon, lat):
    """
    Calculates the Absolute Salinity Anomaly atlas value, SA - SR, in
    the open ocean by spatially interpolating the global reference data set
    of deltaSA_atlas to the location of the seawater sample.

    Parameters
    ----------
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    deltaSA_atlas : array-like, g/kg
        Absolute Salinity Anomaly atlas value


    Notes
    -----
    The Absolute Salinity Anomaly atlas value in the Baltic Sea is
    evaluated separately, since it is a function of Practical Salinity, not
    of space.  The present function returns a deltaSA_atlas of zero for
    data in the Baltic Sea.  The correct way of calculating Absolute
    Salinity in the Baltic Sea is by calling gsw_SA_from_SP.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, F.J. Millero, R. Pawlowicz and
    P.M. Barker, 2012: A global algorithm for estimating Absolute Salinity.
    Ocean Science, 8, 1123-1134.
    https://os.copernicus.org/articles/8/1123/2012/os-8-1123-2012.pdf


    """
    return _gsw_ufuncs.deltasa_atlas(p, lon, lat)
deltaSA_atlas.types = _gsw_ufuncs.deltasa_atlas.types
deltaSA_atlas = match_args_return(deltaSA_atlas)

def deltaSA_from_SP(SP, p, lon, lat):
    """
    Calculates Absolute Salinity Anomaly from Practical Salinity.  Since SP
    is non-negative by definition, this function changes any negative input
    values of SP to be zero.

    Parameters
    ----------
    SP : array-like
        Practical Salinity (PSS-78), unitless
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    deltaSA : array-like, g/kg
        Absolute Salinity Anomaly


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 2.5 and appendices A.4 and A.5 of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, F.J. Millero, R. Pawlowicz and
    P.M. Barker, 2012: A global algorithm for estimating Absolute Salinity.
    Ocean Science, 8, 1117-1128.
    https://os.copernicus.org/articles/8/1117/2012/os-8-1117-2012.pdf


    """
    return _gsw_ufuncs.deltasa_from_sp(SP, p, lon, lat)
deltaSA_from_SP.types = _gsw_ufuncs.deltasa_from_sp.types
deltaSA_from_SP = match_args_return(deltaSA_from_SP)

def dilution_coefficient_t_exact(SA, t, p):
    """
    Calculates the dilution coefficient of seawater.  The dilution
    coefficient of seawater is defined as the Absolute Salinity times the
    second derivative of the Gibbs function with respect to Absolute
    Salinity, that is, SA.*g_SA_SA.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    dilution_coefficient_t_exact : array-like, (J/kg)(kg/g)
        dilution coefficient


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.dilution_coefficient_t_exact(SA, t, p)
dilution_coefficient_t_exact.types = _gsw_ufuncs.dilution_coefficient_t_exact.types
dilution_coefficient_t_exact = match_args_return(dilution_coefficient_t_exact)

def dynamic_enthalpy(SA, CT, p):
    """
    Calculates dynamic enthalpy of seawater using the computationally-
    efficient expression for specific volume in terms of SA, CT and p
    (Roquet et al., 2015).  Dynamic enthalpy is defined as enthalpy minus
    potential enthalpy (Young, 2010).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    dynamic_enthalpy : array-like, J/kg
        dynamic enthalpy


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 3.2 of this TEOS-10 Manual.

    McDougall, T. J., 2003: Potential enthalpy: A conservative oceanic
    variable for evaluating heat content and heat fluxes. Journal of
    Physical Oceanography, 33, 945-963.
    See Eqns. (18) and (22)

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.

    Young, W.R., 2010: Dynamic enthalpy, Conservative Temperature, and the
    seawater Boussinesq approximation. Journal of Physical Oceanography,
    40, 394-400.


    """
    return _gsw_ufuncs.dynamic_enthalpy(SA, CT, p)
dynamic_enthalpy.types = _gsw_ufuncs.dynamic_enthalpy.types
dynamic_enthalpy = match_args_return(dynamic_enthalpy)

def enthalpy(SA, CT, p):
    """
    Calculates specific enthalpy of seawater using the computationally-
    efficient expression for specific volume in terms of SA, CT and p
    (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    enthalpy : array-like, J/kg
        specific enthalpy


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (A.30.6) of this TEOS-10 Manual.

    McDougall, T.J., 2003: Potential enthalpy: A conservative oceanic
    variable for evaluating heat content and heat fluxes. Journal of
    Physical Oceanography, 33, 945-963.
    See Eqns. (18) and (22)

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.enthalpy(SA, CT, p)
enthalpy.types = _gsw_ufuncs.enthalpy.types
enthalpy = match_args_return(enthalpy)

def enthalpy_CT_exact(SA, CT, p):
    """
    Calculates specific enthalpy of seawater from Absolute Salinity and
    Conservative Temperature and pressure.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    enthalpy_CT_exact : array-like, J/kg
        specific enthalpy


    Notes
    -----
    Note that this function uses the full Gibbs function.  There is an
    alternative to calling this function, namely gsw_enthalpy(SA,CT,p),
    which uses the computationally-efficient 75-term expression for specific
    volume in terms of SA, CT and p (Roquet et al., 2015).


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix A.11 of this TEOS-10 Manual.

    McDougall, T. J., 2003: Potential enthalpy: A conservative oceanic
    variable for evaluating heat content and heat fluxes. Journal of
    Physical Oceanography, 33, 945-963.
    See Eqns. (18) and (22)

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.enthalpy_ct_exact(SA, CT, p)
enthalpy_CT_exact.types = _gsw_ufuncs.enthalpy_ct_exact.types
enthalpy_CT_exact = match_args_return(enthalpy_CT_exact)

def enthalpy_diff(SA, CT, p_shallow, p_deep):
    """
    Calculates the difference of the specific enthalpy of seawater between
    two different pressures, p_deep (the deeper pressure) and p_shallow
    (the shallower pressure), at the same values of SA and CT.  This
    function uses the computationally-efficient expression for specific
    volume in terms of SA, CT and p (Roquet et al., 2015).  The output
    (enthalpy_diff) is the specific enthalpy evaluated at (SA,CT,p_deep)
    minus the specific enthalpy at (SA,CT,p_shallow).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p_shallow : array-like
        Upper sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_deep : array-like
        Lower sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    enthalpy_diff : array-like, J/kg
        difference of specific enthalpy
        (deep minus shallow)


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqns. (3.32.2) and (A.30.6) of this TEOS-10 Manual.

    McDougall, T.J., 2003: Potential enthalpy: A conservative oceanic
    variable for evaluating heat content and heat fluxes. Journal of
    Physical Oceanography, 33, 945-963.
    See Eqns. (18) and (22)

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.enthalpy_diff(SA, CT, p_shallow, p_deep)
enthalpy_diff.types = _gsw_ufuncs.enthalpy_diff.types
enthalpy_diff = match_args_return(enthalpy_diff)

def enthalpy_first_derivatives(SA, CT, p):
    """
    Calculates the following two derivatives of specific enthalpy (h) of
    seawater using the computationally-efficient expression for
    specific volume in terms of SA, CT and p (Roquet et al., 2015).
    (1) h_SA, the derivative with respect to Absolute Salinity at
    constant CT and p, and
    (2) h_CT, derivative with respect to CT at constant SA and p.
    Note that h_P is specific volume (1/rho) it can be calculated by calling
    gsw_specvol(SA,CT,p).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    h_SA : array-like, J/g
        The first derivative of specific enthalpy with respect to
        Absolute Salinity at constant CT and p.
        [ J/(kg (g/kg))]  i.e.
    h_CT : array-like, J/(kg K)
        The first derivative of specific enthalpy with respect to
        CT at constant SA and p.


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See Eqns. (A.11.18), (A.11.15) and (A.11.12) of this TEOS-10 Manual.

    McDougall, T.J., 2003: Potential enthalpy: A conservative oceanic
    variable for evaluating heat content and heat fluxes. Journal of
    Physical Oceanography, 33, 945-963.
    See Eqns. (18) and (22)

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling, 90, pp. 29-43.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.enthalpy_first_derivatives(SA, CT, p)
enthalpy_first_derivatives.types = _gsw_ufuncs.enthalpy_first_derivatives.types
enthalpy_first_derivatives = match_args_return(enthalpy_first_derivatives)

def enthalpy_first_derivatives_CT_exact(SA, CT, p):
    """
    Calculates the following two derivatives of specific enthalpy, h,
    (1) h_SA, the derivative with respect to Absolute Salinity at
    constant CT and p, and
    (2) h_CT, derivative with respect to CT at constant SA and p.
    Note that h_P is specific volume, v, it can be calculated by calling
    gsw_specvol_CT_exact(SA,CT,p).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    h_SA : array-like, J/g
        The first derivative of specific enthalpy with respect to
        Absolute Salinity at constant CT and p.
        [ J/(kg (g/kg))]  i.e.
    h_CT : array-like, J/(kg K)
        The first derivative of specific enthalpy with respect to
        CT at constant SA and p.


    Notes
    -----
    Note that this function uses the full Gibbs function.  There is an
    alternative to calling this function, namely
    gsw_enthalpy_first_derivatives(SA,CT,p) which uses the computationally
    efficient 75-term expression for specific volume in terms of SA, CT and
    p (Roquet et al., 2015).


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See Eqns. (A.11.18), (A.11.15) and (A.11.12) of this TEOS-10 Manual.

    McDougall, T.J., 2003: Potential enthalpy: A conservative oceanic
    variable for evaluating heat content and heat fluxes. Journal of
    Physical Oceanography, 33, 945-963.
    See Eqns. (18) and (22)

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.enthalpy_first_derivatives_ct_exact(SA, CT, p)
enthalpy_first_derivatives_CT_exact.types = _gsw_ufuncs.enthalpy_first_derivatives_ct_exact.types
enthalpy_first_derivatives_CT_exact = match_args_return(enthalpy_first_derivatives_CT_exact)

def enthalpy_ice(t, p):
    """
    Calculates the specific enthalpy of ice (h_Ih).

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    enthalpy_ice : array-like, J/kg
        specific enthalpy of ice


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.enthalpy_ice(t, p)
enthalpy_ice.types = _gsw_ufuncs.enthalpy_ice.types
enthalpy_ice = match_args_return(enthalpy_ice)

def enthalpy_second_derivatives(SA, CT, p):
    """
    Calculates the following three second-order derivatives of specific
    enthalpy (h),using the computationally-efficient expression for
    specific volume in terms of SA, CT and p (Roquet et al., 2015).
    (1) h_SA_SA, second-order derivative with respect to Absolute Salinity
    at constant CT & p.
    (2) h_SA_CT, second-order derivative with respect to SA & CT at
    constant p.
    (3) h_CT_CT, second-order derivative with respect to CT at constant SA
    and p.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    h_SA_SA : array-like, J/(kg (g/kg)^2)
        The second derivative of specific enthalpy with respect to
        Absolute Salinity at constant CT & p.
    h_SA_CT : array-like, J/(kg K(g/kg))
        The second derivative of specific enthalpy with respect to
        SA and CT at constant p.
    h_CT_CT : array-like, J/(kg K^2)
        The second derivative of specific enthalpy with respect to
        CT at constant SA and p.


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., 2003: Potential enthalpy: A conservative oceanic
    variable for evaluating heat content and heat fluxes. Journal of
    Physical Oceanography, 33, 945-963.
    See Eqns. (18) and (22)

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.enthalpy_second_derivatives(SA, CT, p)
enthalpy_second_derivatives.types = _gsw_ufuncs.enthalpy_second_derivatives.types
enthalpy_second_derivatives = match_args_return(enthalpy_second_derivatives)

def enthalpy_second_derivatives_CT_exact(SA, CT, p):
    """
    Calculates the following three second-order derivatives of specific
    enthalpy (h),
    (1) h_SA_SA, second-order derivative with respect to Absolute Salinity
    at constant CT & p.
    (2) h_SA_CT, second-order derivative with respect to SA & CT at
    constant p.
    (3) h_CT_CT, second-order derivative with respect to CT at constant SA
    and p.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    h_SA_SA : array-like, J/(kg (g/kg)^2)
        The second derivative of specific enthalpy with respect to
        Absolute Salinity at constant CT & p.
    h_SA_CT : array-like, J/(kg K(g/kg))
        The second derivative of specific enthalpy with respect to
        SA and CT at constant p.
    h_CT_CT : array-like, J/(kg K^2)
        The second derivative of specific enthalpy with respect to
        CT at constant SA and p.


    Notes
    -----
    Note that this function uses the full Gibbs function.  There is an
    alternative to calling this function, namely
    gsw_enthalpy_second_derivatives(SA,CT,p) which uses the computationally
    efficient 75-term expression for specific volume in terms of SA, CT and
    p (Roquet et al., 2015).


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T. J., 2003: Potential enthalpy: A conservative oceanic
    variable for evaluating heat content and heat fluxes. Journal of
    Physical Oceanography, 33, 945-963.
    See Eqns. (18) and (22)

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.enthalpy_second_derivatives_ct_exact(SA, CT, p)
enthalpy_second_derivatives_CT_exact.types = _gsw_ufuncs.enthalpy_second_derivatives_ct_exact.types
enthalpy_second_derivatives_CT_exact = match_args_return(enthalpy_second_derivatives_CT_exact)

def enthalpy_SSO_0(p):
    """
    enthalpy at (SSO,CT=0,p)

    Parameters
    ----------
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    double, array

    Notes
    -----
    gsw_enthalpy_SSO_0                               enthalpy at (SSO,CT=0,p)
                                                               (75-term eqn.)
     This function calculates enthalpy at the Standard Ocean Salinity, SSO,
     and at a Conservative Temperature of zero degrees C, as a function of
     pressure, p, in dbar, using a streamlined version of the 75-term
     computationally-efficient expression for specific volume, that is, a
     streamlined version of the code "gsw_enthalpy(SA,CT,p)".

    VERSION NUMBER: 3.06.12 (25th May, 2020)

    References
    ----------
    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.enthalpy_sso_0(p)
enthalpy_SSO_0.types = _gsw_ufuncs.enthalpy_sso_0.types
enthalpy_SSO_0 = match_args_return(enthalpy_SSO_0)

def enthalpy_t_exact(SA, t, p):
    """
    Calculates the specific enthalpy of seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    enthalpy_t_exact : array-like, J/kg
        specific enthalpy


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.enthalpy_t_exact(SA, t, p)
enthalpy_t_exact.types = _gsw_ufuncs.enthalpy_t_exact.types
enthalpy_t_exact = match_args_return(enthalpy_t_exact)

def entropy_first_derivatives(SA, CT):
    """
    Calculates the following two partial derivatives of specific entropy
    (eta)
    (1) eta_SA, the derivative with respect to Absolute Salinity at
    constant Conservative Temperature, and
    (2) eta_CT, the derivative with respect to Conservative Temperature at
    constant Absolute Salinity.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    eta_SA : array-like, J/(g K)
        The derivative of specific entropy with respect to
        Absolute Salinity (in units of g kg^-1) at constant
        Conservative Temperature.
    eta_CT : array-like, J/(kg K^2)
        The derivative of specific entropy with respect to
        Conservative Temperature at constant Absolute Salinity.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See Eqns. (A.12.8) and (P.14a,c) of this TEOS-10 Manual.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.entropy_first_derivatives(SA, CT)
entropy_first_derivatives.types = _gsw_ufuncs.entropy_first_derivatives.types
entropy_first_derivatives = match_args_return(entropy_first_derivatives)

def entropy_from_CT(SA, CT):
    """
    Calculates specific entropy of seawater from Conservative Temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    entropy : array-like, J/(kg*K)
        specific entropy


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix A.10 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.entropy_from_ct(SA, CT)
entropy_from_CT.types = _gsw_ufuncs.entropy_from_ct.types
entropy_from_CT = match_args_return(entropy_from_CT)

def entropy_from_pt(SA, pt):
    """
    Calculates specific entropy of seawater as a function of potential
    temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    pt : array-like
        Potential temperature referenced to a sea pressure, degrees C

    Returns
    -------
    entropy : array-like, J/(kg*K)
        specific entropy


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix A.10 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.entropy_from_pt(SA, pt)
entropy_from_pt.types = _gsw_ufuncs.entropy_from_pt.types
entropy_from_pt = match_args_return(entropy_from_pt)

def entropy_from_t(SA, t, p):
    """
    Calculates specific entropy of seawater from in-situ temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    entropy : array-like, J/(kg*K)
        specific entropy


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.entropy_from_t(SA, t, p)
entropy_from_t.types = _gsw_ufuncs.entropy_from_t.types
entropy_from_t = match_args_return(entropy_from_t)

def entropy_ice(t, p):
    """
    Calculates specific entropy of ice.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    ice_entropy : array-like, J kg^-1 K^-1
        specific entropy of ice


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.entropy_ice(t, p)
entropy_ice.types = _gsw_ufuncs.entropy_ice.types
entropy_ice = match_args_return(entropy_ice)

def entropy_part(SA, t, p):
    """
    entropy minus the terms that are a function of only SA

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    double, array

    Notes
    -----
    sw_entropy_part    entropy minus the terms that are a function of only SA
    This function calculates entropy, except that it does not evaluate any
    terms that are functions of Absolute Salinity alone.  By not calculating
    these terms, which are a function only of Absolute Salinity, several
    unnecessary computations are avoided (including saving the computation
    of a natural logarithm).  These terms are a necessary part of entropy,
    but are not needed when calculating potential temperature from in-situ
    temperature.

    VERSION NUMBER: 3.06.12 (25th May, 2020)


    """
    return _gsw_ufuncs.entropy_part(SA, t, p)
entropy_part.types = _gsw_ufuncs.entropy_part.types
entropy_part = match_args_return(entropy_part)

def entropy_part_zerop(SA, pt0):
    """
    entropy_part evaluated at the sea surface

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    pt0 : array-like
        Potential temperature with reference pressure of 0 dbar, degrees C

    Returns
    -------
    double, array

    Notes
    -----
    gsw_entropy_part_zerop          entropy_part evaluated at the sea surface
    This function calculates entropy at a sea pressure of zero, except that
    it does not evaluate any terms that are functions of Absolute Salinity
    alone.  By not calculating these terms, which are a function only of
    Absolute Salinity, several unnecessary computations are avoided
    (including saving the computation of a natural logarithm). These terms
    are a necessary part of entropy, but are not needed when calculating
    potential temperature from in-situ temperature.
    The inputs to "gsw_entropy_part_zerop(SA,pt0)" are Absolute Salinity
    and potential temperature with reference sea pressure of zero dbar.

    VERSION NUMBER: 3.06.12 (25th May, 2020)


    """
    return _gsw_ufuncs.entropy_part_zerop(SA, pt0)
entropy_part_zerop.types = _gsw_ufuncs.entropy_part_zerop.types
entropy_part_zerop = match_args_return(entropy_part_zerop)

def entropy_second_derivatives(SA, CT):
    """
    Calculates the following three second-order partial derivatives of
    specific entropy (eta)
    (1) eta_SA_SA, the second derivative with respect to Absolute
    Salinity at constant Conservative Temperature, and
    (2) eta_SA_CT, the derivative with respect to Absolute Salinity and
    Conservative Temperature.
    (3) eta_CT_CT, the second derivative with respect to Conservative
    Temperature at constant Absolute Salinity.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    eta_SA_SA : array-like, J/(kg K(g/kg)^2)
        The second derivative of specific entropy with respect
        to Absolute Salinity (in units of g kg^-1) at constant
        Conservative Temperature.
    eta_SA_CT : array-like, J/(kg (g/kg) K^2)
        The second derivative of specific entropy with respect
        to Conservative Temperature at constant Absolute
    eta_CT_CT : array-like, J/(kg K^3)
        The second derivative of specific entropy with respect
        to Conservative Temperature at constant Absolute


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See Eqns. (P.14b) and (P.15a,b) of this TEOS-10 Manual.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.entropy_second_derivatives(SA, CT)
entropy_second_derivatives.types = _gsw_ufuncs.entropy_second_derivatives.types
entropy_second_derivatives = match_args_return(entropy_second_derivatives)

def Fdelta(p, lon, lat):
    """
    Calculates Fdelta from the Absolute Salinity Anomaly Ratio (SAAR).  It
    finds SAAR by calling the function "gsw_SAAR(p,long,lat)" and then
    simply calculates Fdelta from

    Parameters
    ----------
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    Fdelta : array-like, unitless
        ratio of SA to Sstar, minus 1


    Notes
    -----
    Fdelta = (1 + r1)SAAR/(1 - r1*SAAR)
    = (SA/Sstar) - 1

    with r1 being the constant 0.35 based on the work of Pawlowicz et al.
    (2011). Note that since SAAR is everywhere less than 0.001 in the global
    ocean, Fdelta is only slightly different to 1.35*SAAR.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 2.5 and appendices A.4 and A.5 of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, F.J. Millero, R. Pawlowicz and
    P.M. Barker, 2012: A global algorithm for estimating Absolute Salinity.
    Ocean Science, 8, 1123-1134.
    https://os.copernicus.org/articles/8/1123/2012/os-8-1123-2012.pdf

    Pawlawicz, R., D.G. Wright and F.J. Millero, 2011; The effects of
    biogeochemical processes on oceanic conductivity/salinty/density
    relationships and the characterization of real seawater. Ocean Science,
    7, 363-387.  https://os.copernicus.org/articles/7/363/2011/os-7-363-2011.pdf


    """
    return _gsw_ufuncs.fdelta(p, lon, lat)
Fdelta.types = _gsw_ufuncs.fdelta.types
Fdelta = match_args_return(Fdelta)

def frazil_properties(SA_bulk, h_bulk, p):
    """
    Calculates the mass fraction of ice (mass of ice divided by mass of ice
    plus seawater), w_Ih_final, which results from given values of the bulk
    Absolute Salinity, SA_bulk, bulk enthalpy, h_bulk, occurring at pressure
    p.  The final values of Absolute Salinity, SA_final, and Conservative
    Temperature, CT_final, of the interstitial seawater phase are also
    returned.  This code assumes that there is no dissolved air in the
    seawater (that is, saturation_fraction is assumed to be zero
    throughout the code).

    Parameters
    ----------
    SA_bulk : array-like
        bulk Absolute Salinity of the seawater and ice mixture, g/kg
    h_bulk : array-like
        bulk enthalpy of the seawater and ice mixture, J/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    SA_final : array-like, g/kg
        Absolute Salinity of the seawater in the final state,
        whether or not any ice is present.
    CT_final : array-like, deg C
        Conservative Temperature of the seawater in the the final
        state, whether or not any ice is present.
    w_Ih_final : array-like, unitless
        mass fraction of ice in the final seawater-ice mixture.
        If this ice mass fraction is positive, the system is at
        thermodynamic equilibrium.  If this ice mass fraction is
        zero there is no ice in the final state which consists
        only of seawater which is warmer than the freezing
        temperature.


    Notes
    -----
    When the mass fraction w_Ih_final is calculated as being a positive
    value, the seawater-ice mixture is at thermodynamic equilibrium.

    This code returns w_Ih_final = 0 when the input bulk enthalpy, h_bulk,
    is sufficiently large (i.e. sufficiently "warm") so that there is no ice
    present in the final state.  In this case the final state consists of
    only seawater rather than being an equilibrium mixture of seawater and
    ice which occurs when w_Ih_final is positive.  Note that when
    w_Ih_final = 0, the final seawater is not at the freezing temperature.

    Note that there is another GSW code,
    gsw_frazil_properties_potential_poly(SA_bulk,h_pot_bulk,p) which
    treats potential enthalpy as the conservative variable, while, in
    contrast, the present code treats in situ enthalpy as the conservative
    variable during the interaction of seawater and ice Ih.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of ice and sea ice into seawater, and frazil ice formation.
    Journal of Physical Oceanography, 44, 1751-1775.

    McDougall, T.J., and S.J. Wotherspoon, 2014: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.

    Anonymous, 2014: Modelling the interaction between seawater and frazil
    ice.  Manuscript, March 2015.  See Eqns. (8) - (15) of this manuscript.


    """
    return _gsw_ufuncs.frazil_properties(SA_bulk, h_bulk, p)
frazil_properties.types = _gsw_ufuncs.frazil_properties.types
frazil_properties = match_args_return(frazil_properties)

def frazil_properties_potential(SA_bulk, h_pot_bulk, p):
    """
    Calculates the mass fraction of ice (mass of ice divided by mass of ice
    plus seawater), w_Ih_eq, which results from given values of the bulk
    Absolute Salinity, SA_bulk, bulk potential enthalpy, h_pot_bulk,
    occurring at pressure p.  The final equilibrium values of Absolute
    Salinity, SA_eq, and Conservative Temperature, CT_eq, of the
    interstitial seawater phase are also returned.  This code assumes that
    there is no dissolved air in the seawater (that is, saturation_fraction
    is assumed to be zero throughout the code).

    Parameters
    ----------
    SA_bulk : array-like
        bulk Absolute Salinity of the seawater and ice mixture, g/kg
    h_pot_bulk : array-like
        bulk enthalpy of the seawater and ice mixture, J/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    SA_final : array-like, g/kg
        Absolute Salinity of the seawater in the final state,
        whether or not any ice is present.
    CT_final : array-like, deg C
        Conservative Temperature of the seawater in the the final
        state, whether or not any ice is present.
    w_Ih_final : array-like, unitless
        mass fraction of ice in the final seawater-ice mixture.
        If this ice mass fraction is positive, the system is at
        thermodynamic equilibrium.  If this ice mass fraction is
        zero there is no ice in the final state which consists
        only of seawater which is warmer than the freezing
        temperature.


    Notes
    -----
    When the mass fraction w_Ih_final is calculated as being a positive
    value, the seawater-ice mixture is at thermodynamic equilibrium.

    This code returns w_Ih_final = 0 when the input bulk enthalpy, h_bulk,
    is sufficiently large (i.e. sufficiently "warm") so that there is no ice
    present in the final state.  In this case the final state consists of
    only seawater rather than being an equilibrium mixture of seawater and
    ice which occurs when w_Ih_final is positive.  Note that when
    w_Ih_final = 0, the final seawater is not at the freezing temperature.

    Note that this code uses the exact forms of CT_freezing and
    pot_enthalpy_ice_freezing.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of ice and sea ice into seawater, and frazil ice formation.
    Journal of Physical Oceanography, 44, 1751-1775.

    McDougall, T.J., and S.J. Wotherspoon, 2014: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.

    Anonymous, 2014: Modelling the interaction between seawater and frazil
    ice.  Manuscript, March 2015.  See Eqns. (8)-(15) of this  manuscript.


    """
    return _gsw_ufuncs.frazil_properties_potential(SA_bulk, h_pot_bulk, p)
frazil_properties_potential.types = _gsw_ufuncs.frazil_properties_potential.types
frazil_properties_potential = match_args_return(frazil_properties_potential)

def frazil_properties_potential_poly(SA_bulk, h_pot_bulk, p):
    """
    Calculates the mass fraction of ice (mass of ice divided by mass of ice
    plus seawater), w_Ih_eq, which results from given values of the bulk
    Absolute Salinity, SA_bulk, bulk potential enthalpy, h_pot_bulk,
    occurring at pressure p.  The final equilibrium values of Absolute
    Salinity, SA_eq, and Conservative Temperature, CT_eq, of the
    interstitial seawater phase are also returned.  This code assumes that
    there is no dissolved air in the seawater (that is, saturation_fraction
    is assumed to be zero throughout the code).

    Parameters
    ----------
    SA_bulk : array-like
        bulk Absolute Salinity of the seawater and ice mixture, g/kg
    h_pot_bulk : array-like
        bulk enthalpy of the seawater and ice mixture, J/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    SA_final : array-like, g/kg
        Absolute Salinity of the seawater in the final state,
        whether or not any ice is present.
    CT_final : array-like, deg C
        Conservative Temperature of the seawater in the the final
        state, whether or not any ice is present.
    w_Ih_final : array-like, unitless
        mass fraction of ice in the final seawater-ice mixture.
        If this ice mass fraction is positive, the system is at
        thermodynamic equilibrium.  If this ice mass fraction is
        zero there is no ice in the final state which consists
        only of seawater which is warmer than the freezing
        temperature.


    Notes
    -----
    When the mass fraction w_Ih_final is calculated as being a positive
    value, the seawater-ice mixture is at thermodynamic equilibrium.

    This code returns w_Ih_final = 0 when the input bulk enthalpy, h_bulk,
    is sufficiently large (i.e. sufficiently "warm") so that there is no ice
    present in the final state.  In this case the final state consists of
    only seawater rather than being an equilibrium mixture of seawater and
    ice which occurs when w_Ih_final is positive.  Note that when
    w_Ih_final = 0, the final seawater is not at the freezing temperature.

    Note that this code uses the polynomial forms of CT_freezing and
    pot_enthalpy_ice_freezing.  This code is intended to be used in ocean
    models where the model prognostic variables are SA_bulk and h_pot_bulk.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of ice and sea ice into seawater, and frazil ice formation.
    Journal of Physical Oceanography, 44, 1751-1775.

    McDougall, T.J., and S.J. Wotherspoon, 2014: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.

    Anonymous, 2014: Modelling the interaction between seawater and frazil
    ice.  Manuscript, March 2015.  See Eqns. (8)-(15) of this  manuscript.


    """
    return _gsw_ufuncs.frazil_properties_potential_poly(SA_bulk, h_pot_bulk, p)
frazil_properties_potential_poly.types = _gsw_ufuncs.frazil_properties_potential_poly.types
frazil_properties_potential_poly = match_args_return(frazil_properties_potential_poly)

def frazil_ratios_adiabatic(SA, p, w_Ih):
    """
    Calculates the ratios of SA, CT and P changes when frazil ice forms or
    melts in response to an adiabatic change in pressure of a mixture of
    seawater and frazil ice crystals.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    w_Ih : array-like
        mass fraction of ice: the mass of ice divided by the
        sum of the masses of ice and seawater. 0 <= wIh <= 1. unitless.

    Returns
    -------
    dSA_dCT_frazil : array-like, g/(kg K)
        the ratio of the changes in Absolute Salinity
        to that of Conservative Temperature
    dSA_dP_frazil : array-like, g/(kg Pa)
        the ratio of the changes in Absolute Salinity
        to that of pressure (in Pa)
    dCT_dP_frazil : array-like, K/Pa
        the ratio of the changes in Conservative Temperature
        to that of pressure (in Pa)


    Notes
    -----
    Note that the first output, dSA_dCT_frazil, is dSA/dCT rather than
    dCT/dSA.  This is done so that when SA = 0, the output, dSA/dCT, is zero
    whereas dCT/dSA would then be infinite.

    Also note that both dSA_dP_frazil and dCT_dP_frazil are the pressure
    derivatives with the pressure measured in Pa not dbar.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    See Eqns. (47), (48) and (49) of this manuscript.


    """
    return _gsw_ufuncs.frazil_ratios_adiabatic(SA, p, w_Ih)
frazil_ratios_adiabatic.types = _gsw_ufuncs.frazil_ratios_adiabatic.types
frazil_ratios_adiabatic = match_args_return(frazil_ratios_adiabatic)

def frazil_ratios_adiabatic_poly(SA, p, w_Ih):
    """
    Calculates the ratios of SA, CT and P changes when frazil ice forms or
    melts in response to an adiabatic change in pressure of a mixture of
    seawater and frazil ice crystals.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    w_Ih : array-like
        mass fraction of ice: the mass of ice divided by the
        sum of the masses of ice and seawater. 0 <= wIh <= 1. unitless.

    Returns
    -------
    dSA_dCT_frazil : array-like, g/(kg K)
        the ratio of the changes in Absolute Salinity
        to that of Conservative Temperature
    dSA_dP_frazil : array-like, g/(kg Pa)
        the ratio of the changes in Absolute Salinity
        to that of pressure (in Pa)
    dCT_dP_frazil : array-like, K/Pa
        the ratio of the changes in Conservative Temperature
        to that of pressure (in Pa)


    Notes
    -----
    Note that the first output, dSA_dCT_frazil, is dSA/dCT rather than
    dCT/dSA.  This is done so that when SA = 0, the output, dSA/dCT, is zero
    whereas dCT/dSA would then be infinite.

    Also note that both dSA_dP_frazil and dCT_dP_frazil are the pressure
    derivatives with the pressure measured in Pa not dbar.

    This function uses the computationally-efficient expression for specific
    volume in terms of SA, CT and p (Roquet et al., 2015) and the polynomial
    expression for freezing temperature based on Conservative Temperature
    (McDougall et al., 2015).


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    See Eqns. (47), (48) and (49) of this manuscript.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.frazil_ratios_adiabatic_poly(SA, p, w_Ih)
frazil_ratios_adiabatic_poly.types = _gsw_ufuncs.frazil_ratios_adiabatic_poly.types
frazil_ratios_adiabatic_poly = match_args_return(frazil_ratios_adiabatic_poly)

def gibbs(ns, nt, np, SA, t, p):
    """
    Calculates specific Gibbs energy and its derivatives up to order 3 for
    seawater.  The Gibbs function for seawater is that of TEOS-10
    (IOC et al., 2010), being the sum of IAPWS-08 for the saline part and
    IAPWS-09 for the pure water part.  These IAPWS releases are the
    officially blessed IAPWS descriptions of Feistel (2008) and the pure
    water part of Feistel (2003).  Absolute Salinity, SA, in all of the GSW
    routines is expressed on the Reference-Composition Salinity Scale of
    2008 (RCSS-08) of Millero et al. (2008).

    Parameters
    ----------
    ns : array-like
        order of SA derivative, integer in (0, 1, 2)
    nt : array-like
        order of t derivative, integer in (0, 1, 2)
    np : array-like
        order of p derivative, integer in (0, 1, 2)
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    gibbs : array-like
        Specific Gibbs energy or its derivatives.
        The Gibbs energy (when ns = nt = np = 0) has units of J/kg.
        The Absolute Salinity derivatives are output in units of (J/kg) (g/kg)^(-ns).
        The temperature derivatives are output in units of (J/kg) (K)^(-nt).
        The pressure derivatives are output in units of (J/kg) (Pa)^(-np).
        The mixed derivatives are output in units of (J/kg) (g/kg)^(-ns) (K)^(-nt) (Pa)^(-np).
        Note: The derivatives are taken with respect to pressure in Pa, not
        withstanding that the pressure input into this routine is in dbar.

    References
    ----------
    Feistel, R., 2003: A new extended Gibbs thermodynamic potential of
    seawater,  Progr. Oceanogr., 58, 43-114.

    Feistel, R., 2008: A Gibbs function for seawater thermodynamics
    for -6 to 80°C and salinity up to 120 g kg1, Deep-Sea Res. I,
    55, 1639-1671.

    IAPWS, 2008: Release on the IAPWS Formulation 2008 for the
    Thermodynamic Properties of Seawater. The International Association
    for the Properties of Water and Steam. Berlin, Germany, September
    2008, available from https://iapws.org/.  This Release is referred
    to as IAPWS-08.

    IAPWS, 2009: Supplementary Release on a Computationally Efficient
    Thermodynamic Formulation for Liquid Water for Oceanographic Use.
    The International Association for the Properties of Water and Steam.
    Doorwerth, The Netherlands, September 2009, available from
    https://iapws.org/.  This Release is referred to as IAPWS-09.

    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 2.6 and appendices A.6,  G and H of this TEOS-10 Manual.

    Millero, F.J., R. Feistel, D.G. Wright, and T.J. McDougall, 2008:
    The composition of Standard Seawater and the definition of the
    Reference-Composition Salinity Scale, Deep-Sea Res. I, 55, 50-72.

    Reference page in Help browser
    <a href="matlab:doc gsw_gibbs">doc gsw_gibbs</a>
    Note that this reference page includes the code contained in gsw_gibbs.
    We have opted to encode this programme as it is a global standard and
    such we cannot allow anyone to change it.


    """
    return _gsw_ufuncs.gibbs(ns, nt, np, SA, t, p)
gibbs.types = _gsw_ufuncs.gibbs.types
gibbs = match_args_return(gibbs)

def gibbs_ice(nt, np, t, p):
    """
    Ice specific Gibbs energy and derivatives up to order 2.

    Parameters
    ----------
    nt : array-like
        order of t derivative, integer in (0, 1, 2)
    np : array-like
        order of p derivative, integer in (0, 1, 2)
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    gibbs_ice : array-like
        Specific Gibbs energy of ice or its derivatives.
        The Gibbs energy (when nt = np = 0) has units of J/kg.
        The temperature derivatives are output in units of (J/kg) (K)^(-nt).
        The pressure derivatives are output in units of (J/kg) (Pa)^(-np).
        The mixed derivatives are output in units of (J/kg) (K)^(-nt) (Pa)^(-np).
        Note. The derivatives are taken with respect to pressure in Pa, not
        withstanding that the pressure input into this routine is in dbar.

    References
    ----------
    IAPWS, 2009: Revised release on the Equation of State 2006 for H2O Ice
    Ih. The International Association for the Properties of Water and
    Steam. Doorwerth, The Netherlands, September 2009.

    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See appendix I.

    Reference page in Help browser
    <a href="matlab:doc gsw_gibbs_ice">doc gsw_gibbs_ice</a>
    Note that this reference page includes the code contained in
    gsw_gibbs_ice.  We have opted to encode this programme as it is a global
    standard and such we cannot allow anyone to change it.


    """
    return _gsw_ufuncs.gibbs_ice(nt, np, t, p)
gibbs_ice.types = _gsw_ufuncs.gibbs_ice.types
gibbs_ice = match_args_return(gibbs_ice)

def gibbs_ice_part_t(t, p):
    """
    part of the the first temperature derivative of Gibbs energy of ice
    that is the output is gibbs_ice(1,0,t,p) + S0

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    gibbs_ice_part_t : array-like, J kg^-1 K^-1
        part of temperature derivative


    References
    ----------
    IAPWS, 2009: Revised Release on the Equation of State 2006 for H2O Ice
    Ih. The International Association for the Properties of Water and
    Steam. Doorwerth, The Netherlands, September 2009.

    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See appendix I.


    """
    return _gsw_ufuncs.gibbs_ice_part_t(t, p)
gibbs_ice_part_t.types = _gsw_ufuncs.gibbs_ice_part_t.types
gibbs_ice_part_t = match_args_return(gibbs_ice_part_t)

def gibbs_ice_pt0(pt0):
    """
    part of the the first temperature derivative of Gibbs energy of ice
    that is the output is "gibbs_ice(1,0,pt0,0) + s0"

    Parameters
    ----------
    pt0 : array-like
        Potential temperature with reference pressure of 0 dbar, degrees C

    Returns
    -------
    gibbs_ice_part_pt0 : array-like, J kg^-1 K^-1
        part of temperature derivative


    References
    ----------
    IAPWS, 2009: Revised Release on the Equation of State 2006 for H2O Ice
    Ih. The International Association for the Properties of Water and
    Steam. Doorwerth, The Netherlands, September 2009.

    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See appendix I.


    """
    return _gsw_ufuncs.gibbs_ice_pt0(pt0)
gibbs_ice_pt0.types = _gsw_ufuncs.gibbs_ice_pt0.types
gibbs_ice_pt0 = match_args_return(gibbs_ice_pt0)

def gibbs_ice_pt0_pt0(pt0):
    """
    The second temperature derivative of Gibbs energy of ice at the
    potential temperature with reference sea pressure of zero dbar.  That is
    the output is gibbs_ice(2,0,pt0,0).

    Parameters
    ----------
    pt0 : array-like
        Potential temperature with reference pressure of 0 dbar, degrees C

    Returns
    -------
    gibbs_ice_pt0_pt0 : array-like, J kg^-1 K^-2
        temperature second derivative at pt0


    References
    ----------
    IAPWS, 2009: Revised Release on the Equation of State 2006 for H2O Ice
    Ih. The International Association for the Properties of Water and
    Steam. Doorwerth, The Netherlands, September 2009.

    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See appendix I.


    """
    return _gsw_ufuncs.gibbs_ice_pt0_pt0(pt0)
gibbs_ice_pt0_pt0.types = _gsw_ufuncs.gibbs_ice_pt0_pt0.types
gibbs_ice_pt0_pt0 = match_args_return(gibbs_ice_pt0_pt0)

def gibbs_pt0_pt0(SA, pt0):
    """
    gibbs_tt at (SA,pt,0)

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    pt0 : array-like
        Potential temperature with reference pressure of 0 dbar, degrees C

    Returns
    -------
    double, array

    Notes
    -----
    gsw_gibbs_pt0_pt0                                   gibbs_tt at (SA,pt,0)
    This function calculates the second derivative of the specific Gibbs
    function with respect to temperature at zero sea pressure.  The inputs
    are Absolute Salinity and potential temperature with reference sea
    pressure of zero dbar.

    VERSION NUMBER: 3.06.13 (7th September, 2020)


    """
    return _gsw_ufuncs.gibbs_pt0_pt0(SA, pt0)
gibbs_pt0_pt0.types = _gsw_ufuncs.gibbs_pt0_pt0.types
gibbs_pt0_pt0 = match_args_return(gibbs_pt0_pt0)

def grav(lat, p):
    """
    Calculates acceleration due to gravity as a function of latitude and as
    a function of pressure in the ocean.

    Parameters
    ----------
    lat : array-like
        Latitude, -90 to 90 degrees
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    grav : array-like, m s^-2
        gravitational acceleration


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix D of this TEOS-10 Manual.

    Moritz, H., 2000: Geodetic reference system 1980. J. Geodesy, 74,
    pp. 128-133.

    Saunders, P.M., and N.P. Fofonoff, 1976: Conversion of pressure to
    depth in the ocean. Deep-Sea Res., pp. 109-111.


    """
    return _gsw_ufuncs.grav(lat, p)
grav.types = _gsw_ufuncs.grav.types
grav = match_args_return(grav)

def Helmholtz_energy_ice(t, p):
    """
    Calculates the Helmholtz energy of ice.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    Helmholtz_energy_ice : array-like, J/kg
        Helmholtz energy of ice


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.helmholtz_energy_ice(t, p)
Helmholtz_energy_ice.types = _gsw_ufuncs.helmholtz_energy_ice.types
Helmholtz_energy_ice = match_args_return(Helmholtz_energy_ice)

def Hill_ratio_at_SP2(t):
    """
    Calculates the Hill ratio, which is the adjustment needed to apply for
    Practical Salinities smaller than 2.  This ratio is defined at a
    Practical Salinity = 2 and in-situ temperature, t using PSS-78. The Hill
    ratio is the ratio of 2 to the output of the Hill et al. (1986) formula
    for Practical Salinity at the conductivity ratio, Rt, at which Practical
    Salinity on the PSS-78 scale is exactly 2.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C

    Returns
    -------
    Hill_ratio : array-like, unitless
        Hill ratio at SP of 2


    References
    ----------
    Hill, K.D., T.M. Dauphinee & D.J. Woods, 1986: The extension of the
    Practical Salinity Scale 1978 to low salinities. IEEE J. Oceanic Eng.,
    11, 109 - 112.

    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix E of this TEOS-10 Manual.

    McDougall T.J. and S.J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.

    Unesco, 1983: Algorithms for computation of fundamental properties of
    seawater. Unesco Technical Papers in Marine Science, 44, 53 pp.


    """
    return _gsw_ufuncs.hill_ratio_at_sp2(t)
Hill_ratio_at_SP2.types = _gsw_ufuncs.hill_ratio_at_sp2.types
Hill_ratio_at_SP2 = match_args_return(Hill_ratio_at_SP2)

def ice_fraction_to_freeze_seawater(SA, CT, p, t_Ih):
    """
    Calculates the mass fraction of ice (mass of ice divided by mass of ice
    plus seawater), which, when melted into seawater having (SA,CT,p) causes
    the final dilute seawater to be at the freezing temperature.  The other
    outputs are the Absolute Salinity and Conservative Temperature of the
    final diluted seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    t_Ih : array-like
        In-situ temperature of ice (ITS-90), degrees C

    Returns
    -------
    SA_freeze : array-like, g/kg
        Absolute Salinity of seawater after the mass fraction of
        ice, ice_fraction, at temperature t_Ih has melted into the
        original seawater, and the final mixture is at the freezing
        temperature of seawater.
    CT_freeze : array-like, deg C
        Conservative Temperature of seawater after the mass
        fraction, w_Ih, of ice at temperature t_Ih has melted into
        the original seawater, and the final mixture is at the
        freezing temperature of seawater.
    w_Ih : array-like, unitless
        mass fraction of ice, having in-situ temperature t_Ih,
        which, when melted into seawater at (SA,CT,p) leads to the
        final diluted seawater being at the freezing temperature.
        This output must be between 0 and 1.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall, T.J., and S.J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    See Eqn. (9) of this manuscript.


    """
    return _gsw_ufuncs.ice_fraction_to_freeze_seawater(SA, CT, p, t_Ih)
ice_fraction_to_freeze_seawater.types = _gsw_ufuncs.ice_fraction_to_freeze_seawater.types
ice_fraction_to_freeze_seawater = match_args_return(ice_fraction_to_freeze_seawater)

def infunnel(SA, CT, p):
    """
    "oceanographic funnel" check for the 75-term equation

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    in_funnel : array-like,
        0, if SA, CT and p are outside the "funnel"
        =  1, if SA, CT and p are inside the "funnel"


    Notes
    -----
    gsw_infunnel        "oceanographic funnel" check for the 75-term equation

    USAGE:
    in_funnel = gsw_infunnel(SA,CT,p)

    INPUT:
     SA  =  Absolute Salinity                                     [ g kg^-1 ]
     CT  =  Conservative Temperature (ITS-90)                       [ deg C ]
     p   =  sea pressure                                             [ dbar ]
            ( i.e. absolute pressure - 10.1325 dbar )

     SA & CT need to have the same dimensions.
     p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are MxN.

    OUTPUT:
     in_funnel  =  0, if SA, CT and p are outside the "funnel"
                =  1, if SA, CT and p are inside the "funnel"
     Note. The term "funnel" (McDougall et al., 2003) describes the range of
       SA, CT and p over which the error in the fit of the computationally
       efficient 75-term expression for specific volume in terms of SA, CT
       and p was calculated (Roquet et al., 2015).

    AUTHOR:
     Trevor McDougall and Paul Barker                    [ help@teos-10.org ]

    VERSION NUMBER: 3.06.13 (23rd May, 2021)


    """
    return _gsw_ufuncs.infunnel(SA, CT, p)
infunnel.types = _gsw_ufuncs.infunnel.types
infunnel = match_args_return(infunnel)

def internal_energy(SA, CT, p):
    """
    Calculates specific internal energy of seawater using the
    computationally-efficient expression for specific volume in terms of SA,
    CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    internal_energy : array-like, J/kg
        specific internal energy


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.internal_energy(SA, CT, p)
internal_energy.types = _gsw_ufuncs.internal_energy.types
internal_energy = match_args_return(internal_energy)

def internal_energy_ice(t, p):
    """
    Calculates the specific internal energy of ice.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    internal_energy_ice : array-like, J/kg
        specific internal energy (u)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.internal_energy_ice(t, p)
internal_energy_ice.types = _gsw_ufuncs.internal_energy_ice.types
internal_energy_ice = match_args_return(internal_energy_ice)

def kappa(SA, CT, p):
    """
    Calculates the isentropic compressibility of seawater.  This function
    has inputs of Absolute Salinity and Conservative Temperature.  This
    function uses the computationally-efficient expression for
    specific volume in terms of SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    kappa : array-like, 1/Pa
        isentropic compressibility of seawater


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (2.17.1) of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.kappa(SA, CT, p)
kappa.types = _gsw_ufuncs.kappa.types
kappa = match_args_return(kappa)

def kappa_const_t_ice(t, p):
    """
    Calculates isothermal compressibility of ice.
    Note. This is the compressibility of ice AT CONSTANT IN-SITU
    TEMPERATURE

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    kappa_const_t_ice : array-like, 1/Pa
        isothermal compressibility


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.kappa_const_t_ice(t, p)
kappa_const_t_ice.types = _gsw_ufuncs.kappa_const_t_ice.types
kappa_const_t_ice = match_args_return(kappa_const_t_ice)

def kappa_ice(t, p):
    """
    Calculates the isentropic compressibility of ice.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    kappa_ice : array-like, 1/Pa
        isentropic compressibility


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.kappa_ice(t, p)
kappa_ice.types = _gsw_ufuncs.kappa_ice.types
kappa_ice = match_args_return(kappa_ice)

def kappa_t_exact(SA, t, p):
    """
    Calculates the isentropic compressibility of seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    kappa_t_exact : array-like, 1/Pa
        isentropic compressibility


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqns. (2.16.1) and the row for kappa in Table P.1 of appendix P
    of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.kappa_t_exact(SA, t, p)
kappa_t_exact.types = _gsw_ufuncs.kappa_t_exact.types
kappa_t_exact = match_args_return(kappa_t_exact)

def latentheat_evap_CT(SA, CT):
    """
    Calculates latent heat, or enthalpy, of evaporation at p = 0 (the
    surface).  It is defined as a function of Absolute Salinity, SA, and
    Conservative Temperature, CT, and is valid in the ranges
    0 < SA < 42 g/kg and 0 < CT < 40 deg C.  The errors range between
    -0.4 and 0.6 J/kg.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    latentheat_evap : array-like, J/kg
        latent heat of evaporation


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 3.39 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.latentheat_evap_ct(SA, CT)
latentheat_evap_CT.types = _gsw_ufuncs.latentheat_evap_ct.types
latentheat_evap_CT = match_args_return(latentheat_evap_CT)

def latentheat_evap_t(SA, t):
    """
    Calculates latent heat, or enthalpy, of evaporation at p = 0 (the
    surface).  It is defined as a function of Absolute Salinity, SA, and
    in-situ temperature, t, and is valid in the ranges 0 < SA < 40 g/kg
    and 0 < CT < 42 deg C. The errors range between -0.4 and 0.6 J/kg.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C

    Returns
    -------
    latentheat_evap : array-like, J/kg
        latent heat of evaporation


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 3.39 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.latentheat_evap_t(SA, t)
latentheat_evap_t.types = _gsw_ufuncs.latentheat_evap_t.types
latentheat_evap_t = match_args_return(latentheat_evap_t)

def latentheat_melting(SA, p):
    """
    Calculates latent heat, or enthalpy, of melting.  It is defined in terms
    of Absolute Salinity, SA, and sea pressure, p, and is valid in the
    ranges 0 < SA < 42 g kg^-1 and 0 < p < 10,000 dbar.  This is based on
    the IAPWS Releases IAPWS-09 (for pure water), IAPWS-08 (for the saline
    compoonent of seawater and IAPWS-06 for ice Ih.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    latentheat_melting : array-like, J/kg
        latent heat of melting


    References
    ----------
    IAPWS, 2008: Release on the IAPWS Formulation 2008 for the Thermodynamic
    Properties of Seawater. The International Association for the Properties
    of Water and Steam. Berlin, Germany, September 2008.  This Release is
    known as IAPWS-09.

    IAPWS, 2009a: Revised Release on the Equation of State 2006 for H2O Ice
    Ih. The International Association for the Properties of Water and Steam.
    Doorwerth, The Netherlands, September 2009. This Release is known as
    IAPWS-06

    IAPWS, 2009b: Supplementary Release on a Computationally Efficient
    Thermodynamic Formulation for Liquid Water for Oceanographic Use. The
    International Association for the Properties of Water and Steam.
    Doorwerth, The Netherlands, September 2009.  This Release is known as
    IAPWS-09.

    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 3.34 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.latentheat_melting(SA, p)
latentheat_melting.types = _gsw_ufuncs.latentheat_melting.types
latentheat_melting = match_args_return(latentheat_melting)

def melting_ice_equilibrium_SA_CT_ratio(SA, p):
    """
    Calculates the ratio of SA to CT changes when ice melts into seawater
    with both the seawater and the seaice temperatures being almost equal to
    the equilibrium freezing temperature.  It is assumed that a small mass
    of ice melts into an infinite mass of seawater.  If indeed the
    temperature of the seawater and the ice were both equal to the freezing
    temperature, then no melting or freezing would occur; an imbalance
    between these three temperatures is needed for freezing or melting to
    occur (the three temperatures being (1) the seawater temperature,
    (2) the ice temperature, and (3) the freezing temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    melting_ice_equilibrium_SA_CT_ratio : array-like, g/(kg K)
        the ratio dSA/dCT of SA to CT
        changes when ice melts into seawater, with
        the seawater and seaice being close to the
        freezing temperature.


    Notes
    -----
    The output, melting_ice_equilibrium_SA_CT_ratio, is dSA/dCT rather than
    dCT/dSA.  This is done so that when SA = 0, the output, dSA/dCT is zero
    whereas dCT/dSA would be infinite.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    See Eqn. (16) of this manuscript.


    """
    return _gsw_ufuncs.melting_ice_equilibrium_sa_ct_ratio(SA, p)
melting_ice_equilibrium_SA_CT_ratio.types = _gsw_ufuncs.melting_ice_equilibrium_sa_ct_ratio.types
melting_ice_equilibrium_SA_CT_ratio = match_args_return(melting_ice_equilibrium_SA_CT_ratio)

def melting_ice_equilibrium_SA_CT_ratio_poly(SA, p):
    """
    Calculates the ratio of SA to CT changes when ice melts into seawater
    with both the seawater and the seaice temperatures being almost equal to
    the equilibrium freezing temperature.  It is assumed that a small mass
    of ice melts into an infinite mass of seawater.  If indeed the
    temperature of the seawater and the ice were both equal to the freezing
    temperature, then no melting or freezing would occur; an imbalance
    between these three temperatures is needed for freezing or melting to
    occur (the three temperatures being (1) the seawater temperature,
    (2) the ice temperature, and (3) the freezing temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    melting_ice_equilibrium_SA_CT_ratio : array-like, g/(kg K)
        the ratio dSA/dCT of SA to CT
        changes when ice melts into seawater, with
        the seawater and seaice being close to the
        freezing temperature.


    Notes
    -----
    The output, melting_ice_equilibrium_SA_CT_ratio, is dSA/dCT rather than
    dCT/dSA.  This is done so that when SA = 0, the output, dSA/dCT is zero
    whereas dCT/dSA would be infinite.

    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    See Eqn. (16) of this manuscript.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.melting_ice_equilibrium_sa_ct_ratio_poly(SA, p)
melting_ice_equilibrium_SA_CT_ratio_poly.types = _gsw_ufuncs.melting_ice_equilibrium_sa_ct_ratio_poly.types
melting_ice_equilibrium_SA_CT_ratio_poly = match_args_return(melting_ice_equilibrium_SA_CT_ratio_poly)

def melting_ice_into_seawater(SA, CT, p, w_Ih, t_Ih):
    """
    Calculates the final Absolute Salinity, final Conservative Temperature
    and final ice mass fraction that results when a given mass fraction of
    ice melts and is mixed into seawater whose properties are (SA,CT,p).
    This code takes the seawater to contain no dissolved air.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    w_Ih : array-like
        mass fraction of ice: the mass of ice divided by the
        sum of the masses of ice and seawater. 0 <= wIh <= 1. unitless.
    t_Ih : array-like
        In-situ temperature of ice (ITS-90), degrees C

    Returns
    -------
    SA_final : array-like, g/kg
        Absolute Salinity of the seawater in the final state,
        whether or not any ice is present.
    CT_final : array-like, deg C
        Conservative Temperature of the seawater in the the final
        state, whether or not any ice is present.
    w_Ih_final : array-like, unitless
        mass fraction of ice in the final seawater-ice mixture.
        If this ice mass fraction is positive, the system is at
        thermodynamic equilibrium.  If this ice mass fraction is
        zero there is no ice in the final state which consists
        only of seawater which is warmer than the freezing
        temperature.


    Notes
    -----
    When the mass fraction w_Ih_final is calculated as being a positive
    value, the seawater-ice mixture is at thermodynamic equilibrium.

    This code returns w_Ih_final = 0 when the input bulk enthalpy, h_bulk,
    is sufficiently large (i.e. sufficiently "warm") so that there is no ice
    present in the final state.  In this case the final state consists of
    only seawater rather than being an equilibrium mixture of seawater and
    ice which occurs when w_Ih_final is positive.  Note that when
    w_Ih_final = 0, the final seawater is not at the freezing temperature.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of ice and sea ice into seawater, and frazil ice formation.
    Journal of Physical Oceanography, 44, 1751-1775.


    """
    return _gsw_ufuncs.melting_ice_into_seawater(SA, CT, p, w_Ih, t_Ih)
melting_ice_into_seawater.types = _gsw_ufuncs.melting_ice_into_seawater.types
melting_ice_into_seawater = match_args_return(melting_ice_into_seawater)

def melting_ice_SA_CT_ratio(SA, CT, p, t_Ih):
    """
    Calculates the ratio of SA to CT changes when ice melts into seawater.
    It is assumed that a small mass of ice melts into an infinite mass of
    seawater.  Because of the infinite mass of seawater, the ice will always
    melt.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    t_Ih : array-like
        In-situ temperature of ice (ITS-90), degrees C

    Returns
    -------
    melting_ice_SA_CT_ratio : array-like, g kg^-1 K^-1
        the ratio of SA to CT changes when ice melts
        into a large mass of seawater


    Notes
    -----
    The output, melting_seaice_SA_CT_ratio, is dSA/dCT rather than dCT/dSA.
    This is done so that when SA = 0, the output, dSA/dCT is zero whereas
    dCT/dSA would be infinite.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    See Eqn. (13) of this manuscript.


    """
    return _gsw_ufuncs.melting_ice_sa_ct_ratio(SA, CT, p, t_Ih)
melting_ice_SA_CT_ratio.types = _gsw_ufuncs.melting_ice_sa_ct_ratio.types
melting_ice_SA_CT_ratio = match_args_return(melting_ice_SA_CT_ratio)

def melting_ice_SA_CT_ratio_poly(SA, CT, p, t_Ih):
    """
    Calculates the ratio of SA to CT changes when ice melts into seawater.
    It is assumed that a small mass of ice melts into an infinite mass of
    seawater.  Because of the infinite mass of seawater, the ice will always
    melt.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    t_Ih : array-like
        In-situ temperature of ice (ITS-90), degrees C

    Returns
    -------
    melting_ice_SA_CT_ratio : array-like, g kg^-1 K^-1
        the ratio of SA to CT changes when ice melts
        into a large mass of seawater


    Notes
    -----
    The output, melting_seaice_SA_CT_ratio, is dSA/dCT rather than dCT/dSA.
    This is done so that when SA = 0, the output, dSA/dCT is zero whereas
    dCT/dSA would be infinite.

    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    See Eqn. (13) of this manuscript.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.melting_ice_sa_ct_ratio_poly(SA, CT, p, t_Ih)
melting_ice_SA_CT_ratio_poly.types = _gsw_ufuncs.melting_ice_sa_ct_ratio_poly.types
melting_ice_SA_CT_ratio_poly = match_args_return(melting_ice_SA_CT_ratio_poly)

def melting_seaice_equilibrium_SA_CT_ratio(SA, p):
    """
    Calculates the ratio of SA to CT changes when sea ice melts into
    seawater with both the seawater and the sea ice temperatures being
    almost equal to the equilibrium freezing temperature.  It is assumed
    that a small mass of seaice melts into an infinite mass of seawater.  If
    indeed the temperature of the seawater and the sea ice were both equal
    to the freezing temperature, then no melting or freezing would occur; an
    imbalance between these three temperatures is needed for freezing or
    melting to occur (the three temperatures being (1) the seawater
    temperature, (2) the sea ice temperature, and (3) the freezing
    temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    melting_seaice_equilibrium_SA_CT_ratio : array-like, g/(kg K)
        the ratio dSA/dCT of SA to CT
        changes when sea ice melts into seawater, with
        the seawater and sea ice being close to the
        freezing temperature.


    Notes
    -----
    Note that the output of this function, dSA/dCT is independent of the
    sea ice salinity, SA_seaice.  That is, the output applies equally to
    pure ice Ih and to sea ice with seaice salinity, SA_seaice.  This result
    is proven in McDougall et al. (2014).

    The output, melting_seaice_equilibrium_SA_CT_ratio, is dSA/dCT rather
    than dCT/dSA.  This is done so that when SA = 0, the output, dSA/dCT is
    zero whereas dCT/dSA would be infinite.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    See Eqn. (29) of this manuscript.


    """
    return _gsw_ufuncs.melting_seaice_equilibrium_sa_ct_ratio(SA, p)
melting_seaice_equilibrium_SA_CT_ratio.types = _gsw_ufuncs.melting_seaice_equilibrium_sa_ct_ratio.types
melting_seaice_equilibrium_SA_CT_ratio = match_args_return(melting_seaice_equilibrium_SA_CT_ratio)

def melting_seaice_equilibrium_SA_CT_ratio_poly(SA, p):
    """
    Calculates the ratio of SA to CT changes when sea ice melts into
    seawater with both the seawater and the sea ice temperatures being
    almost equal to the equilibrium freezing temperature.  It is assumed
    that a small mass of seaice melts into an infinite mass of seawater.  If
    indeed the temperature of the seawater and the sea ice were both equal
    to the freezing temperature, then no melting or freezing would occur; an
    imbalance between these three temperatures is needed for freezing or
    melting to occur (the three temperatures being (1) the seawater
    temperature, (2) the sea ice temperature, and (3) the freezing
    temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    melting_seaice_equilibrium_SA_CT_ratio : array-like, g/(kg K)
        the ratio dSA/dCT of SA to CT
        changes when sea ice melts into seawater, with
        the seawater and sea ice being close to the
        freezing temperature.


    Notes
    -----
    Note that the output of this function, dSA/dCT is independent of the
    sea ice salinity, SA_seaice.  That is, the output applies equally to
    pure ice Ih and to sea ice with seaice salinity, SA_seaice.  This result
    is proven in McDougall et al. (2014).

    The output, melting_seaice_equilibrium_SA_CT_ratio, is dSA/dCT rather
    than dCT/dSA.  This is done so that when SA = 0, the output, dSA/dCT is
    zero whereas dCT/dSA would be infinite.

    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    See Eqn. (29) of this manuscript.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.melting_seaice_equilibrium_sa_ct_ratio_poly(SA, p)
melting_seaice_equilibrium_SA_CT_ratio_poly.types = _gsw_ufuncs.melting_seaice_equilibrium_sa_ct_ratio_poly.types
melting_seaice_equilibrium_SA_CT_ratio_poly = match_args_return(melting_seaice_equilibrium_SA_CT_ratio_poly)

def melting_seaice_into_seawater(SA, CT, p, w_seaice, SA_seaice, t_seaice):
    """
    Calculates the Absolute Salinity and Conservative Temperature that
    results when a given mass of sea ice (or ice) melts and is mixed into a
    known mass of seawater (whose properties are (SA,CT,p)).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    w_seaice : array-like
        mass fraction of ice: the mass of sea-ice divided by the sum
        of the masses of sea-ice and seawater. 0 <= wIh <= 1. unitless.
    SA_seaice : array-like
        Absolute Salinity of sea ice: the mass fraction of salt
        in sea ice, expressed in g of salt per kg of sea ice.
    t_seaice : array-like
        In-situ temperature of the sea ice at pressure p (ITS-90), degrees C

    Returns
    -------
    SA_final : array-like, g/kg
        Absolute Salinity of the mixture of the melted sea ice
        (or ice) and the original seawater
    CT_final : array-like, deg C
        Conservative Temperature of the mixture of the melted
        sea ice (or ice) and the original seawater


    Notes
    -----
    If the ice contains no salt (e.g. if it is of glacial origin), then the
    input 'SA_seaice' should be set to zero.

    Ice formed at the sea surface (sea ice) typically contains between 2 g/kg
    and 12 g/kg of salt (defined as the mass of salt divided by the mass of
    ice Ih plus brine) and this programme returns NaN's if the input
    SA_seaice is greater than 15 g/kg.  If the SA_seaice input is not zero,
    usually this would imply that the pressure p should be zero, as sea ice
    only occurs near the sea surface.  The code does not impose that p = 0
    if SA_seaice is non-zero.  Rather, this is left to the user.

    The Absolute Salinity, SA_brine, of the brine trapped in little pockets
    in the sea ice, is in thermodynamic equilibrium with the ice Ih that
    surrounds these pockets.  As the sea ice temperature, t_seaice, may be
    less than the freezing temperature, SA_brine is usually greater than the
    Absolute Salinity of the seawater at the time and place when and where
    the sea ice was formed.  So usually SA_brine will be larger than SA.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    Eqns. (8) and (9) are the simplifications when SA_seaice = 0.


    """
    return _gsw_ufuncs.melting_seaice_into_seawater(SA, CT, p, w_seaice, SA_seaice, t_seaice)
melting_seaice_into_seawater.types = _gsw_ufuncs.melting_seaice_into_seawater.types
melting_seaice_into_seawater = match_args_return(melting_seaice_into_seawater)

def melting_seaice_SA_CT_ratio(SA, CT, p, SA_seaice, t_seaice):
    """
    Calculates the ratio of SA to CT changes when sea ice melts into
    seawater.  It is assumed that a small mass of sea ice melts into an
    infinite mass of seawater.  Because of the infinite mass of seawater,
    the sea ice will always melt.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    SA_seaice : array-like
        Absolute Salinity of sea ice: the mass fraction of salt
        in sea ice, expressed in g of salt per kg of sea ice.
    t_seaice : array-like
        In-situ temperature of the sea ice at pressure p (ITS-90), degrees C

    Returns
    -------
    melting_seaice_SA_CT_ratio : array-like, g/(kg K)
        the ratio dSA/dCT of SA to CT changes when
        sea ice melts into a large mass of seawater


    Notes
    -----
    Ice formed at the sea surface (sea ice) typically contains between 2 g/kg
    and 12 g/kg of salt (defined as the mass of salt divided by the mass of
    ice Ih plus brine) and this programme returns NaN's if the input
    SA_seaice is greater than 15 g/kg.  If the SA_seaice input is not zero,
    usually this would imply that the pressure p should be zero, as sea ice
    only occurs near the sea surface.  The code does not impose that p = 0
    if SA_seaice is non-zero.  Rather, this is left to the user.

    The Absolute Salinity, SA_brine, of the brine trapped in little pockets
    in the sea ice, is in thermodynamic equilibrium with the ice Ih that
    surrounds these pockets.  As the seaice temperature, t_seaice, may be
    less than the freezing temperature, SA_brine is usually greater than the
    Absolute Salinity of the seawater at the time and place when and where
    the sea ice was formed.  So usually SA_brine will be larger than SA.

    The output, melting_seaice_SA_CT_ratio, is dSA/dCT rather than dCT/dSA.
    This is done so that when (SA - seaice_SA) = 0, the output, dSA/dCT is
    zero whereas dCT/dSA would be infinite.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    See Eqn. (28) of this manuscript.


    """
    return _gsw_ufuncs.melting_seaice_sa_ct_ratio(SA, CT, p, SA_seaice, t_seaice)
melting_seaice_SA_CT_ratio.types = _gsw_ufuncs.melting_seaice_sa_ct_ratio.types
melting_seaice_SA_CT_ratio = match_args_return(melting_seaice_SA_CT_ratio)

def melting_seaice_SA_CT_ratio_poly(SA, CT, p, SA_seaice, t_seaice):
    """
    Calculates the ratio of SA to CT changes when sea ice melts into
    seawater.  It is assumed that a small mass of sea ice melts into an
    infinite mass of seawater.  Because of the infinite mass of seawater,
    the sea ice will always melt.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    SA_seaice : array-like
        Absolute Salinity of sea ice: the mass fraction of salt
        in sea ice, expressed in g of salt per kg of sea ice.
    t_seaice : array-like
        In-situ temperature of the sea ice at pressure p (ITS-90), degrees C

    Returns
    -------
    melting_seaice_SA_CT_ratio : array-like, g/(kg K)
        the ratio dSA/dCT of SA to CT changes when
        sea ice melts into a large mass of seawater


    Notes
    -----
    Ice formed at the sea surface (sea ice) typically contains between 2 g/kg
    and 12 g/kg of salt (defined as the mass of salt divided by the mass of
    ice Ih plus brine) and this programme returns NaN's if the input
    SA_seaice is greater than 15 g/kg.  If the SA_seaice input is not zero,
    usually this would imply that the pressure p should be zero, as sea ice
    only occurs near the sea surface.  The code does not impose that p = 0
    if SA_seaice is non-zero.  Rather, this is left to the user.

    The Absolute Salinity, SA_brine, of the brine trapped in little pockets
    in the sea ice, is in thermodynamic equilibrium with the ice Ih that
    surrounds these pockets.  As the seaice temperature, t_seaice, may be
    less than the freezing temperature, SA_brine is usually greater than the
    Absolute Salinity of the seawater at the time and place when and where
    the sea ice was formed.  So usually SA_brine will be larger than SA.

    The output, melting_seaice_SA_CT_ratio, is dSA/dCT rather than dCT/dSA.
    This is done so that when (SA - seaice_SA) = 0, the output, dSA/dCT is
    zero whereas dCT/dSA would be infinite.

    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    See Eqn. (31) of this manuscript.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.melting_seaice_sa_ct_ratio_poly(SA, CT, p, SA_seaice, t_seaice)
melting_seaice_SA_CT_ratio_poly.types = _gsw_ufuncs.melting_seaice_sa_ct_ratio_poly.types
melting_seaice_SA_CT_ratio_poly = match_args_return(melting_seaice_SA_CT_ratio_poly)

def O2sol(SA, CT, p, lon, lat):
    """
    Calculates the oxygen concentration expected at equilibrium with air at
    an Absolute Pressure of 101325 Pa (sea pressure of 0 dbar) including
    saturated water vapor.  This function uses the solubility coefficients
    derived from the data of Benson and Krause (1984), as fitted by Garcia
    and Gordon (1992, 1993).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    O2sol : array-like, umol/kg
        solubility of oxygen in micro-moles per kg


    Notes
    -----
    Note that this algorithm has not been approved by IOC and is not work
    from SCOR/IAPSO Working Group 127.  It is included in the GSW
    Oceanographic Toolbox as it seems to be oceanographic best practice.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    Benson, B.B., and D. Krause, 1984: The concentration and isotopic
    fractionation of oxygen dissolved in freshwater and seawater in
    equilibrium with the atmosphere. Limnology and Oceanography, 29,
    620-632.

    Garcia, H.E., and L.I. Gordon, 1992: Oxygen solubility in seawater:
    Better fitting equations. Limnology and Oceanography, 37, 1307-1312.

    Garcia, H.E., and L.I. Gordon, 1993: Erratum: Oxygen solubility in
    seawater: better fitting equations. Limnology and Oceanography, 38,
    656.


    """
    return _gsw_ufuncs.o2sol(SA, CT, p, lon, lat)
O2sol.types = _gsw_ufuncs.o2sol.types
O2sol = match_args_return(O2sol)

def O2sol_SP_pt(SP, pt):
    """
    Calculates the oxygen concentration expected at equilibrium with air at
    an Absolute Pressure of 101325 Pa (sea pressure of 0 dbar) including
    saturated water vapor.  This function uses the solubility coefficients
    derived from the data of Benson and Krause (1984), as fitted by Garcia
    and Gordon (1992, 1993).

    Parameters
    ----------
    SP : array-like
        Practical Salinity (PSS-78), unitless
    pt : array-like
        Potential temperature referenced to a sea pressure, degrees C

    Returns
    -------
    O2sol : array-like, umol/kg
        solubility of oxygen in micro-moles per kg


    Notes
    -----
    Note that this algorithm has not been approved by IOC and is not work
    from SCOR/IAPSO Working Group 127. It is included in the GSW
    Oceanographic Toolbox as it seems to be oceanographic best practice.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    Benson, B.B., and D. Krause, 1984: The concentration and isotopic
    fractionation of oxygen dissolved in freshwater and seawater in
    equilibrium with the atmosphere. Limnology and Oceanography, 29,
    620-632.

    Garcia, H.E., and L.I. Gordon, 1992: Oxygen solubility in seawater:
    Better fitting equations. Limnology and Oceanography, 37, 1307-1312.

    Garcia, H.E., and L.I. Gordon, 1993: Erratum: Oxygen solubility in
    seawater: better fitting equations. Limnology and Oceanography, 38,
    656.


    """
    return _gsw_ufuncs.o2sol_sp_pt(SP, pt)
O2sol_SP_pt.types = _gsw_ufuncs.o2sol_sp_pt.types
O2sol_SP_pt = match_args_return(O2sol_SP_pt)

def p_from_z(z, lat, geo_strf_dyn_height, sea_surface_geopotential):
    """
    Calculates sea pressure from height using computationally-efficient
    75-term expression for density, in terms of SA, CT and p (Roquet et al.,
    2015).  Dynamic height anomaly, geo_strf_dyn_height, if provided,
    must be computed with its p_ref = 0 (the surface). Also if provided,
    sea_surface_geopotental is the geopotential at zero sea pressure. This
    function solves Eqn.(3.32.3) of IOC et al. (2010) iteratively for p.

    Parameters
    ----------
    z : array-like
        Depth, positive up, m
    lat : array-like
        Latitude, -90 to 90 degrees
    geo_strf_dyn_height : array-like
        dynamic height anomaly, m^2/s^2
            Note that the reference pressure, p_ref, of geo_strf_dyn_height must
            be zero (0) dbar.
    sea_surface_geopotential : array-like
        geopotential at zero sea pressure,  m^2/s^2

    Returns
    -------
    p : array-like, dbar
        sea pressure
        ( i.e. absolute pressure - 10.1325 dbar )


    Notes
    -----
    Note. Height (z) is NEGATIVE in the ocean.  Depth is -z.
    Depth is not used in the GSW computer software library.

    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    McDougall, T.J., and S.J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, pp. 20-25.

    Moritz, H., 2000: Geodetic reference system 1980. J. Geodesy, 74,
    pp. 128-133.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling, 90, pp. 29-43.

    Saunders, P.M., 1981: Practical conversion of pressure to depth.
    Journal of Physical Oceanography, 11, pp. 573-574.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.p_from_z(z, lat, geo_strf_dyn_height, sea_surface_geopotential)
p_from_z.types = _gsw_ufuncs.p_from_z.types
p_from_z = match_args_return(p_from_z)

def pot_enthalpy_from_pt_ice(pt0_ice):
    """
    Calculates the potential enthalpy of ice from potential temperature of
    ice (whose reference sea pressure is zero dbar).

    Parameters
    ----------
    pt0_ice : array-like
        Potential temperature of ice (ITS-90), degrees C

    Returns
    -------
    pot_enthalpy_ice : array-like, J/kg
        potential enthalpy of ice


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.pot_enthalpy_from_pt_ice(pt0_ice)
pot_enthalpy_from_pt_ice.types = _gsw_ufuncs.pot_enthalpy_from_pt_ice.types
pot_enthalpy_from_pt_ice = match_args_return(pot_enthalpy_from_pt_ice)

def pot_enthalpy_from_pt_ice_poly(pt0_ice):
    """
    Calculates the potential enthalpy of ice from potential temperature of
    ice (whose reference sea pressure is zero dbar).  This is a
    compuationally efficient polynomial fit to the potential enthalpy of
    ice.

    Parameters
    ----------
    pt0_ice : array-like
        Potential temperature of ice (ITS-90), degrees C

    Returns
    -------
    pot_enthalpy_ice : array-like, J/kg
        potential enthalpy of ice


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.pot_enthalpy_from_pt_ice_poly(pt0_ice)
pot_enthalpy_from_pt_ice_poly.types = _gsw_ufuncs.pot_enthalpy_from_pt_ice_poly.types
pot_enthalpy_from_pt_ice_poly = match_args_return(pot_enthalpy_from_pt_ice_poly)

def pot_enthalpy_ice_freezing(SA, p):
    """
    Calculates the potential enthalpy of ice at which seawater freezes.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    pot_enthalpy_ice_freezing : array-like, J/kg
        potential enthalpy of ice at freezing
        of seawater


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.


    """
    return _gsw_ufuncs.pot_enthalpy_ice_freezing(SA, p)
pot_enthalpy_ice_freezing.types = _gsw_ufuncs.pot_enthalpy_ice_freezing.types
pot_enthalpy_ice_freezing = match_args_return(pot_enthalpy_ice_freezing)

def pot_enthalpy_ice_freezing_first_derivatives(SA, p):
    """
    Calculates the first derivatives of the potential enthalpy of ice at
    which seawater freezes, with respect to Absolute Salinity SA and
    pressure P (in Pa).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    pot_enthalpy_ice_freezing_SA : array-like, K kg/g
        the derivative of the potential enthalpy
        of ice at freezing (ITS-90) with respect to Absolute
        salinity at fixed pressure  [ K/(g/kg) ] i.e.
    pot_enthalpy_ice_freezing_P : array-like, K/Pa
        the derivative of the potential enthalpy
        of ice at freezing (ITS-90) with respect to pressure
        (in Pa) at fixed Absolute Salinity


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.


    """
    return _gsw_ufuncs.pot_enthalpy_ice_freezing_first_derivatives(SA, p)
pot_enthalpy_ice_freezing_first_derivatives.types = _gsw_ufuncs.pot_enthalpy_ice_freezing_first_derivatives.types
pot_enthalpy_ice_freezing_first_derivatives = match_args_return(pot_enthalpy_ice_freezing_first_derivatives)

def pot_enthalpy_ice_freezing_first_derivatives_poly(SA, p):
    """
    Calculates the first derivatives of the potential enthalpy of ice Ih at
    which ice melts into seawater with Absolute Salinity SA and at pressure
    p.  This code uses the computationally efficient polynomial fit of the
    freezing potential enthalpy of ice Ih (McDougall et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    pot_enthalpy_ice_freezing_SA : array-like, J/g
        the derivative of the potential enthalpy
        of ice at freezing (ITS-90) with respect to Absolute
        salinity at fixed pressure  [ (J/kg)/(g/kg) ] i.e.
    pot_enthalpy_ice_freezing_P : array-like, (J/kg)/Pa
        the derivative of the potential enthalpy
        of ice at freezing (ITS-90) with respect to pressure
        (in Pa) at fixed Absolute Salinity


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.

    McDougall et al. 2015: A reference for this polynomial.


    """
    return _gsw_ufuncs.pot_enthalpy_ice_freezing_first_derivatives_poly(SA, p)
pot_enthalpy_ice_freezing_first_derivatives_poly.types = _gsw_ufuncs.pot_enthalpy_ice_freezing_first_derivatives_poly.types
pot_enthalpy_ice_freezing_first_derivatives_poly = match_args_return(pot_enthalpy_ice_freezing_first_derivatives_poly)

def pot_enthalpy_ice_freezing_poly(SA, p):
    """
    Calculates the potential enthalpy of ice at which seawater freezes.
    The error of this fit ranges between -2.5 and 1 J/kg with an rms of
    1.07, between SA of 0 and 120 g/kg and p between 0 and 10,000 dbar (the
    error in the fit is between -0.7 and 0.7 with an rms of
    0.3, between SA of 0 and 120 g/kg and p between 0 and 5,000 dbar) when
    compared with the potential enthalpy calculated from the exact in-situ
    freezing temperature which is found by a Newton-Raphson iteration of the
    equality of the chemical potentials of water in seawater and in ice.
    Note that the potential enthalpy at freezing can be found
    by this exact method using the function gsw_pot_enthalpy_ice_freezing.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    pot_enthalpy_ice_freezing : array-like, J/kg
        potential enthalpy of ice at freezing
        of seawater


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.


    """
    return _gsw_ufuncs.pot_enthalpy_ice_freezing_poly(SA, p)
pot_enthalpy_ice_freezing_poly.types = _gsw_ufuncs.pot_enthalpy_ice_freezing_poly.types
pot_enthalpy_ice_freezing_poly = match_args_return(pot_enthalpy_ice_freezing_poly)

def pot_rho_t_exact(SA, t, p, p_ref):
    """
    Calculates potential density of seawater.  Note. This function outputs
    potential density, not potential density anomaly; that is, 1000 kg/m^3
    is not subtracted.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_ref : array-like
        Reference pressure, dbar

    Returns
    -------
    pot_rho_t_exact : array-like, kg/m^3
        potential density (not potential density anomaly)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 3.4 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.pot_rho_t_exact(SA, t, p, p_ref)
pot_rho_t_exact.types = _gsw_ufuncs.pot_rho_t_exact.types
pot_rho_t_exact = match_args_return(pot_rho_t_exact)

def pressure_coefficient_ice(t, p):
    """
    Calculates pressure coefficient of ice.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    pressure_coefficient_ice : array-like, Pa/K
        pressure coefficient of ice


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (2.15.1) of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.pressure_coefficient_ice(t, p)
pressure_coefficient_ice.types = _gsw_ufuncs.pressure_coefficient_ice.types
pressure_coefficient_ice = match_args_return(pressure_coefficient_ice)

def pressure_freezing_CT(SA, CT, saturation_fraction):
    """
    Calculates the pressure (in dbar) of seawater at the freezing
    temperature.  That is, the output is the pressure at which seawater,
    with Absolute Salinity SA, Conservative Temperature CT, and with
    saturation_fraction of dissolved air, freezes.  If the input values are
    such that there is no value of pressure in the range between 0 dbar and
    10,000 dbar for which seawater is at the freezing temperature, the
    output, pressure_freezing, is put equal to NaN.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    pressure_freezing : array-like, dbar
        sea pressure at which the seawater freezes
        ( i.e. absolute pressure - 10.1325 dbar )


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See section 3.33 of this TEOS-10 Manual.

    McDougall T. J. and S. J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.pressure_freezing_ct(SA, CT, saturation_fraction)
pressure_freezing_CT.types = _gsw_ufuncs.pressure_freezing_ct.types
pressure_freezing_CT = match_args_return(pressure_freezing_CT)

def pt0_from_t(SA, t, p):
    """
    Calculates potential temperature with reference pressure, p_ref = 0 dbar.
    The present routine is computationally faster than the more general
    function "gsw_pt_from_t(SA,t,p,p_ref)" which can be used for any
    reference pressure value.
    This subroutine calls "gsw_entropy_part(SA,t,p)",
    "gsw_entropy_part_zerop(SA,pt0)" and "gsw_gibbs_pt0_pt0(SA,pt0)".

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    pt0 : array-like, deg C
        potential temperature
        with reference sea pressure (p_ref) = 0 dbar.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 3.1 of this TEOS-10 Manual.

    McDougall T. J. and S. J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.pt0_from_t(SA, t, p)
pt0_from_t.types = _gsw_ufuncs.pt0_from_t.types
pt0_from_t = match_args_return(pt0_from_t)

def pt0_from_t_ice(t, p):
    """
    Calculates potential temperature of ice Ih with a reference pressure of
    0 dbar, from in-situ temperature, t.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    pt0_ice : array-like, deg C
        potential temperature of ice Ih with reference pressure of
        zero dbar (ITS-90)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix I of this TEOS-10 Manual.

    McDougall T. J. and S. J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.pt0_from_t_ice(t, p)
pt0_from_t_ice.types = _gsw_ufuncs.pt0_from_t_ice.types
pt0_from_t_ice = match_args_return(pt0_from_t_ice)

def pt_first_derivatives(SA, CT):
    """
    Calculates the following two partial derivatives of potential
    temperature (the regular potential temperature whose reference sea
    pressure is 0 dbar)
    (1) pt_SA, the derivative with respect to Absolute Salinity at
    constant Conservative Temperature, and
    (2) pt_CT, the derivative with respect to Conservative Temperature at
    constant Absolute Salinity.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    pt_SA : array-like, K/(g/kg)
        The derivative of potential temperature with respect to
        Absolute Salinity at constant Conservative Temperature.
    pt_CT : array-like, unitless
        The derivative of potential temperature with respect to
        Conservative Temperature at constant Absolute Salinity.
        pt_CT is dimensionless.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See Eqns. (A.12.6), (A.12.3), (P.6) and (P.8) of this TEOS-10 Manual.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.pt_first_derivatives(SA, CT)
pt_first_derivatives.types = _gsw_ufuncs.pt_first_derivatives.types
pt_first_derivatives = match_args_return(pt_first_derivatives)

def pt_from_CT(SA, CT):
    """
    Calculates potential temperature (with a reference sea pressure of
    zero dbar) from Conservative Temperature.  This function uses 1.5
    iterations through a modified Newton-Raphson (N-R) iterative solution
    procedure, starting from a rational-function-based initial condition
    for both pt and dCT_dpt.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    pt : array-like, deg C
        potential temperature referenced to a sea pressure
        of zero dbar (ITS-90)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See sections 3.1 and 3.3 of this TEOS-10 Manual.

    McDougall, T.J., and S.J. Wotherspoon, 2014: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.pt_from_ct(SA, CT)
pt_from_CT.types = _gsw_ufuncs.pt_from_ct.types
pt_from_CT = match_args_return(pt_from_CT)

def pt_from_entropy(SA, entropy):
    """
    Calculates potential temperature with reference pressure p_ref = 0 dbar
    and with entropy as an input variable.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    entropy : array-like
        Specific entropy, J/(kg*K)

    Returns
    -------
    pt : array-like, deg C
        potential temperature
        with reference sea pressure (p_ref) = 0 dbar.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix  A.10 of this TEOS-10 Manual.

    McDougall T. J. and S. J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.pt_from_entropy(SA, entropy)
pt_from_entropy.types = _gsw_ufuncs.pt_from_entropy.types
pt_from_entropy = match_args_return(pt_from_entropy)

def pt_from_pot_enthalpy_ice(pot_enthalpy_ice):
    """
    Calculates the potential temperature of ice from the potential enthalpy
    of ice.  The reference sea pressure of both the potential temperature
    and the potential enthalpy is zero dbar.

    Parameters
    ----------
    pot_enthalpy_ice : array-like
        Potential enthalpy of ice, J/kg

    Returns
    -------
    pt0_ice : array-like, deg C
        potential temperature of ice (ITS-90)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.

    McDougall T. J. and S. J. Wotherspoon, 2014: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.pt_from_pot_enthalpy_ice(pot_enthalpy_ice)
pt_from_pot_enthalpy_ice.types = _gsw_ufuncs.pt_from_pot_enthalpy_ice.types
pt_from_pot_enthalpy_ice = match_args_return(pt_from_pot_enthalpy_ice)

def pt_from_pot_enthalpy_ice_poly(pot_enthalpy_ice):
    """
    Calculates the potential temperature of ice (whose reference sea
    pressure is zero dbar) from the potential enthalpy of ice.  This is a
    compuationally efficient polynomial fit to the potential enthalpy of
    ice.

    Parameters
    ----------
    pot_enthalpy_ice : array-like
        Potential enthalpy of ice, J/kg

    Returns
    -------
    pt0_ice : array-like, deg C
        potential temperature of ice (ITS-90)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.pt_from_pot_enthalpy_ice_poly(pot_enthalpy_ice)
pt_from_pot_enthalpy_ice_poly.types = _gsw_ufuncs.pt_from_pot_enthalpy_ice_poly.types
pt_from_pot_enthalpy_ice_poly = match_args_return(pt_from_pot_enthalpy_ice_poly)

def pt_from_t(SA, t, p, p_ref):
    """
    Calculates potential temperature with the general reference pressure,
    p_ref, from in-situ temperature, t.  This function calls
    "gsw_entropy_part" which evaluates entropy except for the parts which
    are a function of Absolute Salinity alone.
    A faster gsw routine exists if p_ref is indeed zero dbar.  This routine
    is "gsw_pt0_from_t(SA,t,p)".

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_ref : array-like
        Reference pressure, dbar

    Returns
    -------
    pt : array-like, deg C
        potential temperature with reference pressure, p_ref, on the
        ITS-90 temperature scale


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 3.1 of this TEOS-10 Manual.

    McDougall T. J. and S. J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.pt_from_t(SA, t, p, p_ref)
pt_from_t.types = _gsw_ufuncs.pt_from_t.types
pt_from_t = match_args_return(pt_from_t)

def pt_from_t_ice(t, p, p_ref):
    """
    Calculates potential temperature of ice Ih with the general reference
    pressure, p_ref, from in-situ temperature, t.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_ref : array-like
        Reference pressure, dbar

    Returns
    -------
    pt_ice : array-like, deg C
        potential temperature of ice Ih with reference pressure,
        p_ref, on the ITS-90 temperature scale


    Notes
    -----
    A faster gsw routine exists if p_ref is indeed zero dbar.  This routine
    is "gsw_pt0_from_t_ice(t,p)".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix I of this TEOS-10 Manual.

    McDougall T. J. and S. J. Wotherspoon, 2014: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.pt_from_t_ice(t, p, p_ref)
pt_from_t_ice.types = _gsw_ufuncs.pt_from_t_ice.types
pt_from_t_ice = match_args_return(pt_from_t_ice)

def pt_second_derivatives(SA, CT):
    """
    Calculates the following three second-order derivatives of potential
    temperature (the regular potential temperature which has a reference
    sea pressure of 0 dbar),
    (1) pt_SA_SA, the second derivative with respect to Absolute Salinity
    at constant Conservative Temperature,
    (2) pt_SA_CT, the derivative with respect to Conservative Temperature
    and Absolute Salinity, and
    (3) pt_CT_CT, the second derivative with respect to Conservative
    Temperature at constant Absolute Salinity.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    pt_SA_SA : array-like, K/((g/kg)^2)
        The second derivative of potential temperature (the
        regular potential temperature which has reference sea
        pressure of 0 dbar) with respect to Absolute Salinity
        at constant Conservative Temperature.
    pt_SA_CT : array-like, 1/(g/kg)
        The derivative of potential temperature with respect
        to Absolute Salinity and Conservative Temperature.
    pt_CT_CT : array-like, 1/K
        The second derivative of potential temperature (the
        regular one with p_ref = 0 dbar) with respect to
        Conservative Temperature at constant SA.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See Eqns. (A.12.9) and (A.12.10) of this TEOS-10 Manual.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.pt_second_derivatives(SA, CT)
pt_second_derivatives.types = _gsw_ufuncs.pt_second_derivatives.types
pt_second_derivatives = match_args_return(pt_second_derivatives)

def rho(SA, CT, p):
    """
    Calculates in-situ density from Absolute Salinity and Conservative
    Temperature, using the computationally-efficient expression for
    specific volume in terms of SA, CT and p  (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    rho : array-like, kg/m^3
        in-situ density


    Notes
    -----
    Note that potential density with respect to reference pressure, pr, is
    obtained by calling this function with the pressure argument being pr
    (i.e. "gsw_rho(SA,CT,pr)").

    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix A.20 and appendix K of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling, 90, pp. 29-43.


    """
    return _gsw_ufuncs.rho(SA, CT, p)
rho.types = _gsw_ufuncs.rho.types
rho = match_args_return(rho)

def rho_alpha_beta(SA, CT, p):
    """
    Calculates in-situ density, the appropriate thermal expansion coefficient
    and the appropriate saline contraction coefficient of seawater from
    Absolute Salinity and Conservative Temperature.  This function uses the
    computationally-efficient expression for specific volume in terms of
    SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    rho : array-like, kg/m^3
        in-situ density
    alpha : array-like, 1/K
        thermal expansion coefficient
        with respect to Conservative Temperature
    beta : array-like, kg/g
        saline (i.e. haline) contraction
        coefficient at constant Conservative Temperature


    Notes
    -----
    Note that potential density (pot_rho) with respect to reference pressure
    p_ref is obtained by calling this function with the pressure argument
    being p_ref as in [pot_rho, ~, ~] = gsw_rho_alpha_beta(SA,CT,p_ref).

    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix A.20 and appendix K of this TEOS-10 Manual.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.rho_alpha_beta(SA, CT, p)
rho_alpha_beta.types = _gsw_ufuncs.rho_alpha_beta.types
rho_alpha_beta = match_args_return(rho_alpha_beta)

def rho_first_derivatives(SA, CT, p):
    """
    Calculates the three (3) partial derivatives of in-situ density with
    respect to Absolute Salinity, Conservative Temperature and pressure.
    Note that the pressure derivative is done with respect to pressure in
    Pa, not dbar.  This function uses the computationally-efficient
    expression for specific volume in terms of SA, CT and p (Roquet et al.,
    2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    rho_SA : array-like, (kg/m^3)(g/kg)^-1
        partial derivative of density
        with respect to Absolute Salinity
    rho_CT : array-like, kg/(m^3 K)
        partial derivative of density
        with respect to Conservative Temperature
    rho_P : array-like, kg/(m^3 Pa)
        partial derivative of density
        with respect to pressure in Pa


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix A.20 and appendix K of this TEOS-10 Manual.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.rho_first_derivatives(SA, CT, p)
rho_first_derivatives.types = _gsw_ufuncs.rho_first_derivatives.types
rho_first_derivatives = match_args_return(rho_first_derivatives)

def rho_first_derivatives_wrt_enthalpy(SA, CT, p):
    """
    Calculates the following two first-order derivatives of rho,
    (1) rho_SA_wrt_h, first-order derivative with respect to Absolute
    Salinity at constant h & p.
    (2) rho_h, first-order derivative with respect to h at
    constant SA & p.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    rho_SA_wrt_h : array-like, ((kg/m^3)(g/kg)^-1
        The first derivative of rho with respect to
        Absolute Salinity at constant CT & p.
    rho_h : array-like, (m^3/kg)(J/kg)^-1
        The first derivative of rho with respect to
        SA and CT at constant p.


    Notes
    -----
    Note that this function uses the using the computationally-efficient
    75 term expression for specific volume (Roquet et al., 2015).  There is
    an alternative to calling this function, namely
    gsw_specvol_first_derivatives_wrt_enthalpy_CT_exact(SA,CT,p) which uses
    the full Gibbs function (IOC et al., 2010).

    This 75-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel" described
    in McDougall et al. (2010).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.rho_first_derivatives_wrt_enthalpy(SA, CT, p)
rho_first_derivatives_wrt_enthalpy.types = _gsw_ufuncs.rho_first_derivatives_wrt_enthalpy.types
rho_first_derivatives_wrt_enthalpy = match_args_return(rho_first_derivatives_wrt_enthalpy)

def rho_ice(t, p):
    """
    Calculates in-situ density of ice from in-situ temperature and pressure.
    Note that the output, rho_ice, is density, not density anomaly;  that
    is, 1000 kg/m^3 is not subtracted from it.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    rho_ice : array-like, kg/m^3
        in-situ density of ice (not density anomaly)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.rho_ice(t, p)
rho_ice.types = _gsw_ufuncs.rho_ice.types
rho_ice = match_args_return(rho_ice)

def rho_second_derivatives(SA, CT, p):
    """
    Calculates the following five second-order derivatives of rho,
    (1) rho_SA_SA, second-order derivative with respect to Absolute
    Salinity at constant CT & p.
    (2) rho_SA_CT, second-order derivative with respect to SA & CT at
    constant p.
    (3) rho_CT_CT, second-order derivative with respect to CT at
    constant SA & p.
    (4) rho_SA_P, second-order derivative with respect to SA & P at
    constant CT.
    (5) rho_CT_P, second-order derivative with respect to CT & P at
    constant SA.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    rho_SA_SA : array-like, (kg/m^3)(g/kg)^-2
        The second-order derivative of rho with respect to
        Absolute Salinity at constant CT & p.
    rho_SA_CT : array-like, (kg/m^3)(g/kg)^-1 K^-1
        The second-order derivative of rho with respect to
        SA and CT at constant p.
    rho_CT_CT : array-like, (kg/m^3) K^-2
        The second-order derivative of rho with respect to CT at
        constant SA & p
    rho_SA_P : array-like, (kg/m^3)(g/kg)^-1 Pa^-1
        The second-order derivative with respect to SA & P at
        constant CT.
    rho_CT_P : array-like, (kg/m^3) K^-1 Pa^-1
        The second-order derivative with respect to CT & P at
        constant SA.


    Notes
    -----
    Note that this function uses the using the computationally-efficient
    expression for specific volume (Roquet et al., 2015).  There is an
    alternative to calling this function, namely
    gsw_rho_second_derivatives_CT_exact(SA,CT,p) which uses the full Gibbs
    function (IOC et al., 2010).

    This 75-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel" described
    in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.rho_second_derivatives(SA, CT, p)
rho_second_derivatives.types = _gsw_ufuncs.rho_second_derivatives.types
rho_second_derivatives = match_args_return(rho_second_derivatives)

def rho_second_derivatives_wrt_enthalpy(SA, CT, p):
    """
    Calculates the following three second-order derivatives of rho with
    respect to enthalpy,
    (1) rho_SA_SA, second-order derivative with respect to Absolute Salinity
    at constant h & p.
    (2) rho_SA_h, second-order derivative with respect to SA & h at
    constant p.
    (3) rho_h_h, second-order derivative with respect to h at
    constant SA & p.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    rho_SA_SA : array-like, (kg/m^3)(g/kg)^-2
        The second-order derivative of rho with respect to
        Absolute Salinity at constant h & p.
    rho_SA_h : array-like, J/(kg K(g/kg))
        The second-order derivative of rho with respect to
        SA and h at constant p.
    rho_h_h : array-like,
        The second-order derivative of rho with respect to h at
        constant SA & p


    Notes
    -----
    Note that this function uses the using the computationally-efficient
    expression for specific volume (Roquet et al., 2015).  There is an
    alternative to calling this function, namely
    gsw_rho_second_derivatives_wrt_enthalpy_CT_exact(SA,CT,p) which uses
    the full Gibbs function (IOC et al., 2010).

    This 75-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel" described
    in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.rho_second_derivatives_wrt_enthalpy(SA, CT, p)
rho_second_derivatives_wrt_enthalpy.types = _gsw_ufuncs.rho_second_derivatives_wrt_enthalpy.types
rho_second_derivatives_wrt_enthalpy = match_args_return(rho_second_derivatives_wrt_enthalpy)

def rho_t_exact(SA, t, p):
    """
    Calculates in-situ density of seawater from Absolute Salinity and
    in-situ temperature.  Note that the output, rho, is density,
    not density anomaly; that is, 1000 kg/m^3 is not subtracted from it.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    rho_t_exact : array-like, kg/m^3
        in-situ density (not density anomaly)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 2.8 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.rho_t_exact(SA, t, p)
rho_t_exact.types = _gsw_ufuncs.rho_t_exact.types
rho_t_exact = match_args_return(rho_t_exact)

def SA_freezing_from_CT(CT, p, saturation_fraction):
    """
    Calculates the Absolute Salinity of seawater at the freezing temperature.
    That is, the output is the Absolute Salinity of seawater, with
    Conservative Temperature CT, pressure p and the fraction
    saturation_fraction of dissolved air, that is in equilibrium
    with ice at the same in situ temperature and pressure.  If the input
    values are such that there is no positive value of Absolute Salinity for
    which seawater is frozen, the output, SA_freezing, is made a NaN.

    Parameters
    ----------
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    SA_freezing : array-like, g/kg
        Absolute Salinity of seawater when it freezes, for
        given input values of its Conservative Temperature,
        pressure and air saturation fraction.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See section 3.33 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.

    McDougall, T.J., and S.J. Wotherspoon, 2014: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.sa_freezing_from_ct(CT, p, saturation_fraction)
SA_freezing_from_CT.types = _gsw_ufuncs.sa_freezing_from_ct.types
SA_freezing_from_CT = match_args_return(SA_freezing_from_CT)

def SA_freezing_from_CT_poly(CT, p, saturation_fraction):
    """
    Calculates the Absolute Salinity of seawater at the freezing temperature.
    That is, the output is the Absolute Salinity of seawater, with the
    fraction saturation_fraction of dissolved air, that is in equilibrium
    with ice at Conservative Temperature CT and pressure p.  If the input
    values are such that there is no positive value of Absolute Salinity for
    which seawater is frozen, the output, SA_freezing, is put equal to NaN.

    Parameters
    ----------
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    SA_freezing : array-like, g/kg
        Absolute Salinity of seawater when it freezes, for
        given input values of Conservative Temperature
        pressure and air saturation fraction.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See section 3.33 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.

    McDougall T. J. and S. J. Wotherspoon, 2014: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.sa_freezing_from_ct_poly(CT, p, saturation_fraction)
SA_freezing_from_CT_poly.types = _gsw_ufuncs.sa_freezing_from_ct_poly.types
SA_freezing_from_CT_poly = match_args_return(SA_freezing_from_CT_poly)

def SA_freezing_from_t(t, p, saturation_fraction):
    """
    Calculates the Absolute Salinity of seawater at the freezing temperature.
    That is, the output is the Absolute Salinity of seawater, with the
    fraction saturation_fraction of dissolved air, that is in equilibrium
    with ice at in-situ temperature t and pressure p.  If the input values
    are such that there is no positive value of Absolute Salinity for which
    seawater is frozen, the output, SA_freezing, is set to NaN.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    SA_freezing : array-like, g/kg
        Absolute Salinity of seawater when it freezes, for
        given input values of in situ temperature, pressure and
        air saturation fraction.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See section 3.33 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.

    McDougall, T.J., and S.J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.sa_freezing_from_t(t, p, saturation_fraction)
SA_freezing_from_t.types = _gsw_ufuncs.sa_freezing_from_t.types
SA_freezing_from_t = match_args_return(SA_freezing_from_t)

def SA_freezing_from_t_poly(t, p, saturation_fraction):
    """
    Calculates the Absolute Salinity of seawater at the freezing temperature.
    That is, the output is the Absolute Salinity of seawater, with the
    fraction saturation_fraction of dissolved air, that is in equilibrium
    with ice at in-situ temperature t and pressure p.  If the input values
    are such that there is no positive value of Absolute Salinity for which
    seawater is frozen, the output, SA_freezing, is put equal to NaN.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    SA_freezing : array-like, g/kg
        Absolute Salinity of seawater when it freezes, for
        given input values of in situ temperature, pressure and
        air saturation fraction.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See section 3.33 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.

    McDougall T. J. and S. J. Wotherspoon, 2014: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.sa_freezing_from_t_poly(t, p, saturation_fraction)
SA_freezing_from_t_poly.types = _gsw_ufuncs.sa_freezing_from_t_poly.types
SA_freezing_from_t_poly = match_args_return(SA_freezing_from_t_poly)

def SA_from_rho(rho, CT, p):
    """
    Calculates the Absolute Salinity of a seawater sample, for given values
    of its density, Conservative Temperature and sea pressure (in dbar).
    This function uses the computationally-efficient 75-term expression for
    specific volume in terms of SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    rho : array-like
        Seawater density (not anomaly) in-situ, e.g., 1026 kg/m^3.
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    SA : array-like, g/kg
        Absolute Salinity.


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 2.5 of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Millero, F.J., R. Feistel, D.G. Wright, and T.J. McDougall, 2008:
    The composition of Standard Seawater and the definition of the
    Reference-Composition Salinity Scale. Deep-Sea Res. I, 55, 50-72.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.sa_from_rho(rho, CT, p)
SA_from_rho.types = _gsw_ufuncs.sa_from_rho.types
SA_from_rho = match_args_return(SA_from_rho)

def SA_from_SP(SP, p, lon, lat):
    """
    Calculates Absolute Salinity from Practical Salinity.  Since SP is
    non-negative by definition, this function changes any negative input
    values of SP to be zero.

    Parameters
    ----------
    SP : array-like
        Practical Salinity (PSS-78), unitless
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    SA : array-like, g/kg
        Absolute Salinity


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 2.5 and appendices A.4 and A.5 of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, F.J. Millero, R. Pawlowicz and
    P.M. Barker, 2012: A global algorithm for estimating Absolute Salinity.
    Ocean Science, 8, 1123-1134.
    https://os.copernicus.org/articles/8/1123/2012/os-8-1123-2012.pdf


    """
    return _gsw_ufuncs.sa_from_sp(SP, p, lon, lat)
SA_from_SP.types = _gsw_ufuncs.sa_from_sp.types
SA_from_SP = match_args_return(SA_from_SP)

def SA_from_SP_Baltic(SP, lon, lat):
    """
    Calculates Absolute Salinity in the Baltic Sea, from Practical Salinity.
    Since SP is non-negative by definition, this function changes any
    negative input values of SP to be zero.
    Note. This programme will only produce Absolute Salinity values for the
    Baltic Sea.

    Parameters
    ----------
    SP : array-like
        Practical Salinity (PSS-78), unitless
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    SA_baltic : array-like, g kg^-1
        Absolute Salinity in the Baltic Sea


    References
    ----------
    Feistel, R., S. Weinreben, H. Wolf, S. Seitz, P. Spitzer, B. Adel,
    G. Nausch, B. Schneider and D. G. Wright, 2010: Density and Absolute
    Salinity of the Baltic Sea 2006-2009.  Ocean Science, 6, 3-24.
    https://os.copernicus.org/articles/6/3/2010/os-6-3-2010.pdf

    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, F.J. Millero, R. Pawlowicz and
    P.M. Barker, 2012: A global algorithm for estimating Absolute Salinity.
    Ocean Science, 8, 1123-1134.
    https://os.copernicus.org/articles/8/1123/2012/os-8-1123-2012.pdf


    """
    return _gsw_ufuncs.sa_from_sp_baltic(SP, lon, lat)
SA_from_SP_Baltic.types = _gsw_ufuncs.sa_from_sp_baltic.types
SA_from_SP_Baltic = match_args_return(SA_from_SP_Baltic)

def SA_from_Sstar(Sstar, p, lon, lat):
    """
    Calculates Absolute Salinity from Preformed Salinity.

    Parameters
    ----------
    Sstar : array-like
        Preformed Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    SA : array-like, g/kg
        Absolute Salinity


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, F.J. Millero, R. Pawlowicz and
    P.M. Barker, 2012: A global algorithm for estimating Absolute Salinity.
    Ocean Science, 8, 1123-1134.
    https://os.copernicus.org/articles/8/1123/2012/os-8-1123-2012.pdf


    """
    return _gsw_ufuncs.sa_from_sstar(Sstar, p, lon, lat)
SA_from_Sstar.types = _gsw_ufuncs.sa_from_sstar.types
SA_from_Sstar = match_args_return(SA_from_Sstar)

def SAAR(p, lon, lat):
    """
    Calculates the Absolute Salinity Anomaly Ratio, SAAR, in the open ocean
    by spatially interpolating the global reference data set of SAAR to the
    location of the seawater sample.

    Parameters
    ----------
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    SAAR : array-like, unitless
        Absolute Salinity Anomaly Ratio


    Notes
    -----
    This function uses version 3.0 of the SAAR look up table (15th May 2011).

    The Absolute Salinity Anomaly Ratio in the Baltic Sea is evaluated
    separately, since it is a function of Practical Salinity, not of space.
    The present function returns a SAAR of zero for data in the Baltic Sea.
    The correct way of calculating Absolute Salinity in the Baltic Sea is by
    calling gsw_SA_from_SP.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, F.J. Millero, R. Pawlowicz and
    P.M. Barker, 2012: A global algorithm for estimating Absolute Salinity.
    Ocean Science, 8, 1123-1134.
    https://os.copernicus.org/articles/8/1123/2012/os-8-1123-2012.pdf

    See also gsw_SA_from_SP, gsw_deltaSA_atlas

    Reference page in Help browser
    <a href="matlab:doc gsw_SAAR">doc gsw_SAAR</a>
    Note that this reference page includes the code contained in gsw_SAAR.
    We have opted to encode this programme as it is a global standard and
    such we cannot allow anyone to change it.


    """
    return _gsw_ufuncs.saar(p, lon, lat)
SAAR.types = _gsw_ufuncs.saar.types
SAAR = match_args_return(SAAR)

def seaice_fraction_to_freeze_seawater(SA, CT, p, SA_seaice, t_seaice):
    """
    Calculates the mass fraction of sea ice (mass of sea ice divided by mass
    of sea ice plus seawater), which, when melted into seawater having the
    properties (SA,CT,p) causes the final seawater to be at the freezing
    temperature.  The other outputs are the Absolute Salinity and
    Conservative Temperature of the final seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    SA_seaice : array-like
        Absolute Salinity of sea ice: the mass fraction of salt
        in sea ice, expressed in g of salt per kg of sea ice.
    t_seaice : array-like
        In-situ temperature of the sea ice at pressure p (ITS-90), degrees C

    Returns
    -------
    SA_freeze : array-like, g/kg
        Absolute Salinity of seawater after the mass fraction of
        sea ice, w_seaice, at temperature t_seaice has melted into
        the original seawater, and the final mixture is at the
        freezing temperature of seawater.
    CT_freeze : array-like, deg C
        Conservative Temperature of seawater after the mass
        fraction, w_seaice, of sea ice at temperature t_seaice has
        melted into the original seawater, and the final mixture
        is at the freezing temperature of seawater.
    w_seaice : array-like, unitless
        mass fraction of sea ice, at SA_seaice and t_seaice,
        which, when melted into seawater at (SA,CT,p) leads to the
        final mixed seawater being at the freezing temperature.
        This output is between 0 and 1.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall T.J. and S.J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.
    See Eqn. (23) of this manuscript.


    """
    return _gsw_ufuncs.seaice_fraction_to_freeze_seawater(SA, CT, p, SA_seaice, t_seaice)
seaice_fraction_to_freeze_seawater.types = _gsw_ufuncs.seaice_fraction_to_freeze_seawater.types
seaice_fraction_to_freeze_seawater = match_args_return(seaice_fraction_to_freeze_seawater)

def sigma0(SA, CT):
    """
    Calculates potential density anomaly with reference pressure of 0 dbar,
    this being this particular potential density minus 1000 kg/m^3.  This
    function has inputs of Absolute Salinity and Conservative Temperature.
    This function uses the computationally-efficient expression for
    specific volume in terms of SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    sigma0 : array-like, kg/m^3
        potential density anomaly with
        respect to a reference pressure of 0 dbar,
        that is, this potential density - 1000 kg/m^3.


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (A.30.1) of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.sigma0(SA, CT)
sigma0.types = _gsw_ufuncs.sigma0.types
sigma0 = match_args_return(sigma0)

def sigma1(SA, CT):
    """
    Calculates potential density anomaly with reference pressure of 1000
    dbar, this being this particular potential density minus 1000 kg/m^3.
    This function has inputs of Absolute Salinity and Conservative
    Temperature.  This function uses the computationally-efficient
    expression for specific volume in terms of SA, CT and p (Roquet et al.,
    2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    sigma1 : array-like, kg/m^3
        potential density anomaly with
        respect to a reference pressure of 1000 dbar,
        that is, this potential density - 1000 kg/m^3.


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (A.30.1) of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.sigma1(SA, CT)
sigma1.types = _gsw_ufuncs.sigma1.types
sigma1 = match_args_return(sigma1)

def sigma2(SA, CT):
    """
    Calculates potential density anomaly with reference pressure of 2000
    dbar, this being this particular potential density minus 1000 kg/m^3.
    Temperature.  This function uses the computationally-efficient
    expression for specific volume in terms of SA, CT and p (Roquet et al.,
    2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    sigma2 : array-like, kg/m^3
        potential density anomaly with
        respect to a reference pressure of 2000 dbar,
        that is, this potential density - 1000 kg/m^3.


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (A.30.1) of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.sigma2(SA, CT)
sigma2.types = _gsw_ufuncs.sigma2.types
sigma2 = match_args_return(sigma2)

def sigma3(SA, CT):
    """
    Calculates potential density anomaly with reference pressure of 3000
    dbar, this being this particular potential density minus 1000 kg/m^3.
    Temperature.  This function uses the computationally-efficient
    expression for specific volume in terms of SA, CT and p (Roquet et al.,
    2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    sigma3 : array-like, kg/m^3
        potential density anomaly with
        respect to a reference pressure of 3000 dbar,
        that is, this potential density - 1000 kg/m^3.


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (A.30.1) of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.sigma3(SA, CT)
sigma3.types = _gsw_ufuncs.sigma3.types
sigma3 = match_args_return(sigma3)

def sigma4(SA, CT):
    """
    Calculates potential density anomaly with reference pressure of 4000
    dbar, this being this particular potential density minus 1000 kg/m^3.
    Temperature.  This function uses the computationally-efficient
    expression for specific volume in terms of SA, CT and p (Roquet et al.,
    2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    sigma4 : array-like, kg/m^3
        potential density anomaly with
        respect to a reference pressure of 4000 dbar,
        that is, this potential density - 1000 kg/m^3.


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (A.30.1) of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.sigma4(SA, CT)
sigma4.types = _gsw_ufuncs.sigma4.types
sigma4 = match_args_return(sigma4)

def sound_speed(SA, CT, p):
    """
    Calculates the speed of sound in seawater.  This function has inputs of
    Absolute Salinity and Conservative Temperature.  This function uses the
    computationally-efficient expression for specific volume in terms of SA,
    CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    sound_speed : array-like, m/s
        speed of sound in seawater


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (2.17.1) of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.sound_speed(SA, CT, p)
sound_speed.types = _gsw_ufuncs.sound_speed.types
sound_speed = match_args_return(sound_speed)

def sound_speed_ice(t, p):
    """
    Calculates the compression speed of sound in ice.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    sound_speed_ice : array-like, m/s
        compression speed of sound in ice


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.sound_speed_ice(t, p)
sound_speed_ice.types = _gsw_ufuncs.sound_speed_ice.types
sound_speed_ice = match_args_return(sound_speed_ice)

def sound_speed_t_exact(SA, t, p):
    """
    Calculates the speed of sound in seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    sound_speed_t_exact : array-like, m/s
        speed of sound in seawater


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (2.17.1) of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.sound_speed_t_exact(SA, t, p)
sound_speed_t_exact.types = _gsw_ufuncs.sound_speed_t_exact.types
sound_speed_t_exact = match_args_return(sound_speed_t_exact)

def SP_from_C(C, t, p):
    """
    Calculates Practical Salinity, SP, from conductivity, C, primarily using
    the PSS-78 algorithm.  Note that the PSS-78 algorithm for Practical
    Salinity is only valid in the range 2 < SP < 42.  If the PSS-78
    algorithm produces a Practical Salinity that is less than 2 then the
    Practical Salinity is recalculated with a modified form of the Hill et
    al. (1986) formula.  The modification of the Hill et al. (1986)
    expression is to ensure that it is exactly consistent with PSS-78
    at SP = 2.  Note that the input values of conductivity need to be in
    units of mS/cm (not S/m).

    Parameters
    ----------
    C : array-like
        Conductivity, mS/cm
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    SP : array-like, unitless
        Practical Salinity on the PSS-78 scale


    References
    ----------
    Culkin and Smith, 1980:  Determination of the Concentration of Potassium
    Chloride Solution Having the Same Electrical Conductivity, at 15C and
    Infinite Frequency, as Standard Seawater of Salinity 35.0000
    (Chlorinity 19.37394), IEEE J. Oceanic Eng, 5, 22-23.

    Hill, K.D., T.M. Dauphinee & D.J. Woods, 1986: The extension of the
    Practical Salinity Scale 1978 to low salinities. IEEE J. Oceanic Eng.,
    11, 109 - 112.

    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix E of this TEOS-10 Manual.

    Unesco, 1983: Algorithms for computation of fundamental properties of
    seawater. Unesco Technical Papers in Marine Science, 44, 53 pp.


    """
    return _gsw_ufuncs.sp_from_c(C, t, p)
SP_from_C.types = _gsw_ufuncs.sp_from_c.types
SP_from_C = match_args_return(SP_from_C)

def SP_from_SA(SA, p, lon, lat):
    """
    Calculates Practical Salinity from Absolute Salinity.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    SP : array-like, unitless
        Practical Salinity  (PSS-78)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, F.J. Millero, R. Pawlowicz and
    P.M. Barker, 2012: A global algorithm for estimating Absolute Salinity.
    Ocean Science, 8, 1123-1134.
    https://os.copernicus.org/articles/8/1123/2012/os-8-1123-2012.pdf


    """
    return _gsw_ufuncs.sp_from_sa(SA, p, lon, lat)
SP_from_SA.types = _gsw_ufuncs.sp_from_sa.types
SP_from_SA = match_args_return(SP_from_SA)

def SP_from_SA_Baltic(SA, lon, lat):
    """
    Calculates Practical Salinity for the Baltic Sea, from a value computed
    analytically from Absolute Salinity.
    Note. This programme will only produce Practical Salinty values for the
    Baltic Sea.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    SP_baltic : array-like, unitless
        Practical Salinity


    References
    ----------
    Feistel, R., S. Weinreben, H. Wolf, S. Seitz, P. Spitzer, B. Adel,
    G. Nausch, B. Schneider and D. G. Wright, 2010c: Density and Absolute
    Salinity of the Baltic Sea 2006-2009.  Ocean Science, 6, 3-24.
    https://os.copernicus.org/articles/6/3/2010/os-6-3-2010.pdf

    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, F.J. Millero, R. Pawlowicz and
    P.M. Barker, 2012: A global algorithm for estimating Absolute Salinity.
    Ocean Science, 8, 1123-1134.
    https://os.copernicus.org/articles/8/1123/2012/os-8-1123-2012.pdf


    """
    return _gsw_ufuncs.sp_from_sa_baltic(SA, lon, lat)
SP_from_SA_Baltic.types = _gsw_ufuncs.sp_from_sa_baltic.types
SP_from_SA_Baltic = match_args_return(SP_from_SA_Baltic)

def SP_from_SK(SK):
    """
    Calculates Practical Salinity from Knudsen Salinity.

    Parameters
    ----------
    SK : array-like
        Knudsen Salinity, ppt

    Returns
    -------
    SP : array-like, unitless
        Practical Salinity  (PSS-78)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Appendix A.3 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.sp_from_sk(SK)
SP_from_SK.types = _gsw_ufuncs.sp_from_sk.types
SP_from_SK = match_args_return(SP_from_SK)

def SP_from_SR(SR):
    """
    Calculates Practical Salinity from Reference Salinity.

    Parameters
    ----------
    SR : array-like
        Reference Salinity, g/kg

    Returns
    -------
    SP : array-like, unitless
        Practical Salinity  (PSS-78)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.sp_from_sr(SR)
SP_from_SR.types = _gsw_ufuncs.sp_from_sr.types
SP_from_SR = match_args_return(SP_from_SR)

def SP_from_Sstar(Sstar, p, lon, lat):
    """
    Calculates Practical Salinity from Preformed Salinity.

    Parameters
    ----------
    Sstar : array-like
        Preformed Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    SP : array-like, unitless
        Practical Salinity  (PSS-78)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, F.J. Millero, R. Pawlowicz and
    P.M. Barker, 2012: A global algorithm for estimating Absolute Salinity.
    Ocean Science, 8, 1123-1134.
    https://os.copernicus.org/articles/8/1123/2012/os-8-1123-2012.pdf


    """
    return _gsw_ufuncs.sp_from_sstar(Sstar, p, lon, lat)
SP_from_Sstar.types = _gsw_ufuncs.sp_from_sstar.types
SP_from_Sstar = match_args_return(SP_from_Sstar)

def SP_salinometer(Rt, t):
    """
    Calculates Practical Salinity SP from a salinometer, primarily using the
    PSS-78 algorithm.  Note that the PSS-78 algorithm for Practical Salinity
    is only valid in the range 2 < SP < 42.  If the PSS-78 algorithm
    produces a Practical Salinity that is less than 2 then the Practical
    Salinity is recalculated with a modified form of the Hill et al. (1986)
    formula.  The modification of the Hill et al. (1986) expression is to
    ensure that it is exactly consistent with PSS-78 at SP = 2.

    Parameters
    ----------
    Rt : array-like
        C(SP,t_68,0)/C(SP=35,t_68,0), unitless
    t : array-like
        In-situ temperature (ITS-90), degrees C

    Returns
    -------
    SP : array-like, unitless
        Practical Salinity on the PSS-78 scale
        t may have dimensions 1x1 or Mx1 or 1xN or MxN, where Rt is MxN.


    Notes
    -----
    A laboratory salinometer has the ratio of conductivities, Rt, as an
    output, and the present function uses this conductivity ratio and the
    temperature t of the salinometer bath as the two input variables.


    References
    ----------
    Fofonoff, P. and R.C. Millard Jr. 1983: Algorithms for computation of
    fundamental properties of seawater. Unesco Tech. Pap. in Mar. Sci., 44,
    53 pp.

    Hill, K.D., T.M. Dauphinee & D.J. Woods, 1986: The extension of the
    Practical Salinity Scale 1978 to low salinities. IEEE J. Oceanic Eng.,
    11, 109 - 112.

    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix E of this TEOS-10 Manual, and in particular,
    Eqns. (E.2.1) and (E.2.6).


    """
    return _gsw_ufuncs.sp_salinometer(Rt, t)
SP_salinometer.types = _gsw_ufuncs.sp_salinometer.types
SP_salinometer = match_args_return(SP_salinometer)

def specvol(SA, CT, p):
    """
    Calculates specific volume from Absolute Salinity, Conservative
    Temperature and pressure, using the computationally-efficient 75-term
    polynomial expression for specific volume (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    specvol : array-like, m^3/kg
        specific volume


    Notes
    -----
    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmos. Ocean. Tech., 20,
    730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.specvol(SA, CT, p)
specvol.types = _gsw_ufuncs.specvol.types
specvol = match_args_return(specvol)

def specvol_alpha_beta(SA, CT, p):
    """
    Calculates specific volume, the appropriate thermal expansion coefficient
    and the appropriate saline contraction coefficient of seawater from
    Absolute Salinity and Conservative Temperature.  This function uses the
    computationally-efficient expression for specific volume in terms of
    SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    specvol : array-like, m/kg
        specific volume
    alpha : array-like, 1/K
        thermal expansion coefficient
        with respect to Conservative Temperature
    beta : array-like, kg/g
        saline (i.e. haline) contraction
        coefficient at constant Conservative Temperature


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix A.20 and appendix K of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.specvol_alpha_beta(SA, CT, p)
specvol_alpha_beta.types = _gsw_ufuncs.specvol_alpha_beta.types
specvol_alpha_beta = match_args_return(specvol_alpha_beta)

def specvol_anom_standard(SA, CT, p):
    """
    Calculates specific volume anomaly from Absolute Salinity, Conservative
    Temperature and pressure. It uses the computationally-efficient
    expression for specific volume as a function of SA, CT and p (Roquet
    et al., 2015).  The reference value to which the anomaly is calculated
    has an Absolute Salinity of SSO and Conservative Temperature equal to
    0 degrees C.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    specvol_anom : array-like, m^3/kg
        specific volume anomaly


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqn. (3.7.3) of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.specvol_anom_standard(SA, CT, p)
specvol_anom_standard.types = _gsw_ufuncs.specvol_anom_standard.types
specvol_anom_standard = match_args_return(specvol_anom_standard)

def specvol_first_derivatives(SA, CT, p):
    """
    Calculates the following three first-order derivatives of specific
    volume (v),
    (1) v_SA, first-order derivative with respect to Absolute Salinity
    at constant CT & p.
    (2) v_CT, first-order derivative with respect to CT at
    constant SA & p.
    (3) v_P, first-order derivative with respect to P at constant SA
    and CT.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    v_SA : array-like, (m^3/kg)(g/kg)^-1
        The first derivative of specific volume with respect to
        Absolute Salinity at constant CT & p.
    v_CT : array-like, m^3/(K kg)
        The first derivative of specific volume with respect to
        CT at constant SA and p.
    v_P : array-like, m^3/(Pa kg)
        The first derivative of specific volume with respect to
        P at constant SA and CT.


    Notes
    -----
    Note that this function uses the using the computationally-efficient
    75-term expression for specific volume (Roquet et al., 2015).  There is
    an alternative to calling this function, namely
    gsw_specvol_first_derivatives_CT_exact(SA,CT,p) which uses the full
    Gibbs function (IOC et al., 2010).

    This 75-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel" described
    in McDougall et al. (2010).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.specvol_first_derivatives(SA, CT, p)
specvol_first_derivatives.types = _gsw_ufuncs.specvol_first_derivatives.types
specvol_first_derivatives = match_args_return(specvol_first_derivatives)

def specvol_first_derivatives_wrt_enthalpy(SA, CT, p):
    """
    Calculates the following two first-order derivatives of specific
    volume (v),
    (1) v_SA_wrt_h, first-order derivative with respect to Absolute Salinity
    at constant h & p.
    (2) v_h, first-order derivative with respect to h at
    constant SA & p.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    v_SA_wrt_h : array-like, (m^3/kg)(g/kg)^-1
        The first derivative of specific volume with respect to
        Absolute Salinity at constant CT & p.
    v_h : array-like, (m^3/kg)(J/kg)^-1
        The first derivative of specific volume with respect to
        SA and CT at constant p.


    Notes
    -----
    Note that this function uses the using the computationally-efficient
    75 term expression for specific volume (Roquet et al., 2015).  There is
    an alternative to calling this function, namely
    gsw_specvol_first_derivatives_wrt_enthalpy_CT_exact(SA,CT,p) which uses
    the full Gibbs function (IOC et al., 2010).

    This 75-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel" described
    in McDougall et al. (2010).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.specvol_first_derivatives_wrt_enthalpy(SA, CT, p)
specvol_first_derivatives_wrt_enthalpy.types = _gsw_ufuncs.specvol_first_derivatives_wrt_enthalpy.types
specvol_first_derivatives_wrt_enthalpy = match_args_return(specvol_first_derivatives_wrt_enthalpy)

def specvol_ice(t, p):
    """
    Calculates the specific volume of ice.

    Parameters
    ----------
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    specvol_ice : array-like, m^3/kg
        specific volume


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.specvol_ice(t, p)
specvol_ice.types = _gsw_ufuncs.specvol_ice.types
specvol_ice = match_args_return(specvol_ice)

def specvol_second_derivatives(SA, CT, p):
    """
    Calculates the following five second-order derivatives of specific
    volume (v),
    (1) v_SA_SA, second-order derivative with respect to Absolute Salinity
    at constant CT & p.
    (2) v_SA_CT, second-order derivative with respect to SA & CT at
    constant p.
    (3) v_CT_CT, second-order derivative with respect to CT at constant SA
    and p.
    (4) v_SA_P, second-order derivative with respect to SA & P at
    constant CT.
    (5) v_CT_P, second-order derivative with respect to CT & P at
    constant SA.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    v_SA_SA : array-like, (m^3/kg)(g/kg)^-2
        The second derivative of specific volume with respect to
        Absolute Salinity at constant CT & p.
    v_SA_CT : array-like, (m^3/kg)(g/kg)^-1 K^-1
        The second derivative of specific volume with respect to
        SA and CT at constant p.
    v_CT_CT : array-like, (m^3/kg) K^-2)
        The second derivative of specific volume with respect to
        CT at constant SA and p.
    v_SA_P : array-like, (m^3/kg)(g/kg)^-1 Pa^-1
        The second derivative of specific volume with respect to
        SA and P at constant CT.
    v_CT_P : array-like, (m^3/kg) K^-1 Pa^-1
        The second derivative of specific volume with respect to
        CT and P at constant SA.


    Notes
    -----
    Note that this function uses the using the computationally-efficient
    75-term expression for specific volume (Roquet et al., 2015).  There is
    an alternative to calling this function, namely
    gsw_specvol_second_derivatives_CT_exact(SA,CT,p) which uses the full
    Gibbs function (IOC et al., 2010).

    Note that the 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.specvol_second_derivatives(SA, CT, p)
specvol_second_derivatives.types = _gsw_ufuncs.specvol_second_derivatives.types
specvol_second_derivatives = match_args_return(specvol_second_derivatives)

def specvol_second_derivatives_wrt_enthalpy(SA, CT, p):
    """
    Calculates the following three first-order derivatives of specific
    volume (v) with respect to enthalpy,
    (1) v_SA_SA_wrt_h, second-order derivative with respect to Absolute Salinity
    at constant h & p.
    (2) v_SA_h, second-order derivative with respect to SA & h at
    constant p.
    (3) v_h_h, second-order derivative with respect to h at
    constant SA & p.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    v_SA_SA_wrt_h : array-like, (m^3/kg)(g/kg)^-2
        The second-order derivative of specific volume with
        respect to Absolute Salinity at constant h & p.
    v_SA_h : array-like, (m^3/kg)(g/kg)^-1 (J/kg)^-1
        The second-order derivative of specific volume with respect to
        SA and h at constant p.
    v_h_h : array-like, (m^3/kg)(J/kg)^-2
        The second-order derivative with respect to h at
        constant SA & p.


    Notes
    -----
    Note that this function uses the using the computationally-efficient
    75 term expression for specific volume (Roquet et al., 2015).  There is
    an alternative to calling this function, namely
    gsw_specvol_second_derivatives_wrt_enthalpy_CT_exact(SA,CT,p) which uses
    the full Gibbs function (IOC et al., 2010).

    This 75-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel" described
    in McDougall et al. (2010).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.specvol_second_derivatives_wrt_enthalpy(SA, CT, p)
specvol_second_derivatives_wrt_enthalpy.types = _gsw_ufuncs.specvol_second_derivatives_wrt_enthalpy.types
specvol_second_derivatives_wrt_enthalpy = match_args_return(specvol_second_derivatives_wrt_enthalpy)

def specvol_SSO_0(p):
    """
    specific volume at (SSO,CT=0,p)

    Parameters
    ----------
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    double, array

    Notes
    -----
    sw_specvol_SSO_0                          specific volume at (SSO,CT=0,p)
                                                           (75-term equation)
     This function calculates specific volume at the Standard Ocean Salinity,
     SSO, and at a Conservative Temperature of zero degrees C, as a function
     of pressure, p, in dbar, using a streamlined version of the 75-term CT
     version of specific volume, that is, a streamlined version of the code
     "gsw_specvol(SA,CT,p)".

    VERSION NUMBER: 3.06.12 (25th May, 2020)

    References
    ----------
    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.specvol_sso_0(p)
specvol_SSO_0.types = _gsw_ufuncs.specvol_sso_0.types
specvol_SSO_0 = match_args_return(specvol_SSO_0)

def specvol_t_exact(SA, t, p):
    """
    Calculates the specific volume of seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    specvol_t_exact : array-like, m^3/kg
        specific volume


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 2.7 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.specvol_t_exact(SA, t, p)
specvol_t_exact.types = _gsw_ufuncs.specvol_t_exact.types
specvol_t_exact = match_args_return(specvol_t_exact)

def spiciness0(SA, CT):
    """
    Calculates spiciness from Absolute Salinity and Conservative
    Temperature at a pressure of 0 dbar, as described by McDougall and
    Krzysik (2015).  This routine is based on the computationally-efficient
    expression for specific volume in terms of SA, CT and p (Roquet et al.,
    2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    spiciness0 : array-like, kg/m^3
        spiciness referenced to a pressure of 0 dbar,
        i.e. the surface


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    McDougall, T.J., and O.A. Krzysik, 2015: Spiciness. Journal of Marine
    Research, 73, 141-152.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.spiciness0(SA, CT)
spiciness0.types = _gsw_ufuncs.spiciness0.types
spiciness0 = match_args_return(spiciness0)

def spiciness1(SA, CT):
    """
    Calculates spiciness from Absolute Salinity and Conservative
    Temperature at a pressure of 1000 dbar, as described by McDougall and
    Krzysik (2015).  This routine is based on the computationally-efficient
    expression for specific volume in terms of SA, CT and p (Roquet et al.,
    2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    spiciness1 : array-like, kg/m^3
        spiciness referenced to a pressure of 1000 dbar


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    McDougall, T.J., and O.A. Krzysik, 2015: Spiciness. Journal of Marine
    Research, 73, 141-152.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.spiciness1(SA, CT)
spiciness1.types = _gsw_ufuncs.spiciness1.types
spiciness1 = match_args_return(spiciness1)

def spiciness2(SA, CT):
    """
    Calculates spiciness from Absolute Salinity and Conservative
    Temperature at a pressure of 2000 dbar, as described by McDougall and
    Krzysik (2015).  This routine is based on the computationally-efficient
    expression for specific volume in terms of SA, CT and p (Roquet et al.,
    2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    spiciness2 : array-like, kg/m^3
        spiciness referenced to a pressure of 2000 dbar


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    McDougall, T.J., and O.A. Krzysik, 2015: Spiciness. Journal of Marine
    Research, 73, 141-152.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.spiciness2(SA, CT)
spiciness2.types = _gsw_ufuncs.spiciness2.types
spiciness2 = match_args_return(spiciness2)

def SR_from_SP(SP):
    """
    Calculates Reference Salinity from Practical Salinity.

    Parameters
    ----------
    SP : array-like
        Practical Salinity (PSS-78), unitless

    Returns
    -------
    SR : array-like, g/kg
        Reference Salinity


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.sr_from_sp(SP)
SR_from_SP.types = _gsw_ufuncs.sr_from_sp.types
SR_from_SP = match_args_return(SR_from_SP)

def Sstar_from_SA(SA, p, lon, lat):
    """
    Converts Preformed Salinity from Absolute Salinity.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    Sstar : array-like, g/kg
        Preformed Salinity


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, F.J. Millero, R. Pawlowicz and
    P.M. Barker, 2012: A global algorithm for estimating Absolute Salinity.
    Ocean Science, 8, 1123-1134.
    https://os.copernicus.org/articles/8/1123/2012/os-8-1123-2012.pdf


    """
    return _gsw_ufuncs.sstar_from_sa(SA, p, lon, lat)
Sstar_from_SA.types = _gsw_ufuncs.sstar_from_sa.types
Sstar_from_SA = match_args_return(Sstar_from_SA)

def Sstar_from_SP(SP, p, lon, lat):
    """
    Calculates Preformed Salinity from Absolute Salinity.
    Since SP is non-negative by definition, this function changes any
    negative input values of SP to be zero.

    Parameters
    ----------
    SP : array-like
        Practical Salinity (PSS-78), unitless
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    Sstar : array-like, g/kg
        Preformed Salinity


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See section 2.5 and appendices A.4 and A.5 of this TEOS-10 Manual.

    McDougall, T.J., D.R. Jackett, F.J. Millero, R. Pawlowicz and
    P.M. Barker, 2012: A global algorithm for estimating Absolute Salinity.
    Ocean Science, 8, 1123-1134.
    https://os.copernicus.org/articles/8/1123/2012/os-8-1123-2012.pdf


    """
    return _gsw_ufuncs.sstar_from_sp(SP, p, lon, lat)
Sstar_from_SP.types = _gsw_ufuncs.sstar_from_sp.types
Sstar_from_SP = match_args_return(Sstar_from_SP)

def t_deriv_chem_potential_water_t_exact(SA, t, p):
    """
    Calculates the temperature derivative of the chemical potential of water
    in seawater so that it is valid at exactly SA = 0.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    chem_potential_water_dt : array-like, J g^-1 K^-1
        temperature derivative of the chemical
        potential of water in seawater


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.t_deriv_chem_potential_water_t_exact(SA, t, p)
t_deriv_chem_potential_water_t_exact.types = _gsw_ufuncs.t_deriv_chem_potential_water_t_exact.types
t_deriv_chem_potential_water_t_exact = match_args_return(t_deriv_chem_potential_water_t_exact)

def t_freezing(SA, p, saturation_fraction):
    """
    Calculates the in-situ temperature at which seawater freezes. The
    in-situ temperature freezing point is calculated from the exact
    in-situ freezing temperature which is found by a modified Newton-Raphson
    iteration (McDougall and Wotherspoon, 2013) of the equality of the
    chemical potentials of water in seawater and in ice.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    t_freezing : array-like, deg C
        in-situ temperature at which seawater freezes.
        (ITS-90)


    Notes
    -----
    An alternative GSW function, gsw_t_freezing_poly, it is based on a
    computationally-efficient polynomial, and is accurate to within -5e-4 K
    and 6e-4 K, when compared with this function.


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall T.J., and S.J. Wotherspoon, 2013: A simple modification of
    Newton's method to achieve convergence of order 1 + sqrt(2).  Applied
    Mathematics Letters, 29, 20-25.


    """
    return _gsw_ufuncs.t_freezing(SA, p, saturation_fraction)
t_freezing.types = _gsw_ufuncs.t_freezing.types
t_freezing = match_args_return(t_freezing)

def t_freezing_first_derivatives(SA, p, saturation_fraction):
    """
    Calculates the first derivatives of the in-situ temperature at which
    seawater freezes with respect to Absolute Salinity SA and pressure P (in
    Pa).  These expressions come from differentiating the expression that
    defines the freezing temperature, namely the equality between the
    chemical potentials of water in seawater and in ice.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    tfreezing_SA : array-like, K kg/g
        the derivative of the in-situ freezing temperature
        (ITS-90) with respect to Absolute Salinity at fixed
        pressure                     [ K/(g/kg) ] i.e.
    tfreezing_P : array-like, K/Pa
        the derivative of the in-situ freezing temperature
        (ITS-90) with respect to pressure (in Pa) at fixed
        Absolute Salinity


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.


    """
    return _gsw_ufuncs.t_freezing_first_derivatives(SA, p, saturation_fraction)
t_freezing_first_derivatives.types = _gsw_ufuncs.t_freezing_first_derivatives.types
t_freezing_first_derivatives = match_args_return(t_freezing_first_derivatives)

def t_freezing_first_derivatives_poly(SA, p, saturation_fraction):
    """
    Calculates the first derivatives of the in-situ temperature at which
    seawater freezes with respect to Absolute Salinity SA and pressure P (in
    Pa).  These expressions come from differentiating the expression that
    defines the freezing temperature, namely the equality between the
    chemical potentials of water in seawater and in ice.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    tfreezing_SA : array-like, K kg/g
        the derivative of the in-situ freezing temperature
        (ITS-90) with respect to Absolute Salinity at fixed
        pressure                     [ K/(g/kg) ] i.e.
    tfreezing_P : array-like, K/Pa
        the derivative of the in-situ freezing temperature
        (ITS-90) with respect to pressure (in Pa) at fixed
        Absolute Salinity


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.


    """
    return _gsw_ufuncs.t_freezing_first_derivatives_poly(SA, p, saturation_fraction)
t_freezing_first_derivatives_poly.types = _gsw_ufuncs.t_freezing_first_derivatives_poly.types
t_freezing_first_derivatives_poly = match_args_return(t_freezing_first_derivatives_poly)

def t_freezing_poly(SA, p, saturation_fraction):
    """
    Calculates the in-situ temperature at which seawater freezes from a
    comptationally efficient polynomial.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    saturation_fraction : array-like
        Saturation fraction of dissolved air in seawater. (0..1)

    Returns
    -------
    t_freezing : array-like, deg C
        in-situ temperature at which seawater freezes.
        (ITS-90)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.

    McDougall, T.J., P.M. Barker, R. Feistel and B.K. Galton-Fenzi, 2014:
    Melting of Ice and Sea Ice into Seawater and Frazil Ice Formation.
    Journal of Physical Oceanography, 44, 1751-1775.


    """
    return _gsw_ufuncs.t_freezing_poly(SA, p, saturation_fraction)
t_freezing_poly.types = _gsw_ufuncs.t_freezing_poly.types
t_freezing_poly = match_args_return(t_freezing_poly)

def t_from_CT(SA, CT, p):
    """
    Calculates in-situ temperature from the Conservative Temperature of
    seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    t : array-like, deg C
        in-situ temperature (ITS-90)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See sections 3.1 and 3.3 of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.t_from_ct(SA, CT, p)
t_from_CT.types = _gsw_ufuncs.t_from_ct.types
t_from_CT = match_args_return(t_from_CT)

def t_from_pt0_ice(pt0_ice, p):
    """
    Calculates in-situ temperature from the potential temperature of ice Ih
    with reference pressure, p_ref, of 0 dbar (the surface), and the
    in-situ pressure.

    Parameters
    ----------
    pt0_ice : array-like
        Potential temperature of ice (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    t : array-like, deg C
        in-situ temperature (ITS-90)


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See appendix I of this TEOS-10 Manual.


    """
    return _gsw_ufuncs.t_from_pt0_ice(pt0_ice, p)
t_from_pt0_ice.types = _gsw_ufuncs.t_from_pt0_ice.types
t_from_pt0_ice = match_args_return(t_from_pt0_ice)

def thermobaric(SA, CT, p):
    """
    Calculates the thermobaric coefficient of seawater with respect to
    Conservative Temperature.  This routine is based on the
    computationally-efficient expression for specific volume in terms of
    SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    thermobaric : array-like, 1/(K Pa)
        thermobaric coefficient with
        respect to Conservative Temperature.


    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/
    See Eqns. (3.8.2) and (P.2) of this TEOS-10 manual.

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.


    """
    return _gsw_ufuncs.thermobaric(SA, CT, p)
thermobaric.types = _gsw_ufuncs.thermobaric.types
thermobaric = match_args_return(thermobaric)

def z_from_p(p, lat, geo_strf_dyn_height, sea_surface_geopotential):
    """
    Calculates height from sea pressure using the computationally-efficient
    75-term expression for specific volume in terms of SA, CT and p
    (Roquet et al., 2015).  Dynamic height anomaly, geo_strf_dyn_height, if
    provided, must be computed with its p_ref = 0 (the surface).  Also if
    provided, sea_surface_geopotental is the geopotential at zero sea
    pressure. This function solves Eqn.(3.32.3) of IOC et al. (2010).

    Parameters
    ----------
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lat : array-like
        Latitude, -90 to 90 degrees
    geo_strf_dyn_height : array-like
        dynamic height anomaly, m^2/s^2
            Note that the reference pressure, p_ref, of geo_strf_dyn_height must
            be zero (0) dbar.
    sea_surface_geopotential : array-like
        geopotential at zero sea pressure,  m^2/s^2

    Returns
    -------
    z : array-like, m
        height


    Notes
    -----
    Note. Height z is NEGATIVE in the ocean. i.e. Depth is -z.
    Depth is not used in the GSW computer software library.

    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "gsw_infunnel(SA,CT,p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
    seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from https://www.teos-10.org/

    McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
    pp. 730-741.

    Moritz, H., 2000: Geodetic reference system 1980. J. Geodesy, 74,
    pp. 128-133.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling, 90, pp. 29-43.

    This software is available from https://www.teos-10.org/


    """
    return _gsw_ufuncs.z_from_p(p, lat, geo_strf_dyn_height, sea_surface_geopotential)
z_from_p.types = _gsw_ufuncs.z_from_p.types
z_from_p = match_args_return(z_from_p)
