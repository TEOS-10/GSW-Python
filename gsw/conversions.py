"""
Conversions involving temperature, salinity, entropy, pressure,
and height.
"""

from ._utilities import match_args_return

from ._wrapped_ufuncs import (
adiabatic_lapse_rate_from_CT,
C_from_SP,
CT_from_enthalpy,
CT_from_entropy,
CT_from_pt,
CT_from_rho,
CT_from_t,
deltaSA_from_SP,
entropy_from_pt,
entropy_from_t,
pt0_from_t,
pt_from_CT,
pt_from_entropy,
pt_from_t,
SA_from_rho,
SA_from_SP,
SA_from_Sstar,
SP_from_C,
SP_from_SA,
SP_from_SK,
SP_from_SR,
SP_from_Sstar,
SR_from_SP,
Sstar_from_SA,
Sstar_from_SP,
t_from_CT,
z_from_p,
)


@match_args_return
def t90_from_t68(t68):
    """
    ITS-90 temperature from IPTS-68 temperature

    This conversion should be applied to all in-situ
    data collected between 1/1/1968 and 31/12/1989.

    """
    return t68 / 1.00024
