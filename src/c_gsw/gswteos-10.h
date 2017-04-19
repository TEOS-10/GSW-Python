/*
**  $Id: gswteos-10.h,v 045882f2da27 2015/09/13 23:47:38 fdelahoyde $
**  $Version: 3.05.0-2 $
**
**  GSW TEOS-10 V3.05 definitions and prototypes.
*/
#ifndef GSWTEOS_10_H
#define GSWTEOS_10_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <errno.h>

#define	GSW_INVALID_VALUE	9e15	/* error return from gsw_saar et al. */
#define GSW_ERROR_LIMIT		1e10

/*
**  Prototypes:
*/
extern void   gsw_add_barrier(double *input_data, double lon, double lat,
		double long_grid, double lat_grid, double dlong_grid,
		double dlat_grid, double *output_data);
extern void   gsw_add_mean(double *data_in, double *data_out);
extern double gsw_adiabatic_lapse_rate_from_ct(double sa, double ct, double p);
extern double gsw_adiabatic_lapse_rate_ice(double t, double p);
extern double gsw_alpha(double sa, double ct, double p);
extern double gsw_alpha_on_beta(double sa, double ct, double p);
extern double gsw_alpha_wrt_t_exact(double sa, double t, double p);
extern double gsw_alpha_wrt_t_ice(double t, double p);
extern double gsw_beta_const_t_exact(double sa, double t, double p);
extern double gsw_beta(double sa, double ct, double p);
extern double gsw_cabbeling(double sa, double ct, double p);
extern double gsw_c_from_sp(double sp, double t, double p);
extern double gsw_chem_potential_water_ice(double t, double p);
extern double gsw_chem_potential_water_t_exact(double sa, double t, double p);
extern double gsw_cp_ice(double t, double p);
extern double gsw_cp_t_exact(double sa, double t, double p);
extern void   gsw_ct_first_derivatives (double sa, double pt, double *ct_sa,
		double *ct_pt);
extern void   gsw_ct_first_derivatives_wrt_t_exact(double sa, double t,
		double p, double *ct_sa_wrt_t, double *ct_t_wrt_t,
		double *ct_p_wrt_t);
extern double gsw_ct_freezing(double sa, double p, double saturation_fraction);
extern void   gsw_ct_freezing_first_derivatives(double sa, double p,
		double saturation_fraction, double *ctfreezing_sa,
		double *ctfreezing_p);
extern void   gsw_ct_freezing_first_derivatives_poly(double sa, double p,
		double saturation_fraction, double *ctfreezing_sa,
		double *ctfreezing_p);
extern double gsw_ct_freezing_poly(double sa, double p,
		double saturation_fraction);
extern double gsw_ct_from_enthalpy(double sa, double h, double p);
extern double gsw_ct_from_enthalpy_exact(double sa, double h, double p);
extern double gsw_ct_from_entropy(double sa, double entropy);
extern double gsw_ct_from_pt(double sa, double pt);
extern void   gsw_ct_from_rho(double rho, double sa, double p, double *ct,
		double *ct_multiple);
extern double gsw_ct_from_t(double sa, double t, double p);
extern double gsw_ct_maxdensity(double sa, double p);
extern void   gsw_ct_second_derivatives(double sa, double pt, double *ct_sa_sa,
		double *ct_sa_pt, double *ct_pt_pt);
extern double gsw_deltasa_atlas(double p, double lon, double lat);
extern double gsw_deltasa_from_sp(double sp, double p, double lon, double lat);
extern double gsw_dilution_coefficient_t_exact(double sa, double t, double p);
extern double gsw_dynamic_enthalpy(double sa, double ct, double p);
extern double gsw_enthalpy_ct_exact(double sa, double ct, double p);
extern double gsw_enthalpy_diff(double sa, double ct, double p_shallow,
		double p_deep);
extern double gsw_enthalpy(double sa, double ct, double p);
extern void   gsw_enthalpy_first_derivatives_ct_exact(double sa, double ct,
		double p, double *h_sa, double *h_ct);
extern void   gsw_enthalpy_first_derivatives(double sa, double ct, double p,
		double *h_sa, double *h_ct);
extern double gsw_enthalpy_ice(double t, double p);
extern void   gsw_enthalpy_second_derivatives_ct_exact(double sa, double ct,
		double p, double *h_sa_sa, double *h_sa_ct, double *h_ct_ct);
extern void   gsw_enthalpy_second_derivatives(double sa, double ct, double p,
		double *h_sa_sa, double *h_sa_ct, double *h_ct_ct);
extern double gsw_enthalpy_sso_0(double p);
extern double gsw_enthalpy_t_exact(double sa, double t, double p);
extern void   gsw_entropy_first_derivatives(double sa, double ct,
		double *eta_sa, double *eta_ct);
extern double gsw_entropy_from_ct(double sa, double ct);
extern double gsw_entropy_from_pt(double sa, double pt);
extern double gsw_entropy_from_t(double sa, double t, double p);
extern double gsw_entropy_ice(double t, double p);
extern double gsw_entropy_part(double sa, double t, double p);
extern double gsw_entropy_part_zerop(double sa, double pt0);
extern void   gsw_entropy_second_derivatives(double sa, double ct,
		double *eta_sa_sa, double *eta_sa_ct, double *eta_ct_ct);
extern double gsw_fdelta(double p, double lon, double lat);
extern void   gsw_frazil_properties(double sa_bulk, double h_bulk, double p,
		double *sa_final, double *ct_final, double *w_ih_final);
extern void   gsw_frazil_properties_potential(double sa_bulk, double h_pot_bulk,
		double p, double *sa_final, double *ct_final,
		double *w_ih_final);
extern void   gsw_frazil_properties_potential_poly(double sa_bulk,
		double h_pot_bulk, double p, double *sa_final, double *ct_final,
		double *w_ih_final);
extern void   gsw_frazil_ratios_adiabatic(double sa, double p, double w_ih,
		double *dsa_dct_frazil, double *dsa_dp_frazil,
		double *dct_dp_frazil);
extern void   gsw_frazil_ratios_adiabatic_poly(double sa, double p,
		double w_ih, double *dsa_dct_frazil, double *dsa_dp_frazil,
		double *dct_dp_frazil);
extern double *gsw_geo_strf_dyn_height(double *sa, double *ct, double *p,
		double p_ref, int n_levels, double *dyn_height);
extern double *gsw_geo_strf_dyn_height_pc(double *sa, double *ct,
		double *delta_p, int n_levels, double *geo_strf_dyn_height_pc,
		double *p_mid);
extern double gsw_gibbs_ice (int nt, int np, double t, double p);
extern double gsw_gibbs_ice_part_t(double t, double p);
extern double gsw_gibbs_ice_pt0(double pt0);
extern double gsw_gibbs_ice_pt0_pt0(double pt0);
extern double gsw_gibbs(int ns, int nt, int np, double sa, double t, double p);
extern double gsw_gibbs_pt0_pt0(double sa, double pt0);
extern double gsw_grav(double lat, double p);
extern double gsw_helmholtz_energy_ice(double t, double p);
extern double gsw_hill_ratio_at_sp2(double t);
extern void   gsw_ice_fraction_to_freeze_seawater(double sa, double ct,
		double p, double t_ih, double *sa_freeze, double *ct_freeze,
		double *w_ih);
extern double gsw_internal_energy(double sa, double ct, double p);
extern double gsw_internal_energy_ice(double t, double p);
extern void   gsw_ipv_vs_fnsquared_ratio(double *sa, double *ct, double *p,
		double p_ref, int nz, double *ipv_vs_fnsquared_ratio,
		double *p_mid);
extern double gsw_kappa_const_t_ice(double t, double p);
extern double gsw_kappa(double sa, double ct, double p);
extern double gsw_kappa_ice(double t, double p);
extern double gsw_kappa_t_exact(double sa, double t, double p);
extern double gsw_latentheat_evap_ct(double sa, double ct);
extern double gsw_latentheat_evap_t(double sa, double t);
extern double gsw_latentheat_melting(double sa, double p);
extern void   gsw_linear_interp_sa_ct(double *sa, double *ct, double *p, int np,
		double *p_i, int npi, double *sa_i, double *ct_i);
extern double gsw_melting_ice_equilibrium_sa_ct_ratio(double sa, double p);
extern double gsw_melting_ice_equilibrium_sa_ct_ratio_poly(double sa, double p);
extern void   gsw_melting_ice_into_seawater(double sa, double ct, double p,
		double w_ih, double t_ih, double *sa_final, double *ct_final,
		double *w_ih_final);
extern double gsw_melting_ice_sa_ct_ratio(double sa, double ct, double p,
		double t_ih);
extern double gsw_melting_ice_sa_ct_ratio_poly(double sa, double ct, double p,
		double t_ih);
extern double gsw_melting_seaice_equilibrium_sa_ct_ratio(double sa, double p);
extern double gsw_melting_seaice_equilibrium_sa_ct_ratio_poly(double sa,
		double p);
extern void   gsw_melting_seaice_into_seawater(double sa, double ct, double p,
		double w_seaice, double sa_seaice, double t_seaice,
		double *sa_final, double *ct_final);
extern double gsw_melting_seaice_sa_ct_ratio(double sa, double ct, double p,
		double sa_seaice, double t_seaice);
extern double gsw_melting_seaice_sa_ct_ratio_poly(double sa, double ct,
		double p, double sa_seaice, double t_seaice);
extern void   gsw_nsquared(double *sa, double *ct, double *p, double *lat,
		int nz, double *n2, double *p_mid);
extern double gsw_pot_enthalpy_from_pt_ice(double pt0_ice);
extern double gsw_pot_enthalpy_from_pt_ice_poly(double pt0_ice);
extern double gsw_pot_enthalpy_ice_freezing(double sa, double p);
extern void   gsw_pot_enthalpy_ice_freezing_first_derivatives(double sa,
		double p, double *pot_enthalpy_ice_freezing_sa,
		double *pot_enthalpy_ice_freezing_p);
extern void   gsw_pot_enthalpy_ice_freezing_first_derivatives_poly(double sa,
		double p, double *pot_enthalpy_ice_freezing_sa,
		double *pot_enthalpy_ice_freezing_p);
extern double gsw_pot_enthalpy_ice_freezing_poly(double sa, double p);
extern double gsw_pot_rho_t_exact(double sa, double t, double p, double p_ref);
extern double gsw_pressure_coefficient_ice(double t, double p);
extern double gsw_pressure_freezing_ct(double sa, double ct,
		double saturation_fraction);
extern double gsw_pt0_cold_ice_poly(double pot_enthalpy_ice);
extern double gsw_pt0_from_t(double sa, double t, double p);
extern double gsw_pt0_from_t_ice(double t, double p);
extern void   gsw_pt_first_derivatives (double sa, double ct, double *pt_sa,
		double *pt_ct);
extern double gsw_pt_from_ct(double sa, double ct);
extern double gsw_pt_from_entropy(double sa, double entropy);
extern double gsw_pt_from_pot_enthalpy_ice(double pot_enthalpy_ice);
extern double gsw_pt_from_pot_enthalpy_ice_poly_dh(double pot_enthalpy_ice);
extern double gsw_pt_from_pot_enthalpy_ice_poly(double pot_enthalpy_ice);
extern double gsw_pt_from_t(double sa, double t, double p, double p_ref);
extern double gsw_pt_from_t_ice(double t, double p, double p_ref);
extern void   gsw_pt_second_derivatives (double sa, double ct, double *pt_sa_sa,
		double *pt_sa_ct, double *pt_ct_ct);
extern void   gsw_rho_alpha_beta (double sa, double ct, double p, double *rho,
		double *alpha, double *beta);
extern double gsw_rho(double sa, double ct, double p);
extern void   gsw_rho_first_derivatives(double sa, double ct, double p,
		double *drho_dsa, double *drho_dct, double *drho_dp);
extern void   gsw_rho_first_derivatives_wrt_enthalpy (double sa, double ct,
		double p, double *rho_sa, double *rho_h);
extern double gsw_rho_ice(double t, double p);
extern void   gsw_rho_second_derivatives(double sa, double ct, double p,
		double *rho_sa_sa, double *rho_sa_ct, double *rho_ct_ct,
		double *rho_sa_p, double *rho_ct_p);
extern void   gsw_rho_second_derivatives_wrt_enthalpy(double sa, double ct,
		double p, double *rho_sa_sa, double *rho_sa_h, double *rho_h_h);
extern double gsw_rho_t_exact(double sa, double t, double p);
extern void   gsw_rr68_interp_sa_ct(double *sa, double *ct, double *p, int mp,
		double *p_i, int mp_i, double *sa_i, double *ct_i);
extern double gsw_saar(double p, double lon, double lat);
extern double gsw_sa_freezing_estimate(double p, double saturation_fraction,
		double *ct, double *t);
extern double gsw_sa_freezing_from_ct(double ct, double p,
		double saturation_fraction);
extern double gsw_sa_freezing_from_ct_poly(double ct, double p,
		double saturation_fraction);
extern double gsw_sa_freezing_from_t(double t, double p,
		double saturation_fraction);
extern double gsw_sa_freezing_from_t_poly(double t, double p,
		double saturation_fraction);
extern double gsw_sa_from_rho(double rho, double ct, double p);
extern double gsw_sa_from_sp_baltic(double sp, double lon, double lat);
extern double gsw_sa_from_sp(double sp, double p, double lon, double lat);
extern double gsw_sa_from_sstar(double sstar, double p,double lon,double lat);
extern int    gsw_sa_p_inrange(double sa, double p);
extern void   gsw_seaice_fraction_to_freeze_seawater(double sa, double ct,
		double p, double sa_seaice, double t_seaice, double *sa_freeze,
		double *ct_freeze, double *w_seaice);
extern double gsw_sigma0(double sa, double ct);
extern double gsw_sigma1(double sa, double ct);
extern double gsw_sigma2(double sa, double ct);
extern double gsw_sigma3(double sa, double ct);
extern double gsw_sigma4(double sa, double ct);
extern double gsw_sound_speed(double sa, double ct, double p);
extern double gsw_sound_speed_ice(double t, double p);
extern double gsw_sound_speed_t_exact(double sa, double t, double p);
extern void   gsw_specvol_alpha_beta(double sa, double ct, double p,
		double *specvol, double *alpha, double *beta);
extern double gsw_specvol_anom_standard(double sa, double ct, double p);
extern double gsw_specvol(double sa, double ct, double p);
extern void   gsw_specvol_first_derivatives(double sa, double ct, double p,
		double *v_sa, double *v_ct, double *v_p);
extern void   gsw_specvol_first_derivatives_wrt_enthalpy(double sa, double ct,
		double p, double *v_sa, double *v_h);
extern double gsw_specvol_ice(double t, double p);
extern void   gsw_specvol_second_derivatives (double sa, double ct, double p,
		double *v_sa_sa, double *v_sa_ct, double *v_ct_ct,
		double *v_sa_p, double *v_ct_p);
extern void   gsw_specvol_second_derivatives_wrt_enthalpy(double sa, double ct,
		double p, double *v_sa_sa, double *v_sa_h, double *v_h_h);
extern double gsw_specvol_sso_0(double p);
extern double gsw_specvol_t_exact(double sa, double t, double p);
extern double gsw_sp_from_c(double c, double t, double p);
extern double gsw_sp_from_sa_baltic(double sa, double lon, double lat);
extern double gsw_sp_from_sa(double sa, double p, double lon, double lat);
extern double gsw_sp_from_sk(double sk);
extern double gsw_sp_from_sr(double sr);
extern double gsw_sp_from_sstar(double sstar, double p,double lon,double lat);
extern double gsw_spiciness0(double sa, double ct);
extern double gsw_spiciness1(double sa, double ct);
extern double gsw_spiciness2(double sa, double ct);
extern double gsw_sr_from_sp(double sp);
extern double gsw_sstar_from_sa(double sa, double p, double lon, double lat);
extern double gsw_sstar_from_sp(double sp, double p, double lon, double lat);
extern double gsw_t_deriv_chem_potential_water_t_exact(double sa, double t,
		double p);
extern double gsw_t_freezing(double sa, double p, double saturation_fraction);
extern void   gsw_t_freezing_first_derivatives_poly(double sa, double p,
		double saturation_fraction, double *tfreezing_sa,
		double *tfreezing_p);
extern void   gsw_t_freezing_first_derivatives(double sa, double p,
		double saturation_fraction, double *tfreezing_sa,
		double *tfreezing_p);
extern double gsw_t_freezing_poly(double sa, double p,
		double saturation_fraction);
extern double gsw_t_from_ct(double sa, double ct, double p);
extern double gsw_t_from_pt0_ice(double pt0_ice, double p);
extern double gsw_thermobaric(double sa, double ct, double p);
extern void   gsw_turner_rsubrho(double *sa, double *ct, double *p, int nz,
		double *tu, double *rsubrho, double *p_mid);
extern int    gsw_util_indx(double *x, int n, double z);
extern double *gsw_util_interp1q_int(int nx, double *x, int *iy, int nxi,
		double *x_i, double *y_i);
extern double *gsw_util_linear_interp(int nx, double *x, int ny, double *y,
		int nxi, double *x_i, double *y_i);
extern void   gsw_util_sort_real(double *rarray, int nx, int *iarray);
extern double gsw_util_xinterp1(double *x, double *y, int n, double x0);
extern double gsw_z_from_p(double p, double lat);
extern double gsw_p_from_z(double z, double lat);

#ifdef __cplusplus
}
#endif

#endif /* GSWTEOS_10_H */
