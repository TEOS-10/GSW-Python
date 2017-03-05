/*
**  $Id: gsw_check_functions.c,v 0db1b20bdf1b 2015/08/26 21:39:20 fdelahoyde $
**  $Version: 3.05.0-1 $
*/
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <gswteos-10.h>
#include <gsw_check_data.c>

#define test_func(name, arglist, value, var) \
	for (i=0; i<count; i++) { \
	    value[i] = gsw_ ## name arglist; \
	} \
	check_accuracy(#name, var ## _ca, #var, count, value, var)

#define test_sub1(name,arglist,val1,var1) \
	for (i=0; i<count; i++) { \
	    gsw_ ## name arglist; \
	} \
	check_accuracy(#name, var1 ## _ca, #var1, count, val1, var1)

#define VALS1 &val1[i]

#define test_sub2(name,arglist,val1,var1,val2,var2) \
	for (i=0; i<count; i++) { \
	    gsw_ ## name arglist; \
	} \
	check_accuracy(#name, var1 ## _ca, #var1, count, val1, var1); \
	check_accuracy(#name, var2 ## _ca, #var2, count, val2, var2)

#define VALS2 &val1[i],&val2[i]

#define test_sub3(name,arglist,val1,var1,val2,var2,val3,var3) \
	for (i=0; i<count; i++) { \
	    gsw_ ## name arglist; \
	} \
	check_accuracy(#name, var1 ## _ca, #var1, count, val1, var1); \
	check_accuracy(#name, var2 ## _ca, #var2, count, val2, var2); \
	check_accuracy(#name, var3 ## _ca, #var3, count, val3, var3)

#define VALS3 &val1[i],&val2[i],&val3[i]

#define test_sub4(name,arglist,val1,var1,val2,var2,val3,var3,val4,var4) \
	for (i=0; i<count; i++) { \
	    gsw_ ## name arglist; \
	} \
	check_accuracy(#name, var1 ## _ca, #var1, count, val1, var1); \
	check_accuracy(#name, var2 ## _ca, #var2, count, val2, var2); \
	check_accuracy(#name, var3 ## _ca, #var3, count, val3, var3); \
	check_accuracy(#name, var4 ## _ca, #var4, count, val4, var4)

#define VALS4 &val1[i],&val2[i],&val3[i],&val4[i]

#define test_sub5(name,arglist,val1,var1,val2,var2,val3,var3,val4,var4,\
		val5,var5) \
	for (i=0; i<count; i++) { \
	    gsw_ ## name arglist; \
	} \
	check_accuracy(#name, var1 ## _ca, #var1, count, val1, var1); \
	check_accuracy(#name, var2 ## _ca, #var2, count, val2, var2); \
	check_accuracy(#name, var3 ## _ca, #var3, count, val3, var3); \
	check_accuracy(#name, var4 ## _ca, #var4, count, val4, var4); \
	check_accuracy(#name, var5 ## _ca, #var5, count, val5, var5)

#define VALS5 &val1[i],&val2[i],&val3[i],&val4[i],&val5[i]

typedef struct gsw_error_info {
	int	ncomp, flags;
#define GSW_ERROR_LIMIT_FLAG	1
#define GSW_ERROR_ERROR_FLAG	2
	double	max,
		rel,
		limit,
		rlimit;
}	gsw_error_info;
	
void report(char *funcname, char *varname, gsw_error_info *errs);
void check_accuracy(char *funcname, double accuracy, char *varname,
			int count, double *calcval, double *refval);
void section_title(char *title);

int		debug=0, check_count, gsw_error_flag=0;
double		c[cast_m*cast_n];
double		sr[cast_m*cast_n];
double		sstar[cast_m*cast_n];
double		pt[cast_m*cast_n];
double		pt0[cast_m*cast_n];
double		entropy[cast_m*cast_n];
double		ctf[cast_m*cast_n];
double		tf[cast_m*cast_n];
double		ctf_poly[cast_m*cast_n];
double		tf_poly[cast_m*cast_n];
double		h[cast_m*cast_n];

int
main(int argc, char **argv)
{
	int	count = cast_m*cast_n, i, j, k, l, n;
	double	saturation_fraction, value[count], lat[count],
		lon[count], val1[count], val2[count], val3[count],
		val4[count], val5[count];

	if (argc==2 && !strcmp(argv[1],"-debug"))
	    debug	= 1;

	for (i=0; i<cast_n; i++) {
	    for (j=i*cast_m, k=j+cast_m; j<k; j++) {
		lat[j]	= lat_cast[i];
		lon[j]	= long_cast[i];
	    }
	}

	printf(
"============================================================================\n"
" Gibbs SeaWater (GSW) Oceanographic Toolbox of TEOS-10 version 3.05 (C)\n"
"============================================================================\n"
"\n"
" These are the check values for the subset of functions that have been \n"
" converted into C from the Gibbs SeaWater (GSW) Oceanographic Toolbox \n"
" of TEOS-10 (version 3.05).\n");

	check_count = 1;

	section_title("Practical Salinity, PSS-78");

	test_func(c_from_sp, (sp[i],t[i],p[i]), c,c_from_sp);
	test_func(sp_from_c, (c[i],t[i],p[i]), value,sp_from_c);
	test_func(sp_from_sk, (sk[i]), value,sp_from_sk);

	section_title(
	  "Absolute Salinity, Preformed Salinity and Conservative Temperature");

	test_func(sa_from_sp, (sp[i],p[i],lon[i],lat[i]), value,sa_from_sp);
	test_func(sstar_from_sp,(sp[i],p[i],lon[i],lat[i]),value,sstar_from_sp);
	test_func(ct_from_t, (sa[i],t[i],p[i]), value,ct_from_t);

	section_title(
	  "Other conversions between Temperatures, Salinities, Entropy, "
	  "Pressure and Height");

	test_func(deltasa_from_sp, (sp[i],p[i],lon[i],lat[i]), value,
	    deltasa_from_sp);
	test_func(sr_from_sp, (sp[i]), sr,sr_from_sp);
	test_func(sp_from_sr, (sr[i]), value,sp_from_sr);
	test_func(sp_from_sa, (sa[i],p[i],lon[i],lat[i]), value,sp_from_sa);
	test_func(sstar_from_sa,(sa[i],p[i],lon[i],lat[i]),sstar,sstar_from_sa);
	test_func(sa_from_sstar, (sstar[i],p[i],lon[i],lat[i]), value,
	    sa_from_sstar);
	test_func(sp_from_sstar, (sstar[i],p[i],lon[i],lat[i]), value,
	    sp_from_sstar);
	test_func(pt_from_ct, (sa[i],ct[i]), pt,pt_from_ct);
	test_func(t_from_ct, (sa[i],ct[i],p[i]), value,t_from_ct);
	test_func(ct_from_pt, (sa[i],pt[i]), value,ct_from_pt);
	test_func(pt0_from_t, (sa[i],t[i],p[i]), value,pt0_from_t);
	test_func(pt_from_t, (sa[i],t[i],p[i],pref[0]), value,pt_from_t);
	test_func(z_from_p, (p[i],lat[i]), value,z_from_p);
	test_func(entropy_from_pt, (sa[i],pt[i]), entropy,entropy_from_pt);
	test_func(pt_from_entropy, (sa[i],entropy[i]), value,pt_from_entropy);
	test_func(ct_from_entropy, (sa[i],entropy[i]), value,ct_from_entropy);
	test_func(entropy_from_t, (sa[i],t[i],p[i]), value,entropy_from_t);
	test_func(adiabatic_lapse_rate_from_ct, (sa[i],ct[i],p[i]), value,
	    adiabatic_lapse_rate_from_ct);

	section_title("Specific Volume, Density and Enthalpy");

	test_func(specvol, (sa[i],ct[i],p[i]), value,specvol);
	test_func(alpha, (sa[i],ct[i],p[i]), value,alpha);
	test_func(beta, (sa[i],ct[i],p[i]), value,beta);
	test_func(alpha_on_beta, (sa[i],ct[i],p[i]), value,alpha_on_beta);
	test_sub3(specvol_alpha_beta, (sa[i],ct[i],p[i],VALS3),
	    val1,v_vab,val2,alpha_vab,val3,beta_vab);
	test_sub3(specvol_first_derivatives,(sa[i],ct[i],p[i],VALS3),
	    val1,v_sa, val2,v_ct, val3,v_p);
	test_sub5(specvol_second_derivatives, (sa[i],ct[i],p[i],VALS5),
	    val1,v_sa_sa,val2,v_sa_ct,val3,v_ct_ct,val4,v_sa_p,val5,v_ct_p);
	test_sub2(specvol_first_derivatives_wrt_enthalpy,
	    (sa[i],ct[i],p[i],VALS2),val1,v_sa_wrt_h, val2,v_h);
	test_sub3(specvol_second_derivatives_wrt_enthalpy,
	    (sa[i],ct[i],p[i],VALS3),
	    val1,v_sa_sa_wrt_h, val2,v_sa_h, val3,v_h_h);
	test_func(specvol_anom_standard, (sa[i],ct[i],p[i]), value,
	    specvol_anom_standard);
	test_func(rho, (sa[i],ct[i],p[i]), rho,rho);
	test_sub3(rho_alpha_beta, (sa[i],ct[i],p[i],VALS3),
	    val1,rho_rab,val2,alpha_rab,val3,beta_rab);
	test_sub3(rho_first_derivatives,(sa[i],ct[i],p[i],VALS3),
	    val1,rho_sa,val2,rho_ct,val3,rho_p);
	test_sub5(rho_second_derivatives,
	    (sa[i],ct[i],p[i],VALS5),val1,rho_sa_sa,val2,rho_sa_ct,
	    val3,rho_ct_ct,val4,rho_sa_p,val5, rho_ct_p);
	test_sub2(rho_first_derivatives_wrt_enthalpy,
	    (sa[i],ct[i],p[i],VALS2),val1,rho_sa_wrt_h,val2,rho_h);
	test_sub3(rho_second_derivatives_wrt_enthalpy,
		(sa[i],ct[i],p[i],VALS3),val1,rho_sa_sa_wrt_h,
		val2,rho_sa_h, val3,rho_h_h);
	test_func(sigma0, (sa[i],ct[i]), value,sigma0);
	test_func(sigma1, (sa[i],ct[i]), value,sigma1);
	test_func(sigma2, (sa[i],ct[i]), value,sigma2);
	test_func(sigma3, (sa[i],ct[i]), value,sigma3);
	test_func(sigma4, (sa[i],ct[i]), value,sigma4);
	test_func(sound_speed, (sa[i],ct[i],p[i]), value,sound_speed);
	test_func(kappa, (sa[i],ct[i],p[i]), value,kappa);
	test_func(cabbeling, (sa[i],ct[i],p[i]), value,cabbeling);
	test_func(thermobaric, (sa[i],ct[i],p[i]), value,thermobaric);
	test_func(sa_from_rho, (rho[i],ct[i],p[i]), value,sa_from_rho);
	test_sub1(ct_from_rho, (rho[i],sa[i],p[i],VALS1,NULL),val1,ct_from_rho);
	test_func(ct_maxdensity, (sa[i],p[i]), value,ct_maxdensity);
	test_func(internal_energy, (sa[i],ct[i],p[i]), value,internal_energy);
	test_func(enthalpy, (sa[i],ct[i],p[i]), h,enthalpy);
	test_func(enthalpy_diff,
	    (sa[i],ct[i],p_shallow[i],p_deep[i]), value,enthalpy_diff);
	test_func(ct_from_enthalpy, (sa[i],h[i],p[i]), value,ct_from_enthalpy);
	test_func(dynamic_enthalpy, (sa[i],ct[i],p[i]), value,dynamic_enthalpy);
	test_sub2(enthalpy_first_derivatives, (sa[i],ct[i],p[i],VALS2),
	    val1,h_sa, val2,h_ct);
	test_sub3(enthalpy_second_derivatives,(sa[i],ct[i],p[i],VALS3),
	    val1,h_sa_sa,val2,h_sa_ct, val3,h_ct_ct);

	section_title("Derivatives of entropy, CT and pt");

	test_sub2(ct_first_derivatives, (sa[i],pt[i],VALS2),
	    val1,ct_sa, val2,ct_pt);
	test_sub3(ct_second_derivatives, (sa[i],pt[i],VALS3),
	    val1,ct_sa_sa, val2,ct_sa_pt, val3,ct_pt_pt);
	test_sub2(entropy_first_derivatives, (sa[i],ct[i],VALS2),
	    val1,eta_sa, val2,eta_ct);
	test_sub3(entropy_second_derivatives, (sa[i],ct[i],VALS3),
	    val1,eta_sa_sa, val2,eta_sa_ct, val3,eta_ct_ct);
	test_sub2(pt_first_derivatives, (sa[i],ct[i],VALS2),
	    val1,pt_sa, val2,pt_ct);
	test_sub3(pt_second_derivatives, (sa[i],ct[i],VALS3),
	    val1,pt_sa_sa,val2,pt_sa_ct,val3,pt_ct_ct);

	section_title("Freezing temperatures");

	saturation_fraction = 0.5;

	test_func(ct_freezing_exact,(sa[i],p[i],saturation_fraction),
	    ctf,ct_freezing);
	test_func(ct_freezing_poly, (sa[i],p[i],saturation_fraction), ctf_poly,
	    ct_freezing_poly);
	test_func(t_freezing_exact, (sa[i],p[i],saturation_fraction), tf,
	    t_freezing);
	test_func(t_freezing_poly, (sa[i],p[i],saturation_fraction, 0), tf_poly,
	    t_freezing_poly);
	test_func(pot_enthalpy_ice_freezing, (sa[i],p[i]), value,
	    pot_enthalpy_ice_freezing);
	test_func(pot_enthalpy_ice_freezing_poly, (sa[i],p[i]), value,
	    pot_enthalpy_ice_freezing_poly);
	test_func(sa_freezing_from_ct, (ctf[i],p[i],saturation_fraction),value,
	    sa_freezing_from_ct);
	test_func(sa_freezing_from_ct_poly,
	    (ctf_poly[i],p[i],saturation_fraction),value,
	    sa_freezing_from_ct_poly);
	test_func(sa_freezing_from_t, (tf[i],p[i],saturation_fraction),value,
	    sa_freezing_from_t);
	test_func(sa_freezing_from_t_poly,
	    (tf_poly[i],p[i],saturation_fraction),value,
	    sa_freezing_from_t_poly);
	test_sub2(ct_freezing_first_derivatives,
	    (sa[i],p[i],saturation_fraction,VALS2),
		val1,ctfreezing_sa,val2,ctfreezing_p);
	test_sub2(ct_freezing_first_derivatives_poly,
	    (sa[i],p[i],saturation_fraction,VALS2),val1,ctfreezing_sa_poly,
		val2,ctfreezing_p_poly);
	test_sub2(t_freezing_first_derivatives,
	    (sa[i],p[i],saturation_fraction,VALS2),
	    val1,tfreezing_sa,val2,tfreezing_p);
	test_sub2(t_freezing_first_derivatives_poly,
	    (sa[i],p[i],saturation_fraction,VALS2),
	    val1,tfreezing_sa_poly,val2,tfreezing_p_poly);
	test_sub2(pot_enthalpy_ice_freezing_first_derivatives,
		(sa[i],p[i],VALS2),
		val1, pot_enthalpy_ice_freezing_sa,
		val2, pot_enthalpy_ice_freezing_p);
	test_sub2(pot_enthalpy_ice_freezing_first_derivatives_poly,
		(sa[i],p[i],VALS2),
		val1, pot_enthalpy_ice_freezing_sa_poly,
		val2, pot_enthalpy_ice_freezing_p_poly);

	section_title(
	    "Isobaric Melting Enthalpy and Isobaric Evaporation Enthalpy");

	test_func(latentheat_melting, (sa[i],p[i]), value,latentheat_melting);
	test_func(latentheat_evap_ct, (sa[i],ct[i]), value,latentheat_evap_ct);
	test_func(latentheat_evap_t, (sa[i],t[i]), value,latentheat_evap_t);

	section_title("Planet Earth properties");

	test_func(grav, (lat[i],p[i]), value,grav);

	section_title(
	    "Density and enthalpy in terms of CT, derived from the "
	    "exact Gibbs function");

	test_func(enthalpy_ct_exact,(sa[i],ct[i],p[i]),value,enthalpy_ct_exact);
	test_sub2(enthalpy_first_derivatives_ct_exact, (sa[i],ct[i],p[i],VALS2),
	    val1,h_sa_ct_exact,val2,h_ct_ct_exact);
	test_sub3(enthalpy_second_derivatives_ct_exact, (sa[i],ct[i],p[i],VALS3),
	    val1,h_sa_sa_ct_exact, val2,h_sa_ct_ct_exact,val3,h_ct_ct_ct_exact);

	section_title(
	    "Basic thermodynamic properties in terms of in-situ t,\n"
	    "based on the exact Gibbs function");

	test_func(rho_t_exact, (sa[i],t[i],p[i]),value,rho_t_exact);
	test_func(pot_rho_t_exact, (sa[i],t[i],p[i],pref[0]),value,
	    pot_rho_t_exact);
	test_func(alpha_wrt_t_exact, (sa[i],t[i],p[i]),value,alpha_wrt_t_exact);
	test_func(beta_const_t_exact, (sa[i],t[i],p[i]),value,
	    beta_const_t_exact);
	test_func(specvol_t_exact, (sa[i],t[i],p[i]),value,specvol_t_exact);
	test_func(sound_speed_t_exact, (sa[i],t[i],p[i]),value,
	    sound_speed_t_exact);
	test_func(kappa_t_exact, (sa[i],t[i],p[i]),value,kappa_t_exact);
	test_func(enthalpy_t_exact, (sa[i],t[i],p[i]),value,enthalpy_t_exact);
	test_sub3(ct_first_derivatives_wrt_t_exact, (sa[i],t[i],p[i],VALS3),
	    val1,ct_sa_wrt_t, val2,ct_t_wrt_t,val3,ct_p_wrt_t);
	test_func(chem_potential_water_t_exact, (sa[i],t[i],p[i]),value,
	    chem_potential_water_t_exact);
	test_func(t_deriv_chem_potential_water_t_exact, (sa[i],t[i],p[i]),
	    value,t_deriv_chem_potential_water_t_exact);
	test_func(dilution_coefficient_t_exact, (sa[i],t[i],p[i]),value,
	    dilution_coefficient_t_exact);

	section_title("Library functions of the GSW Toolbox");

	test_func(deltasa_atlas, (p[i],lon[i],lat[i]),value,deltasa_atlas);
	test_func(fdelta, (p[i],lon[i],lat[i]),value,fdelta);

	section_title(
	    "Water column properties, based on the 75-term polynomial "
	    "for specific volume");

	count	= cast_mpres_m*cast_mpres_n;
	for (j = 0; j<cast_mpres_n; j++) {
	    k = j*cast_m; l = j*cast_mpres_m;
    	    gsw_nsquared(&sa[k],&ct[k],&p[k],&lat[k],cast_m,&val1[l],&val2[l]);
	}
	check_accuracy("nsquared",n2_ca,"n2",count, val1, n2);
	check_accuracy("nsquared",p_mid_n2_ca,"p_mid_n2",count, val2, p_mid_n2);

	for (j = 0; j<cast_mpres_n; j++) {
	    k = j*cast_m; l = j*cast_mpres_m;
	    gsw_turner_rsubrho(&sa[k],&ct[k],&p[k],cast_m,&val1[l],&val2[l],
				&val3[l]);
	}
	check_accuracy("turner_rsubrho",tu_ca,"tu",count, val1, tu);
	check_accuracy("rsubrhorner_rsubrho",rsubrho_ca,"rsubrho",count, val2,
		rsubrho);
	check_accuracy("p_mid_tursrrner_rsubrho",p_mid_tursr_ca,"p_mid_tursr",
		count, val3, p_mid_tursr);

	for (j = 0; j<cast_mpres_n; j++) {
	    k = j*cast_m; l = j*cast_mpres_m;
	    gsw_ipv_vs_fnsquared_ratio(&sa[k],&ct[k],&p[k],pref[0],cast_m,
		&val1[l], &val2[l]);
	}
	check_accuracy("ipv_vs_fnsquared_ratio",ipvfn2_ca,"ipvfn2",count,
		val1, ipvfn2);
	check_accuracy("ipv_vs_fnsquared_ratio",p_mid_ipvfn2_ca,"p_mid_ipvfn2",
		count, val2, p_mid_ipvfn2);

	for (j = 0; j<cast_mpres_n; j++) {
	    k = j*cast_m;
	    for (n=0; n<cast_m; n++)
		if (isnan(sa[k+n]) || fabs(sa[k+n]) >= GSW_ERROR_LIMIT)
		    break;
	    if (gsw_geo_strf_dyn_height(&sa[k],&ct[k],&p[k],pref[0],n,
		&val1[k]) == NULL)
		printf("geo_strf_dyn_height returned NULL.\n");
	}
	check_accuracy("geo_strf_dyn_height",geo_strf_dyn_height_ca,
		"geo_strf_dyn_height",count, val1, geo_strf_dyn_height);

	for (j = 0; j<cast_mpres_n; j++) {
	    k = j*cast_m;
	    for (n=0; n<cast_m; n++)
		if (isnan(sa[k+n]) || fabs(sa[k+n]) >= GSW_ERROR_LIMIT)
		    break;
	    gsw_geo_strf_dyn_height_pc(&sa[k],&ct[k],&delta_p[k],n,
		&val1[k], &val2[k]);
	}
	check_accuracy("geo_strf_dyn_height_pc",geo_strf_dyn_height_pc_ca,
		"geo_strf_dyn_height_pc",count, val1, geo_strf_dyn_height_pc);
	check_accuracy("geo_strf_dyn_height_pc",geo_strf_dyn_height_pc_p_mid_ca,
		"geo_strf_dyn_height_pc_p_mid",count, val2,
		geo_strf_dyn_height_pc_p_mid);

	section_title("Thermodynamic properties of ice Ih");

	count = cast_ice_m*cast_ice_n;

	test_func(rho_ice, (t_seaice[i],p_arctic[i]),value,rho_ice);
	test_func(alpha_wrt_t_ice, (t_seaice[i],p_arctic[i]),value,
	    alpha_wrt_t_ice);
	test_func(specvol_ice, (t_seaice[i],p_arctic[i]),value,specvol_ice);
	test_func(pressure_coefficient_ice, (t_seaice[i],p_arctic[i]),value,
	    pressure_coefficient_ice);
	test_func(sound_speed_ice, (t_seaice[i],p_arctic[i]),value,
	    sound_speed_ice);
	test_func(kappa_ice, (t_seaice[i],p_arctic[i]),value,kappa_ice);
	test_func(kappa_const_t_ice, (t_seaice[i],p_arctic[i]),value,
	    kappa_const_t_ice);
	test_func(internal_energy_ice, (t_seaice[i],p_arctic[i]),value,
	    internal_energy_ice);
	test_func(enthalpy_ice, (t_seaice[i],p_arctic[i]),value,enthalpy_ice);
	test_func(entropy_ice, (t_seaice[i],p_arctic[i]),value,entropy_ice);
	test_func(cp_ice, (t_seaice[i],p_arctic[i]),value,cp_ice);
	test_func(chem_potential_water_ice,
	    (t_seaice[i],p_arctic[i]),value,chem_potential_water_ice);
	test_func(helmholtz_energy_ice, (t_seaice[i],p_arctic[i]),value,
	    helmholtz_energy_ice);
	test_func(adiabatic_lapse_rate_ice,
	    (t_seaice[i],p_arctic[i]),value,adiabatic_lapse_rate_ice);
	test_func(pt0_from_t_ice,(t_seaice[i],p_arctic[i]),pt0, pt0_from_t_ice);
	test_func(pt_from_t_ice, (t_seaice[i],p_arctic[i],pref[0]),value,
	    pt_from_t_ice);
	test_func(t_from_pt0_ice, (pt0[i],p_arctic[i]),value,t_from_pt0_ice);
	test_func(pot_enthalpy_from_pt_ice, (pt0[i]), h,
	    pot_enthalpy_from_pt_ice);
	test_func(pt_from_pot_enthalpy_ice, (h[i]), value,
	    pt_from_pot_enthalpy_ice);
	test_func(pot_enthalpy_from_pt_ice_poly, (pt0[i]), h,
	    pot_enthalpy_from_pt_ice_poly);
	test_func(pt_from_pot_enthalpy_ice_poly, (h[i]),value,
	    pt_from_pot_enthalpy_ice_poly);

	saturation_fraction = 0.5;

	test_func(pressure_freezing_ct,
	    (sa_arctic[i],ct_arctic[i]-1.0,saturation_fraction),value,
	    pressure_freezing_ct);

	section_title("Thermodynamic interaction between ice and seawater");

	test_func(melting_ice_sa_ct_ratio,
	    (sa_arctic[i],ct_arctic[i],p_arctic[i],t_ice[i]),value,
	    melting_ice_sa_ct_ratio);
	test_func(melting_ice_sa_ct_ratio_poly,
	    (sa_arctic[i],ct_arctic[i],p_arctic[i],t_ice[i]),value,
	    melting_ice_sa_ct_ratio_poly);
	test_func(melting_ice_equilibrium_sa_ct_ratio,
	    (sa_arctic[i],p_arctic[i]),value,
	    melting_ice_equilibrium_sa_ct_ratio);
	test_func(melting_ice_equilibrium_sa_ct_ratio_poly,
	    (sa_arctic[i],p_arctic[i]),value,
	    melting_ice_equilibrium_sa_ct_ratio_poly);
	test_sub2(melting_ice_into_seawater,
	    (sa_arctic[i],ct_arctic[i]+0.1,p_arctic[i],w_ice[i],t_ice[i],VALS3),
	    val1, melting_ice_into_seawater_sa_final,
	    val2, melting_ice_into_seawater_ct_final);
	    /*val3, melting_ice_into_seawater_w_ih);*/
	test_sub3(ice_fraction_to_freeze_seawater,
	    (sa_arctic[i],ct_arctic[i],p_arctic[i],t_ice[i],VALS3),
	    val1, ice_fraction_to_freeze_seawater_sa_freeze,
	    val2, ice_fraction_to_freeze_seawater_ct_freeze,
	    val3, ice_fraction_to_freeze_seawater_w_ih);
	test_sub3(frazil_ratios_adiabatic,
	    (sa_arctic[i],p_arctic[i],w_ice[i],VALS3),
	    val1,dsa_dct_frazil, val2,dsa_dp_frazil,
	    val3,dct_dp_frazil);
	test_sub3(frazil_ratios_adiabatic_poly,
	    (sa_arctic[i],p_arctic[i],w_ice[i],VALS3),
	    val1,dsa_dct_frazil_poly, val2,dsa_dp_frazil_poly,
	    val3,dct_dp_frazil_poly);
	test_sub3(frazil_properties_potential,
	    (sa_bulk[i],h_pot_bulk[i],p_arctic[i], VALS3),
	    val1, frazil_properties_potential_sa_final,
	    val2, frazil_properties_potential_ct_final,
	    val3, frazil_properties_potential_w_ih_final);
	test_sub3(frazil_properties_potential_poly,
	    (sa_bulk[i],h_pot_bulk[i], p_arctic[i],VALS3),
	    val1, frazil_properties_potential_poly_sa_final,
	    val2, frazil_properties_potential_poly_ct_final,
	    val3, frazil_properties_potential_poly_w_ih_final);
	test_sub3(frazil_properties,
	    (sa_bulk[i],h_bulk[i],p_arctic[i],VALS3),
	    val1,frazil_properties_sa_final,
	    val2,frazil_properties_ct_final,
	    val3,frazil_properties_w_ih_final);

	section_title("Thermodynamic interaction between seaice and seawater");

	test_func(melting_seaice_sa_ct_ratio,
	    (sa_arctic[i],ct_arctic[i],p_arctic[i], sa_seaice[i],t_seaice[i]),
	    value,melting_seaice_sa_ct_ratio);
	test_func(melting_seaice_sa_ct_ratio_poly,
	    (sa_arctic[i],ct_arctic[i],p_arctic[i], sa_seaice[i],t_seaice[i]),
	    value,melting_seaice_sa_ct_ratio_poly);
	test_func(melting_seaice_equilibrium_sa_ct_ratio,
	    (sa_arctic[i],p_arctic[i]),value,
	    melting_seaice_equilibrium_sa_ct_ratio);
	test_func(melting_seaice_equilibrium_sa_ct_ratio_poly,
	    (sa_arctic[i],p_arctic[i]),value,
	    melting_seaice_equilibrium_sa_ct_ratio_poly);
	test_sub2(melting_seaice_into_seawater, (sa_arctic[i],ct_arctic[i],
	    p_arctic[i], w_seaice[i],sa_seaice[i],t_seaice[i],VALS2),
	    val1, melting_seaice_into_seawater_sa_final,
	    val2, melting_seaice_into_seawater_ct_final);
	test_sub3(seaice_fraction_to_freeze_seawater,(sa_arctic[i],ct_arctic[i],
	    p_arctic[i], sa_seaice[i],t_seaice[i],VALS3),
	    val1, seaice_fraction_to_freeze_seawater_sa_freeze,
	    val2, seaice_fraction_to_freeze_seawater_ct_freeze,
	    val3, seaice_fraction_to_freeze_seawater_w_ih);

	if (gsw_error_flag)
	    printf("\nYour installation of the Gibbs SeaWater (GSW) "
		"Oceanographic Toolbox has errors !\n");
	else
	    printf("\nWell done! The gsw_check_functions confirms that the\n"
		"Gibbs SeaWater (GSW) Oceanographic Toolbox is "
		"installed correctly.\n");

	return (0);
}

void
section_title(char *title)
{
	printf("\n------------------------------------------------"
               "----------------------------\n%s\n\n",title);
}

void
report(char *funcname, char *varname, gsw_error_info *errs)
{
	int	msglen = strlen(funcname)+((varname==NULL)?0:strlen(varname)),
		k, ndots;
	char	message[81], *dots, infoflg[8];

	dots ="...............................................................";
	strcpy(message, funcname);
	if (strcmp(funcname, varname)) {
	    msglen += 5;
	    if (msglen > 62) {
		k = strlen(varname) - (msglen - 62);
		strcat(message, " (..");
		strncat(message, varname, k);
	    } else {
		strcat(message, " (");
		strcat(message, varname);
	    }
	    strcat(message, ")");
	}
	sprintf(infoflg,"(%s%3d)",(errs->flags & GSW_ERROR_LIMIT_FLAG)?"*":"",
		errs->ncomp);
	ndots = 65 - strlen(message);
	if (errs->flags & GSW_ERROR_ERROR_FLAG) {
	    gsw_error_flag = 1;
	    if (ndots > 3)
	        strncat(message, dots, ndots-3);
	    printf("%s << failed >>\n",message);
	    printf("\n  Max difference = %.17g, limit = %.17g\n",
		errs->max,errs->limit);
	    printf("  Max diff (rel) = %.17g, limit = %.17g\n",
		errs->rel,errs->rlimit);
	} else {
	    if (ndots > 0)
		strncat(message, dots, ndots);
	    printf("%s passed %s\n",message,infoflg);
	}
}

void
check_accuracy(char *funcname, double accuracy, char *varname, int count,
	double *calcval, double *refval)
{
	int		i;
	double		diff;
	gsw_error_info	errs;

	memset(&errs, 0, sizeof (errs));
	for (i=0; i<count; i++) {
	    if (fabs(refval[i]) >= GSW_ERROR_LIMIT || isnan(refval[i]))
		continue;
	    errs.ncomp++;
	    diff	= fabs(calcval[i] - refval[i]);
	    if (calcval[i] >= GSW_ERROR_LIMIT)
		errs.flags	|= GSW_ERROR_LIMIT_FLAG;
	    else if (isnan(diff) || diff >= accuracy) {
		errs.flags	|= GSW_ERROR_ERROR_FLAG;
		if (isnan(diff) || diff > errs.max) {
		    errs.max	= diff;
		    errs.limit = accuracy;
		    errs.rel = diff*100.0/fabs(calcval[i]);
		    errs.rlimit = accuracy*100.0/fabs(calcval[i]);
		}
	    }
	}
	report(funcname, varname, &errs);
}
/*
**  The End.
*/
