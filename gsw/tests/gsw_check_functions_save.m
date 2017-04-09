if isempty(which('gsw_gibbs.html'))
    fprintf(2,'You need to add the GSW "html" subdirectory to your path. \n');
end

if isempty(which('gsw_gibbs.p'))
    fprintf(2,'You need to add the GSW "library" subdirectory to your path. \n');
end

if isempty(which('gibbs.pdf'))
    fprintf(2,'You need to add the GSW "pdf" subdirectory to your path. \n');
end

if isempty(which('gsw_rho_t_exact.m'))
    fprintf(2,'You need to add the GSW "thermodynamics_from_t" subdirectory to your path. \n');
end

if isempty(which('gsw_gibbs.html')) | isempty(which('gsw_gibbs.p')) | ...
        isempty(which('gibbs.pdf')) | isempty(which('gsw_rho_t_exact.m'))
    error('You have not added the GSW subdirectories to you MATLAB Path')
end

try
    gsw_installation_dir = which ('gsw_gibbs.html');
    builddocsearchdb ([gsw_installation_dir(1:end-14)])
    clear gsw_installation_dir
end

gsw_data = 'gsw_data_v3_0.mat';
gsw_data_file = which(gsw_data);
load (gsw_data_file,'gsw_cv');

gsw_ver

fprintf(1,' \n');
fprintf(1,'This function is running three stored vertical profiles through\n');
fprintf(1,'all the functions in the GSW Oceanographic Toolbox, and then checks\n');
fprintf(1,'that the outputs are all within pre-defined limits of the correct\n');
fprintf(1,'answers.  These pre-defined limits are a factor of approximately\n');
fprintf(1,'a hundred larger than the errors expected from numerical round-off.\n');
fprintf(1,' \n');
fprintf(1,' checking ');

gsw_cf.gsw_chks = 1;

%% Practical Salinity (SP):- PSS-78

gsw_cf.C = gsw_C_from_SP(gsw_cv.SP_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.IC_from_SP] = find(abs(gsw_cv.C_from_SP - gsw_cf.C) >= gsw_cv.C_from_SP_ca);
if ~isempty(gsw_cf.IC_from_SP)
    fprintf(2,'gsw_C_from_SP:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SP_from_C = gsw_SP_from_C(gsw_cf.C,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.ISP_from_C] = find(abs(gsw_cv.SP_from_C - gsw_cf.SP_from_C) >= gsw_cv.SP_from_C_ca);
if ~isempty(gsw_cf.ISP_from_C)
    fprintf(2,'gsw_SP_from_C:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.R = gsw_R_from_SP(gsw_cv.SP_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.IR_from_SP] = find(abs(gsw_cv.R_from_SP - gsw_cf.R) >= gsw_cv.R_from_SP_ca);
if ~isempty(gsw_cf.IR_from_SP)
    fprintf(2,'gsw_R_from_SP:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SP_from_R = gsw_SP_from_R(gsw_cf.R,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.ISP_from_R] = find(abs(gsw_cv.SP_from_R - gsw_cf.SP_from_R) >= gsw_cv.SP_from_R_ca);
if ~isempty(gsw_cf.ISP_from_R)
        fprintf(2,'gsw_SP_from_R:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.SP_salinometer = gsw_SP_salinometer(gsw_cv.Rt_chck_cast,gsw_cv.t_chck_cast);
[gsw_cf.ISP_salinometer] = find(abs(gsw_cv.SP_salinometer - gsw_cf.SP_salinometer) >= gsw_cv.SP_salinometer_ca);
if ~isempty(gsw_cf.ISP_salinometer)
    fprintf(2,'gsw_SP_salinometer:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SP_from_SK = gsw_SP_from_SK(gsw_cv.SK_chck_cast);
[gsw_cf.ISP_from_SK] = find(abs(gsw_cv.SP_from_SK - gsw_cf.SP_from_SK) >= gsw_cv.SP_from_SK_ca);
if ~isempty(gsw_cf.ISP_from_SK)
    fprintf(2,'gsw_SP_from_SK:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%% Absolute Salinity (SA), Preformed Salinity (Sstar) and Conservative Temperature (CT)

gsw_cf.SA_from_SP = gsw_SA_from_SP(gsw_cv.SP_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.ISA_from_SP] = find(abs(gsw_cv.SA_from_SP - gsw_cf.SA_from_SP) >= gsw_cv.SA_from_SP_ca);
if ~isempty(gsw_cf.ISA_from_SP)
    fprintf(2,'gsw_SA_from_SP:   Failed. Note that this will cause many other programmes in the GSW toolbox to fail.\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.Sstar_from_SP = gsw_Sstar_from_SP(gsw_cv.SP_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.ISstar_from_SP] = find(abs(gsw_cv.Sstar_from_SP - gsw_cf.Sstar_from_SP) >= gsw_cv.Sstar_from_SP_ca);
if ~isempty(gsw_cf.ISstar_from_SP)
    fprintf(2,'gsw_Sstar_from_SP:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.CT_chck_cast = gsw_CT_from_t(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.ICT_from_t] = find(abs(gsw_cv.CT_from_t - gsw_cf.CT_chck_cast) >= gsw_cv.CT_from_t_ca);
if ~isempty(gsw_cf.ICT_from_t)
    fprintf(2,'gsw_CT_from_t:   Failed. Note that this will cause many other programmes in the GSW toolbox to fail.\n');
    gsw_cf.gsw_chks = 0;
end

%% other conversions between temperatures, salinities, pressure and height

gsw_cf.deltaSA_from_SP = gsw_deltaSA_from_SP(gsw_cv.SP_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.IdeltaSA_from_SP] = find(abs(gsw_cv.deltaSA_from_SP - gsw_cf.deltaSA_from_SP) >= gsw_cv.deltaSA_from_SP_ca);
if ~isempty(gsw_cf.IdeltaSA_from_SP)
    fprintf(2,'gsw_deltaSA_from_SP:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.SA_SA_Sstar_from_SP, gsw_cf.Sstar_SA_Sstar_from_SP] = gsw_SA_Sstar_from_SP(gsw_cv.SP_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.ISA_Sstar_from_SP] = find(abs(gsw_cv.SA_SA_Sstar_from_SP - gsw_cf.SA_SA_Sstar_from_SP) >= gsw_cv.SA_SA_Sstar_from_SP_ca | ...
    abs(gsw_cv.Sstar_SA_Sstar_from_SP - gsw_cf.Sstar_SA_Sstar_from_SP) >= gsw_cv.Sstar_SA_Sstar_from_SP_ca);
if ~isempty(gsw_cf.ISA_Sstar_from_SP)
    fprintf(2,'gsw_SA_Sstar_from_SP:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SR_from_SP = gsw_SR_from_SP(gsw_cv.SP_chck_cast);
[gsw_cf.ISR_from_SP] = find(abs(gsw_cv.SR_from_SP - gsw_cf.SR_from_SP) >= gsw_cv.SR_from_SP_ca);
if ~isempty(gsw_cf.ISR_from_SP)
    fprintf(2,'gsw_SR_from_SP:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SP_from_SR = gsw_SP_from_SR(gsw_cv.SR_from_SP);
[gsw_cf.ISP_from_SR] = find(abs(gsw_cv.SP_from_SR - gsw_cf.SP_from_SR) >= gsw_cv.SP_from_SR_ca);
if ~isempty(gsw_cf.ISP_from_SR)
    fprintf(2,'gsw_SP_from_SR:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SP_from_SA = gsw_SP_from_SA(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.ISP_from_SA] = find(abs(gsw_cv.SP_chck_cast - gsw_cf.SP_from_SA) >= gsw_cv.SP_from_SA_ca);
if ~isempty(gsw_cf.ISP_from_SA)
    fprintf(2,'gsw_SP_from_SA:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.Sstar_from_SA = gsw_Sstar_from_SA(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.ISstar_from_SA] = find(abs(gsw_cv.Sstar_from_SA - gsw_cf.Sstar_from_SA) >= gsw_cv.Sstar_from_SA_ca);
if ~isempty(gsw_cf.ISstar_from_SA)
    fprintf(2,'gsw_Sstar_from_SA:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SA_from_Sstar = gsw_SA_from_Sstar(gsw_cv.Sstar_from_SA,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.ISA_from_Sstar] = find(abs(gsw_cv.SA_from_Sstar - gsw_cf.SA_from_Sstar) >= gsw_cv.SA_from_Sstar_ca);
if ~isempty(gsw_cf.ISA_from_Sstar)
    fprintf(2,'gsw_SA_from_Sstar:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SP_from_Sstar = gsw_SP_from_Sstar(gsw_cv.Sstar_from_SA,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.ISP_from_Sstar] = find(abs(gsw_cv.SP_from_Sstar - gsw_cf.SP_from_Sstar) >= gsw_cv.SP_from_Sstar_ca);
if ~isempty(gsw_cf.ISP_from_Sstar)
    fprintf(2,'gsw_SP_from_Sstar:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.t_from_CT =  gsw_t_from_CT(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.It_from_CT] = find(abs(gsw_cv.t_chck_cast - gsw_cf.t_from_CT) >= gsw_cv.t_from_CT_ca);
if ~isempty(gsw_cf.It_from_CT)
    fprintf(2,'gsw_t_from_CT:   Failed.\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.pt = gsw_pt_from_CT(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Ipt_from_CT] = find(abs(gsw_cv.pt_from_CT - gsw_cf.pt) >= gsw_cv.pt_from_CT_ca);
if ~isempty(gsw_cf.Ipt_from_CT)
    fprintf(2,'gsw_pt_from_CT:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.CT_from_pt = gsw_CT_from_pt(gsw_cv.SA_chck_cast,gsw_cf.pt);
[gsw_cf.ICT_from_pt] = find(abs(gsw_cv.CT_from_pt - gsw_cf.CT_from_pt) >= gsw_cv.CT_from_pt_ca);
if ~isempty(gsw_cf.ICT_from_pt)
    fprintf(2,'gsw_CT_from_pt:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.pot_enthalpy_from_pt = gsw_pot_enthalpy_from_pt(gsw_cv.SA_chck_cast,gsw_cf.pt);
[gsw_cf.Ipot_enthalpy] = find(abs(gsw_cv.pot_enthalpy_from_pt - gsw_cf.pot_enthalpy_from_pt) >= gsw_cv.pot_enthalpy_from_pt_ca);
if ~isempty(gsw_cf.Ipot_enthalpy)
    fprintf(2,'gsw_pot_enthalpy_from_pt:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.pt0_from_t = gsw_pt0_from_t(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ipt0] = find(abs(gsw_cv.pt0_from_t - gsw_cf.pt0_from_t) >= gsw_cv.pt0_from_t_ca);
if ~isempty(gsw_cf.Ipt0)
    fprintf(2,'gsw_pt0_from_t:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.pt_from_t = gsw_pt_from_t(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast,gsw_cv.pr);
[gsw_cf.Ipt_from_t] = find(abs(gsw_cv.pt_from_t - gsw_cf.pt_from_t) >= gsw_cv.pt_from_t_ca);
if ~isempty(gsw_cf.Ipt_from_t)
    fprintf(2,'gsw_pt_from_t:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

if gsw_cf.gsw_chks == 1 ;
    fprintf(1,'.');
end

gsw_cf.t90_from_t48 = gsw_t90_from_t48(gsw_cv.t_chck_cast);
[gsw_cf.It90_from_t48] = find(abs(gsw_cv.t90_from_t48 - gsw_cf.t90_from_t48) >= gsw_cv.t90_from_t48_ca);
if ~isempty(gsw_cf.It90_from_t48)
    fprintf(2,'gsw_t90_from_t48:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.t90_from_t68 = gsw_t90_from_t68(gsw_cv.t_chck_cast);
[gsw_cf.It90_from_t68] = find(abs(gsw_cv.t90_from_t68 - gsw_cf.t90_from_t68) >= gsw_cv.t90_from_t68_ca);
if ~isempty(gsw_cf.It90_from_t68)
    fprintf(2,'gsw_t90_from_t68:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.z_from_p = gsw_z_from_p(gsw_cv.p_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.Iz_from_p] = find(abs(gsw_cv.z_from_p - gsw_cf.z_from_p) >= gsw_cv.z_from_p_ca);
if ~isempty(gsw_cf.Iz_from_p)
    fprintf(2,'gsw_z_from_p:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.p_from_z = gsw_p_from_z(gsw_cf.z_from_p,gsw_cv.lat_chck_cast);
[gsw_cf.Ip_from_z] = find(abs(gsw_cv.p_from_z - gsw_cf.p_from_z) >= gsw_cv.p_from_z_ca);
if ~isempty(gsw_cf.Ip_from_z)
    fprintf(2,'gsw_p_from_z:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.depth_from_z = gsw_depth_from_z(gsw_cf.z_from_p);
[gsw_cf.Idepth_from_z] = find(abs(gsw_cv.depth_from_z - gsw_cf.depth_from_z) >= gsw_cv.depth_from_z_ca);
if ~isempty(gsw_cf.Idepth_from_z)
    fprintf(2,'gsw_depth_from_z:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.z_from_depth = gsw_z_from_depth(gsw_cf.depth_from_z);
[gsw_cf.Iz_from_depth] = find(abs(gsw_cv.z_from_depth - gsw_cf.z_from_depth) >= gsw_cv.z_from_depth_ca);
if ~isempty(gsw_cf.Iz_from_depth)
    fprintf(2,'gsw_z_from_depth:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.Abs_Pressure_from_p = gsw_Abs_Pressure_from_p(gsw_cv.p_chck_cast);
[gsw_cf.IAbs_Pressure_from_p] = find(abs(gsw_cv.Abs_Pressure_from_p - gsw_cf.Abs_Pressure_from_p) >= gsw_cv.Abs_Pressure_from_p_ca);
if ~isempty(gsw_cf.IAbs_Pressure_from_p)
    fprintf(2,'gsw_Abs_Pressure_from_p:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.p_from_Abs_Pressure = gsw_p_from_Abs_Pressure(gsw_cf.Abs_Pressure_from_p);
[gsw_cf.Ip_from_Abs_Pressure] = find(abs(gsw_cv.p_from_Abs_Pressure - gsw_cf.p_from_Abs_Pressure) >= gsw_cv.p_from_Abs_Pressure_ca);
if ~isempty(gsw_cf.Ip_from_Abs_Pressure)
    fprintf(2,'gsw_p_from_Abs_Pressure:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.entropy_from_CT = gsw_entropy_from_CT(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Ientropy_from_CT] = find(abs(gsw_cv.entropy_from_CT - gsw_cf.entropy_from_CT) >= gsw_cv.entropy_from_CT_ca);
if ~isempty(gsw_cf.Ientropy_from_CT)
    fprintf(2,'gsw_entropy_from_CT:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.CT_from_entropy = gsw_CT_from_entropy(gsw_cv.SA_chck_cast,gsw_cf.entropy_from_CT);
[gsw_cf.ICT_from_entropy] = find(abs(gsw_cv.CT_from_entropy - gsw_cf.CT_from_entropy) >= gsw_cv.CT_from_entropy_ca);
if ~isempty(gsw_cf.ICT_from_entropy)
    fprintf(2,'gsw_CT_from_entropy:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.entropy_from_pt = gsw_entropy_from_pt(gsw_cv.SA_chck_cast,gsw_cf.pt_from_t);
[gsw_cf.Ientropy_from_pt] = find(abs(gsw_cv.entropy_from_pt - gsw_cf.entropy_from_pt) >= gsw_cv.entropy_from_pt_ca);
if ~isempty(gsw_cf.Ientropy_from_pt)
    fprintf(2,'gsw_entropy_from_pt:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.pt_from_entropy = gsw_pt_from_entropy(gsw_cv.SA_chck_cast,gsw_cf.entropy_from_pt);
[gsw_cf.Ipt_from_entropy] = find(abs(gsw_cv.pt_from_entropy - gsw_cf.pt_from_entropy) >= gsw_cv.pt_from_entropy_ca);
if ~isempty(gsw_cf.Ipt_from_entropy)
    fprintf(2,'gsw_pt_from_entropy:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.entropy_from_t = gsw_entropy_from_t(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ientropy_from_t] = find(abs(gsw_cv.entropy_from_t - gsw_cf.entropy_from_t) >= gsw_cv.entropy_from_t_ca);
if ~isempty(gsw_cf.Ientropy_from_t)
    fprintf(2,'gsw_entropy_from_t:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.t_from_entropy = gsw_t_from_entropy(gsw_cv.SA_chck_cast,gsw_cf.entropy_from_t,gsw_cv.p_chck_cast);
[gsw_cf.It_from_entropy] = find(abs(gsw_cv.t_from_entropy - gsw_cf.t_from_entropy) >= gsw_cv.t_from_entropy_ca);
if ~isempty(gsw_cf.It_from_entropy)
    fprintf(2,'gsw_t_from_entropy:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.adiabatic_lapse_rate_from_CT = gsw_adiabatic_lapse_rate_from_CT(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iadiabatic_lapse_rate_from_CT] = find(abs(gsw_cv.adiabatic_lapse_rate_from_CT - gsw_cf.adiabatic_lapse_rate_from_CT) >= gsw_cv.adiabatic_lapse_rate_from_CT_ca);
if ~isempty(gsw_cf.Iadiabatic_lapse_rate_from_CT)
    fprintf(2,'gsw_adiabatic_lapse_rate_from_CT:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.adiabatic_lapse_rate_from_t = gsw_adiabatic_lapse_rate_from_t(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iadiabatic_lapse_rate_from_t] = find(abs(gsw_cv.adiabatic_lapse_rate_from_t - gsw_cf.adiabatic_lapse_rate_from_t) >= gsw_cv.adiabatic_lapse_rate_from_t_ca);
if ~isempty(gsw_cf.Iadiabatic_lapse_rate_from_t)
    fprintf(2,'gsw_adiabatic_lapse_rate_from_t:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.molality_from_SA = gsw_molality_from_SA(gsw_cv.SA_chck_cast);
[gsw_cf.Imolality_from_SA] = find(abs(gsw_cv.molality_from_SA - gsw_cf.molality_from_SA) >= gsw_cv.molality_from_SA_ca);
if ~isempty(gsw_cf.Imolality_from_SA)
    fprintf(2,'gsw_molality_from_SA:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.ionic_strength_from_SA = gsw_ionic_strength_from_SA(gsw_cv.SA_chck_cast);
[gsw_cf.Iionic_strength_from_SA] = find(abs(gsw_cv.ionic_strength_from_SA - gsw_cf.ionic_strength_from_SA) >= gsw_cv.ionic_strength_from_SA_ca);
if ~isempty(gsw_cf.Iionic_strength_from_SA)
    fprintf(2,'gsw_ionic_strength_from_SA:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%% specific volume, density and enthalpy
if gsw_cf.gsw_chks == 1;
    fprintf(1,'.');
end

gsw_cf.specvol = gsw_specvol(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ispecvol] = find(abs(gsw_cv.specvol - gsw_cf.specvol) >= gsw_cv.specvol_ca);
if ~isempty(gsw_cf.Ispecvol)
    fprintf(2,'gsw_specvol:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.alpha = gsw_alpha(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ialpha] = find(abs(gsw_cv.alpha - gsw_cf.alpha) >= gsw_cv.alpha_ca);
if ~isempty(gsw_cf.Ialpha)
    fprintf(2,'gsw_alpha:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.beta = gsw_beta(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ibeta] = find(abs(gsw_cv.beta - gsw_cf.beta) >= gsw_cv.beta_ca);
if ~isempty(gsw_cf.Ibeta)
    fprintf(2,'gsw_beta:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.alpha_on_beta = gsw_alpha_on_beta(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ialpha_on_beta] = find(abs(gsw_cv.alpha_on_beta - gsw_cf.alpha_on_beta) >= gsw_cv.alpha_on_beta_ca);
if ~isempty(gsw_cf.Ialpha_on_beta)
    fprintf(2,'gsw_alpha_on_beta:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.v_vab, gsw_cf.alpha_vab, gsw_cf.beta_vab] = gsw_specvol_alpha_beta(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ivab] = find(abs(gsw_cv.v_vab - gsw_cf.v_vab) >= gsw_cv.v_vab_ca | ...
    abs(gsw_cv.alpha_vab - gsw_cf.alpha_vab) >= gsw_cv.alpha_vab_ca | ...
    abs(gsw_cv.beta_vab - gsw_cf.beta_vab) >= gsw_cv.beta_vab_ca);
if ~isempty(gsw_cf.Ivab)
    fprintf(2,'gsw_specvol_alpha_beta:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.v_SA, gsw_cf.v_CT, gsw_cf.v_P] = gsw_specvol_first_derivatives(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iv_fd] = find(abs(gsw_cv.v_SA - gsw_cf.v_SA) >= gsw_cv.v_SA_ca | ...
    abs(gsw_cv.v_CT - gsw_cf.v_CT) >= gsw_cv.v_CT_ca | ...
    abs(gsw_cv.v_P - gsw_cf.v_P) >= gsw_cv.v_P_ca);
if ~isempty(gsw_cf.Iv_fd)
    fprintf(2,'gsw_specvol_first_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.v_SA_SA, gsw_cf.v_SA_CT, gsw_cf.v_CT_CT, gsw_cf.v_SA_P, gsw_cf.v_CT_P] = gsw_specvol_second_derivatives(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iv_sd] = find(abs(gsw_cv.v_SA_SA - gsw_cf.v_SA_SA) >= gsw_cv.v_SA_SA_ca | ...
    abs(gsw_cv.v_SA_CT - gsw_cf.v_SA_CT) >= gsw_cv.v_SA_CT_ca | ...
    abs(gsw_cv.v_CT_CT - gsw_cf.v_CT_CT) >= gsw_cv.v_CT_CT_ca | ...
    abs(gsw_cv.v_SA_P - gsw_cf.v_SA_P) >= gsw_cv.v_SA_P_ca | ...
    abs(gsw_cv.v_CT_P - gsw_cf.v_CT_P) >= gsw_cv.v_CT_P_ca);
if ~isempty(gsw_cf.Iv_sd)
    fprintf(2,'gsw_specvol_second_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.v_SA_wrt_h, gsw_cf.v_h] = gsw_specvol_first_derivatives_wrt_enthalpy(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iv_fd_wrt_h] = find(abs(gsw_cv.v_SA_wrt_h - gsw_cf.v_SA_wrt_h) >= gsw_cv.v_SA_wrt_h_ca | ...
    abs(gsw_cv.v_h - gsw_cf.v_h) >= gsw_cv.v_h_ca);
if ~isempty(gsw_cf.Iv_fd_wrt_h)
    fprintf(2,'gsw_specvol_first_derivatives_wrt_enthalpy:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.v_SA_SA_wrt_h, gsw_cf.v_SA_h, gsw_cf.v_h_h] = gsw_specvol_second_derivatives_wrt_enthalpy(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iv_sd_wrt_h] = find(abs(gsw_cv.v_SA_SA - gsw_cf.v_SA_SA) >= gsw_cv.v_SA_SA_ca | ...
    abs(gsw_cv.v_SA_CT - gsw_cf.v_SA_CT) >= gsw_cv.v_SA_CT_ca | ...
    abs(gsw_cv.v_CT_CT - gsw_cf.v_CT_CT) >= gsw_cv.v_CT_CT_ca | ...
    abs(gsw_cv.v_SA_P - gsw_cf.v_SA_P) >= gsw_cv.v_SA_P_ca | ...
    abs(gsw_cv.v_CT_P - gsw_cf.v_CT_P) >= gsw_cv.v_CT_P_ca);
if ~isempty(gsw_cf.Iv_sd_wrt_h)
    fprintf(2,'gsw_specvol_second_derivatives_wrt_enthalpy:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.specvol_anom = gsw_specvol_anom(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.SA_ref,gsw_cv.CT_ref);
[gsw_cf.Ispecvol_anom] = find(abs(gsw_cv.specvol_anom - gsw_cf.specvol_anom) >= gsw_cv.specvol_anom_ca);
if ~isempty(gsw_cf.Ispecvol_anom)
    fprintf(2,'gsw_specvol_anom:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.specvol_anom_standard = gsw_specvol_anom_standard(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ispecvol_anom_standard] = find(abs(gsw_cv.specvol_anom_standard - gsw_cf.specvol_anom_standard) >= gsw_cv.specvol_anom_standard_ca);
if ~isempty(gsw_cf.Ispecvol_anom_standard)
    fprintf(2,'gsw_specvol_anom_standard:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.rho = gsw_rho(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho] = find(abs(gsw_cv.rho - gsw_cf.rho) >= gsw_cv.rho_ca);
if ~isempty(gsw_cf.Irho)
    fprintf(2,'gsw_rho:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.rho_rab, gsw_cf.alpha_rab, gsw_cf.beta_rab] = gsw_rho_alpha_beta(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho_rab] = find(abs(gsw_cv.rho_rab - gsw_cf.rho_rab) >= gsw_cv.rho_rab_ca | ...
    abs(gsw_cv.alpha_rab - gsw_cf.alpha_rab) >= gsw_cv.alpha_rab_ca | ...
    abs(gsw_cv.beta_rab - gsw_cf.beta_rab) >= gsw_cv.beta_rab_ca);
if ~isempty(gsw_cf.Irho_rab)
    fprintf(2,'gsw_rho_alpha_beta:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.rho_SA, gsw_cf.rho_CT, gsw_cf.rho_P] = gsw_rho_first_derivatives(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho_fd] = find(abs(gsw_cv.rho_SA - gsw_cf.rho_SA) >= gsw_cv.rho_SA_ca | ...
    abs(gsw_cv.rho_CT - gsw_cf.rho_CT) >= gsw_cv.rho_CT_ca | ...
    abs(gsw_cv.rho_P - gsw_cf.rho_P) >= gsw_cv.rho_P_ca);
if ~isempty(gsw_cf.Irho_fd)
    fprintf(2,'gsw_rho_first_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.rho_SA_SA, gsw_cf.rho_SA_CT, gsw_cf.rho_CT_CT, gsw_cf.rho_SA_P, gsw_cf.rho_CT_P] = gsw_rho_second_derivatives(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho_sd] = find(abs(gsw_cv.rho_SA_SA - gsw_cf.rho_SA_SA) >= gsw_cv.rho_SA_SA_ca | ...
    abs(gsw_cv.rho_SA_CT - gsw_cf.rho_SA_CT) >= gsw_cv.rho_SA_CT_ca | ...
    abs(gsw_cv.rho_CT_CT - gsw_cf.rho_CT_CT) >= gsw_cv.rho_CT_CT_ca | ...
    abs(gsw_cv.rho_SA_P - gsw_cf.rho_SA_P) >= gsw_cv.rho_SA_P_ca | ...
    abs(gsw_cv.rho_CT_P - gsw_cf.rho_CT_P) >= gsw_cv.rho_CT_P_ca);
if ~isempty(gsw_cf.Irho_sd)
    fprintf(2,'gsw_rho_second_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.rho_SA_wrt_h, gsw_cf.rho_h] = gsw_rho_first_derivatives_wrt_enthalpy(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho_fd_wrt_h] = find(abs(gsw_cv.rho_SA_wrt_h - gsw_cf.rho_SA_wrt_h) >= gsw_cv.rho_SA_wrt_h_ca | ...
    abs(gsw_cv.rho_h - gsw_cf.rho_h) >= gsw_cv.rho_h_ca);
if ~isempty(gsw_cf.Irho_fd_wrt_h)
    fprintf(2,'gsw_rho_first_derivatives_wrt_enthalpy:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.v_SA_SA_wrt_h, gsw_cf.v_SA_h, gsw_cf.v_h_h] = gsw_rho_second_derivatives_wrt_enthalpy(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho_sd_wrt_h] = find(abs(gsw_cv.rho_SA_SA - gsw_cf.rho_SA_SA) >= gsw_cv.rho_SA_SA_ca | ...
    abs(gsw_cv.rho_SA_CT - gsw_cf.rho_SA_CT) >= gsw_cv.rho_SA_CT_ca | ...
    abs(gsw_cv.rho_CT_CT - gsw_cf.rho_CT_CT) >= gsw_cv.rho_CT_CT_ca | ...
    abs(gsw_cv.rho_SA_P - gsw_cf.rho_SA_P) >= gsw_cv.rho_SA_P_ca | ...
    abs(gsw_cv.rho_CT_P - gsw_cf.rho_CT_P) >= gsw_cv.rho_CT_P_ca);
if ~isempty(gsw_cf.Irho_sd_wrt_h)
    fprintf(2,'gsw_rho_second_derivatives_wrt_enthalpy:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.sigma0 = gsw_sigma0(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Isigma0] = find(abs(gsw_cv.sigma0 - gsw_cf.sigma0) >= gsw_cv.sigma0_ca);
if ~isempty(gsw_cf.Isigma0)
    fprintf(2,'gsw_sigma0:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.sigma1 = gsw_sigma1(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Isigma1] = find(abs(gsw_cv.sigma1 - gsw_cf.sigma1) >= gsw_cv.sigma1_ca);
if ~isempty(gsw_cf.Isigma1)
    fprintf(2,'gsw_sigma1:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.sigma2 = gsw_sigma2(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Isigma2] = find(abs(gsw_cv.sigma2 - gsw_cf.sigma2) >= gsw_cv.sigma2_ca);
if ~isempty(gsw_cf.Isigma2)
    fprintf(2,'gsw_sigma2:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.sigma3 = gsw_sigma3(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Isigma3] = find(abs(gsw_cv.sigma3 - gsw_cf.sigma3) >= gsw_cv.sigma3_ca);
if ~isempty(gsw_cf.Isigma3)
    fprintf(2,'gsw_sigma3:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.sigma4 = gsw_sigma4(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Isigma4] = find(abs(gsw_cv.sigma4 - gsw_cf.sigma4) >= gsw_cv.sigma4_ca);
if ~isempty(gsw_cf.Isigma4)
    fprintf(2,'gsw_sigma4:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.cabbeling = gsw_cabbeling(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Icabbeling] = find(abs(gsw_cv.cabbeling - gsw_cf.cabbeling) >= gsw_cv.cabbeling_ca);
if ~isempty(gsw_cf.Icabbeling)
    fprintf(2,'gsw_cabbeling:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.thermobaric = gsw_thermobaric(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ithermobaric] = find(abs(gsw_cv.thermobaric - gsw_cf.thermobaric) >= gsw_cv.thermobaric_ca);
if ~isempty(gsw_cf.Ithermobaric)
    fprintf(2,'gsw_thermobaric:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.enthalpy = gsw_enthalpy(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ienthalpy] = find(abs(gsw_cv.enthalpy - gsw_cf.enthalpy) >= gsw_cv.enthalpy_ca);
if ~isempty(gsw_cf.Ienthalpy)
    fprintf(2,'gsw_enthalpy:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.enthalpy_diff = gsw_enthalpy_diff(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast_shallow,gsw_cv.p_chck_cast_deep);
[gsw_cf.Ienthalpy_diff] = find(abs(gsw_cv.enthalpy_diff - gsw_cf.enthalpy_diff) >= gsw_cv.enthalpy_diff_ca);
if ~isempty(gsw_cf.Ienthalpy_diff)
    fprintf(2,'gsw_enthalpy_diff:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.dynamic_enthalpy = gsw_dynamic_enthalpy(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Idynamic_enthalpy] = find(abs(gsw_cv.dynamic_enthalpy - gsw_cf.dynamic_enthalpy) >= gsw_cv.dynamic_enthalpy_ca);
if ~isempty(gsw_cf.Idynamic_enthalpy)
    fprintf(2,'gsw_dynamic_enthalpy:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.h_SA, gsw_cf.h_CT] = gsw_enthalpy_first_derivatives(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ienthalpy_first_deriv] = find(abs(gsw_cv.h_SA - gsw_cf.h_SA) >= gsw_cv.h_SA_ca | ...
    abs(gsw_cv.h_CT - gsw_cf.h_CT) >= gsw_cv.h_CT_ca);
if ~isempty(gsw_cf.Ienthalpy_first_deriv)
    fprintf(2,'gsw_enthalpy_first_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.h_SA_SA, gsw_cf.h_SA_CT, gsw_cf.h_CT_CT] = gsw_enthalpy_second_derivatives(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ienthalpy_second_deriv] = find(abs(gsw_cv.h_SA_SA - gsw_cf.h_SA_SA) >= gsw_cv.h_SA_SA_ca  | ...
    abs(gsw_cv.h_SA_CT - gsw_cf.h_SA_CT) >= gsw_cv.h_SA_CT_ca | ...
    abs(gsw_cv.h_CT_CT - gsw_cf.h_CT_CT) >= gsw_cv.h_CT_CT_ca);
if ~isempty(gsw_cf.Ienthalpy_second_deriv)
    fprintf(2,'gsw_enthalpy_second_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.sound_speed = gsw_sound_speed(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Isound_speed] = find(abs(gsw_cv.sound_speed - gsw_cf.sound_speed) >= gsw_cv.sound_speed_ca);
if ~isempty(gsw_cf.Isound_speed)
    fprintf(2,'gsw_sound_speed:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.kappa = gsw_kappa(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ikappa] = find(abs(gsw_cv.kappa - gsw_cf.kappa) >= gsw_cv.kappa_ca);
if ~isempty(gsw_cf.Ikappa)
    fprintf(2,'gsw_kappa:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.internal_energy = gsw_internal_energy(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iinternal_energy] = find(abs(gsw_cv.internal_energy - gsw_cf.internal_energy) >= gsw_cv.internal_energy_ca);
if ~isempty(gsw_cf.Iinternal_energy)
    fprintf(2,'gsw_internal_energy:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.u_SA, gsw_cf.u_CT, gsw_cf.u_P] = gsw_internal_energy_first_derivatives(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iu_fd] = find(abs(gsw_cv.u_SA - gsw_cf.u_SA) >= gsw_cv.u_SA_ca | ...
    abs(gsw_cv.u_CT - gsw_cf.u_CT) >= gsw_cv.u_CT_ca | ...
    abs(gsw_cv.u_P - gsw_cf.u_P) >= gsw_cv.u_P_ca);
if ~isempty(gsw_cf.Iu_fd)
    fprintf(2,'gsw_rho_first_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.u_SA_SA, gsw_cf.u_SA_CT, gsw_cf.u_CT_CT, gsw_cf.u_SA_P, gsw_cf.u_CT_P] = gsw_internal_energy_second_derivatives(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iu_sd] = find(abs(gsw_cv.u_SA_SA - gsw_cf.u_SA_SA) >= gsw_cv.rho_SA_SA_ca | ...
    abs(gsw_cv.u_SA_CT - gsw_cf.u_SA_CT) >= gsw_cv.u_SA_CT_ca | ...
    abs(gsw_cv.u_CT_CT - gsw_cf.u_CT_CT) >= gsw_cv.u_CT_CT_ca | ...
    abs(gsw_cv.u_SA_P - gsw_cf.u_SA_P) >= gsw_cv.u_SA_P_ca | ...
    abs(gsw_cv.u_CT_P - gsw_cf.u_CT_P) >= gsw_cv.u_CT_P_ca);
if ~isempty(gsw_cf.Iu_sd)
    fprintf(2,'gsw_internal_energy_second_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%%% EF fix: swapped argument order between S and h
%%%gsw_cf.CT_from_enthalpy = gsw_CT_from_enthalpy(gsw_cf.enthalpy,gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast);

gsw_cf.CT_from_enthalpy = gsw_CT_from_enthalpy(gsw_cv.SA_chck_cast,gsw_cf.enthalpy,gsw_cv.p_chck_cast);
[gsw_cf.ICT_from_h] = find(abs(gsw_cv.CT_from_enthalpy - gsw_cf.CT_from_enthalpy) >= gsw_cv.CT_from_enthalpy_ca);
if ~isempty(gsw_cf.ICT_from_h)
    fprintf(2,'gsw_CT_from_enthalpy:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SA_from_rho = gsw_SA_from_rho(gsw_cf.rho,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.ISA_from_rho] = find(abs(gsw_cv.SA_from_rho - gsw_cf.SA_from_rho) >= gsw_cv.SA_from_rho_ca);
if ~isempty(gsw_cf.ISA_from_rho)
    fprintf(2,'gsw_SA_from_rho:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.CT_from_rho = gsw_CT_from_rho(gsw_cf.rho,gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.ICT_from_rho] = find(abs(gsw_cv.CT_from_rho - gsw_cf.CT_from_rho) >= gsw_cv.CT_from_rho_ca);
if ~isempty(gsw_cf.ICT_from_rho)
    fprintf(2,'gsw_CT_from_rho:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.CT_maxdensity = gsw_CT_maxdensity(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.ICT_maxdensity] = find(abs(gsw_cv.CT_maxdensity - gsw_cf.CT_maxdensity) >= gsw_cv.CT_maxdensity_ca);
if ~isempty(gsw_cf.ICT_maxdensity)
    fprintf(2,'gsw_CT_maxdensity:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%% vertical stability

[gsw_cf.Tu, gsw_cf.Rsubrho, gsw_cf.p_mid_TuRsr] = gsw_Turner_Rsubrho(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.ITurner] = find(abs(gsw_cv.Tu - gsw_cf.Tu) >= gsw_cv.Tu_ca | abs(gsw_cv.Rsubrho - gsw_cf.Rsubrho) >= gsw_cv.Rsubrho_ca | ...
    abs(gsw_cv.p_mid_TuRsr - gsw_cf.p_mid_TuRsr) >= gsw_cv.p_mid_TuRsr_ca);
if ~isempty(gsw_cf.ITurner)
    fprintf(2,'gsw_Turner_Rsubrho:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.n2, gsw_cf.p_mid_n2] = gsw_Nsquared(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.INsquared] = find(abs(gsw_cv.n2 - gsw_cf.n2) >= gsw_cv.n2_ca | abs(gsw_cv.p_mid_n2 - gsw_cf.p_mid_n2) >= gsw_cv.p_mid_n2_ca);
if ~isempty(gsw_cf.INsquared)
    fprintf(2,'gsw_Nsquared:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.n2min, gsw_cf.n2min_pmid, gsw_cf.n2min_specvol, gsw_cf.n2min_alpha, gsw_cf.n2min_beta, gsw_cf.n2min_dsa, gsw_cf.n2min_dct, gsw_cf.n2min_dp] = gsw_Nsquared_min(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.INsquared_min] = find(abs(gsw_cv.n2min - gsw_cf.n2min) >= gsw_cv.n2min_ca | abs(gsw_cv.n2min_pmid - gsw_cf.n2min_pmid) >= gsw_cv.n2min_pmid_ca | ...
    abs(gsw_cv.n2min_specvol - gsw_cf.n2min_specvol) >= gsw_cv.n2min_specvol_ca | abs(gsw_cv.n2min_alpha - gsw_cf.n2min_alpha) >= gsw_cv.n2min_alpha_ca | ...
    abs(gsw_cv.n2min_beta - gsw_cf.n2min_beta) >= gsw_cv.n2min_beta_ca | abs(gsw_cv.n2min_dsa - gsw_cf.n2min_dsa) >= gsw_cv.n2min_dsa_ca | ...
    abs(gsw_cv.n2min_dct - gsw_cf.n2min_dct) >= gsw_cv.n2min_dct_ca | abs(gsw_cv.n2min_dp - gsw_cf.n2min_dp) >= gsw_cv.n2min_dp_ca);
if ~isempty(gsw_cf.INsquared_min)
    fprintf(2,'gsw_Nsquared_min:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.mlp = gsw_mlp(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Imlp] = find(abs(gsw_cv.mlp - gsw_cf.mlp) >= gsw_cv.mlp_ca);
if ~isempty(gsw_cf.Imlp)
    fprintf(2,'gsw_mlp:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.n2_lowerlimit = gsw_Nsquared_lowerlimit(gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.INsquared_lowerlimit] = find(abs(gsw_cv.n2_lowerlimit - gsw_cf.n2_lowerlimit) >= gsw_cv.n2_lowerlimit_ca);
if ~isempty(gsw_cf.INsquared_lowerlimit)
    fprintf(2,'gsw_Nsquared_lowerlimit:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.IPVfN2, gsw_cf.p_mid_IPVfN2] = gsw_IPV_vs_fNsquared_ratio(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.pr);
[gsw_cf.IIPVfN2] = find(abs(gsw_cv.IPVfN2 - gsw_cf.IPVfN2) >= gsw_cv.IPVfN2_ca | ...
    abs(gsw_cv.p_mid_IPVfN2 - gsw_cf.p_mid_IPVfN2) >= gsw_cv.p_mid_IPVfN2_ca);
if ~isempty(gsw_cf.IIPVfN2)
    fprintf(2,'gsw_IPV_vs_fNsquared_ratio:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

if gsw_cf.gsw_chks == 1 ;
    fprintf(1,'.');
end

%% geostrophic streamfunctions and acoustic travel time

gsw_cf.geo_strf_dyn_height = gsw_geo_strf_dyn_height(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.pr);
[gsw_cf.Igeo_strf_dyn_height] = find(abs(gsw_cv.geo_strf_dyn_height - gsw_cf.geo_strf_dyn_height) >= gsw_cv.geo_strf_dyn_height_ca);
if ~isempty(gsw_cf.Igeo_strf_dyn_height)
    fprintf(2,'gsw_geo_strf_dyn_height:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.geo_strf_dyn_height_pc, gsw_cf.geo_strf_dyn_height_pc_p_mid] = gsw_geo_strf_dyn_height_pc(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.delta_p_chck_cast);
[gsw_cf.Igeo_strf_dyn_height_pc] = find(abs(gsw_cv.geo_strf_dyn_height_pc - gsw_cf.geo_strf_dyn_height_pc) >= gsw_cv.geo_strf_dyn_height_pc_ca | ...
    abs(gsw_cv.geo_strf_dyn_height_pc_p_mid - gsw_cf.geo_strf_dyn_height_pc_p_mid) >= gsw_cv.geo_strf_dyn_height_pc_p_mid_ca);
if ~isempty(gsw_cf.Igeo_strf_dyn_height_pc)
    fprintf(2,'gsw_geo_strf_dyn_height_pc:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.geo_strf_isopycnal = gsw_geo_strf_isopycnal(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.pr,gsw_cv.Neutral_Density,gsw_cv.p_Neutral_Density);
[gsw_cf.Igeo_strf_isopycnal] = find(abs(gsw_cv.geo_strf_isopycnal - gsw_cf.geo_strf_isopycnal) >= gsw_cv.geo_strf_isopycnal_ca);
if ~isempty(gsw_cf.Igeo_strf_isopycnal)
    fprintf(2,'gsw_geo_strf_isopycnal:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.geo_strf_isopycnal_pc, gsw_cf.geo_strf_isopycnal_pc_p_mid] = gsw_geo_strf_isopycnal_pc(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.delta_p_chck_cast,gsw_cv.Neutral_Density(1),3);
[gsw_cf.Igeo_strf_isopycnal_pc] = find(abs(gsw_cv.geo_strf_isopycnal_pc - gsw_cf.geo_strf_isopycnal_pc) >= gsw_cv.geo_strf_isopycnal_pc_ca |...
    abs(gsw_cv.geo_strf_isopycnal_pc_p_mid - gsw_cf.geo_strf_isopycnal_pc_p_mid) >= gsw_cv.geo_strf_isopycnal_pc_p_mid_ca);
if ~isempty(gsw_cf.Igeo_strf_isopycnal_pc)
    fprintf(2,'gsw_geo_strf_isopycnal_pc:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.geo_strf_Montgomery = gsw_geo_strf_Montgomery(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.pr);
[gsw_cf.Igeo_strf_Montgomery] = find(abs(gsw_cv.geo_strf_Montgomery - gsw_cf.geo_strf_Montgomery) >= gsw_cv.geo_strf_Montgomery_ca);
if ~isempty(gsw_cf.Igeo_strf_Montgomery)
    fprintf(2,'gsw_geo_strf_Montgomery:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.geo_strf_Cunningham = gsw_geo_strf_Cunningham(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.pr);
[gsw_cf.Igeo_strf_Cunningham] = find(abs(gsw_cv.geo_strf_Cunningham - gsw_cf.geo_strf_Cunningham) >= gsw_cv.geo_strf_Cunningham_ca);
if ~isempty(gsw_cf.Igeo_strf_Cunningham)
    fprintf(2,'gsw_geo_strf_Cunningham:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.geo_strf_steric_height = gsw_geo_strf_steric_height(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.pr);
[gsw_cf.Igeo_strf_steric_height] = find(abs(gsw_cv.geo_strf_steric_height - gsw_cf.geo_strf_steric_height) >= gsw_cv.geo_strf_steric_height_ca);
if ~isempty(gsw_cf.Igeo_strf_steric_height)
    fprintf(2,'gsw_geo_strf_steric_height:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.geo_strf_PISH = gsw_geo_strf_PISH(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.pr_05);
[gsw_cf.Igeo_strf_PISH] = find(abs(gsw_cv.geo_strf_PISH - gsw_cf.geo_strf_PISH) >= gsw_cv.geo_strf_PISH_ca);
if ~isempty(gsw_cf.Igeo_strf_PISH)
    fprintf(2,'gsw_geo_strf_PISH:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.travel_time = gsw_travel_time(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.lat_chck_cast(1));
[gsw_cf.Itravel_time] = find(abs(gsw_cv.travel_time - gsw_cf.travel_time) >= gsw_cv.travel_time_ca);
if ~isempty(gsw_cf.Itravel_time)
    fprintf(2,'gsw_travel_time:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%% Geostrophic velocity

[gsw_cf.geo_strf_velocity, gsw_cf.geo_strf_velocity_mid_lat, gsw_cf.geo_strf_velocity_mid_long] = gsw_geostrophic_velocity(gsw_cf.geo_strf_dyn_height,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Igeostrophic_velo] = find(abs(gsw_cv.geo_strf_velocity - gsw_cf.geo_strf_velocity) >= gsw_cv.geo_strf_velocity_ca | ...
    abs(gsw_cv.geo_strf_velocity_mid_lat - gsw_cf.geo_strf_velocity_mid_lat) >= gsw_cv.geo_strf_velocity_mid_lat_ca  | ...
    abs(gsw_cv.geo_strf_velocity_mid_long - gsw_cf.geo_strf_velocity_mid_long) >= gsw_cv.geo_strf_velocity_mid_long_ca);
if ~isempty(gsw_cf.Igeostrophic_velo)
    fprintf(2,'gsw_geostrophic_velocity:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%% neutral versus isopycnal slopes and ratios

if gsw_cf.gsw_chks == 1 ;
    fprintf(1,'.');
end

gsw_cf.isopycnal_slope_ratio = gsw_isopycnal_slope_ratio(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.pr);
[gsw_cf.Iisopycnal_slope_ratio] = find(abs(gsw_cv.isopycnal_slope_ratio - gsw_cf.isopycnal_slope_ratio) >= gsw_cv.isopycnal_slope_ratio_ca);
if ~isempty(gsw_cf.Iisopycnal_slope_ratio)
    fprintf(2,'gsw_isopycnal_slope_ratio:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.G_CT, gsw_cf.p_mid_G_CT] = gsw_isopycnal_vs_ntp_CT_ratio(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.pr);
[gsw_cf.IG_CT] = find(abs(gsw_cv.G_CT - gsw_cf.G_CT) >= gsw_cv.G_CT_ca | ...
    (gsw_cv.p_mid_G_CT - gsw_cf.p_mid_G_CT) >= gsw_cv.p_mid_G_CT_ca);
if ~isempty(gsw_cf.IG_CT)
    fprintf(2,'gsw_isopycnal_vs_ntp_CT_ratio:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.ntpptCT = gsw_ntp_pt_vs_CT_ratio(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.IntpptCT] = find(abs(gsw_cv.ntpptCT - gsw_cf.ntpptCT) >= gsw_cv.ntpptCT_ca);
if ~isempty(gsw_cf.IntpptCT)
    fprintf(2,'gsw_ntp_pt_vs_CT_ratio:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%% derivatives of enthalpy, entropy, CT and pt

[gsw_cf.CT_SA, gsw_cf.CT_pt] = gsw_CT_first_derivatives(gsw_cv.SA_chck_cast,gsw_cf.pt);
[gsw_cf.ICT_first_deriv] = find(abs(gsw_cv.CT_SA - gsw_cf.CT_SA) >= gsw_cv.CT_SA_ca | ...
    (gsw_cv.CT_pt - gsw_cf.CT_pt) >= gsw_cv.CT_pt_ca);
if ~isempty(gsw_cf.ICT_first_deriv)
    fprintf(2,'gsw_CT_first_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.CT_SA_SA, gsw_cf.CT_SA_pt, gsw_cf.CT_pt_pt] = gsw_CT_second_derivatives(gsw_cv.SA_chck_cast,gsw_cf.pt);
[gsw_cf.ICT_second_deriv] = find(abs(gsw_cv.CT_SA_SA - gsw_cf.CT_SA_SA) >= gsw_cv.CT_SA_SA_ca | ...
    abs(gsw_cv.CT_SA_pt - gsw_cf.CT_SA_pt) >= gsw_cv.CT_SA_pt_ca | ...
    abs(gsw_cv.CT_pt_pt - gsw_cf.CT_pt_pt) >= gsw_cv.CT_pt_pt_ca);
if ~isempty(gsw_cf.ICT_second_deriv)
    fprintf(2,'gsw_CT_second_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.eta_SA, gsw_cf.eta_CT] = gsw_entropy_first_derivatives(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Ientropy_first_deriv] = find(abs(gsw_cv.eta_SA - gsw_cf.eta_SA) >= gsw_cv.eta_SA_ca | ...
    abs(gsw_cv.eta_CT - gsw_cf.eta_CT) >= gsw_cv.eta_CT_ca);
if ~isempty(gsw_cf.Ientropy_first_deriv)
    fprintf(2,'gsw_entropy_first_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

if gsw_cf.gsw_chks == 1 ;
    fprintf(1,'.');
end

[gsw_cf.eta_SA_SA, gsw_cf.eta_SA_CT, gsw_cf.eta_CT_CT] = gsw_entropy_second_derivatives(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Ientropy_second_deriv] = find((abs(gsw_cv.eta_SA_SA - gsw_cf.eta_SA_SA)) >= gsw_cv.eta_SA_SA_ca |...
    abs(gsw_cv.eta_SA_CT - gsw_cf.eta_SA_CT) >= gsw_cv.eta_SA_CT_ca |...
    abs(gsw_cv.eta_CT_CT - gsw_cf.eta_CT_CT) >= gsw_cv.eta_CT_CT_ca);
if ~isempty(gsw_cf.Ientropy_second_deriv)
    fprintf(2,'gsw_entropy_second_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.pt_SA, gsw_cf.pt_CT] = gsw_pt_first_derivatives(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Ipt_first_deriv] = find(abs(gsw_cv.pt_SA - gsw_cf.pt_SA) >= gsw_cv.pt_SA_ca |...
    abs(gsw_cv.pt_CT - gsw_cf.pt_CT) >= gsw_cv.pt_CT_ca);
if ~isempty(gsw_cf.Ipt_first_deriv)
    fprintf(2,'gsw_pt_first_derivatives:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.pt_SA_SA, gsw_cf.pt_SA_CT, gsw_cf.pt_CT_CT] = gsw_pt_second_derivatives(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Ipt_second_deriv] = find(abs(gsw_cv.pt_SA_SA - gsw_cf.pt_SA_SA) >= gsw_cv.pt_SA_SA_ca  | ...
    abs(gsw_cv.pt_SA_CT - gsw_cf.pt_SA_CT) >= gsw_cv.pt_SA_CT_ca | ...
    abs(gsw_cv.pt_CT_CT - gsw_cf.pt_CT_CT) >= gsw_cv.pt_CT_CT_ca);
if ~isempty(gsw_cf.Ipt_second_deriv)
    fprintf(2,'gsw_pt_second_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%% seawater properties at freezing temperatures

gsw_cf.CT_freezing = gsw_CT_freezing(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast,0.5);
[gsw_cf.ICT_freezing] = find(abs(gsw_cv.CT_freezing - gsw_cf.CT_freezing) >= gsw_cv.CT_freezing_ca);
if ~isempty(gsw_cf.ICT_freezing)
    fprintf(2,'gsw_CT_freezing:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.CT_freezing_poly = gsw_CT_freezing_poly(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast,0.5);
[gsw_cf.ICT_freezing_poly] = find(abs(gsw_cv.CT_freezing_poly - gsw_cf.CT_freezing_poly) >= gsw_cv.CT_freezing_poly_ca);
if ~isempty(gsw_cf.ICT_freezing_poly)
    fprintf(2,'gsw_CT_freezing_poly:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.t_freezing = gsw_t_freezing(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast,0.5);
[gsw_cf.It_freezing] = find(abs(gsw_cv.t_freezing - gsw_cf.t_freezing) >= gsw_cv.t_freezing_ca);
if ~isempty(gsw_cf.It_freezing)
    fprintf(2,'gsw_t_freezing:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.t_freezing_poly = gsw_t_freezing_poly(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast,0.5);
[gsw_cf.It_freezing_poly] = find(abs(gsw_cv.t_freezing_poly - gsw_cf.t_freezing_poly) >= gsw_cv.t_freezing_poly_ca);
if ~isempty(gsw_cf.It_freezing_poly)
    fprintf(2,'gsw_t_freezing_poly:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.pot_enthalpy_ice_freezing = gsw_pot_enthalpy_ice_freezing(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ipot_enthalpy_ice_freezing] = find(abs(gsw_cv.pot_enthalpy_ice_freezing - gsw_cf.pot_enthalpy_ice_freezing) >= gsw_cv.pot_enthalpy_ice_freezing_ca);
if ~isempty(gsw_cf.Ipot_enthalpy_ice_freezing)
    fprintf(2,'gsw_pot_enthalpy_ice_freezing:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.pot_enthalpy_ice_freezing_poly = gsw_pot_enthalpy_ice_freezing_poly(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ipot_enthalpy_ice_freezing_poly] = find(abs(gsw_cv.pot_enthalpy_ice_freezing_poly - gsw_cf.pot_enthalpy_ice_freezing_poly) >= gsw_cv.pot_enthalpy_ice_freezing_poly_ca);
if ~isempty(gsw_cf.Ipot_enthalpy_ice_freezing_poly)
    fprintf(2,'gsw_pot_enthalpy_ice_freezing_poly:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.SA_freezing_from_CT = gsw_SA_freezing_from_CT(gsw_cf.CT_freezing,gsw_cv.p_chck_cast,0.5);
[gsw_cf.ISA_freezing_from_CT] = find(abs(gsw_cv.SA_freezing_from_CT - gsw_cf.SA_freezing_from_CT) >= gsw_cv.SA_freezing_from_CT_ca);
if ~isempty(gsw_cf.ISA_freezing_from_CT)
    fprintf(2,'gsw_SA_freezing_from_CT:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.SA_freezing_from_CT_poly = gsw_SA_freezing_from_CT_poly(gsw_cf.CT_freezing_poly,gsw_cv.p_chck_cast,0.5);
[gsw_cf.ISA_freezing_from_CT_poly] = find(abs(gsw_cv.SA_freezing_from_CT_poly - gsw_cf.SA_freezing_from_CT_poly) >= gsw_cv.SA_freezing_from_CT_poly_ca);
if ~isempty(gsw_cf.ISA_freezing_from_CT_poly)
    fprintf(2,'gsw_SA_freezing_from_CT_poly:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.SA_freezing_from_t = gsw_SA_freezing_from_t(gsw_cf.t_freezing,gsw_cv.p_chck_cast,0.5);
[gsw_cf.ISA_freezing_from_t] = find(abs(gsw_cv.SA_freezing_from_t - gsw_cf.SA_freezing_from_t) >= gsw_cv.SA_freezing_from_t_ca);
if ~isempty(gsw_cf.ISA_freezing_from_t)
    fprintf(2,'gsw_SA_freezing_from_t:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SA_freezing_from_t_poly = gsw_SA_freezing_from_t_poly(gsw_cf.t_freezing_poly,gsw_cv.p_chck_cast,0.5);
[gsw_cf.ISA_freezing_from_t_poly] = find(abs(gsw_cv.SA_freezing_from_t_poly - gsw_cf.SA_freezing_from_t_poly) >= gsw_cv.SA_freezing_from_t_poly_ca);
if ~isempty(gsw_cf.ISA_freezing_from_t_poly)
    fprintf(2,'gsw_SA_freezing_from_t_poly:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.pressure_freezing_CT = gsw_pressure_freezing_CT(gsw_cv.SA_Arctic,(gsw_cv.CT_Arctic - 1),0.5);
[gsw_cf.Ipressure_freezing_CT] = find(abs(gsw_cv.pressure_freezing_CT - gsw_cf.pressure_freezing_CT) >= gsw_cv.pressure_freezing_CT_ca);
if ~isempty(gsw_cf.Ipressure_freezing_CT)
    fprintf(2,'gsw_pressure_freezing_CT:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.CTfreezing_SA, gsw_cf.CTfreezing_P] = gsw_CT_freezing_first_derivatives(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast,0.5);
[gsw_cf.ICT_freezing_first_derivatives] = find(abs(gsw_cv.CTfreezing_SA - gsw_cf.CTfreezing_SA) >= gsw_cv.CTfreezing_SA_ca | ...
    abs(gsw_cv.CTfreezing_P - gsw_cf.CTfreezing_P) >= gsw_cv.CTfreezing_P_ca);
if ~isempty(gsw_cf.ICT_freezing_first_derivatives)
    fprintf(2,'gsw_CT_freezing_first_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.CTfreezing_SA_poly, gsw_cf.CTfreezing_P_poly] = gsw_CT_freezing_first_derivatives_poly(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast,0.5);
[gsw_cf.ICT_freezing_first_derivatives_poly] = find(abs(gsw_cv.CTfreezing_SA_poly - gsw_cf.CTfreezing_SA_poly) >= gsw_cv.CTfreezing_SA_poly_ca | ...
    abs(gsw_cv.CTfreezing_P_poly - gsw_cf.CTfreezing_P_poly) >= gsw_cv.CTfreezing_P_poly_ca);
if ~isempty(gsw_cf.ICT_freezing_first_derivatives_poly)
    fprintf(2,'gsw_CT_freezing_first_derivatives_poly:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.tfreezing_SA, gsw_cf.tfreezing_P] = gsw_t_freezing_first_derivatives(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast,0.5);
[gsw_cf.It_freezing_first_derivatives] = find(abs(gsw_cv.tfreezing_SA - gsw_cf.tfreezing_SA) >= gsw_cv.tfreezing_SA_ca | ...
    abs(gsw_cv.tfreezing_P - gsw_cf.tfreezing_P) >= gsw_cv.tfreezing_P_ca);
if ~isempty(gsw_cf.It_freezing_first_derivatives)
    fprintf(2,'gsw_t_freezing_first_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.tfreezing_SA_poly, gsw_cf.tfreezing_P_poly] = gsw_t_freezing_first_derivatives_poly(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast,0.5);
[gsw_cf.It_freezing_first_derivatives_poly] = find(abs(gsw_cv.tfreezing_SA_poly - gsw_cf.tfreezing_SA_poly) >= gsw_cv.tfreezing_SA_poly_ca | ...
    abs(gsw_cv.tfreezing_P_poly - gsw_cf.tfreezing_P_poly) >= gsw_cv.tfreezing_P_poly_ca);
if ~isempty(gsw_cf.It_freezing_first_derivatives_poly)
    fprintf(2,'gsw_t_freezing_first_derivatives_poly:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.pot_enthalpy_ice_freezing_SA, gsw_cf.pot_enthalpy_ice_freezing_P] = gsw_pot_enthalpy_ice_freezing_first_derivatives(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ipot_enthalpy_ice_freezing_first_derivatives] = find(abs(gsw_cv.pot_enthalpy_ice_freezing_SA - gsw_cf.pot_enthalpy_ice_freezing_SA) >= gsw_cv.pot_enthalpy_ice_freezing_SA_ca | ...
    abs(gsw_cv.pot_enthalpy_ice_freezing_P - gsw_cf.pot_enthalpy_ice_freezing_P) >= gsw_cv.pot_enthalpy_ice_freezing_P_ca);
if ~isempty(gsw_cf.Ipot_enthalpy_ice_freezing_first_derivatives)
    fprintf(2,'gsw_pot_enthalpy_ice_freezing_first_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.pot_enthalpy_ice_freezing_SA_poly, gsw_cf.pot_enthalpy_ice_freezing_P_poly] = gsw_pot_enthalpy_ice_freezing_first_derivatives_poly(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ipot_enthalpy_ice_freezing_first_derivatives_poly] = find(abs(gsw_cv.pot_enthalpy_ice_freezing_SA_poly - gsw_cf.pot_enthalpy_ice_freezing_SA_poly) >= gsw_cv.pot_enthalpy_ice_freezing_SA_poly_ca | ...
    abs(gsw_cv.pot_enthalpy_ice_freezing_P_poly - gsw_cf.pot_enthalpy_ice_freezing_P_poly) >= gsw_cv.pot_enthalpy_ice_freezing_P_poly_ca);
if ~isempty(gsw_cf.Ipot_enthalpy_ice_freezing_first_derivatives_poly)
    fprintf(2,'gsw_pot_enthalpy_ice_freezing_first_derivatives_poly:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.latentheat_melting = gsw_latentheat_melting(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ilatentheat_melting] = find(abs(gsw_cv.latentheat_melting - gsw_cf.latentheat_melting) >= gsw_cv.latentheat_melting_ca);
if ~isempty(gsw_cf.Ilatentheat_melting)
    fprintf(2,'gsw_latentheat_melting:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%%  thermodynamic interaction between ice and seawater

gsw_cf.melting_ice_SA_CT_ratio = gsw_melting_ice_SA_CT_ratio(gsw_cv.SA_Arctic,gsw_cv.CT_Arctic,gsw_cv.p_Arctic,gsw_cv.t_ice);
[gsw_cf.Imelting_ice_SA_CT_ratio] = find(abs(gsw_cv.melting_ice_SA_CT_ratio - gsw_cf.melting_ice_SA_CT_ratio) >= gsw_cv.melting_ice_SA_CT_ratio_ca);
if ~isempty(gsw_cf.Imelting_ice_SA_CT_ratio)
    fprintf(2,'gsw_melting_ice_SA_CT_ratio:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.melting_ice_SA_CT_ratio_poly = gsw_melting_ice_SA_CT_ratio_poly(gsw_cv.SA_Arctic,gsw_cv.CT_Arctic,gsw_cv.p_Arctic,gsw_cv.t_ice);
[gsw_cf.Imelting_ice_SA_CT_ratio_poly] = find(abs(gsw_cv.melting_ice_SA_CT_ratio_poly - gsw_cf.melting_ice_SA_CT_ratio_poly) >= gsw_cv.melting_ice_SA_CT_ratio_poly_ca);
if ~isempty(gsw_cf.Imelting_ice_SA_CT_ratio_poly)
    fprintf(2,'gsw_melting_ice_SA_CT_ratio_poly:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.melting_ice_equilibrium_SA_CT_ratio = gsw_melting_ice_equilibrium_SA_CT_ratio(gsw_cv.SA_Arctic,gsw_cv.p_Arctic);
[gsw_cf.Imelting_ice_equilibrium_SA_CT_ratio] = find(abs(gsw_cv.melting_ice_equilibrium_SA_CT_ratio - gsw_cf.melting_ice_equilibrium_SA_CT_ratio) >= gsw_cv.melting_ice_equilibrium_SA_CT_ratio_ca);
if ~isempty(gsw_cf.Imelting_ice_equilibrium_SA_CT_ratio)
    fprintf(2,'gsw_melting_ice_equilibrium_SA_CT_ratio:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.melting_ice_equilibrium_SA_CT_ratio_poly = gsw_melting_ice_equilibrium_SA_CT_ratio_poly(gsw_cv.SA_Arctic,gsw_cv.p_Arctic);
[gsw_cf.Imelting_ice_equilibrium_SA_CT_ratio_poly] = find(abs(gsw_cv.melting_ice_equilibrium_SA_CT_ratio_poly - gsw_cf.melting_ice_equilibrium_SA_CT_ratio_poly) >= gsw_cv.melting_ice_equilibrium_SA_CT_ratio_poly_ca);
if ~isempty(gsw_cf.Imelting_ice_equilibrium_SA_CT_ratio_poly)
    fprintf(2,'gsw_melting_ice_equilibrium_SA_CT_ratio_poly:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.melting_ice_into_seawater_SA_final, gsw_cf.melting_ice_into_seawater_CT_final] = gsw_melting_ice_into_seawater(gsw_cv.SA_Arctic,(gsw_cv.CT_Arctic+0.1),gsw_cv.p_Arctic,gsw_cv.w_ice,gsw_cv.t_ice);
[gsw_cf.Imelting_ice_into_seawater] = find(abs(gsw_cv.melting_ice_into_seawater_SA_final - gsw_cf.melting_ice_into_seawater_SA_final) >= gsw_cv.melting_ice_into_seawater_SA_final_ca | ...
    abs(gsw_cv.melting_ice_into_seawater_CT_final - gsw_cf.melting_ice_into_seawater_CT_final) >= gsw_cv.melting_ice_into_seawater_CT_final_ca);
if ~isempty(gsw_cf.Imelting_ice_into_seawater)
    fprintf(2,'gsw_melting_ice_into_seawater:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.ice_fraction_to_freeze_seawater_SA_freeze, gsw_cf.ice_fraction_to_freeze_seawater_CT_freeze, gsw_cf.ice_fraction_to_freeze_seawater_w_Ih]  = gsw_ice_fraction_to_freeze_seawater(gsw_cv.SA_Arctic,gsw_cv.CT_Arctic,gsw_cv.p_Arctic,gsw_cv.t_ice);
[gsw_cf.Iice_fraction_to_freeze_seawater] = find(abs(gsw_cv.ice_fraction_to_freeze_seawater_SA_freeze - gsw_cf.ice_fraction_to_freeze_seawater_SA_freeze) >= gsw_cv.ice_fraction_to_freeze_seawater_SA_freeze_ca | ...
    abs(gsw_cv.ice_fraction_to_freeze_seawater_CT_freeze - gsw_cf.ice_fraction_to_freeze_seawater_CT_freeze) >= gsw_cv.ice_fraction_to_freeze_seawater_CT_freeze_ca | ...
    abs(gsw_cv.ice_fraction_to_freeze_seawater_w_Ih - gsw_cf.ice_fraction_to_freeze_seawater_w_Ih) >= gsw_cv.ice_fraction_to_freeze_seawater_w_Ih_ca );
if ~isempty(gsw_cf.Imelting_ice_into_seawater)
    fprintf(2,'gsw_melting_ice_into_seawater:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.dSA_dCT_frazil, gsw_cf.dSA_dP_frazil, gsw_cf.dCT_dP_frazil] = gsw_frazil_ratios_adiabatic(gsw_cv.SA_Arctic,gsw_cv.p_Arctic,gsw_cv.w_ice);
[gsw_cf.Ifrazil_ratios] = find(abs(gsw_cv.dSA_dCT_frazil - gsw_cf.dSA_dCT_frazil) >= gsw_cv.dSA_dCT_frazil_ca | ...
    abs(gsw_cv.dSA_dP_frazil - gsw_cf.dSA_dP_frazil) >= gsw_cv.dSA_dP_frazil_ca | ...
    abs(gsw_cv.dCT_dP_frazil - gsw_cf.dCT_dP_frazil) >= gsw_cv.dCT_dP_frazil_ca );
if ~isempty(gsw_cf.Ifrazil_ratios)
    fprintf(2,'gsw_frazil_ratios_adiabatic:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.dSA_dCT_frazil_poly, gsw_cf.dSA_dP_frazil_poly, gsw_cf.dCT_dP_frazil_poly] = gsw_frazil_ratios_adiabatic_poly(gsw_cv.SA_Arctic,gsw_cv.p_Arctic,gsw_cv.w_ice);
[gsw_cf.Ifrazil_ratios_poly] = find(abs(gsw_cv.dSA_dCT_frazil_poly - gsw_cf.dSA_dCT_frazil_poly) >= gsw_cv.dSA_dCT_frazil_poly_ca | ...
    abs(gsw_cv.dSA_dP_frazil_poly - gsw_cf.dSA_dP_frazil_poly) >= gsw_cv.dSA_dP_frazil_poly_ca | ...
    abs(gsw_cv.dCT_dP_frazil_poly - gsw_cf.dCT_dP_frazil_poly) >= gsw_cv.dCT_dP_frazil_poly_ca );
if ~isempty(gsw_cf.Ifrazil_ratios_poly)
    fprintf(2,'gsw_frazil_ratios_adiabatic_poly:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.frazil_properties_potential_SA_final, gsw_cf.frazil_properties_potential_CT_final, gsw_cf.frazil_properties_potential_w_Ih_final] = gsw_frazil_properties_potential(gsw_cv.SA_bulk,gsw_cv.h_pot_bulk,gsw_cv.p_Arctic);
[gsw_cf.Ifrazil_properties_potential] = find(abs(gsw_cv.frazil_properties_potential_SA_final - gsw_cf.frazil_properties_potential_SA_final) >= gsw_cv.frazil_properties_potential_SA_final_ca | ...
    abs(gsw_cv.frazil_properties_potential_CT_final - gsw_cf.frazil_properties_potential_CT_final) >= gsw_cv.frazil_properties_potential_CT_final_ca | ...
    abs(gsw_cv.frazil_properties_potential_w_Ih_final - gsw_cf.frazil_properties_potential_w_Ih_final) >= gsw_cv.frazil_properties_potential_w_Ih_final_ca );
if ~isempty(gsw_cf.Ifrazil_properties_potential)
    fprintf(2,'gsw_frazil_properties_potential:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.frazil_properties_potential_poly_SA_final, gsw_cf.frazil_properties_potential_poly_CT_final, gsw_cf.frazil_properties_potential_poly_w_Ih_final] = gsw_frazil_properties_potential_poly(gsw_cv.SA_bulk,gsw_cv.h_pot_bulk,gsw_cv.p_Arctic);
[gsw_cf.Ifrazil_properties_potential_poly] = find(abs(gsw_cv.frazil_properties_potential_poly_SA_final - gsw_cf.frazil_properties_potential_poly_SA_final) >= gsw_cv.frazil_properties_potential_poly_SA_final_ca | ...
    abs(gsw_cv.frazil_properties_potential_poly_CT_final - gsw_cf.frazil_properties_potential_poly_CT_final) >= gsw_cv.frazil_properties_potential_poly_CT_final_ca | ...
    abs(gsw_cv.frazil_properties_potential_poly_w_Ih_final - gsw_cf.frazil_properties_potential_poly_w_Ih_final) >= gsw_cv.frazil_properties_potential_poly_w_Ih_final_ca );
if ~isempty(gsw_cf.Ifrazil_properties_potential_poly)
    fprintf(2,'gsw_frazil_properties_potential_poly:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.frazil_properties_SA_final, gsw_cf.frazil_properties_CT_final, gsw_cf.frazil_properties_w_Ih_final] = gsw_frazil_properties(gsw_cv.SA_bulk,gsw_cv.h_bulk,gsw_cv.p_Arctic);
[gsw_cf.Ifrazil_properties] = find(abs(gsw_cv.frazil_properties_SA_final - gsw_cf.frazil_properties_SA_final) >= gsw_cv.frazil_properties_SA_final_ca | ...
    abs(gsw_cv.frazil_properties_CT_final - gsw_cf.frazil_properties_CT_final) >= gsw_cv.frazil_properties_CT_final_ca | ...
    abs(gsw_cv.frazil_properties_w_Ih_final - gsw_cf.frazil_properties_w_Ih_final) >= gsw_cv.frazil_properties_w_Ih_final_ca );
if ~isempty(gsw_cf.Ifrazil_properties)
    fprintf(2,'gsw_frazil_properties:   Failed\n');
    gsw_chks = 0;
end

%%  thermodynamic interaction between sea ice and seawater

gsw_cf.melting_seaice_SA_CT_ratio = gsw_melting_seaice_SA_CT_ratio(gsw_cv.SA_Arctic,gsw_cv.CT_Arctic,gsw_cv.p_Arctic,gsw_cv.SA_seaice,gsw_cv.t_seaice);
[gsw_cf.Imelting_seaice_SA_CT_ratio] = find(abs(gsw_cv.melting_seaice_SA_CT_ratio - gsw_cf.melting_seaice_SA_CT_ratio) >= gsw_cv.melting_seaice_SA_CT_ratio_ca);
if ~isempty(gsw_cf.Imelting_seaice_SA_CT_ratio)
    fprintf(2,'gsw_melting_seaice_SA_CT_ratio:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.melting_seaice_SA_CT_ratio_poly = gsw_melting_seaice_SA_CT_ratio_poly(gsw_cv.SA_Arctic,gsw_cv.CT_Arctic,gsw_cv.p_Arctic,gsw_cv.SA_seaice,gsw_cv.t_seaice);
[gsw_cf.Imelting_seaice_SA_CT_ratio_poly] = find(abs(gsw_cv.melting_seaice_SA_CT_ratio_poly - gsw_cf.melting_seaice_SA_CT_ratio_poly) >= gsw_cv.melting_seaice_SA_CT_ratio_poly_ca);
if ~isempty(gsw_cf.Imelting_seaice_SA_CT_ratio_poly)
    fprintf(2,'gsw_melting_seaice_SA_CT_ratio_poly:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.melting_seaice_equilibrium_SA_CT_ratio = gsw_melting_seaice_equilibrium_SA_CT_ratio(gsw_cv.SA_Arctic,gsw_cv.p_Arctic);
[gsw_cf.Imelting_seaice_equilibrium_SA_CT_ratio] = find(abs(gsw_cv.melting_seaice_equilibrium_SA_CT_ratio - gsw_cf.melting_seaice_equilibrium_SA_CT_ratio) >= gsw_cv.melting_seaice_equilibrium_SA_CT_ratio_ca);
if ~isempty(gsw_cf.Imelting_seaice_equilibrium_SA_CT_ratio)
    fprintf(2,'gsw_melting_seaice_equilibrium_SA_CT_ratio:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.melting_seaice_equilibrium_SA_CT_ratio_poly = gsw_melting_seaice_equilibrium_SA_CT_ratio_poly(gsw_cv.SA_Arctic,gsw_cv.p_Arctic);
[gsw_cf.Imelting_seaice_equilibrium_SA_CT_ratio_poly] = find(abs(gsw_cv.melting_seaice_equilibrium_SA_CT_ratio_poly - gsw_cf.melting_seaice_equilibrium_SA_CT_ratio_poly) >= gsw_cv.melting_seaice_equilibrium_SA_CT_ratio_poly_ca);
if ~isempty(gsw_cf.Imelting_seaice_equilibrium_SA_CT_ratio_poly)
    fprintf(2,'gsw_melting_seaice_equilibrium_SA_CT_ratio_poly:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.melting_seaice_into_seawater_SA_final, gsw_cf.melting_seaice_into_seawater_CT_final] = gsw_melting_seaice_into_seawater(gsw_cv.SA_Arctic,gsw_cv.CT_Arctic,gsw_cv.p_Arctic,gsw_cv.w_seaice,gsw_cv.SA_seaice,gsw_cv.t_seaice);
[gsw_cf.Imelting_seaice_into_seawater] = find(abs(gsw_cv.melting_seaice_into_seawater_SA_final - gsw_cf.melting_seaice_into_seawater_SA_final) >= gsw_cv.melting_seaice_into_seawater_SA_final_ca | ...
    abs(gsw_cv.melting_seaice_into_seawater_CT_final - gsw_cf.melting_seaice_into_seawater_CT_final) >= gsw_cv.melting_seaice_into_seawater_CT_final_ca);
if ~isempty(gsw_cf.Imelting_seaice_into_seawater)
    fprintf(2,'gsw_melting_seaice_into_seawater:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.seaice_fraction_to_freeze_seawater_SA_freeze, gsw_cf.seaice_fraction_to_freeze_seawater_CT_freeze, gsw_cf.seaice_fraction_to_freeze_seawater_w_Ih]  = gsw_seaice_fraction_to_freeze_seawater(gsw_cv.SA_Arctic,gsw_cv.CT_Arctic,gsw_cv.p_Arctic,gsw_cv.SA_seaice,gsw_cv.t_seaice);
[gsw_cf.Iseaice_fraction_to_freeze_seawater] = find(abs(gsw_cv.seaice_fraction_to_freeze_seawater_SA_freeze - gsw_cf.seaice_fraction_to_freeze_seawater_SA_freeze) >= gsw_cv.seaice_fraction_to_freeze_seawater_SA_freeze_ca | ...
    abs(gsw_cv.seaice_fraction_to_freeze_seawater_CT_freeze - gsw_cf.seaice_fraction_to_freeze_seawater_CT_freeze) >= gsw_cv.seaice_fraction_to_freeze_seawater_CT_freeze_ca | ...
    abs(gsw_cv.seaice_fraction_to_freeze_seawater_w_Ih - gsw_cf.seaice_fraction_to_freeze_seawater_w_Ih) >= gsw_cv.seaice_fraction_to_freeze_seawater_w_Ih_ca );
if ~isempty(gsw_cf.Imelting_seaice_into_seawater)
    fprintf(2,'gsw_melting_seaice_into_seawater:   Failed\n');
    gsw_chks = 0;
end

%% themodynamic properties of ice Ih

gsw_cf.rho_ice = gsw_rho_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
[gsw_cf.Irho_ice] = find(abs(gsw_cv.rho_ice - gsw_cf.rho_ice) >= gsw_cv.rho_ice_ca);
if ~isempty(gsw_cf.Irho_ice)
    fprintf(2,'gsw_rho_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.alpha_wrt_t_ice = gsw_alpha_wrt_t_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
[gsw_cf.Ialpha_wrt_t_ice] = find(abs(gsw_cv.alpha_wrt_t_ice - gsw_cf.alpha_wrt_t_ice) >= gsw_cv.alpha_wrt_t_ice_ca);
if ~isempty(gsw_cf.Ialpha_wrt_t_ice)
    fprintf(2,'gsw_alpha_wrt_t_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.specvol_ice = gsw_specvol_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
[gsw_cf.Ispecvol_ice] = find(abs(gsw_cv.specvol_ice - gsw_cf.specvol_ice) >= gsw_cv.specvol_ice_ca);
if ~isempty(gsw_cf.Ispecvol_ice)
    fprintf(2,'gsw_specvol_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.pressure_coefficient_ice = gsw_pressure_coefficient_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
[gsw_cf.Ipressure_coefficient_ice] = find(abs(gsw_cv.pressure_coefficient_ice - gsw_cf.pressure_coefficient_ice) >= gsw_cv.pressure_coefficient_ice_ca);
if ~isempty(gsw_cf.Ipressure_coefficient_ice)
    fprintf(2,'gsw_pressure_coefficient_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.sound_speed_ice = gsw_sound_speed_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
[gsw_cf.Isound_speed_ice] = find(abs(gsw_cv.sound_speed_ice - gsw_cf.sound_speed_ice) >= gsw_cv.sound_speed_ice_ca);
if ~isempty(gsw_cf.Isound_speed_ice)
    fprintf(2,'gsw_sound_speed_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.kappa_ice = gsw_kappa_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
[gsw_cf.Ikappa_ice] = find(abs(gsw_cv.kappa_ice - gsw_cf.kappa_ice) >= gsw_cv.kappa_ice_ca);
if ~isempty(gsw_cf.Ikappa_ice)
    fprintf(2,'gsw_kappa_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.kappa_const_t_ice = gsw_kappa_const_t_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
[gsw_cf.Ikappa_const_t_ice] = find(abs(gsw_cv.kappa_const_t_ice - gsw_cf.kappa_const_t_ice) >= gsw_cv.kappa_const_t_ice_ca);
if ~isempty(gsw_cf.Ikappa_const_t_ice)
    fprintf(2,'gsw_kappa_const_t_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.internal_energy_ice = gsw_internal_energy_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
[gsw_cf.Iinternal_energy_ice] = find(abs(gsw_cv.internal_energy_ice - gsw_cf.internal_energy_ice) >= gsw_cv.internal_energy_ice_ca);
if ~isempty(gsw_cf.Iinternal_energy_ice)
    fprintf(2,'gsw_internal_energy_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.enthalpy_ice = gsw_enthalpy_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
 [gsw_cf.Ienthalpy_ice] = find(abs(gsw_cv.enthalpy_ice - gsw_cf.enthalpy_ice) >= gsw_cv.enthalpy_ice_ca);
if ~isempty(gsw_cf.Ienthalpy_ice)
    fprintf(2,'gsw_enthalpy_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.entropy_ice = gsw_entropy_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
[gsw_cf.Ientropy_ice] = find(abs(gsw_cv.entropy_ice - gsw_cf.entropy_ice) >= gsw_cv.entropy_ice_ca);
if ~isempty(gsw_cf.Ientropy_ice)
    fprintf(2,'gsw_entropy_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.cp_ice = gsw_cp_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
 [gsw_cf.Icp_ice] = find(abs(gsw_cv.cp_ice - gsw_cf.cp_ice) >= gsw_cv.cp_ice_ca);
if ~isempty(gsw_cf.Icp_ice)
    fprintf(2,'gsw_cp_ice:   Failed\n');
    gsw_chks = 0;
end

%%% EF: another bug is fixed in the block below.

gsw_cf.chem_potential_water_ice = gsw_chem_potential_water_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
 [gsw_cf.Ichem_potential_water_ice] = find(abs(gsw_cv.chem_potential_water_ice - gsw_cf.chem_potential_water_ice) >= gsw_cv.chem_potential_water_ice_ca);
if ~isempty(gsw_cf.Ichem_potential_water_ice)
    fprintf(2,'gsw_chem_potential_water_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.Helmholtz_energy_ice = gsw_Helmholtz_energy_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
[gsw_cf.IHelmholtz_energy_ice] = find(abs(gsw_cv.Helmholtz_energy_ice - gsw_cf.Helmholtz_energy_ice) >= gsw_cv.Helmholtz_energy_ice_ca);
if ~isempty(gsw_cf.IHelmholtz_energy_ice)
    fprintf(2,'gsw_Helmholtz_energy_ice:   Failed\n');
    gsw_chks = 0;
end

%%% EF: and here
gsw_cf.adiabatic_lapse_rate_ice = gsw_adiabatic_lapse_rate_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
[gsw_cf.Iadiabatic_lapse_rate_ice] = find(abs(gsw_cv.adiabatic_lapse_rate_ice - gsw_cf.adiabatic_lapse_rate_ice) >= gsw_cv.adiabatic_lapse_rate_ice_ca);
if ~isempty(gsw_cf.Iadiabatic_lapse_rate_ice)
    fprintf(2,'gsw_adiabatic_lapse_rate_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.pt0_from_t_ice = gsw_pt0_from_t_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic);
[gsw_cf.Ipt0_from_t_ice] = find(abs(gsw_cv.pt0_from_t_ice - gsw_cf.pt0_from_t_ice) >= gsw_cv.pt0_from_t_ice_ca);
if ~isempty(gsw_cf.Ipt0_from_t_ice)
    fprintf(2,'gsw_pt0_from_t_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.pt_from_t_ice = gsw_pt_from_t_ice(gsw_cv.t_seaice,gsw_cv.p_Arctic,gsw_cv.pr);
[gsw_cf.Ipt_from_t_ice] = find(abs(gsw_cv.pt_from_t_ice - gsw_cf.pt_from_t_ice) >= gsw_cv.pt_from_t_ice_ca);
if ~isempty(gsw_cf.Ipt_from_t_ice)
    fprintf(2,'gsw_pt_from_t_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.t_from_pt0_ice = gsw_t_from_pt0_ice(gsw_cv.pt0_from_t_ice,gsw_cv.p_Arctic);
[gsw_cf.It_from_pt0_ice] = find(abs(gsw_cv.t_from_pt0_ice - gsw_cf.t_from_pt0_ice) >= gsw_cv.t_from_pt0_ice_ca);
if ~isempty(gsw_cf.It_from_pt0_ice)
    fprintf(2,'gsw_t_from_pt0_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.t_from_rho_ice = gsw_t_from_rho_ice(gsw_cv.rho_ice,gsw_cv.p_Arctic);
[gsw_cf.It_from_rho_ice] = find(abs(gsw_cv.t_from_rho_ice - gsw_cf.t_from_rho_ice) >= gsw_cv.t_from_rho_ice_ca);
if ~isempty(gsw_cf.It_from_rho_ice)
    fprintf(2,'gsw_t_from_rho_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.pot_enthalpy_from_pt_ice = gsw_pot_enthalpy_from_pt_ice(gsw_cv.pt0_from_t_ice);
[gsw_cf.Ipot_enthalpy_from_pt_ice] = find(abs(gsw_cv.pot_enthalpy_from_pt_ice - gsw_cf.pot_enthalpy_from_pt_ice) >= gsw_cv.pot_enthalpy_from_pt_ice_ca);
if ~isempty(gsw_cf.Ipot_enthalpy_from_pt_ice)
    fprintf(2,'gsw_pot_enthalpy_from_pt_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.pt_from_pot_enthalpy_ice = gsw_pt_from_pot_enthalpy_ice (gsw_cv.pot_enthalpy_from_pt_ice);
[gsw_cf.Ipt_from_pot_enthalpy_ice] = find(abs(gsw_cv.pt_from_pot_enthalpy_ice - gsw_cf.pt_from_pot_enthalpy_ice) >= gsw_cv.pt_from_pot_enthalpy_ice_ca);
if ~isempty(gsw_cf.Ipt_from_pot_enthalpy_ice)
    fprintf(2,'gsw_pt_from_pot_enthalpy_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.pot_enthalpy_from_pt_ice_poly = gsw_pot_enthalpy_from_pt_ice_poly(gsw_cv.pt0_from_t_ice);
[gsw_cf.Ipot_enthalpy_from_pt_ice_poly] = find(abs(gsw_cv.pot_enthalpy_from_pt_ice_poly - gsw_cf.pot_enthalpy_from_pt_ice_poly) >= gsw_cv.pot_enthalpy_from_pt_ice_poly_ca);
if ~isempty(gsw_cf.Ipot_enthalpy_from_pt_ice_poly)
    fprintf(2,'gsw_pot_enthalpy_from_pt_ice_poly:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.pt_from_pot_enthalpy_ice_poly = gsw_pt_from_pot_enthalpy_ice_poly(gsw_cv.pot_enthalpy_from_pt_ice_poly);
[gsw_cf.Ipt_from_pot_enthalpy_ice_poly] = find(abs(gsw_cv.pt_from_pot_enthalpy_ice_poly - gsw_cf.pt_from_pot_enthalpy_ice_poly) >= gsw_cv.pt_from_pot_enthalpy_ice_poly_ca);
if ~isempty(gsw_cf.Ipt_from_pot_enthalpy_ice_poly)
    fprintf(2,'gsw_pt_from_pot_enthalpy_ice_poly:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.pot_enthalpy_from_specvol_ice = gsw_pot_enthalpy_from_specvol_ice(gsw_cv.specvol_ice,gsw_cv.p_Arctic);
[gsw_cf.Ipot_enthalpy_from_specvol_ice] = find(abs(gsw_cv.pot_enthalpy_from_specvol_ice - gsw_cf.pot_enthalpy_from_specvol_ice) >= gsw_cv.pot_enthalpy_from_pt_ice_ca);
if ~isempty(gsw_cf.Ipot_enthalpy_from_specvol_ice)
    fprintf(2,'gsw_pot_enthalpy_from_specvol_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.specvol_from_pot_enthalpy_ice = gsw_specvol_from_pot_enthalpy_ice(gsw_cv.pot_enthalpy_from_specvol_ice,gsw_cv.p_Arctic);
[gsw_cf.Ispecvol_from_pot_enthalpy_ice] = find(abs(gsw_cv.specvol_from_pot_enthalpy_ice - gsw_cf.specvol_from_pot_enthalpy_ice) >= gsw_cv.pt_from_pot_enthalpy_ice_ca);
if ~isempty(gsw_cf.Ispecvol_from_pot_enthalpy_ice)
    fprintf(2,'gsw_specvol_from_pot_enthalpy_ice:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.pot_enthalpy_from_specvol_ice_poly = gsw_pot_enthalpy_from_specvol_ice_poly(gsw_cv.specvol_ice,gsw_cv.p_Arctic);
[gsw_cf.Ipot_enthalpy_from_specvol_ice_poly] = find(abs(gsw_cv.pot_enthalpy_from_specvol_ice_poly - gsw_cf.pot_enthalpy_from_specvol_ice_poly) >= gsw_cv.pot_enthalpy_from_pt_ice_poly_ca);
if ~isempty(gsw_cf.Ipot_enthalpy_from_specvol_ice_poly)
    fprintf(2,'gsw_pot_enthalpy_from_specvol_ice_poly:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.specvol_from_pot_enthalpy_ice_poly = gsw_specvol_from_pot_enthalpy_ice_poly(gsw_cv.pot_enthalpy_from_specvol_ice_poly,gsw_cv.p_Arctic);
[gsw_cf.Ispecvol_from_pot_enthalpy_ice_poly] = find(abs(gsw_cv.specvol_from_pot_enthalpy_ice_poly - gsw_cf.specvol_from_pot_enthalpy_ice_poly) >= gsw_cv.pt_from_pot_enthalpy_ice_poly_ca);
if ~isempty(gsw_cf.Ispecvol_from_pot_enthalpy_ice_poly)
    fprintf(2,'gsw_specvol_from_pot_enthalpy_ice_poly:   Failed\n');
    gsw_chks = 0;
end

%% isobaric evaporation enthalpy

gsw_cf.latentheat_evap_CT = gsw_latentheat_evap_CT(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Ilatentheat_evap_CT] = find(abs(gsw_cv.latentheat_evap_CT - gsw_cf.latentheat_evap_CT) >= gsw_cv.latentheat_evap_CT_ca);
if ~isempty(gsw_cf.Ilatentheat_evap_CT)
    fprintf(2,'gsw_latentheat_evap_CT:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.latentheat_evap_t = gsw_latentheat_evap_t(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast);
[gsw_cf.Ilatentheat_evap_t] = find(abs(gsw_cv.latentheat_evap_t - gsw_cf.latentheat_evap_t) >= gsw_cv.latentheat_evap_t_ca);
if ~isempty(gsw_cf.Ilatentheat_evap_t)
    fprintf(2,'gsw_latentheat_evap_t:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%% spiciness

gsw_cf.spiciness0 = gsw_spiciness0(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Ispiciness0] = find(abs(gsw_cv.spiciness0 - gsw_cf.spiciness0) >= gsw_cv.spiciness0_ca);
if ~isempty(gsw_cf.Ispiciness0)
    fprintf(2,'gsw_spiciness0:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.spiciness1 = gsw_spiciness1(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Ispiciness1] = find(abs(gsw_cv.spiciness1 - gsw_cf.spiciness1) >= gsw_cv.spiciness1_ca);
if ~isempty(gsw_cf.Ispiciness1)
    fprintf(2,'gsw_spiciness1:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.spiciness2 = gsw_spiciness2(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Ispiciness2] = find(abs(gsw_cv.spiciness2 - gsw_cf.spiciness2) >= gsw_cv.spiciness2_ca);
if ~isempty(gsw_cf.Ispiciness2)
    fprintf(2,'gsw_spiciness2:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%% planet earth properties

gsw_cf.f = gsw_f(gsw_cv.lat_chck_cast);
[gsw_cf.If] = find(abs(gsw_cv.f - gsw_cf.f) >= gsw_cv.f_ca);
if ~isempty(gsw_cf.If)
    fprintf(2,'gsw_f:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.grav = gsw_grav(gsw_cv.lat_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Igrav] = find(abs(gsw_cv.grav - gsw_cf.grav) >= gsw_cv.grav_ca);
if ~isempty(gsw_cf.Igrav)
    fprintf(2,'gsw_grav:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.distance = gsw_distance(gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Idistance] = find(abs(gsw_cv.distance - gsw_cf.distance) >= gsw_cv.distance_ca);
if ~isempty(gsw_cf.Idistance)
    fprintf(2,'gsw_distance:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%% TEOS-10 constants

gsw_cf.T0 = gsw_T0;
[gsw_cf.IT0] = find(abs(gsw_cv.T0 - gsw_cf.T0) > 1e-13);
if ~isempty(gsw_cf.IT0)
    fprintf(2,'gsw_T0:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.P0 = gsw_P0;
[gsw_cf.IP0] = find(abs(gsw_cv.P0 - gsw_cf.P0) > 1e-13);
if ~isempty(gsw_cf.IP0)
    fprintf(2,'gsw_P0:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SSO = gsw_SSO;
[gsw_cf.ISSO] = find(abs(gsw_cv.SSO - gsw_cf.SSO) > 1e-13);
if ~isempty(gsw_cf.ISSO)
    fprintf(2,'gsw_SSO:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.uPS = gsw_uPS;
[gsw_cf.IuPS] = find(abs(gsw_cv.uPS - gsw_cf.uPS) > 1e-13);
if ~isempty(gsw_cf.IuPS)
    fprintf(2,'gsw_uPS:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.cp0 = gsw_cp0;
[gsw_cf.Icp0] = find(abs(gsw_cv.cp0 - gsw_cf.cp0) > 1e-13);
if ~isempty(gsw_cf.Icp0)
    fprintf(2,'gsw_cp0:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.C3515 = gsw_C3515;
[gsw_cf.IC3515] = find(abs(gsw_cv.C3515 - gsw_cf.C3515) > 1e-13);
if ~isempty(gsw_cf.IC3515)
    fprintf(2,'gsw_C3515:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SonCl = gsw_SonCl;
[gsw_cf.ISonCl] = find(abs(gsw_cv.SonCl - gsw_cf.SonCl) > 1e-13);
if ~isempty(gsw_cf.ISonCl)
    fprintf(2,'gsw_SonCl:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.valence_factor = gsw_valence_factor;
[gsw_cf.Ivalence_factor] = find(abs(gsw_cv.valence_factor - gsw_cf.valence_factor) > 1e-13);
if ~isempty(gsw_cf.Ivalence_factor)
    fprintf(2,'gsw_valence_factor:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.atomic_weight = gsw_atomic_weight;
[gsw_cf.Iatomic_weight] = find(abs(gsw_cv.atomic_weight - gsw_cf.atomic_weight) > 1e-13);
if ~isempty(gsw_cf.Iatomic_weight)
    fprintf(2,'gsw_atomic_weight:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%% dissolved gasses

gsw_cf.Arsol = gsw_Arsol(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.IArsol] = find(abs(gsw_cv.Arsol - gsw_cf.Arsol) >= gsw_cv.Arsol_ca);
if ~isempty(gsw_cf.IArsol)
    fprintf(2,'gsw_Arsol:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.Arsol_SP_pt = gsw_Arsol_SP_pt(gsw_cv.SP_chck_cast,gsw_cv.pt0_from_t);
[gsw_cf.IArsol_SP_pt] = find(abs(gsw_cv.Arsol_SP_pt - gsw_cf.Arsol_SP_pt) >= gsw_cv.Arsol_SP_pt_ca);
if ~isempty(gsw_cf.IArsol_SP_pt)
    fprintf(2,'gsw_Arsol_SP_pt:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.Hesol = gsw_Hesol(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.IHesol] = find(abs(gsw_cv.Hesol - gsw_cf.Hesol) >= gsw_cv.Hesol_ca);
if ~isempty(gsw_cf.IHesol)
    fprintf(2,'gsw_Hesol:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.Hesol_SP_pt = gsw_Hesol_SP_pt(gsw_cv.SP_chck_cast,gsw_cv.pt0_from_t);
[gsw_cf.IHesol_SP_pt] = find(abs(gsw_cv.Hesol_SP_pt - gsw_cf.Hesol_SP_pt) >= gsw_cv.Hesol_SP_pt_ca);
if ~isempty(gsw_cf.IHesol_SP_pt)
    fprintf(2,'gsw_Hesol_SP_pt:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.Krsol = gsw_Krsol(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.IKrsol] = find(abs(gsw_cv.Krsol - gsw_cf.Krsol) >= gsw_cv.Krsol_ca);
if ~isempty(gsw_cf.IKrsol)
    fprintf(2,'gsw_Krsol:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.Krsol_SP_pt = gsw_Krsol_SP_pt(gsw_cv.SP_chck_cast,gsw_cv.pt0_from_t);
[gsw_cf.IKrsol_SP_pt] = find(abs(gsw_cv.Krsol_SP_pt - gsw_cf.Krsol_SP_pt) >= gsw_cv.Krsol_SP_pt_ca);
if ~isempty(gsw_cf.IKrsol_SP_pt)
    fprintf(2,'gsw_Krsol_SP_pt:   Failed\n');
    gsw_chks = 0;
end

% gsw_cf.N2Osol = gsw_N2Osol(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
% [gsw_cf.IN2Osol] = find(abs(gsw_cv.N2Osol - gsw_cf.N2Osol) >= gsw_cv.N2Osol_ca);
% if ~isempty(gsw_cf.IN2Osol)
%     fprintf(2,'gsw_N2Osol:   Failed\n');
%     gsw_chks = 0;
% end

% gsw_cf.N2Osol_SP_pt = gsw_N2Osol_SP_pt(gsw_cv.SP_chck_cast,gsw_cv.pt0_from_t);
% [gsw_cf.IN2Osol_SP_pt] = find(abs(gsw_cv.N2Osol_SP_pt - gsw_cf.N2Osol_SP_pt) >= gsw_cv.N2Osol_SP_pt_ca);
% if ~isempty(gsw_cf.IN2Osol_SP_pt)
%     fprintf(2,'gsw_N2Osol_SP_pt:   Failed\n');
%     gsw_chks = 0;
% end

gsw_cf.N2sol = gsw_N2sol(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.IN2sol] = find(abs(gsw_cv.N2sol - gsw_cf.N2sol) >= gsw_cv.N2sol_ca);
if ~isempty(gsw_cf.IN2sol)
    fprintf(2,'gsw_N2sol:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.N2sol_SP_pt = gsw_N2sol_SP_pt(gsw_cv.SP_chck_cast,gsw_cv.pt0_from_t);
[gsw_cf.IN2sol_SP_pt] = find(abs(gsw_cv.N2sol_SP_pt - gsw_cf.N2sol_SP_pt) >= gsw_cv.N2sol_SP_pt_ca);
if ~isempty(gsw_cf.IN2sol_SP_pt)
    fprintf(2,'gsw_N2sol_SP_pt:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.Nesol = gsw_Nesol(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.INesol] = find(abs(gsw_cv.Nesol - gsw_cf.Nesol) >= gsw_cv.Nesol_ca);
if ~isempty(gsw_cf.INesol)
    fprintf(2,'gsw_Nesol:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.Nesol_SP_pt = gsw_Nesol_SP_pt(gsw_cv.SP_chck_cast,gsw_cv.pt0_from_t);
[gsw_cf.INesol_SP_pt] = find(abs(gsw_cv.Nesol_SP_pt - gsw_cf.Nesol_SP_pt) >= gsw_cv.Nesol_SP_pt_ca);
if ~isempty(gsw_cf.INesol_SP_pt)
    fprintf(2,'gsw_Nesol_SP_pt:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.O2sol = gsw_O2sol(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.IO2sol] = find(abs(gsw_cv.O2sol - gsw_cf.O2sol) >= gsw_cv.O2sol_ca);
if ~isempty(gsw_cf.IO2sol)
    fprintf(2,'gsw_O2sol:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.O2sol_SP_pt = gsw_O2sol_SP_pt(gsw_cv.SP_chck_cast,gsw_cv.pt0_from_t);
[gsw_cf.IO2sol_SP_pt] = find(abs(gsw_cv.O2sol_SP_pt - gsw_cf.O2sol_SP_pt) >= gsw_cv.O2sol_SP_pt_ca);
if ~isempty(gsw_cf.IO2sol_SP_pt)
    fprintf(2,'gsw_O2sol_SP_pt:   Failed\n');
    gsw_chks = 0;
end

%% density and enthalpy in terms of CT, derived from the exact Gibbs function

gsw_cf.specvol_CT_exact = gsw_specvol_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ispecvol_CT_exact] = find(abs(gsw_cv.specvol_CT_exact - gsw_cf.specvol_CT_exact) >= gsw_cv.specvol_CT_exact_ca);
if ~isempty(gsw_cf.Ispecvol_CT_exact)
    fprintf(2,'gsw_specvol_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.alpha_CT_exact = gsw_alpha_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ialpha_CT_exact] = find(abs(gsw_cv.alpha_CT_exact - gsw_cf.alpha_CT_exact) >= gsw_cv.alpha_CT_exact_ca);
if ~isempty(gsw_cf.Ialpha_CT_exact)
    fprintf(2,'gsw_alpha_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.beta_CT_exact = gsw_beta_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ibeta_CT_exact] = find(abs(gsw_cv.beta_CT_exact - gsw_cf.beta_CT_exact) >= gsw_cv.beta_CT_exact_ca);
if ~isempty(gsw_cf.Ibeta_CT_exact)
    fprintf(2,'gsw_beta_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.alpha_on_beta_CT_exact = gsw_alpha_on_beta_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ialpha_on_beta_CT_exact] = find(abs(gsw_cv.alpha_on_beta_CT_exact - gsw_cf.alpha_on_beta_CT_exact) >= gsw_cv.alpha_on_beta_CT_exact_ca);
if ~isempty(gsw_cf.Ialpha_on_beta_CT_exact)
    fprintf(2,'gsw_alpha_on_beta_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.v_vab_CT_exact, gsw_cf.alpha_vab_CT_exact, gsw_cf.beta_vab_CT_exact] = gsw_specvol_alpha_beta_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iv_vab_CT_exact] = find(abs(gsw_cv.v_vab_CT_exact - gsw_cf.v_vab_CT_exact) >= gsw_cv.v_vab_CT_exact_ca | ...
    abs(gsw_cv.alpha_vab_CT_exact - gsw_cf.alpha_vab_CT_exact) >= gsw_cv.alpha_vab_CT_exact_ca | ...
    abs(gsw_cv.beta_vab_CT_exact - gsw_cf.beta_vab_CT_exact) >= gsw_cv.beta_vab_CT_exact_ca);
if ~isempty(gsw_cf.Iv_vab_CT_exact)
    fprintf(2,'gsw_specvol_alpha_beta_CT_exact:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.v_SA_CT_exact, gsw_cf.v_CT_CT_exact, gsw_cf.v_P_CT_exact] = gsw_specvol_first_derivatives_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iv_fd_CT_exact] = find(abs(gsw_cv.v_SA_CT_exact - gsw_cf.v_SA_CT_exact) >= gsw_cv.v_SA_CT_exact_ca | ...
    abs(gsw_cv.v_CT_CT_exact - gsw_cf.v_CT_CT_exact) >= gsw_cv.v_CT_CT_exact_ca | ...
    abs(gsw_cv.v_P_CT_exact - gsw_cf.v_P_CT_exact) >= gsw_cv.v_P_CT_exact_ca);
if ~isempty(gsw_cf.Iv_fd_CT_exact)
    fprintf(2,'gsw_specvol_first_derivatives_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.v_SA_SA_CT_exact, gsw_cf.v_SA_CT_CT_exact, gsw_cf.v_CT_CT_CT_exact, gsw_cf.v_SA_P_CT_exact, gsw_cf.v_CT_P_CT_exact] = gsw_specvol_second_derivatives_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iv_sd_CT_exact] = find(abs(gsw_cv.v_SA_SA_CT_exact - gsw_cf.v_SA_SA_CT_exact) >= gsw_cv.v_SA_SA_CT_exact_ca | ...
    abs(gsw_cv.v_SA_CT_CT_exact - gsw_cf.v_SA_CT_CT_exact) >= gsw_cv.v_SA_CT_CT_exact_ca | ...
    abs(gsw_cv.v_CT_CT_CT_exact - gsw_cf.v_CT_CT_CT_exact) >= gsw_cv.v_CT_CT_CT_exact_ca | ...
    abs(gsw_cv.v_SA_P_CT_exact - gsw_cf.v_SA_P_CT_exact) >= gsw_cv.v_SA_P_CT_exact_ca | ...
    abs(gsw_cv.v_CT_P_CT_exact - gsw_cf.v_CT_P_CT_exact) >= gsw_cv.v_CT_P_CT_exact_ca);
if ~isempty(gsw_cf.Iv_sd_CT_exact)
    fprintf(2,'gsw_specvol_second_derivatives_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.v_SA_wrt_h_CT_exact, gsw_cf.v_h_CT_exact] = gsw_specvol_first_derivatives_wrt_enthalpy_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iv_fd_wrt_h_CT_exact] = find(abs(gsw_cv.v_SA_wrt_h_CT_exact - gsw_cf.v_SA_wrt_h_CT_exact) >= gsw_cv.v_SA_wrt_h_CT_exact_ca | ...
    abs(gsw_cv.v_h_CT_exact - gsw_cf.v_h_CT_exact) >= gsw_cv.v_h_CT_exact_ca);
if ~isempty(gsw_cf.Iv_fd_wrt_h_CT_exact)
    fprintf(2,'gsw_specvol_first_derivatives_wrt_enthalpy_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.v_SA_SA_wrt_h_CT_exact, gsw_cf.v_SA_h_CT_exact, gsw_cf.v_h_h_CT_exact] = gsw_specvol_second_derivatives_wrt_enthalpy_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iv_sd_wrt_h_CT_exact] = find(abs(gsw_cv.v_SA_SA_CT_exact - gsw_cf.v_SA_SA_CT_exact) >= gsw_cv.v_SA_SA_CT_exact_ca | ...
    abs(gsw_cv.v_SA_CT_CT_exact - gsw_cf.v_SA_CT_CT_exact) >= gsw_cv.v_SA_CT_CT_exact_ca | ...
    abs(gsw_cv.v_CT_CT_CT_exact - gsw_cf.v_CT_CT_CT_exact) >= gsw_cv.v_CT_CT_CT_exact_ca | ...
    abs(gsw_cv.v_SA_P_CT_exact - gsw_cf.v_SA_P_CT_exact) >= gsw_cv.v_SA_P_CT_exact_ca | ...
    abs(gsw_cv.v_CT_P_CT_exact - gsw_cf.v_CT_P_CT_exact) >= gsw_cv.v_CT_P_CT_exact_ca);
if ~isempty(gsw_cf.Iv_sd_wrt_h_CT_exact)
    fprintf(2,'gsw_specvol_second_derivatives_wrt_enthalpy_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.specvol_anom_CT_exact = gsw_specvol_anom_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast,gsw_cv.SA_ref,gsw_cv.CT_ref);
[gsw_cf.Ispecvol_anom_CT_exact] = find(abs(gsw_cv.specvol_anom_CT_exact - gsw_cf.specvol_anom_CT_exact) >= gsw_cv.specvol_anom_CT_exact_ca);
if ~isempty(gsw_cf.Ispecvol_anom_CT_exact)
    fprintf(2,'gsw_specvol_anom_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.specvol_anom_standard_CT_exact = gsw_specvol_anom_standard_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ispecvol_anom_standard_CT_exact] = find(abs(gsw_cv.specvol_anom_standard_CT_exact - gsw_cf.specvol_anom_standard_CT_exact) >= gsw_cv.specvol_anom_standard_CT_exact_ca);
if ~isempty(gsw_cf.Ispecvol_anom_standard_CT_exact)
    fprintf(2,'gsw_specvol_anom_standard_CT_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.rho_CT_exact = gsw_rho_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho_CT_exact] = find(abs(gsw_cv.rho_CT_exact - gsw_cf.rho_CT_exact) >= gsw_cv.rho_CT_exact_ca);
if ~isempty(gsw_cf.Irho_CT_exact)
    fprintf(2,'gsw_rho_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.rho_rab_CT_exact, gsw_cf.alpha_rab_CT_exact, gsw_cf.beta_rab_CT_exact] = gsw_rho_alpha_beta_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho_rab_CT_exact] = find(abs(gsw_cv.rho_rab_CT_exact - gsw_cf.rho_rab_CT_exact) >= gsw_cv.rho_rab_CT_exact_ca | ...
    abs(gsw_cv.alpha_rab_CT_exact - gsw_cf.alpha_rab_CT_exact) >= gsw_cv.alpha_rab_CT_exact_ca | ...
    abs(gsw_cv.beta_rab_CT_exact - gsw_cf.beta_rab_CT_exact) >= gsw_cv.beta_rab_CT_exact_ca);
if ~isempty(gsw_cf.Irho_rab_CT_exact)
    fprintf(2,'gsw_rho_alpha_beta_CT_exact:   Failed\n');
    gsw_chks = 0;
end

[gsw_cf.rho_SA_CT_exact, gsw_cf.rho_CT_CT_exact, gsw_cf.rho_P_CT_exact] = gsw_rho_first_derivatives_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho_fd_CT_exact] = find(abs(gsw_cv.rho_SA_CT_exact - gsw_cf.rho_SA_CT_exact) >= gsw_cv.rho_SA_CT_exact_ca | ...
    abs(gsw_cv.rho_CT_CT_exact - gsw_cf.rho_CT_CT_exact) >= gsw_cv.rho_CT_CT_exact_ca | ...
    abs(gsw_cv.rho_P_CT_exact - gsw_cf.rho_P_CT_exact) >= gsw_cv.rho_P_CT_exact_ca);
if ~isempty(gsw_cf.Irho_fd_CT_exact)
    fprintf(2,'gsw_rho_first_derivatives_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.rho_SA_SA_CT_exact, gsw_cf.rho_SA_CT_CT_exact, gsw_cf.rho_CT_CT_CT_exact, gsw_cf.rho_SA_P_CT_exact, gsw_cf.rho_CT_P_CT_exact] = gsw_rho_second_derivatives_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho_sd_CT_exact] = find(abs(gsw_cv.rho_SA_SA_CT_exact - gsw_cf.rho_SA_SA_CT_exact) >= gsw_cv.rho_SA_SA_CT_exact_ca | ...
    abs(gsw_cv.rho_SA_CT_CT_exact - gsw_cf.rho_SA_CT_CT_exact) >= gsw_cv.rho_SA_CT_CT_exact_ca | ...
    abs(gsw_cv.rho_CT_CT_CT_exact - gsw_cf.rho_CT_CT_CT_exact) >= gsw_cv.rho_CT_CT_CT_exact_ca | ...
    abs(gsw_cv.rho_SA_P_CT_exact - gsw_cf.rho_SA_P_CT_exact) >= gsw_cv.rho_SA_P_CT_exact_ca | ...
    abs(gsw_cv.rho_CT_P_CT_exact - gsw_cf.rho_CT_P_CT_exact) >= gsw_cv.rho_CT_P_CT_exact_ca);
if ~isempty(gsw_cf.Irho_sd_CT_exact)
    fprintf(2,'gsw_rho_second_derivatives_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.rho_SA_wrt_h_CT_exact, gsw_cf.rho_h_CT_exact] = gsw_rho_first_derivatives_wrt_enthalpy_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho_fd_wrt_h_CT_exact] = find(abs(gsw_cv.rho_SA_wrt_h_CT_exact - gsw_cf.rho_SA_wrt_h_CT_exact) >= gsw_cv.rho_SA_wrt_h_CT_exact_ca | ...
    abs(gsw_cv.rho_h_CT_exact - gsw_cf.rho_h_CT_exact) >= gsw_cv.rho_h_CT_exact_ca);
if ~isempty(gsw_cf.Irho_fd_wrt_h_CT_exact)
    fprintf(2,'gsw_rho_first_derivatives_wrt_enthalpy_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.v_SA_SA_wrt_h_CT_exact, gsw_cf.v_SA_h_CT_exact, gsw_cf.v_h_h_CT_exact] = gsw_rho_second_derivatives_wrt_enthalpy_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho_sd_wrt_h_CT_exact] = find(abs(gsw_cv.rho_SA_SA_CT_exact - gsw_cf.rho_SA_SA_CT_exact) >= gsw_cv.rho_SA_SA_CT_exact_ca | ...
    abs(gsw_cv.rho_SA_CT_CT_exact - gsw_cf.rho_SA_CT_CT_exact) >= gsw_cv.rho_SA_CT_CT_exact_ca | ...
    abs(gsw_cv.rho_CT_CT_CT_exact - gsw_cf.rho_CT_CT_CT_exact) >= gsw_cv.rho_CT_CT_CT_exact_ca | ...
    abs(gsw_cv.rho_SA_P_CT_exact - gsw_cf.rho_SA_P_CT_exact) >= gsw_cv.rho_SA_P_CT_exact_ca | ...
    abs(gsw_cv.rho_CT_P_CT_exact - gsw_cf.rho_CT_P_CT_exact) >= gsw_cv.rho_CT_P_CT_exact_ca);
if ~isempty(gsw_cf.Irho_sd_wrt_h_CT_exact)
    fprintf(2,'gsw_rho_second_derivatives_wrt_enthalpy_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.sigma0_CT_exact = gsw_sigma0_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Isigma0_CT_exact] = find(abs(gsw_cv.sigma0_CT_exact - gsw_cf.sigma0_CT_exact) >= gsw_cv.sigma0_CT_exact_ca);
if ~isempty(gsw_cf.Isigma0_CT_exact)
    fprintf(2,'gsw_sigma0_CT_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.sigma1_CT_exact = gsw_sigma1_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Isigma1_CT_exact] = find(abs(gsw_cv.sigma1_CT_exact - gsw_cf.sigma1_CT_exact) >= gsw_cv.sigma1_CT_exact_ca);
if ~isempty(gsw_cf.Isigma1_CT_exact)
    fprintf(2,'gsw_sigma1_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.sigma2_CT_exact = gsw_sigma2_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Isigma2_CT_exact] = find(abs(gsw_cv.sigma2_CT_exact - gsw_cf.sigma2_CT_exact) >= gsw_cv.sigma2_CT_exact_ca);
if ~isempty(gsw_cf.Isigma2_CT_exact)
    fprintf(2,'gsw_sigma2_CT_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.sigma3_CT_exact = gsw_sigma3_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Isigma3_CT_exact] = find(abs(gsw_cv.sigma3_CT_exact - gsw_cf.sigma3_CT_exact) >= gsw_cv.sigma3_CT_exact_ca);
if ~isempty(gsw_cf.Isigma3_CT_exact)
    fprintf(2,'gsw_sigma3_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.sigma4_CT_exact = gsw_sigma4_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast);
[gsw_cf.Isigma4_CT_exact] = find(abs(gsw_cv.sigma4_CT_exact - gsw_cf.sigma4_CT_exact) >= gsw_cv.sigma4_CT_exact_ca);
if ~isempty(gsw_cf.Isigma4_CT_exact)
    fprintf(2,'gsw_sigma4_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.cabbeling_CT_exact = gsw_cabbeling_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Icabbeling_CT_exact] = find(abs(gsw_cv.cabbeling_CT_exact - gsw_cf.cabbeling_CT_exact) >= gsw_cv.cabbeling_CT_exact_ca);
if ~isempty(gsw_cf.Icabbeling_CT_exact)
    fprintf(2,'gsw_cabbeling_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.thermobaric_CT_exact = gsw_thermobaric_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ithermobaric_CT_exact] = find(abs(gsw_cv.thermobaric_CT_exact - gsw_cf.thermobaric_CT_exact) >= gsw_cv.thermobaric_CT_exact_ca);
if ~isempty(gsw_cf.Ithermobaric_CT_exact)
    fprintf(2,'gsw_thermobaric_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.enthalpy_CT_exact = gsw_enthalpy_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ienthalpy_CT_exact] = find(abs(gsw_cv.enthalpy_CT_exact - gsw_cf.enthalpy_CT_exact) >= gsw_cv.enthalpy_CT_exact_ca);
if ~isempty(gsw_cf.Ienthalpy_CT_exact)
    fprintf(2,'gsw_enthalpy_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.enthalpy_diff_CT_exact = gsw_enthalpy_diff_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast_shallow,gsw_cv.p_chck_cast_deep);
[gsw_cf.Ienthalpy_diff_CT_exact] = find(abs(gsw_cv.enthalpy_diff_CT_exact - gsw_cf.enthalpy_diff_CT_exact) >= gsw_cv.enthalpy_diff_CT_exact_ca);
if ~isempty(gsw_cf.Ienthalpy_diff_CT_exact)
    fprintf(2,'gsw_enthalpy_diff_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.dynamic_enthalpy_CT_exact = gsw_dynamic_enthalpy_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Idynamic_enthalpy_CT_exact] = find(abs(gsw_cv.dynamic_enthalpy_CT_exact - gsw_cf.dynamic_enthalpy_CT_exact) >= gsw_cv.dynamic_enthalpy_CT_exact_ca);
if ~isempty(gsw_cf.Idynamic_enthalpy_CT_exact)
    fprintf(2,'gsw_dynamic_enthalpy_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.h_SA_CT_exact, gsw_cf.h_CT_CT_exact] = gsw_enthalpy_first_derivatives_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ienthalpy_first_deriv_CT_exact] = find(abs(gsw_cv.h_SA_CT_exact - gsw_cf.h_SA_CT_exact) >= gsw_cv.h_SA_CT_exact_ca | ...
    abs(gsw_cv.h_CT_CT_exact - gsw_cf.h_CT_CT_exact) >= gsw_cv.h_CT_CT_exact_ca);
if ~isempty(gsw_cf.Ienthalpy_first_deriv_CT_exact)
    fprintf(2,'gsw_enthalpy_first_derivatives_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.h_SA_SA_CT_exact, gsw_cf.h_SA_CT_CT_exact, gsw_cf.h_CT_CT_CT_exact] = gsw_enthalpy_second_derivatives_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ienthalpy_second_deriv_CT_exact] = find(abs(gsw_cv.h_SA_SA_CT_exact - gsw_cf.h_SA_SA_CT_exact) >= gsw_cv.h_SA_SA_CT_exact_ca  | ...
    abs(gsw_cv.h_SA_CT_CT_exact - gsw_cf.h_SA_CT_CT_exact) >= gsw_cv.h_SA_CT_CT_exact_ca | ...
    abs(gsw_cv.h_CT_CT_CT_exact - gsw_cf.h_CT_CT_CT_exact) >= gsw_cv.h_CT_CT_CT_exact_ca);
if ~isempty(gsw_cf.Ienthalpy_second_deriv_CT_exact)
    fprintf(2,'gsw_enthalpy_second_derivatives_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.sound_speed_CT_exact = gsw_sound_speed_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Isound_speed_CT_exact] = find(abs(gsw_cv.sound_speed_CT_exact - gsw_cf.sound_speed_CT_exact) >= gsw_cv.sound_speed_CT_exact_ca);
if ~isempty(gsw_cf.Isound_speed_CT_exact)
    fprintf(2,'gsw_sound_speed_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.kappa_CT_exact = gsw_kappa_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ikappa_CT_exact] = find(abs(gsw_cv.kappa_CT_exact - gsw_cf.kappa_CT_exact) >= gsw_cv.kappa_CT_exact_ca);
if ~isempty(gsw_cf.Ikappa_CT_exact)
    fprintf(2,'gsw_kappa_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.internal_energy_CT_exact = gsw_internal_energy_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iinternal_energy_CT_exact] = find(abs(gsw_cv.internal_energy_CT_exact - gsw_cf.internal_energy_CT_exact) >= gsw_cv.internal_energy_CT_exact_ca);
if ~isempty(gsw_cf.Iinternal_energy_CT_exact)
    fprintf(2,'gsw_internal_energy_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.u_SA_CT_exact, gsw_cf.u_CT_CT_exact, gsw_cf.u_P_CT_exact] = gsw_internal_energy_first_derivatives_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iu_fd_CT_exact] = find(abs(gsw_cv.u_SA_CT_exact - gsw_cf.u_SA_CT_exact) >= gsw_cv.u_SA_CT_exact_ca | ...
    abs(gsw_cv.u_CT_CT_exact - gsw_cf.u_CT_CT_exact) >= gsw_cv.u_CT_CT_exact_ca | ...
    abs(gsw_cv.u_P_CT_exact - gsw_cf.u_P_CT_exact) >= gsw_cv.u_P_CT_exact_ca);
if ~isempty(gsw_cf.Iu_fd_CT_exact)
    fprintf(2,'gsw_rho_first_derivatives_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.u_SA_SA_CT_exact, gsw_cf.u_SA_CT_CT_exact, gsw_cf.u_CT_CT_CT_exact, gsw_cf.u_SA_P_CT_exact, gsw_cf.u_CT_P_CT_exact] = gsw_internal_energy_second_derivatives_CT_exact(gsw_cv.SA_chck_cast,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iu_sd_CT_exact] = find(abs(gsw_cv.u_SA_SA_CT_exact - gsw_cf.u_SA_SA_CT_exact) >= gsw_cv.rho_SA_SA_CT_exact_ca | ...
    abs(gsw_cv.u_SA_CT_CT_exact - gsw_cf.u_SA_CT_CT_exact) >= gsw_cv.u_SA_CT_CT_exact_ca | ...
    abs(gsw_cv.u_CT_CT_CT_exact - gsw_cf.u_CT_CT_CT_exact) >= gsw_cv.u_CT_CT_CT_exact_ca | ...
    abs(gsw_cv.u_SA_P_CT_exact - gsw_cf.u_SA_P_CT_exact) >= gsw_cv.u_SA_P_CT_exact_ca | ...
    abs(gsw_cv.u_CT_P_CT_exact - gsw_cf.u_CT_P_CT_exact) >= gsw_cv.u_CT_P_CT_exact_ca);
if ~isempty(gsw_cf.Iu_sd_CT_exact)
    fprintf(2,'gsw_internal_energy_second_derivatives_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

%%% EF: fixed order of arguments in the following

gsw_cf.CT_from_enthalpy_exact = gsw_CT_from_enthalpy_exact(gsw_cv.SA_chck_cast,gsw_cf.enthalpy_CT_exact,gsw_cv.p_chck_cast);
[gsw_cf.ICT_from_enthalpy_exact] = find(abs(gsw_cv.CT_from_enthalpy_exact - gsw_cf.CT_from_enthalpy_exact) >= gsw_cv.CT_from_enthalpy_exact_ca);
if ~isempty(gsw_cf.ICT_from_enthalpy_exact)
    fprintf(2,'gsw_CT_from_enthalpy_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SA_from_rho_CT_exact = gsw_SA_from_rho_CT_exact(gsw_cf.rho_CT_exact,gsw_cv.CT_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.ISA_from_rho_CT_exact] = find(abs(gsw_cv.SA_from_rho_CT_exact - gsw_cf.SA_from_rho_CT_exact) >= gsw_cv.SA_from_rho_CT_exact_ca);
if ~isempty(gsw_cf.ISA_from_rho_CT_exact)
    fprintf(2,'gsw_SA_from_rho_CT_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.CT_from_rho_exact = gsw_CT_from_rho_exact(gsw_cf.rho_CT_exact,gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.ICT_from_rho_exact] = find(abs(gsw_cv.CT_from_rho_exact - gsw_cf.CT_from_rho_exact) >= gsw_cv.CT_from_rho_exact_ca);
if ~isempty(gsw_cf.ICT_from_rho_exact)
    fprintf(2,'gsw_CT_from_rho_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.CT_maxdensity_exact = gsw_CT_maxdensity_exact(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.ICT_maxdensity_exact] = find(abs(gsw_cv.CT_maxdensity_exact - gsw_cf.CT_maxdensity_exact) >= gsw_cv.CT_maxdensity_exact_ca);
if ~isempty(gsw_cf.ICT_maxdensity_exact)
    fprintf(2,'gsw_CT_maxdensity_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end



%% Labrortory functions

gsw_cf.rho_t_exact = gsw_rho_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Irho_t_exact] = find(abs(gsw_cv.rho_t_exact - gsw_cf.rho_t_exact) >= gsw_cv.rho_t_exact_ca);
if ~isempty(gsw_cf.Irho_t_exact)
    fprintf(2,'gsw_rho_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.SA_from_rho_t_exact = gsw_SA_from_rho_t_exact(gsw_cf.rho_t_exact,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.ISA_from_rho_t_exact] = find(abs(gsw_cv.SA_from_rho_t_exact - gsw_cf.SA_from_rho_t_exact) >= gsw_cv.SA_from_rho_t_exact_ca);
if ~isempty(gsw_cf.ISA_from_rho_t_exact)
    fprintf(2,'gsw_SA_from_rho_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.deltaSA_from_rho_t_exact = gsw_deltaSA_from_rho_t_exact(gsw_cf.rho_t_exact,gsw_cv.SP_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.IdeltaSA_from_rho_t_exact] = find(abs(gsw_cv.deltaSA_from_rho_t_exact - gsw_cf.deltaSA_from_rho_t_exact) >= gsw_cv.deltaSA_from_rho_t_exact_ca);
if ~isempty(gsw_cf.IdeltaSA_from_rho_t_exact)
    fprintf(2,'gsw_deltaSA_from_rho_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end


%% basic thermodynamic properties interms of in-situ t, derived from the exact Gibbs function

% gsw_cf.rho_t_exact = gsw_rho_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
% [gsw_cf.Irho_t_exact] = find(abs(gsw_cv.rho_t_exact - gsw_cf.rho_t_exact) >= gsw_cv.rho_t_exact_ca);
% if ~isempty(gsw_cf.Irho_t_exact)
%     fprintf(2,'gsw_rho_t_exact:   Failed\n');
%     gsw_cf.gsw_chks = 0;
% end

gsw_cf.pot_rho_t_exact = gsw_pot_rho_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast,gsw_cv.pr);
[gsw_cf.Ipot_rho_t_exact] = find(abs(gsw_cv.pot_rho_t_exact - gsw_cf.pot_rho_t_exact) >= gsw_cv.pot_rho_t_exact_ca);
if ~isempty(gsw_cf.Ipot_rho_t_exact)
    fprintf(2,'gsw_pot_rho_t_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.sigma0_pt0_exact = gsw_sigma0_pt0_exact(gsw_cv.SA_chck_cast,gsw_cf.pt0_from_t);
[gsw_cf.Isigma0_pt0_exact] = find(abs(gsw_cv.sigma0_pt0_exact - gsw_cf.sigma0_pt0_exact) >= gsw_cv.sigma0_pt0_exact_ca);
if ~isempty(gsw_cf.Isigma0_pt0_exact)
    fprintf(2,'gsw_sigma0_pt0_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.alpha_wrt_CT_t_exact = gsw_alpha_wrt_CT_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ialpha_wrt_CT_t_exact] = find(abs(gsw_cv.alpha_wrt_CT_t_exact - gsw_cf.alpha_wrt_CT_t_exact) >= gsw_cv.alpha_wrt_CT_t_exact_ca);
if ~isempty(gsw_cf.Ialpha_wrt_CT_t_exact)
    fprintf(2,'gsw_alpha_wrt_CT_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.alpha_wrt_pt_t_exact = gsw_alpha_wrt_pt_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ialpha_wrt_pt_t_exact] = find(abs(gsw_cv.alpha_wrt_pt_t_exact - gsw_cf.alpha_wrt_pt_t_exact) >= gsw_cv.alpha_wrt_pt_t_exact_ca);
if ~isempty(gsw_cf.Ialpha_wrt_pt_t_exact)
    fprintf(2,'gsw_alpha_wrt_pt_t_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.alpha_wrt_t_exact = gsw_alpha_wrt_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ialpha_wrt_t_exact] = find(abs(gsw_cv.alpha_wrt_t_exact - gsw_cf.alpha_wrt_t_exact) >= gsw_cv.alpha_wrt_t_exact_ca);
if ~isempty(gsw_cf.Ialpha_wrt_t_exact)
    fprintf(2,'gsw_alpha_wrt_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.beta_const_CT_t_exact = gsw_beta_const_CT_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ibeta_const_CT_t_exact] = find(abs(gsw_cv.beta_const_CT_t_exact - gsw_cf.beta_const_CT_t_exact) >= gsw_cv.beta_const_CT_t_exact_ca);
if ~isempty(gsw_cf.Ibeta_const_CT_t_exact)
    fprintf(2,'gsw_beta_const_CT_t_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.beta_const_pt_t_exact = gsw_beta_const_pt_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ibeta_const_pt_t_exact] = find(abs(gsw_cv.beta_const_pt_t_exact - gsw_cf.beta_const_pt_t_exact) >= gsw_cv.beta_const_pt_t_exact_ca);
if ~isempty(gsw_cf.Ibeta_const_pt_t_exact)
    fprintf(2,'gsw_beta_const_pt_t_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.beta_const_t_exact = gsw_beta_const_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ibeta_const_t_exact] = find(abs(gsw_cv.beta_const_t_exact - gsw_cf.beta_const_t_exact) >= gsw_cv.beta_const_t_exact_ca);
if ~isempty(gsw_cf.Ibeta_const_t_exact)
    fprintf(2,'gsw_beta_const_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.specvol_t_exact = gsw_specvol_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ispecvol_t_exact] = find(abs(gsw_cv.specvol_t_exact - gsw_cf.specvol_t_exact) >= gsw_cv.specvol_t_exact_ca);
if ~isempty(gsw_cf.Ispecvol_t_exact)
    fprintf(2,'gsw_specvol_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.specvol_anom_standard_t_exact = gsw_specvol_anom_standard_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ispecvol_anom_standard_t_exact] = find(abs(gsw_cv.specvol_anom_standard_t_exact - gsw_cf.specvol_anom_standard_t_exact) >= gsw_cv.specvol_anom_standard_t_exact_ca);
if ~isempty(gsw_cf.Ispecvol_anom_standard_t_exact)
    fprintf(2,'gsw_specvol_anom_standard_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

if gsw_cf.gsw_chks == 1 ;
    fprintf(1,'.');
end

gsw_cf.sound_speed_t_exact = gsw_sound_speed_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Isound_speed_t_exact] = find(abs(gsw_cv.sound_speed_t_exact - gsw_cf.sound_speed_t_exact) >= gsw_cv.sound_speed_t_exact_ca);
if ~isempty(gsw_cf.Isound_speed_t_exact)
    fprintf(2,'gsw_sound_speed_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.kappa_t_exact = gsw_kappa_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ikappa_t_exact] = find(abs(gsw_cv.kappa_t_exact - gsw_cf.kappa_t_exact) >= gsw_cv.kappa_t_exact_ca);
if ~isempty(gsw_cf.Ikappa_t_exact)
    fprintf(2,'gsw_kappa_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.kappa_const_t_exact = gsw_kappa_const_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ikappa_const_t_exact] = find(abs(gsw_cv.kappa_const_t_exact - gsw_cf.kappa_const_t_exact) >= gsw_cv.kappa_const_t_exact_ca);
if ~isempty(gsw_cf.Ikappa_const_t_exact)
    fprintf(2,'gsw_kappa_const_t_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.t_from_rho_exact = gsw_t_from_rho_exact(gsw_cf.rho_t_exact,gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.It_from_rho_exact] = find(abs(gsw_cv.t_from_rho_exact - gsw_cf.t_from_rho_exact) >= gsw_cv.t_from_rho_exact_ca);
if ~isempty(gsw_cf.It_from_rho_exact)
    fprintf(2,'gsw_t_from_rho_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.t_maxdensity_exact = gsw_t_maxdensity_exact(gsw_cv.SA_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.It_maxdensity_exact] = find(abs(gsw_cv.t_maxdensity_exact - gsw_cf.t_maxdensity_exact) >= gsw_cv.t_maxdensity_exact_ca);
if ~isempty(gsw_cf.It_maxdensity_exact)
    fprintf(2,'gsw_t_maxdensity_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.internal_energy_t_exact = gsw_internal_energy_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iinternal_energy_t_exact] = find(abs(gsw_cv.internal_energy_t_exact - gsw_cf.internal_energy_t_exact) >= gsw_cv.internal_energy_t_exact_ca);
if ~isempty(gsw_cf.Iinternal_energy_t_exact)
    fprintf(2,'gsw_internal_energy_t_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.enthalpy_t_exact = gsw_enthalpy_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ienthalpy_t_exact] = find(abs(gsw_cv.enthalpy_t_exact - gsw_cf.enthalpy_t_exact) >= gsw_cv.enthalpy_t_exact_ca);
if ~isempty(gsw_cf.Ienthalpy_t_exact)
    fprintf(2,'gsw_enthalpy_t_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.dynamic_enthalpy_t_exact = gsw_dynamic_enthalpy_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Idynamic_enthalpy_t_exact] = find(abs(gsw_cv.dynamic_enthalpy_t_exact - gsw_cf.dynamic_enthalpy_t_exact) >= gsw_cv.dynamic_enthalpy_t_exact_ca);
if ~isempty(gsw_cf.Idynamic_enthalpy_t_exact)
    fprintf(2,'gsw_dynamic_enthalpy_t_exact:   Failed\n');
    gsw_chks = 0;
end

% gsw_cf.SA_from_rho_t_exact = gsw_SA_from_rho_t_exact(gsw_cf.rho_t_exact,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
% [gsw_cf.ISA_from_rho_t_exact] = find(abs(gsw_cv.SA_from_rho_t_exact - gsw_cf.SA_from_rho_t_exact) >= gsw_cv.SA_from_rho_t_exact_ca);
% if ~isempty(gsw_cf.ISA_from_rho_t_exact)
%     fprintf(2,'gsw_SA_from_rho_t_exact:   Failed\n');
%     gsw_cf.gsw_chks = 0;
% end

[gsw_cf.CT_SA_wrt_t, gsw_cf.CT_T_wrt_t, gsw_cf.CT_P_wrt_t] = gsw_CT_first_derivatives_wrt_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.ICT_first_derivatives_wrt_t_exact] = find(abs(gsw_cv.CT_SA_wrt_t - gsw_cf.CT_SA_wrt_t) >= gsw_cv.CT_SA_wrt_t_ca | ...
    abs(gsw_cv.CT_T_wrt_t - gsw_cf.CT_T_wrt_t) >= gsw_cv.CT_T_wrt_t_ca | ...
    abs(gsw_cv.CT_P_wrt_t - gsw_cf.CT_P_wrt_t) >= gsw_cv.CT_P_wrt_t_ca);
if ~isempty(gsw_cf.ICT_first_derivatives_wrt_t_exact)
    fprintf(2,'gsw_CT_first_derivatives_wrt_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

[gsw_cf.h_SA_wrt_t, gsw_cf.h_T_wrt_t, gsw_cf.h_P_wrt_t] = gsw_enthalpy_first_derivatives_wrt_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ienthalpy_first_derivatives_wrt_t_exact] = find(abs(gsw_cv.h_SA_wrt_t - gsw_cf.h_SA_wrt_t) >= gsw_cv.h_SA_wrt_t_ca | ...
    abs(gsw_cv.h_T_wrt_t - gsw_cf.h_T_wrt_t) >= gsw_cv.h_T_wrt_t_ca | ...
    abs(gsw_cv.h_P_wrt_t - gsw_cf.h_P_wrt_t) >= gsw_cv.h_P_wrt_t_ca);
if ~isempty(gsw_cf.Ienthalpy_first_derivatives_wrt_t_exact)
    fprintf(2,'gsw_enthalpy_first_derivatives_wrt_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.cp_t_exact = gsw_cp_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Icp_t_exact] = find(abs(gsw_cv.cp_t_exact - gsw_cf.cp_t_exact) >= gsw_cv.cp_t_exact_ca);
if ~isempty(gsw_cf.Icp_t_exact)
    fprintf(2,'gsw_cp_t_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.isochoric_heat_cap_t_exact = gsw_isochoric_heat_cap_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iisochoric_heat_cap_t_exact] = find(abs(gsw_cv.isochoric_heat_cap_t_exact - gsw_cf.isochoric_heat_cap_t_exact) >= gsw_cv.isochoric_heat_cap_t_exact_ca);
if ~isempty(gsw_cf.Iisochoric_heat_cap_t_exact)
    fprintf(2,'gsw_isochoric_heat_cap_t_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.chem_potential_relative_t_exact = gsw_chem_potential_relative_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ichem_potential_relative_t_exact] = find(abs(gsw_cv.chem_potential_relative_t_exact - gsw_cf.chem_potential_relative_t_exact) >= gsw_cv.chem_potential_relative_t_exact_ca);
if ~isempty(gsw_cf.Ichem_potential_relative_t_exact)
    fprintf(2,'gsw_chem_potential_relative_t_exact:   Failed\n');
    gsw_chks = 0;
end

gsw_cf.chem_potential_water_t_exact = gsw_chem_potential_water_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ichem_potential_water_t_exact] = find(abs(gsw_cv.chem_potential_water_t_exact - gsw_cf.chem_potential_water_t_exact) >= gsw_cv.chem_potential_water_t_exact_ca);
if ~isempty(gsw_cf.Ichem_potential_water_t_exact)
    fprintf(2,'gsw_chem_potential_water_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

if gsw_cf.gsw_chks == 1 ;
    fprintf(1,'.');
end

gsw_cf.chem_potential_salt_t_exact = gsw_chem_potential_salt_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Ichem_potential_salt_t_exact] = find(abs(gsw_cv.chem_potential_salt_t_exact - gsw_cf.chem_potential_salt_t_exact) >= gsw_cv.chem_potential_salt_t_exact_ca);
if ~isempty(gsw_cf.Ichem_potential_salt_t_exact)
    fprintf(2,'gsw_chem_potential_salt_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.t_deriv_chem_potential_water_t_exact =  gsw_t_deriv_chem_potential_water_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.It_deriv_chem_potential_water_t_exact] = find(abs(gsw_cv.t_deriv_chem_potential_water_t_exact - gsw_cf.t_deriv_chem_potential_water_t_exact) >= gsw_cv.t_deriv_chem_potential_water_t_exact_ca);
if ~isempty(gsw_cf.It_deriv_chem_potential_water_t_exact)
    fprintf(2,'gsw_t_deriv_chem_potential_water_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.dilution_coefficient_t_exact = gsw_dilution_coefficient_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Idilution_coefficient_t_exact] = find(abs(gsw_cv.dilution_coefficient_t_exact - gsw_cf.dilution_coefficient_t_exact) >= gsw_cv.dilution_coefficient_t_exact_ca);
if ~isempty(gsw_cf.Idilution_coefficient_t_exact)
    fprintf(2,'gsw_dilution_coefficient_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.Helmholtz_energy_t_exact = gsw_Helmholtz_energy_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.IHelmholtz_energy_t_exact] = find(abs(gsw_cv.Helmholtz_energy_t_exact - gsw_cf.Helmholtz_energy_t_exact) >= gsw_cv.Helmholtz_energy_t_exact_ca);
if ~isempty(gsw_cf.IHelmholtz_energy_t_exact)
    fprintf(2,'gsw_Helmholtz_energy_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.osmotic_coefficient_t_exact = gsw_osmotic_coefficient_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iosmotic_coefficient_t_exact] = find(abs(gsw_cv.osmotic_coefficient_t_exact - gsw_cf.osmotic_coefficient_t_exact) >= gsw_cv.osmotic_coefficient_t_exact_ca);
if ~isempty(gsw_cf.Iosmotic_coefficient_t_exact)
    fprintf(2,'gsw_osmotic_coefficient_t_exact:   Failed\n');
    gsw_cf.gsw_chks = 0;
end

gsw_cf.osmotic_pressure_t_exact = gsw_osmotic_pressure_t_exact(gsw_cv.SA_chck_cast,gsw_cv.t_chck_cast,gsw_cv.p_chck_cast);
[gsw_cf.Iosmotic_pressure_t_exact] = find(abs(gsw_cv.osmotic_pressure_t_exact - gsw_cf.osmotic_pressure_t_exact) >= gsw_cv.osmotic_pressure_t_exact_ca);
if ~isempty(gsw_cf.Iosmotic_pressure_t_exact)
    fprintf(2,'gsw_osmotic_pressure_t_exact:   Failed\n');
    gsw_chks = 0;
end

if gsw_cf.gsw_chks == 1 ;
    fprintf(1,'.');
end



%% library
gsw_cf.Fdelta = gsw_Fdelta(gsw_cv.p_chck_cast,gsw_cv.long_chck_cast,gsw_cv.lat_chck_cast);
[gsw_cf.IFdelta] = find(abs(gsw_cv.Fdelta - gsw_cf.Fdelta) >= gsw_cv.Fdelta_ca);
if ~isempty(gsw_cf.IFdelta)
    fprintf(2,'gsw_Fdelta:   Failed. \n');
    gsw_cf.gsw_chks = 0;
end

for I = 1:45
    gsw_cf.long_chck_cast_temp(I,:) = gsw_cv.long_chck_cast(1,:);
    gsw_cf.lat_chck_cast_temp(I,:) = gsw_cv.lat_chck_cast(1,:);
end
[I] = find(~isnan(gsw_cv.p_chck_cast));
gsw_cf.deltaSA_atlas = nan(45,3);
gsw_cf.deltaSA_atlas(I) = gsw_deltaSA_atlas(gsw_cv.p_chck_cast(I),gsw_cf.long_chck_cast_temp(I),gsw_cf.lat_chck_cast_temp(I));
[gsw_cf.IdeltaSA_atlas] = find(abs(gsw_cv.deltaSA_atlas - gsw_cf.deltaSA_atlas) >= gsw_cv.deltaSA_atlas_ca);
if ~isempty(gsw_cf.IdeltaSA_atlas)
    fprintf(2,'gsw_deltaSA_atlas:   Failed. \n');
    gsw_cf.gsw_chks = 0;
end

clear I



%%
if gsw_cf.gsw_chks == 1 ;
    fprintf(1,' Finished.\n');
    fprintf(1,'\n');
end

if gsw_cf.gsw_chks == 0
    fprintf(2,'Your installation of the Gibbs SeaWater (GSW) Oceanographic Toolbox has errors !\n');
    demo = 0;
else
    fprintf(1,'Well done! The gsw_check_fuctions confirms that the \n');
    fprintf(1,'Gibbs SeaWater (GSW) Oceanographic Toolbox is installed correctly.\n');
    fprintf(1,'\n');
    save('gsw_cv_cf.mat', 'gsw_cv', 'gsw_cf', '-v6')
end
