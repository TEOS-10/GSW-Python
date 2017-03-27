"""
Blocks of text for assembling docstrings.
"""

parameters = dict(
SP = "Practical Salinity (PSS-78), unitless",
SA = "Absolute Salinity, g/kg",
SR = "Reference Salinity, g/kg",
SK = "Knudsen Salinity, ppt",
Sstar = "Preformed Salinity, g/kg",
SA_seaice =
"""Absolute Salinity of sea ice: the mass fraction of salt
in sea ice, expressed in g of salt per kg of sea ice.""",
t_seaice =
"In-situ temperature of the sea ice at pressure p (ITS-90), degrees C",
t = "In-situ temperature (ITS-90), degrees C",
CT = "Conservative Temperature (ITS-90), degrees C",
C = "Conductivity, mS/cm",
p = "Sea pressure (absolute pressure minus 10.1325 dbar), dbar",
lon = "Longitude, -360 to 360 degrees",
lat = "Latitude, -90 to 90 degrees",
saturation_fraction =
"Saturation fraction of dissolved air in seawater. (0..1)",
p_ref = "Reference pressure, dbar",
p_shallow = "Upper sea pressure (absolute pressure minus 10.1325 dbar), dbar",
p_deep = "Lower sea pressure (absolute pressure minus 10.1325 dbar), dbar",

enthalpy_diff = "Specific enthalpy, deep minus shallow, J/kg",
pot_enthalpy_ice =  "Potential enthalpy of ice, J/kg",
h = "Specific enthalpy, J/kg",
entropy = "Specific entropy, J/(kg*K)",
pt0 = "Potential temperature with reference pressure of 0 dbar, degrees C",
pt0_ice = "Potential temperature of ice (ITS-90), degrees C",
# TODO: Check the functions using this to see if any customizations are needed.
pt = "Potential temperature referenced to a sea pressure, degrees C",
rho = "Seawater density (not anomaly) in-situ, e.g., 1026 kg/m^3.",
t_Ih = "In-situ temperature of ice (ITS-90), degrees C",
)
