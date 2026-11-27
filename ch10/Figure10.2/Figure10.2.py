"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 10 Particle Formation from Supercritical Solutions
Figure 10.2
Solubility of a model solute in a CO2-expanded liquid from the regular-solution model of Eq. 10.1,
relative to its solubility in the pure organic solvent, as a function of the volume fraction phi_3
of CO2 in the liquid (V_2=250 cm^3/mol, 313 K). Solubility parameters of the solvents (values in
MPa^1/2 in the legends) are Hansen total parameters; that of dense CO2 follows from Giddings'
density correlation with Span-Wagner densities from CoolProp. A Three solvents for a solute with
delta_2=24 MPa^1/2 and CO2 at 10 MPa (delta_3=11.0 MPa^1/2). B Solutes of different delta_2 in
ethanol. C Effect of the CO2 pressure for the solute of panel A in ethanol. The volume expansion
that relates phi_3 to the CO2 mole fraction is not computed.

No command-line arguments: edit the SETTINGS block below and run   python Figure10.2.py
For a solute at low concentration in a mixture of an organic solvent (1) and CO2 (3), Eq. 10.1 gives
the activity coefficient
ln gamma_2 = V2 [ phi1 (d2 - d1) + phi3 (d2 - d3) ]^2 / RT (d = Hildebrand solubility parameter, phi
= volume fraction in the CO2-expanded liquid), so that the solubility relative to the pure solvent
is  x2/x2(0) = exp{ -V2 [ (d2 - d_mix)^2 - (d2 - d1)^2 ] / RT },  d_mix = phi1 d1 + phi3 d3. The
solubility parameter of dense CO2 follows from Giddings' correlation d3 = 1.25 sqrt(pc/atm)
(rho_r/2.66) (cal cm-3)^(1/2), evaluated with the Span-Wagner density from CoolProp. Volume
fractions phi are the abscissa (the volume expansion of the liquid that connects phi3 with the CO2
mole fraction is not computed here); the solute is a model compound with V2 and d2 set below.
Produces Fig. 10.2.   Requires: numpy, matplotlib, CoolProp.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

font = {'family':'Dejavu Sans','weight':'normal','size':6}
mpl.rc('font',**font)
for ax_ in ('xtick','ytick'):
    mpl.rcParams[ax_+'.major.size'] = 3
    mpl.rcParams[ax_+'.minor.size'] = 3
    mpl.rcParams[ax_+'.major.width'] = .5
    mpl.rcParams[ax_+'.minor.width'] = .5
mpl.rcParams['lines.linewidth']=.75

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Settings
Edit the parameters below to change the calculation.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ============================== SETTINGS ====================================
T          = 313.15                                  # K
P_CO2      = 10e6                                    # Pa, pressure of the CO2-rich phase (Panels A and B)
V2         = 250.0                                   # cm3/mol, molar volume of the model solute
D2         = 24.0                                    # MPa^1/2, solubility parameter of the model solute (Panel A)
SOLVENTS   = [("DMSO", 26.7), ("Ethanol", 26.5), ("Acetone", 20.0)]     # Hansen total solubility parameters, MPa^1/2
D2_SCAN    = [14.0, 18.0, 22.0, 26.0]                # MPa^1/2, solutes of Panel B (solvent: ethanol)
P_SCAN     = [8e6, 10e6, 15e6, 25e6]                 # Pa, pressures of Panel C
SAVE_FIG   = "figure10.2.pdf"
# ============================================================================

R = 8.314462618
HERE = os.path.dirname(os.path.abspath(__file__))
cm = 1/2.54
Pc_atm = PropsSI("pcrit", "CO2") / 101325.0
rho_c = PropsSI("rhocrit", "CO2")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- delta_co2: MPa^1/2 (1 (cal/cm3)^1/2 = 2.0455 MPa^1/2)
- rel_sol: x2/x2(0); V2 in cm3/mol, d in MPa^1/2 = (J/cm3)^1/2
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def delta_co2(P, T=T):                                    # MPa^1/2 (1 (cal/cm3)^1/2 = 2.0455 MPa^1/2)
    rr = PropsSI("Dmass", "T", T, "P", P, "CO2") / rho_c
    return 2.0455 * 1.25 * np.sqrt(Pc_atm) * rr / 2.66
def rel_sol(phi3, d2, d1, d3):                            # x2/x2(0); V2 in cm3/mol, d in MPa^1/2 = (J/cm3)^1/2
    dmix = (1 - phi3) * d1 + phi3 * d3
    return np.exp(-V2 * ((d2 - dmix) ** 2 - (d2 - d1) ** 2) / (R * T))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

d3 = delta_co2(P_CO2)
print(f"delta(CO2) at {T:.0f} K, {P_CO2/1e6:.0f} MPa: {d3:.1f} MPa^1/2")
phi = np.linspace(0, 0.95, 300)
fig, axs = plt.subplots(1, 3, figsize=(19 * cm, 7.0 * cm))
for (name, d1), c in zip(SOLVENTS, ["tab:blue", "tab:orange", "tab:green"]):
    axs[0].plot(phi, rel_sol(phi, D2, d1, d3), color=c, label=f"{name} ({d1:.1f})")
    print(f"{name:8s}: x2/x2(0) at phi3 = 0.5: {rel_sol(0.5, D2, d1, d3):.2e}, at 0.8: {rel_sol(0.8, D2, d1, d3):.2e}")
axs[0].set_yscale("log")
axs[0].set_ylim(1e-5, 10)
axs[0].set_xlabel(r"CO$_2$ volume fraction $\phi_3$")
axs[0].set_ylabel(r"Relative solubility $x_2/x_2(\phi_3 = 0)$")
axs[0].set_xlim([0,0.9])
axs[0].minorticks_off()
axs[0].legend(frameon=False, loc="lower left", title=rf"solvent ($\delta_1$, MPa$^{{1/2}}$); $\delta_2$ = {D2:.0f}", title_fontsize=5.5)
for d2, c in zip(D2_SCAN, plt.cm.plasma(np.linspace(0.1, 0.8, len(D2_SCAN)))):
    axs[1].plot(phi, rel_sol(phi, d2, 26.5, d3), color=c, label=rf"$\delta_2$ = {d2:.0f}")
axs[1].set_yscale("log")
axs[1].set_ylim(1e-6, 1e6)
axs[1].set_xlabel(r"CO$_2$ volume fraction $\phi_3$")
axs[1].set_ylabel(r"Relative solubility $x_2/x_2(\phi_3 = 0)$")
axs[1].legend(frameon=False, loc="lower left", title=r"Solute in Ethanol", title_fontsize=5.5)
axs[1].set_xlim([0,0.9])
axs[1].hlines(1,0,0.9,ls='--',color='k',lw=0.5)
axs[0].hlines(1,0,0.9,ls='--',color='k',lw=0.5)
axs[2].hlines(1,0,0.9,ls='--',color='k',lw=0.5)
for P, c in zip(P_SCAN, plt.cm.viridis(np.linspace(0.0, 0.85, len(P_SCAN)))):
    axs[2].plot(phi, rel_sol(phi, D2, 26.5, delta_co2(P)), color=c, label=f"{P/1e6:.0f} MPa ({delta_co2(P):.1f})")
axs[2].set_yscale("log")
axs[2].set_ylim(1e-5, 5)
axs[2].set_xlabel(r"CO$_2$ volume fraction $\phi_3$")
axs[2].set_ylabel(r"Relative solubility $x_2/x_2(\phi_3 = 0)$")
axs[2].legend(frameon=False, loc="lower left")
axs[2].text(0.95,0.925,rf"$T=${T:.0f} K; Ethanol",
            transform=axs[2].transAxes,
            horizontalalignment='right')
axs[2].set_xlim([0,0.9])
axs[2].minorticks_off()

for a, L in zip(axs, "ABC"): 
    a.text(-0.24, 0.95, L, 
           transform=a.transAxes, fontsize=8, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(HERE, SAVE_FIG))
print("saved", SAVE_FIG)
