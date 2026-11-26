"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 9 Supercritical Fluid Extraction and Fractionation
Example 9.3
Hansen solubility parameters of CO2 as functions of pressure and temperature: In this example, we
calculate delta_d, delta_p, delta_h, and delta of CO2 from Eq. 9.12 of Williams et al., using the
molar volume from the Span-Wagner Equation of State in the CoolProp library, and compare the total
delta with the value obtained from the residual internal energy of the same Equation of State. The
result is shown in Fig. 9.4; the script Example9.3.py is provided in the folder Example9.3 on the GitHub repository.

Ind. Eng. Chem. Res. 43, 4967 (2004)) with the density of CO2 from CoolProp.   Produces Fig. 9.4.
No command-line arguments: edit the SETTINGS block below and run   python Example9.3.py
Williams et al. scale the Hansen parameters of CO2 at a reference state (T0, molar volume V0) to any
other state (T, V) with
    delta_d = delta_d0 (V0/V)^1.25
    delta_p = delta_p0 (V0/V)^0.5
    delta_h = delta_h0 (V0/V)^0.5 exp[-1.32e-3 (T - T0)]
delta^2 = delta_d^2 + delta_p^2 + delta_h^2                         (Eq. 9.10) where V = M/rho is
the molar volume of CO2 at (T, P), here from CoolProp (Span-Wagner equation of state; Williams et
al. used the equation of state of Huang et al.).  Reference state: T0 = 298.15 K, delta_d0 = 15.6,
delta_p0 = 5.2, delta_h0 = 5.8 MPa^0.5 (Hansen's values for CO2), V0 = 39.13 cm3/mol, as quoted by
Marcus, ACS Omega 3, 524 (2018). As an independent check, the total parameter is also computed from
the residual internal energy of the equation of state,
delta_EOS^2 = (U_ideal gas - U)/V = -u_res/V,    u_res = R T tau (d alpha_r / d tau)    (CoolProp,
Helmholtz-energy form), i.e. the Hildebrand definition of Eq. (9.8) with the energy of vaporization
to the ideal-gas state evaluated at the same T and density.
Requires: numpy, matplotlib, CoolProp.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

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
T_REF      = 298.15                 # K, reference temperature
V_REF      = 39.13                  # cm3/mol, reference molar volume of CO2
HSP_REF    = (15.6, 5.2, 5.8)       # MPa^0.5, delta_d0, delta_p0, delta_h0 at the reference state
EXP_D, EXP_P, EXP_H, COEF_T = 1.25, 0.5, 0.5, 1.32e-3      # exponents of V0/V and temperature coefficient (1/K) of delta_h
T_A        = 318                    # K, isotherm of Panel A
T_B        = [308, 318, 338, 373]   # K, isotherms of Panel B
P_RANGE    = (0.1, 60.0)            # MPa
SAVE_FIG   = "figure9.4.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
cm = 1/2.54
M = CP.PropsSI("molar_mass", "CO2") * 1e3                                    # g/mol

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- hsp: Williams et al. scaling, P in Pa; returns delta_d, delta_p, delta_h, delta, V
- delta_eos: sqrt of the cohesive energy density -u_res/V, in MPa^0.5
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def hsp(T, P):                                                               # Williams et al. scaling, P in Pa; returns delta_d, delta_p, delta_h, delta, V
    V = M / CP.PropsSI("Dmass", "T", T, "P", P, "CO2") * 1e3                 # cm3/mol
    d0, p0, h0 = HSP_REF
    dd = d0 * (V_REF / V)**EXP_D
    dp = p0 * (V_REF / V)**EXP_P
    dh = h0 * (V_REF / V)**EXP_H * np.exp(-COEF_T * (T - T_REF))
    return dd, dp, dh, np.sqrt(dd**2 + dp**2 + dh**2), V

AS = CP.AbstractState("HEOS", "CO2")
def delta_eos(T, P):                                                         # sqrt of the cohesive energy density -u_res/V, in MPa^0.5
    AS.update(CP.PT_INPUTS, P, T)
    u_res = AS.gas_constant() * T * AS.tau() * AS.dalphar_dTau()  # J/mol
    return np.sqrt(-u_res / (M / AS.rhomass() * 1e-3) / 1e6)                 # (J/mol)/(m3/mol) = Pa -> MPa

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Pg = np.linspace(P_RANGE[0], P_RANGE[1], 240) * 1e6
res = {T: np.array([hsp(T, p) for p in Pg]) for T in sorted(set([T_A] + T_B))}
eos = {T: np.array([delta_eos(T, p) for p in Pg]) for T in sorted(set([T_A] + T_B))}

# ---- printed checks: reference state and the values quoted by Marcus for 298.15 K and 20 MPa
for T, P in ((T_REF, 0.101325e6), (T_REF, 20e6), (318.15, 20e6)):
    dd, dp, dh, d, V = hsp(T, P)
    print(f"{T:.2f} K, {P / 1e6:7.3f} MPa:  V = {V:5.1f} cm3/mol,  d_d = {dd:5.2f},  d_p = {dp:4.2f},  d_h = {dh:4.2f},  d = {d:5.2f},  d(EOS energy) = {delta_eos(T, P):5.2f} MPa^0.5")
for T in T_B:
    i = np.argmin(abs(Pg - 10e6))
    j = np.argmin(abs(Pg - 30e6))
    print(f"{T} K: d = {res[T][i, 3]:.2f} at 10 MPa, {res[T][j, 3]:.2f} at 30 MPa;  d(EOS) = {eos[T][i]:.2f}, {eos[T][j]:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(19*cm*0.8, 7*cm), gridspec_kw={'wspace':0.3})
# ---- Panel A: components along one isotherm
a = ax[0]
r = res[T_A]
for k, (lab, col) in enumerate((("$\\delta_\\mathrm{d}$", "tab:blue"), ("$\\delta_\\mathrm{p}$", "tab:green"), ("$\\delta_\\mathrm{h}$", "tab:orange"))):
    a.plot(Pg / 1e6, r[:, k], color=col, label=lab)
a.plot(Pg / 1e6, r[:, 3], color="k", lw=1.2, label=r"$\delta$ (Eq. 9.12)")
a.plot(Pg / 1e6, eos[T_A], color="0.5", ls="--", label=r"$\delta$ from the residual energy")
a.axvline(CP.PropsSI("pcrit", "CO2") / 1e6, color="0.4", ls=(0, (4, 3)), lw=.5)
a.text(CP.PropsSI("pcrit", "CO2") / 1e6 + 0.8, 0.3, r"$p_{\mathrm{crit},1}$", fontsize=6, color="0.3")
a.set_xlabel("$p$ / MPa")
a.set_ylabel(r"solubility parameter / MPa$^{1/2}$")
a.set_xlim(*P_RANGE)
a.set_ylim(0, 16)
a.legend(frameon=False, fontsize=5.5, ncol=1, handletextpad=0.4, 
         loc="center right", bbox_to_anchor=(1.0, 0.5))                       # between delta_d and delta_h, clear of the p_crit label
a.text(0.05,0.9,f"CO$_2$, {T_A} K",transform=a.transAxes)

# ---- Panel B: total parameter at several temperatures
b = ax[1]
cols = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
for T, col in zip(T_B, cols):
    b.plot(Pg / 1e6, res[T][:, 3], color=col, label=f"{T} K")
    b.plot(Pg / 1e6, eos[T], color=col, ls="--", lw=.5)
b.set_xlabel("$p$ / MPa")
b.set_ylabel(r"$\delta$ / MPa$^{1/2}$")
b.set_xlim(*P_RANGE)
b.set_ylim(0, 17)
b.legend(frameon=False, fontsize=5.2, loc="lower right", title="full: Eq. (9.12), dashed: residual energy", title_fontsize=5.2)
for a_, lab in zip(ax, "AB"):
    a_.text(-0.2, 1.0, lab, transform=a_.transAxes, fontweight="bold", fontsize=7)
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
