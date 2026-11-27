"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 10 Particle Formation from Supercritical Solutions
Example 10.2
Classical nucleation theory for a RESS solute: In this example, we evaluate Eqs. 10.12 and 10.13 for
a model solute with the molecular volume of naphthalene (v_m=1.86x10^-28 m^3, molecular diameter
0.71 nm) at 300 K, and ask which supersaturation is needed before nucleation becomes observable. We
take J=10^6 m^-3 s^-1 (1 cm^-3 s^-1) as the onset of nucleation, and use interfacial tensions of 40,
50, and 60 mJ m^-2, the lowest standing for a co-solvent or a more favorable solid-fluid interface;
gamma and the pre-exponential factor A are illustrative values of the usual order of magnitude, not
fitted to a solute. The script is Example10.2.py in Examples/ch10. The results are shown in Fig.
10.4.

No command-line arguments: edit the SETTINGS block below and run   python Example10.2.py
The Gibbs energy of forming a spherical nucleus of n molecules from a supersaturated fluid is (Eq.
10.12)
DG(n) = -n kT ln S + (36 pi)^(1/3) vm^(2/3) n^(2/3) gamma = -(4 pi/3) r^3 kT ln S / vm + 4 pi r^2
gamma, with the maximum DG* at r* = 2 gamma vm / (kT ln S) and the rate J = A exp(-DG*/kT) (Eq.
10.13). The script
  (A) plots DG(r)/kT for one interfacial tension and three supersaturations,
(B) plots J(S) for three interfacial tensions (a lower gamma stands for a co-solvent or a more
favorable solute-fluid interface)
and marks the critical supersaturation S_crit at which J reaches J_CRIT, the conventional onset of
observable nucleation,
(C) plots r* and n* against ln S, with the molecular size of the solute for comparison. Model
solute: the molecular volume of naphthalene in the solid (vm = M/(rho_s N_A)); gamma and the
pre-exponential factor A are illustrative values of the usual order of magnitude, not fitted to a
particular solute (they are set in the settings block).
Prints the numbers quoted in the text. Panels A and B of Fig. 10.4 are drawn by Example10.3.py with
the same parameters; this script saves the three-panel figure of the class version
(figure10.4AB.pdf).   Requires: numpy, matplotlib.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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
T        = 300.0                     # K, temperature of the expanded fluid
M_SOLUTE = 128.17e-3                 # kg/mol, naphthalene
RHO_S    = 1145.0                    # kg/m3, solid density
SIGMA    = [0.04, 0.05, 0.06]        # J/m2, solid-fluid interfacial tensions of Panel B (illustrative)
SIGMA_A  = 0.05                      # J/m2, interfacial tension of Panel A
S_A      = [50, 100, 1000]           # supersaturation ratios of Panel A
A_PRE    = 1e28                      # m-3 s-1, pre-exponential factor of Eq. 10.13 (illustrative)
J_CRIT   = 1e6                       # m-3 s-1 (1 cm-3 s-1), onset of observable nucleation
SAVE_FIG = "figure10.4AB.pdf"
# ============================================================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

kB, NA = 1.380649e-23, 6.02214076e23
HERE = os.path.dirname(os.path.abspath(__file__))
cm = 1/2.54
vm = M_SOLUTE / (RHO_S * NA)
kT = kB * T
d_mol = (6 * vm / np.pi) ** (1 / 3)                      # molecular diameter of the solute
print(f"vm = {vm:.3e} m3, molecular diameter {d_mol*1e9:.2f} nm, kT = {kT:.3e} J")

rstar = lambda sig, S: 2 * sig * vm / (kT * np.log(S))
nstar = lambda sig, S: 4 * np.pi / 3 * rstar(sig, S) ** 3 / vm
dGstar = lambda sig, S: 16 * np.pi * vm ** 2 * sig ** 3 / (3 * kT ** 2 * np.log(S) ** 2)     # J
J = lambda sig, S: A_PRE * np.exp(-dGstar(sig, S) / kT)

fig, axs = plt.subplots(1, 3, figsize=(19 * cm, 7.5 * cm))
# Panel A
r = np.linspace(1e-12, 6e-9, 600)
for S, c in zip(S_A, ["tab:blue", "tab:orange", "tab:red"]):
    dG = (4 * np.pi * r ** 2 * SIGMA_A - 4 * np.pi / 3 * r ** 3 * kT * np.log(S) / vm) / kT
    axs[0].plot(r * 1e9, dG, color=c, label=rf"$S$ = {S}")
    axs[0].plot(rstar(SIGMA_A, S) * 1e9, dGstar(SIGMA_A, S) / kT, "o", ms=3, color=c)
    print(f"sigma = {SIGMA_A}, S = {S:5d}: r* = {rstar(SIGMA_A,S)*1e9:.2f} nm, n* = {nstar(SIGMA_A,S):6.1f}, DG*/kT = {dGstar(SIGMA_A,S)/kT:7.1f}, J = {J(SIGMA_A,S):.2e} m-3 s-1")
axs[0].axhline(0, color="0.6", lw=.4)
axs[0].set_ylim(-60, 80)
axs[0].set_xlim(0, 2)
axs[0].set_xlabel("Nucleus radius $r$ (nm)")
axs[0].set_ylabel(r"$\Delta G(r)/k_\mathrm{B}T$")
axs[0].legend(frameon=False, loc="lower left", title=rf"$\gamma$ = {SIGMA_A*1e3:.0f} mJ m$^{{-2}}$", title_fontsize=6)
# Panel B
lnS = np.linspace(1.0, 14, 500)
S = np.exp(lnS)
for sig, c in zip(SIGMA, ["tab:blue", "tab:orange", "tab:red"]):
    axs[1].plot(lnS, np.log10(np.maximum(J(sig, S), 1e-300)), color=c, label=rf"{sig*1e3:.0f} mJ m$^{{-2}}$")
    Sc = np.exp(np.sqrt(16 * np.pi * vm ** 2 * sig ** 3 / (3 * kT ** 3 * np.log(A_PRE / J_CRIT))))
    axs[1].plot(np.log(Sc), np.log10(J_CRIT), "o", ms=3, color=c)
    print(f"sigma = {sig}: S_crit(J = {J_CRIT:.0e}) = {Sc:.1f}  (ln S = {np.log(Sc):.2f})")
axs[1].axhline(np.log10(J_CRIT), color="0.6", lw=.4, ls="--")
axs[1].set_ylim(-5, 29)
axs[1].set_xlabel(r"$\ln S$")
axs[1].set_ylabel(r"$\log_{10}\,(J / \mathrm{m^{-3}\,s^{-1}})$")
axs[1].set_xlim([2.5,12.5])
axs[1].legend(frameon=False, loc="lower right", title=r"$\gamma$", title_fontsize=6)
# Panel C
lnS2 = np.linspace(1.5, 14, 300)
for sig, c in zip(SIGMA, ["tab:blue", "tab:orange", "tab:red"]):
    axs[2].plot(lnS2, 2 * rstar(sig, np.exp(lnS2)) * 1e9, color=c, label=rf"{sig*1e3:.0f} mJ m$^{{-2}}$")
axs[2].axhline(d_mol * 1e9, color="0.5", lw=.5, ls="--")
axs[2].text(2.5, d_mol * 1e9 * 1.08, "$d_\mathrm{mol}$", color="k", fontsize=6)
axs[2].set_yscale("log")
axs[2].set_ylim(0.5,10)
axs[2].set_xlim([1.5,14])
axs[2].set_xlabel(r"$\ln S$")
axs[2].set_ylabel(r"Critical nucleus diameter $2r^{*}$ (nm)")
ax2 = axs[2].twinx()
ax2.set_yscale("log")
ax2.set_ylim(*[4 * np.pi / 3 * (y / 2e9) ** 3 / vm for y in axs[2].get_ylim()])
ax2.set_ylabel(r"Molecules in a sphere of that diameter")
for a, L in zip(axs, "ABC"):
    a.text(-0.24, 0.95, L,
                                    transform=a.transAxes, fontsize=8, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HERE, SAVE_FIG))
print("saved", SAVE_FIG)
