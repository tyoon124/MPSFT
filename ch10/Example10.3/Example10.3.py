"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 10 Particle Formation from Supercritical Solutions
Example 10.3
Population balance of a RESS aerosol: In this example, we advance the first three moments of the
particle-volume distribution (Eqs. 10.9-10.10 and the equation for M_2) in a fluid parcel of the jet
that is quenched at t = 0 to the supersaturation S_0 = 2000 and then evolves in time, which stands
for the axial position divided by the jet velocity. Nucleation follows Eq. 10.13 with the parameters
of Example 10.2; growth follows the free-molecular counterpart of Eq. 10.14, the impingement flux of
solute molecules on the particle surface (sticking coefficient 0.1), which applies when the particles
are small compared with the mean free path of the expanded fluid; coagulation uses the free-molecular
Brownian kernel; the integrals are evaluated with the log-normal distribution of Eq. 10.11.

No command-line arguments: edit the SETTINGS block below and run   python Example10.3.py
A fluid parcel of the expanded jet is quenched at t = 0 to the supersaturation S0 and then evolves
in time (the time stands for the axial position divided by the jet velocity). Three moments M0, M1,
M2 of the particle-volume distribution (Eq. 10.8) are advanced:
dM0/dt = J - (1/2) int int beta n n'                           (nucleation, loss by coagulation)
dM1/dt = J v* + int G(v) n dv                                   (nucleation, growth; coagulation
                                                                 conserves M1)
dM2/dt = J v*^2 + 2 int G(v) v n dv + int int beta v v' n n'    (nucleation, growth, coagulation)
with the nucleation rate J of Eq. 10.13 (v* = critical-nucleus volume), the growth law of direct
molecular attachment G(v) = vm (36 pi)^(1/3) v^(2/3) sqrt(kT/(2 pi m)) stick (c - c_sat)
(impingement flux on the surface of the particle), and the free-molecular Brownian kernel beta.
The distribution entering the integrals is reconstructed from M0, M1, M2 with the log-normal
closure of Eq. 10.11 (geometric standard deviation bounded below by SIGMA_MIN, because nuclei of a
single size would give a delta function). The solute is consumed by the particles,
c = c0 - M1/vm, so that S = c/c_sat falls as they grow. The run is repeated without coagulation to
show its effect on the mean size. Model solute: naphthalene-like molecular properties; gamma, A, S0,
the sticking coefficient, and the solute concentration are illustrative values of the usual order
of magnitude, not fitted to experiment.
Produces Fig. 10.4: panels A and B show the classical nucleation theory of Example 10.2 (same
parameters), panels C and D the population balance of this example.
Requires: numpy, scipy, matplotlib.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
from scipy.integrate import solve_ivp
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
T        = 300.0                     # K, temperature of the expanded jet (held constant)
M_SOLUTE = 128.17e-3                 # kg/mol
RHO_S    = 1145.0                    # kg/m3, solid density
GAMMA    = 0.05                      # J/m2, solid-fluid interfacial tension (illustrative; Panels A, C, D)
GAMMA_B  = [0.04, 0.05, 0.06]        # J/m2, interfacial tensions of Panel B (as in Example 10.2)
S_A      = [50, 100, 1000]           # supersaturation ratios of Panel A (as in Example 10.2)
J_CRIT   = 1e6                       # m-3 s-1, onset of observable nucleation (Panel B)
A_PRE    = 1e28                      # m-3 s-1, pre-exponential factor of the nucleation rate (illustrative)
S0       = 2000.0                    # supersaturation right after the quench
Y0       = 1e-3                      # solute mole fraction in the jet
N_FLUID  = 2.4e25                    # m-3, number density of the fluid in the jet
STICK    = 0.1                       # sticking coefficient of the growth law
SIGMA_MIN= 1.2                       # lower bound of the geometric standard deviation used for the closure
T_END    = 1e-2                      # s
T_PSD    = [1e-4, 3e-4, 1e-3, 1e-2]  # s, times at which d_g and sigma_g are printed
SAVE_FIG = "figure10.4.pdf"
# ============================================================================

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Set constants and grids
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

kB, NA = 1.380649e-23, 6.02214076e23
HERE = os.path.dirname(os.path.abspath(__file__))
cm = 1/2.54
m = M_SOLUTE / NA
vm = m / RHO_S
kT = kB * T
c0 = Y0 * N_FLUID
csat = c0 / S0
vimp = np.sqrt(kT / (2 * np.pi * m))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define functions
- J
- vstar
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def J(S):
    return 0.0 if S <= 1.0001 else A_PRE * np.exp(-16 * np.pi * vm ** 2 * GAMMA ** 3 / (3 * kT ** 3 * np.log(S) ** 2))
def vstar(S):
    return max(4 * np.pi / 3 * (2 * GAMMA * vm / (kT * np.log(S))) ** 3, 2 * vm)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Set constants and grids
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

xg = np.logspace(-29, -17, 160)
lx = np.log(xg)
dl = lx[1] - lx[0]               # volume grid for the integrals
BET = (3 / (4 * np.pi)) ** (1 / 6) * np.sqrt(6 * kT / RHO_S) * np.sqrt(1 / xg[:, None] + 1 / xg[None, :]) * (xg[:, None] ** (1 / 3) + xg[None, :] ** (1 / 3)) ** 2
sc = np.array([1e15, 1e-6, 1e-27])                                                  # scales of M0, M1, M2

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Define functions
- lognormal: Eq. 10.11; returns dN/dln v, v_g, ln sigma_g (diameter based)
- run
- moments
- stats
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def lognormal(M0, M1, M2):                                                          # Eq. 10.11; returns dN/dln v, v_g, ln sigma_g (diameter based)
    vg = M1 ** 2 / (M0 ** 1.5 * M2 ** 0.5)
    ls = max(np.sqrt(max(np.log(M0 * M2 / M1 ** 2), 0) / 9.0), np.log(SIGMA_MIN))
    lsv = 3 * ls
    return M0 / (np.sqrt(2 * np.pi) * lsv) * np.exp(-(lx - np.log(vg)) ** 2 / (2 * lsv ** 2)), vg, ls

def run(coag):
    def f(t, y):
        M0, M1, M2 = np.maximum(y * sc, 1e-300)
        S = max((c0 - M1 / vm) / csat, 1.0)
        j = J(S)
        vs = vstar(S) if S > 1.0001 else 2 * vm
        d0, d1, d2 = j, j * vs, j * vs ** 2
        if M0 > 1.0 and S > 1.0:
            nl, _, _ = lognormal(M0, M1, M2)
            G = vm * (36 * np.pi) ** (1 / 3) * xg ** (2 / 3) * vimp * STICK * csat * (S - 1)
            d1 += np.sum(G * nl) * dl
            d2 += 2 * np.sum(G * nl * xg) * dl
            if coag:
                W = nl[:, None] * nl[None, :] * dl * dl
                d0 -= 0.5 * np.sum(BET * W)
                d2 += np.sum(BET * W * xg[:, None] * xg[None, :])
        return np.array([d0, d1, d2]) / sc
    n0, v0 = 1e10, vstar(S0)                                                         # negligible seed of critical nuclei
    return solve_ivp(f, [0, T_END], np.array([n0, n0 * v0, n0 * v0 ** 2]) / sc, method="LSODA", rtol=1e-6, atol=[1e-20, 1e-40, 1e-60], dense_output=True)

def moments(sol, t):
    return sol.sol(t) * sc
def stats(M):
    M0, M1, M2 = M
    vg = M1 ** 2 / (M0 ** 1.5 * M2 ** 0.5)
    return (6 * vg / np.pi) ** (1 / 3), np.exp(np.sqrt(max(np.log(M0 * M2 / M1 ** 2), 0) / 9.0))   # d_g, sigma_g

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
7. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

sols = {True: run(True), False: run(False)}
tt = np.logspace(-6, np.log10(T_END), 300)
print(f"S0 = {S0:g}: critical nucleus d* = {(6*vstar(S0)/np.pi)**(1/3)*1e9:.2f} nm, J(S0) = {J(S0):.2e} m-3 s-1")
for coag in (False, True):
    for t in T_PSD:
        M = moments(sols[coag], t)
        dg, sg = stats(M)
        print(f"coagulation {str(coag):5s} t = {t:7.1e} s: M0 = {M[0]:.2e} m-3, d_g = {dg*1e9:6.1f} nm, sigma_g = {sg:.2f}, S = {(c0-M[1]/vm)/csat:8.1f}")

fig, axs = plt.subplots(1, 4, figsize=(19 * cm, 5.6 * cm))
cols3 = ["tab:blue", "tab:orange", "tab:red"]
dGs = lambda g, S: 16 * np.pi * vm ** 2 * g ** 3 / (3 * kT ** 2 * np.log(S) ** 2)    # barrier of Eq. 10.12, J
# Panel A: Gibbs energy of a spherical nucleus (Example 10.2)
r = np.linspace(1e-12, 6e-9, 600)
for S_, c in zip(S_A, cols3):
    dG = (4 * np.pi * r ** 2 * GAMMA - 4 * np.pi / 3 * r ** 3 * kT * np.log(S_) / vm) / kT
    axs[0].plot(r * 1e9, dG, color=c, label=rf"$S$ = {S_}")
    axs[0].plot(2 * GAMMA * vm / (kT * np.log(S_)) * 1e9, dGs(GAMMA, S_) / kT, "o", ms=3, color=c)
axs[0].axhline(0, color="0.6", lw=.4)
axs[0].set_ylim(-60, 80)
axs[0].set_xlim(0, 2)
axs[0].set_xlabel("Nucleus radius $r$ (nm)")
axs[0].set_ylabel(r"$\Delta G(r)/k_\mathrm{B}T$")
axs[0].legend(frameon=False, loc="lower left", title=rf"$\gamma$ = {GAMMA*1e3:.0f} mJ m$^{{-2}}$", title_fontsize=6)
# Panel B: nucleation rate against ln S (Example 10.2)
lnS = np.linspace(1.0, 14, 500)
for g, c in zip(GAMMA_B, cols3):
    axs[1].plot(lnS, np.log10(np.maximum(A_PRE * np.exp(-dGs(g, np.exp(lnS)) / kT), 1e-300)), color=c, label=rf"{g*1e3:.0f} mJ m$^{{-2}}$")
    Sc = np.exp(np.sqrt(16 * np.pi * vm ** 2 * g ** 3 / (3 * kT ** 3 * np.log(A_PRE / J_CRIT))))
    axs[1].plot(np.log(Sc), np.log10(J_CRIT), "o", ms=3, color=c)
axs[1].axhline(np.log10(J_CRIT), color="0.6", lw=.4, ls="--")
axs[1].set_ylim(-5, 29)
axs[1].set_xlim(2.5, 12.5)
axs[1].set_xlabel(r"$\ln S$")
axs[1].set_ylabel(r"$\log_{10}\,(J / \mathrm{m^{-3}\,s^{-1}})$")
axs[1].legend(frameon=False, loc="lower right", title=r"$\gamma$", title_fontsize=6)
# Panel C: supersaturation in the quenched parcel (with coagulation)
M = np.array([moments(sols[True], t) for t in tt])
S = np.array([max((c0 - x[1] / vm) / csat, 1.0) for x in M])
axs[2].plot(tt * 1e3, S, color="tab:blue")
axs[2].set_xscale("log")
axs[2].set_yscale("log")
axs[2].set_ylim(0.95, 3e3)
axs[2].set_xlim(1e-1, 1e1)
axs[2].set_xlabel("Time (ms)")
axs[2].set_ylabel(r"Supersaturation $S$")
# Panel D: geometric mean diameter with and without coagulation
for coag, ls_, lab in ((True, "-", "with coagulation"), (False, "--", "without")):
    st = np.array([stats(moments(sols[coag], t)) for t in tt])
    axs[3].plot(tt * 1e3, st[:, 0] * 1e9, ls_, color="tab:orange", label=lab)
axs[3].set_xscale("log")
axs[3].set_yscale("log")
axs[3].set_xlim(1e-2, T_END * 1e3)
axs[3].set_ylim(1, 100)
axs[3].set_xlabel("Time (ms)")
axs[3].set_ylabel(r"Geometric mean diameter $d_\mathrm{g}$ (nm)")
axs[3].legend(frameon=False, loc="upper left")
for a, L in zip(axs, "ABCD"):
    a.text(-0.3, 0.95, L, transform=a.transAxes, fontsize=8, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HERE, SAVE_FIG))
print("saved", SAVE_FIG)
