"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 9 Supercritical Fluid Extraction and Fractionation
Example 9.1
Retrograde solubility of naphthalene in supercritical CO2 from the Peng-Robinson Equation of State:
In this example, we solve Eq. 9.3 with the Peng-Robinson (PR) Equation of State of the PhasePy
package (see Chapter 2), the van der Waals one-fluid mixing rules of Chapter 2, and two binary
parameters, k_12 in the cross attraction and l_12 in the cross covolume, and compare the calculated
solubility of solid naphthalene in CO2 with the measurements of Sako et al. between 308 and 338 K.
The decomposition of the solubility along the 318 K isotherm is shown in Fig. 9.2A and the four
isotherms in Fig. 9.2B.

crossover and the enhancement factor.   Produces Fig. 9.2 (Panel A: Psub/P, Psub/P with the Poynting
factor and the PR solubility y2 along one isotherm, with data; Panel B: four isotherms vs data).
No command-line arguments: edit the SETTINGS block below and run   python Example9.1.py
Eqs. (9.1)-(9.3):  y2 = [Psub(T)/P] * exp[v_s (P - Psub)/RT] / phi2(T,P,y2). phi2 is the fugacity
coefficient of naphthalene in the CO2-rich fluid, computed with the Peng-Robinson EoS object of the
phasepy package (phasepy.preos; component, mixture and logfugef) with van der Waals one-fluid mixing
rules and two binary parameters k12 and l12; the fluid is assumed not to dissolve in the solid, and
Psub is a pure-solid property.  The solid-fluid equilibrium itself (Eq. 9.3) is not part of phasepy
and is solved here for y2 at every (T,P), because phi2 depends on y2. Two binary parameters are
used: k12 in the cross attraction, a12 = (1-k12) sqrt(a1 a2) (phasepy: mixture.kij_cubic), and l12
in the cross covolume, b12 = (1-l12)(b1+b2)/2.  The quadratic mixing rule of phasepy has a linear
covolume (l12 = 0); l12 is introduced by replacing the mixrule function of the EoS object with a
copy of phasepy's qmr in which the covolume is quadratic (function qmr_l12 below).  k12 and l12 are
strongly correlated; the defaults K12, L12 are the values of the valley of the regression (rms
deviation of ln y2) that give solvable and monotonic isotherms in the whole pressure range (the
unconstrained minimum, k12 = -0.01, l12 = -0.19, gives isotherms with wiggles).  k12 may depend
linearly on temperature, K12 = (k0, k1); on this data set the slope of the best solvable and
monotonic fit is zero within 2e-4 1/K (rms deviation of ln y2: 0.21-0.26 for k1 = 0, no improvement
for k1 != 0).  Set K12 = None to run the unconstrained two-parameter regression. Data: Sako, Yamane,
Negishi, Sato, Sekiyu Gakkaishi 37(3), 321-327 (1994): solid solubility of naphthalene in CO2 at
308, 318, 328 and 338 K (T, P, y2 in the fluid phase; 38 points), as tabulated in the data
collection N-13.  A single source is used.  Psub of naphthalene: ln(Psub/Pa) = A + B/T through the
sublimation-pressure values of the Landolt-Boernstein correlation (11.08, 28.71, 69.77, 159.0 Pa at
298.15, 308.15, 318.15, 328.15 K).
Requires: numpy, scipy, matplotlib, phasepy, CoolProp (only for Tc, pc, omega of CO2).
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from CoolProp.CoolProp import PropsSI
from phasepy import component, mixture, preos

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
K12        = (0.02, 0.0)    # k12(T) = K12[0] + K12[1]*(T - T_REF), PR binary parameter in the cross attraction (a number = constant); None = unconstrained regression of K12 (constant) and L12 (slow)
T_REF      = 318.0          # K
L12        = -0.14          # PR binary parameter in the cross covolume
T_ISO      = [308, 318, 328, 338]            # K, isotherms of Panel B
T_DECOMP   = 318                             # K, isotherm decomposed in Panel A
P_DECOMP   = (1.0, 30.0)                     # MPa, pressure range of Panel A
P_RANGE    = (7.5, 30.0)    # MPa, pressure range of the isotherms
NAPH       = dict(Tc=748.4, Pc=4.05e6, omega=0.302, M=128.17e-3, rho_solid=1175.0)     # K, Pa, -, kg/mol, kg/m3
PSUB_AB    = None           # (A, B) of ln(Psub/Pa) = A + B/T; None = fitted to PSUB_POINTS
PSUB_POINTS = [(298.15, 11.08), (308.15, 28.71), (318.15, 69.77), (328.15, 159.02)]     # K, Pa
EXCLUDE    = []             # list of (T/K, P/kPa) of data points left out of the regression
# Sako, Yamane, Negishi, Sato, Sekiyu Gakkaishi 37(3), 321-327 (1994), as tabulated in the data collection N-13:
# (T/K, [P/bar], [y2 x 10^6])
DATA_SAKO = [
 (308, [88.3, 114.7, 134.4, 157.9, 186.3, 197.1, 228.5, 251.0, 288.3], [9750, 13080, 14570, 15850, 16230, 16320, 16620, 17890, 18290]),
 (318, [83.4, 89.2, 92.2, 97.1, 100.0, 122.3, 151.0, 204.0, 227.5, 265.8, 300.1], [1110, 1960, 5710, 8090, 9950, 17020, 20690, 24160, 26640, 28680, 29350]),
 (328, [77.5, 98.1, 109.8, 117.7, 133.4, 138.3, 154.9, 181.4, 218.7, 268.7], [930, 3140, 7610, 12530, 22360, 24340, 30120, 37420, 43980, 51390]),
 (338, [117.7, 127.5, 147.1, 160.8, 187.3, 199.1, 218.7, 225.6], [7340, 10820, 25950, 33500, 46310, 51040, 58940, 63940]),
]
DATA = [(T, pb * 100.0, y * 1e-6) for T, ps, ys in DATA_SAKO for pb, y in zip(ps, ys)]     # (T/K, P/kPa, y2)
SAVE_FIG   = "figure9.2.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
cm = 1/2.54
R = 8.314462618
D = np.array(DATA)
vs = NAPH["M"] / NAPH["rho_solid"]                                           # m3/mol, molar volume of the solid
if PSUB_AB is None:
    pt = np.array(PSUB_POINTS)
    B_, A_ = np.polyfit(1 / pt[:, 0], np.log(pt[:, 1]), 1)
else:
    A_, B_ = PSUB_AB
Psub = lambda T: np.exp(A_ + B_ / T)

# ---- phasepy: components (Tc in K, Pc in bar), binary mixture and Peng-Robinson EoS object
co2 = component(name="CO2", Tc=PropsSI("Tcrit", "CO2"), Pc=PropsSI("pcrit", "CO2") / 1e5, w=PropsSI("acentric", "CO2"))
naph = component(name="naphthalene", Tc=NAPH["Tc"], Pc=NAPH["Pc"] / 1e5, w=NAPH["omega"])
mix = mixture(co2, naph)
mix.kij_cubic(np.zeros((2, 2)))
eos = preos(mix)                                                             # Peng-Robinson, quadratic mixing rule (phasepy)
eos.lij = np.zeros((2, 2))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define functions
- qmr_l12: phasepy's qmr with a quadratic covolume, b_ij = (1-l_ij)(b_i+b_j)/2
- ln_phi2: ln phi_2 of naphthalene in the fluid (phasepy logfugef, P in bar)
- y_sat: Eq. (9.3): y2 = Psub/(P phi2) exp[vs (P-Psub)/RT], smallest root with y2 < 0.1
- rms
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def qmr_l12(X, RT, ai, bi, order, Kij):                                      # phasepy's qmr with a quadratic covolume, b_ij = (1-l_ij)(b_i+b_j)/2
    aij = np.sqrt(np.outer(ai, ai)) * (1 - Kij)
    ax = aij * X
    D_ = np.sum(ax.T * X)
    bij = 0.5 * (bi[:, None] + bi[None, :]) * (1 - eos.lij)
    B = X @ bij @ X
    if order == 0:
        return D_, B
    Di = 2 * np.sum(ax, axis=1)
    Bi = 2 * (bij @ X) - B                      # d(n a)/dn_i and d(n b)/dn_i
    if order == 1:
        return D_, Di, B, Bi
    raise Exception("derivative order not implemented for l12 != 0")
eos.mixrule = qmr_l12

def ln_phi2(T, P, y2, k12, l12):                                             # ln phi_2 of naphthalene in the fluid (phasepy logfugef, P in bar)
    eos.kij = np.array([[0., k12], [k12, 0.]])
    eos.lij = np.array([[0., l12], [l12, 0.]])
    return eos.logfugef(np.array([1 - y2, y2]), T, P / 1e5, "V")[0][1]

def y_sat(T, P, k12, l12):                                                   # Eq. (9.3): y2 = Psub/(P phi2) exp[vs (P-Psub)/RT], smallest root with y2 < 0.1
    ps = Psub(T)
    rhs = np.log(ps / P) + vs * (P - ps) / (R * T)
    f = lambda ly: ly + ln_phi2(T, P, np.exp(ly), k12, l12) - rhs
    g = np.linspace(-30, np.log(0.1), 100)
    v = [f(t) for t in g]
    for i in range(len(g) - 1):
        if v[i] * v[i + 1] < 0:
            return np.exp(brentq(f, g[i], g[i + 1]))
    return np.nan

keep = np.array([not any(abs(d[0] - t) < 0.02 and abs(d[1] - p) < 1 for t, p in EXCLUDE) for d in D])
kT = lambda T, k=None: (lambda q: q[0] + q[1] * (T - T_REF) if isinstance(q, (tuple, list)) else q)(K12 if k is None else k)
def rms(p):
    e = np.array([np.log(y_sat(T, P * 1e3, kT(T, p[0]), p[1]) / y) for T, P, y in D[keep]])
    return np.sqrt(np.mean(e**2)) if np.isfinite(e).all() else 9.9

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if K12 is None:
    from scipy.optimize import minimize
    r_ = minimize(rms, [0.0, -0.18], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-4, maxiter=150))
    K12, L12 = float(r_.x[0]), float(r_.x[1])
dev = np.array([y_sat(T, P * 1e3, kT(T), L12) / y - 1 for T, P, y in D])
print(f"Psub: ln(Psub/Pa) = {A_:.3f} + {B_:.1f}/T   (Psub(308.15 K) = {Psub(308.15):.1f} Pa)")
print(f"k12 = {K12}, l12 = {L12:.4f};  root-mean-square deviation of ln y2 = {rms([K12, L12]):.3f};  AARD = {100 * np.mean(np.abs(dev[keep])):.0f} % ({keep.sum()} points)")
for Ti in T_ISO:
    print(f"  {Ti} K: AARD = {100 * np.mean(np.abs(dev[keep & (D[:, 0] == Ti)])):.0f} %")
Pg = np.linspace(P_RANGE[0], P_RANGE[1], 400) * 1e6
Y = {Ti: np.array([y_sat(Ti, p, kT(Ti), L12) for p in Pg]) for Ti in T_ISO}
for a, b in zip(T_ISO[:-1], T_ISO[1:]):
    s = Y[b] - Y[a]
    ok = (Pg >= 10e6) & np.isfinite(s)
    sg = np.sign(s[ok])
    i = np.where(sg[:-1] != sg[1:])[0]
    Pc_ = Pg[ok]
    if len(i):
        print(f"model crossover between {a} and {b} K at about {Pc_[i[0]] / 1e6:.1f} MPa")
    else:
        print(f"no crossover between {a} and {b} K in the pressure range")
# Decomposition of Eq. (9.3) along the isotherm T_DECOMP:  y2 = (Psub/P) * E,   E = Poynting factor / phi2
Pd = np.linspace(P_DECOMP[0], P_DECOMP[1], 90) * 1e6
yd = np.array([y_sat(T_DECOMP, p, kT(T_DECOMP), L12) for p in Pd])
yd0 = np.array([y_sat(T_DECOMP, p, 0.0, 0.0) for p in Pd])
y_id = Psub(T_DECOMP) / Pd                                                   # ideal-gas solubility (sublimation pressure only)
Poy = np.exp(vs * (Pd - Psub(T_DECOMP)) / (R * T_DECOMP))                    # Poynting factor of the solid
E = yd * Pd / Psub(T_DECOMP)                                                 # enhancement factor
for p_ in (5e6, 10e6, 15e6, 20e6):
    i = np.argmin(abs(Pd - p_))
    print(f"{T_DECOMP} K, {p_/1e6:.0f} MPa:  Psub/P = {y_id[i]:.2e},  E = {E[i]:.2e},  Poynting = {Poy[i]:.2f},  1/phi2 = {E[i]/Poy[i]:.2e},  y2 = {yd[i]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(19*cm*0.8, 8*cm), gridspec_kw={'wspace':0.28})
# ---- Panel A: contributions to the solubility along one isotherm
a = ax[0]
Pc1_MPa = PropsSI("pcrit", "CO2") / 1e6
a.semilogy(Pd / 1e6, y_id, color="tab:brown", label=r"$p_2^{\mathrm{sub}}/p$")
a.semilogy(Pd / 1e6, y_id * Poy, color="tab:brown", ls="--", label=r"$p_2^{\mathrm{sub}}/p$ with Poynting factor")
a.semilogy(Pd / 1e6, yd0, color="0.55", label=r"PR, $k_{12} = l_{12} = 0$")
a.semilogy(Pd / 1e6, yd, color="k", lw=1.2, label=f"PR, $k_{{12}}$ = {kT(T_DECOMP):g}, $l_{{12}}$ = {L12:g}")
m = np.abs(D[:, 0] - T_DECOMP) < 0.5
a.semilogy(D[m, 1] / 1e3, D[m, 2], "o", ms=2.2, mfc="none", mec="k", mew=.5, label="data")
a.plot([Pc1_MPa, Pc1_MPa], [1e-6, 4e-2], color="0.4", ls=(0, (4, 3)), lw=.5)        # stops below the legend
a.text(Pc1_MPa + 0.2, 1.5e-6, r"$p_{\mathrm{crit},1}$", fontsize=6, color="0.3")
i = np.argmin(abs(Pd - 20e6))
a.annotate("", xy=(20, yd[i]), xytext=(20, y_id[i]), arrowprops=dict(arrowstyle="<->", lw=.6, color="tab:blue", shrinkA=0, shrinkB=0))
a.text(19.7, np.sqrt(yd[i] * y_id[i]), f"$E$ = {E[i]:.1e}".replace("e+0", r"$\times 10^{").replace("e+", r"$\times 10^{") + "}$", color="tab:blue", fontsize=5.5, ha="right", va="center")
a.set_xlabel("$p$ / MPa")
a.set_xlim(*P_DECOMP)
a.set_ylim(1e-6, 10.0)                                                     # top decade left free for the legend
a.set_xticks([1, 5, 10, 15, 20, 25, 30])
a.set_ylabel(r"$y_2$ (naphthalene, mole fraction)")
a.minorticks_off()
a.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])                            # no ticks above y2 = 0.1 (the top is space for the legend)
a.legend(frameon=False, fontsize=5.0, loc="upper left", bbox_to_anchor=(0.0, 1.0), title=f"naphthalene in CO$_2$, {T_DECOMP} K", title_fontsize=5.2, borderaxespad=0.2)

# ---- Panel B: calculation vs data
b = ax[1]
cols = dict(zip(T_ISO, ["tab:blue", "tab:green", "tab:orange", "tab:red"]))
symbols = ["o",">","s","p"]
for counter,Ti in enumerate(T_ISO):
    b.plot(Pg / 1e6, Y[Ti], color=cols[Ti], label=f"{Ti} K")
    m = np.abs(D[:, 0] - Ti) < 2.5
    b.plot(D[m, 1] / 1e3, D[m, 2], symbols[counter], ms=3, mfc="none", mec=cols[Ti], mew=.5)
mx = np.array([abs(d[0] - EXCLUDE[0][0]) < 0.02 and abs(d[1] - EXCLUDE[0][1]) < 1 for d in D]) if EXCLUDE else np.zeros(len(D), bool)
b.plot(D[mx, 1] / 1e3, D[mx, 2], "x", ms=3, color="0.3", mew=.6)
b.set_xlabel("$p$ / MPa")
b.set_ylabel(r"$y_2$ (naphthalene, mole fraction)")
b.set_xlim(*P_RANGE)
b.set_ylim(0,0.07)
#b.set_yscale('log')
b.legend(frameon=False, fontsize=5.2, loc="upper left", title="PR model (lines), data (open markers)", title_fontsize=5.2)
b.text(0.97, 0.05, f"$k_{{12}}$ = {kT(T_ISO[0]):g}, $l_{{12}}$ = {L12:g}", transform=b.transAxes, ha="right", fontsize=5.5)
for a_, lab in zip(ax, "AB"):
    a_.text(-0.225, 1.02, lab, transform=a_.transAxes, fontweight="bold", fontsize=7)
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
