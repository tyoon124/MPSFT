"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 7 Reactions in Supercritical Fluids
Example 7.2
Activation volume and apparent activation energy from the compressibility of the solvent: In this
example, we evaluate Eqs. 7.9, 7.11, and 7.14 for scCO_2 and water with CoolProp, which provides
rho, kappa_T, and alpha_p from the Span-Wagner and IAPWS-95 Equations of State, and show how an
activation volume is extracted from rate constants measured at several pressures. The results are
shown in Figs. 7.2 and 7.3.

No command-line arguments: edit the SETTINGS block below and run   python Example7.2.py
Part 1 (Fig. 7.2): activation volume of a bimolecular reaction in scCO2 at constant T from the
compressibility link
DV = -kappa_T R T (1 + rho*dc)             (Eq. 7.11; dc = c_TS - c_A - c_B, direct-correlation
integrals)
with kappa_T and rho from CoolProp (Span-Wagner), and the resulting ln[k_c(p)/k_c(p0)] from
integrating Eq. 7.9.
Panels C and D of the same figure analyze measured rate constants: the second-order rate constant
k_x (mole-fraction
basis) of the Diels-Alder reaction maleic anhydride + isoprene in scCO2 at 308, 318, and 333 K
(Paulaitis and Alexander,
Pure Appl. Chem. 59, 61 (1987), Fig. 4; values digitized from the figure and converted from atm to
bar, so that the
   digitization error is not included). They are fitted with
        d ln k_x/dp = -DV/RT ,   DV(p) = DV0 - kappa_T R T (1 + rho*dc)           (Eq. 7.13)
where DV0 is a pressure-independent ("intrinsic") activation volume and dc is taken as constant; the
model is
integrated over p and fitted to ln k_x by least squares (one offset ln k_x(p0) per temperature, DV0
and dc shared
by all temperatures if FIT_MODE = "global", fitted separately if "per_T"). The fit with dc = 0 is
shown for comparison. Part 2 (Fig. 7.3): apparent activation energy along isobars for k_c = A
exp(-E_v/RT) (rho/rho_0)^m,
        E_app = -R d ln k/d(1/T)|_p = E_v - m R T^2 alpha_p                 (Eq. 7.14)
   for scCO2 and for water (alpha_p = isobaric thermal expansivity).
Requires: numpy, scipy, matplotlib, CoolProp.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from CoolProp.CoolProp import PropsSI

font = {'family':'Dejavu Sans',
        'weight':'normal',
        'size':6}
mpl.rc('font',**font)

mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = .5
mpl.rcParams['xtick.minor.width'] = .5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = .5
mpl.rcParams['ytick.minor.width'] = .5
mpl.rcParams['lines.linewidth']=.75

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Settings
Edit the parameters below to change the calculation.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ============================== SETTINGS ====================================
# --- Part 1: activation volume
T1          = 313.15                     # K
P_RANGE     = (7.5, 25.0, 0.05)          # MPa: start, stop, step of the model curves
DC_VALUES   = [0.0, 0.05, 0.10]          # nm^3, dc = c_TS - c_A - c_B  (0 = no specific attraction of the TS)
N_MOLECULAR = 2                          # molecularity n of the reaction (bimolecular)
# --- Part 1, panels C and D: fit of measured k_x(p)  (Paulaitis and Alexander 1987, digitized; T in C -> (p / bar, ln k_x))
PAULAITIS = {
    35: ([80.8, 81.7, 84.1, 87.4, 104.4, 118.2, 125.1, 143.4, 176.2, 203.5, 261.2, 271.7, 345.7, 432.7],
         [0.686, 0.746, 0.839, 1.061, 1.167, 1.190, 1.223, 1.322, 1.430, 1.449, 1.590, 1.594, 1.750, 1.888]),
    45: ([103.0, 111.7, 123.3, 148.4, 230.0, 336.4, 437.6],
         [1.493, 1.586, 1.682, 1.816, 2.087, 2.279, 2.467]),
    60: ([135.4, 172.3, 210.0, 262.8, 332.7, 414.7],
         [2.462, 2.620, 2.779, 2.956, 3.164, 3.331]),
}
FIT_MODE    = "global"                   # "global": dc, DV0 shared by all temperatures; "per_T": fitted separately
COLORS_T    = {35: "tab:blue", 45: "tab:orange", 60: "tab:red"}
MARKERS_T   = {35: "o", 45: "s", 60: ">"}
XLIM_C      = (0, 450)                   # bar
YLIM_C      = (0.5, 3.5)
YLIM_D      = (5, 3000)                  # cm3/mol, -DV on a log axis
# --- Part 2: apparent activation energy
E_V, M_RHO  = 40e3, 2.0                  # J/mol (isochoric activation energy), density exponent m
CO2_ISOBARS = [8.0, 10.0, 15.0]          # MPa
H2O_ISOBARS = [23.0, 25.0, 30.0]         # MPa
CO2_T       = (295.0, 420.0)             # K
H2O_T       = (550.0, 800.0)             # K
N_T_CO2     = 400                        # temperature points along the CO2 isobars
N_T_H2O     = 20001                      # temperature points along the water isobars (the minima are very narrow; 400 points give -2500 instead of -3600 kJ/mol at 23 MPa)
EAPP_YLIM_B = (-500.0, 50.0)             # kJ/mol, y-range of Panel B (CO2)
EAPP_YLIM_C = (-1000.0, 50.0)            # kJ/mol, y-range of Panel C (water; the minimum at 23 MPa is cut, its value is printed)
SAVE_FIG_1  = "figure7.2.pdf"
SAVE_FIG_2  = "figure7.3.pdf"
# ============================================================================

R, NA = 8.314462618, 6.02214076e23
HERE = os.path.dirname(os.path.abspath(__file__))
cm = 1/2.54

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- kappa_T
- dV
- ln_k_model: d ln k_c/dp = kappa_T [rho dc - ... ] (n = 2)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def kappa_T(T, P, fl):
    return PropsSI("isothermal_compressibility", "T", T, "P", P, fl)

# ---------------------------------------------------------------- part 1
p = np.arange(*P_RANGE) * 1e6
rho = np.array([PropsSI("Dmolar", "T", T1, "P", x, "CO2") for x in p])          # mol/m3
kap = np.array([kappa_T(T1, x, "CO2") for x in p])
rhoN = rho * NA * 1e-27                                                           # nm^-3

def dV(dc):
    return -kap * R * T1 * (1 + rhoN * dc) * 1e6                          # cm3/mol

def ln_k_model(dc, pp):                                                           # d ln k_c/dp = kappa_T [rho dc - ... ] (n = 2)
    pg = np.arange(P_RANGE[0], pp.max() + 0.01, 0.01) * 1e6
    r = np.array([PropsSI("Dmolar", "T", T1, "P", x, "CO2") for x in pg]) * NA * 1e-27
    k = np.array([kappa_T(T1, x, "CO2") for x in pg])
    integrand = k * ((1 + r * dc) - (N_MOLECULAR - 1))                            # = -DV/RT - (n-1) kappa_T
    cum = np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(pg))])
    return np.interp(pp * 1e6, pg, cum)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

pc = PropsSI("pcrit", "CO2") / 1e6
Tr = T1 / PropsSI("Tcrit", "CO2")
imax = np.argmax(kap)
print(f"kappa_T max at {p[imax]/1e6:.2f} MPa: {kap[imax]:.3e} 1/Pa; DV(dc=0) = {dV(0.0)[imax]:.0f} cm3/mol")

# ---- panels C, D: fit of measured k_x(p) with  d ln k_x/dp = -DV/RT,  DV = DV0 - kappa_T R T (1 + rho dc)
Ts = sorted(PAULAITIS)
G = {}                                           # property grids over the pressure range of the data of each isotherm
for T in Ts:
    pb = np.array(PAULAITIS[T][0])
    pg = np.linspace(pb.min(), pb.max(), 800) * 1e5
    G[T] = (pg, np.array([kappa_T(T + 273.15, x, "CO2") for x in pg]),
            np.array([PropsSI("Dmolar", "T", T + 273.15, "P", x, "CO2") for x in pg]) * NA * 1e-27)   # Pa, 1/Pa, nm^-3

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Define functions
- dV_fit: cm3/mol on the grid of temperature T
- lnk_fit: P in bar
- fit_k
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def dV_fit(T, dc, dV0):                          # cm3/mol on the grid of temperature T
    pg, k, r = G[T]
    return dV0 - k * R * (T + 273.15) * (1 + r * dc) * 1e6

def lnk_fit(T, P, dc, dV0, a):                   # P in bar
    pg, k, r = G[T]
    f = k * (1 + r * dc) - dV0 * 1e-6 / (R * (T + 273.15))
    cum = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(pg))])
    return a + np.interp(np.asarray(P) * 1e5, pg, cum)

def fit_k(dc_free=True):
    P = {T: np.array(PAULAITIS[T][0]) for T in Ts}
    Y = {T: np.array(PAULAITIS[T][1]) for T in Ts}
    def stderr(res, free):
        J = res.jac[:, free]
        se = np.zeros(res.jac.shape[1])
        se[free] = np.sqrt(np.diag(np.linalg.inv(J.T @ J) * np.sum(res.fun**2) / (len(res.fun) - len(free))))
        return se
    out, par = {}, {}
    if FIT_MODE == "global":
        def resid(x):
            return np.concatenate([lnk_fit(T, P[T], x[0] if dc_free else 0.0, x[1], x[2+i]) - Y[T] for i, T in enumerate(Ts)])
        res = least_squares(resid, [0.0, -30.0] + [Y[T][0] for T in Ts])
        se = stderr(res, ([0] if dc_free else []) + list(range(1, 2 + len(Ts))))
        for i, T in enumerate(Ts):
            out[T] = (res.x[0] if dc_free else 0.0, res.x[1], res.x[2+i])
            par[T] = (out[T][0], se[0], res.x[1], se[1])
        return out, np.sqrt(np.mean(res.fun**2)), par
    allres = []
    for T in Ts:
        def resid(x):
            return lnk_fit(T, P[T], x[0] if dc_free else 0.0, x[1], x[2]) - Y[T]
        res = least_squares(resid, [0.0, -30.0, Y[T][0]])
        se = stderr(res, [0, 1, 2] if dc_free else [1, 2])
        out[T] = (res.x[0] if dc_free else 0.0, res.x[1], res.x[2])
        par[T] = (out[T][0], se[0], res.x[1], se[1])
        allres.append(res.fun)
    return out, np.sqrt(np.mean(np.concatenate(allres)**2)), par

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fit1, rms1, par1 = fit_k(True)
fit0, rms0, par0 = fit_k(False)
print(f"k_x fit, FIT_MODE = {FIT_MODE}:  dc free: rms(ln k_x) = {rms1:.3f};  dc = 0: rms(ln k_x) = {rms0:.3f}")
for T in Ts:
    dc, sdc, d0, sd0 = par1[T]
    dc_0, _, d0_0, sd0_0 = par0[T]
    pg = G[T][0]
    dVT = dV_fit(T, *fit1[T][:2])
    i = np.argmax(-dVT)
    slope = np.polyfit(np.array(PAULAITIS[T][0]) * 1e5, PAULAITIS[T][1], 1)[0]
    print(f"   {T} C: dc = {dc:+.4f} +- {sdc:.4f} nm^3, DV0 = {d0:.1f} +- {sd0:.1f} cm3/mol | dc = 0: DV0 = {d0_0:.1f} +- {sd0_0:.1f} cm3/mol"
          f" | -DV = {-dVT[i]:.0f} cm3/mol at {pg[i]/1e5:.0f} bar, {-dVT[-1]:.0f} at {pg[-1]/1e5:.0f} bar"
          f" | linear fit of ln k_x(p): DV = {-R*(T+273.15)*slope*1e6:.0f} cm3/mol")

fig, axs = plt.subplots(2, 2, figsize=(19*cm*0.7, 12*cm), gridspec_kw={'wspace':0.38, 'hspace':0.4})
ax = axs[0]
cols = ["tab:gray", "tab:blue", "tab:red"]
for dc, c in zip(DC_VALUES, cols):
    ax[0].plot(p/1e6, -dV(dc), color=c, label=rf"$\Delta c$ = {dc:.2f} nm$^3$")
    ax[1].plot(p/1e6, ln_k_model(dc, p/1e6), color=c)
ax[0].set_xlim([5,25])
ax[0].set_yscale("log")
ax[0].axvline(pc, color="k", ls=":", lw=.5)
ax[1].axvline(pc, color="k", ls=":", lw=.5)
ax[1].text(0.97, 0.05, r"$\Delta c = 0$: $k_c$ independent of $p$", color="tab:gray", fontsize=6,
           transform=ax[1].transAxes, ha="right", va="bottom")
ax[0].set_xlabel("$p$ / MPa")
ax[0].set_ylabel(r"$-\Delta V^{\ddagger}$ / cm$^3$ mol$^{-1}$")
ax[0].legend(frameon=False, fontsize=6)
ax[1].set_xlabel("$p$ / MPa")
ax[1].set_ylabel(r"$\ln\,[k_c(p)/k_c(p_0)]$")
ax[1].set_xlim([5,25])
ax[1].set_ylim([-0.05,0.9])
for a, lab in zip(ax, "AB"):
    a.text(-0.26, 0.975, lab, transform=a.transAxes, fontweight="bold", fontsize=7)
ax = axs[1]
for T in Ts:
    c = COLORS_T[T]
    pg = G[T][0] / 1e5
    ax[0].plot(pg, lnk_fit(T, pg, *fit1[T]), color=c)
    ax[0].plot(pg, lnk_fit(T, pg, *fit0[T]), color=c, ls="--", lw=.5)
    ax[0].plot(PAULAITIS[T][0], PAULAITIS[T][1], ls="", color=c, 
               marker=MARKERS_T[T], ms=4, mew=.75, mfc="w", label=f"{T + 273.15:.0f} K")
    ax[1].plot(pg, -dV_fit(T, *fit1[T][:2]), color=c)
    if FIT_MODE != "global":
        ax[1].axhline(-fit1[T][1], color=c, ls=":", lw=.5)
if FIT_MODE == "global":
    ax[1].axhline(-fit1[Ts[0]][1], color="k", ls=":", lw=.5)
    ax[1].text(XLIM_C[1] - 5, -fit1[Ts[0]][1] * 0.8, r"$-\Delta V_0^{\ddagger}$ (intrinsic)", ha="right", va="top", fontsize=6)
for a in ax:
    a.axvline(pc * 10, color="k", ls=":", lw=.5)
    a.set_xlim(*XLIM_C)
    a.set_xlabel("$p$ / bar")
ax[0].set_ylim(*YLIM_C)
ax[0].set_ylabel(r"$\ln\,k_x$")
ax[0].plot([], [], color="k", label=r"fit, $\Delta c$ free")
ax[0].legend(frameon=False, fontsize=6, loc="lower right", ncol=2)
ax[1].set_yscale("log")
ax[1].set_ylim(*YLIM_D)
ax[1].set_ylabel(r"$-\Delta V^{\ddagger}$ / cm$^3$ mol$^{-1}$")
for a, lab in zip(ax, "CD"):
    a.text(-0.26, 0.975, lab, transform=a.transAxes, fontweight="bold", fontsize=7)
plt.savefig(os.path.join(HERE, SAVE_FIG_1), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG_1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
7. Define functions
- alpha_p
- lnk_iso
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ---------------------------------------------------------------- part 2
def alpha_p(T, P, fl):
    return PropsSI("isobaric_expansion_coefficient", "T", T, "P", P, fl)
def lnk_iso(T, P, fl, ref):
    rho_ = PropsSI("Dmolar", "T", T, "P", P, fl)
    return -E_V / (R * T) + M_RHO * np.log(rho_ / ref)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
8. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig, ax = plt.subplots(1, 3, figsize=(19*cm, 6.5*cm), gridspec_kw={'wspace':0.38})
T = np.linspace(*CO2_T, N_T_CO2)
ref = PropsSI("rhomolar_critical", "CO2")
for P, c in zip(CO2_ISOBARS, cols):
    lk = np.array([lnk_iso(t, P*1e6, "CO2", ref) for t in T])
    Eapp = np.array([E_V - M_RHO * R * t**2 * alpha_p(t, P*1e6, "CO2") for t in T]) / 1e3
    ax[0].plot(1000/T, lk, color=c, label=f"{P:.0f} MPa")
    ax[1].plot(T, Eapp, color=c, label=f"{P:.0f} MPa")
    print(f"CO2 {P} MPa: min E_app = {Eapp.min():.0f} kJ/mol at {T[np.argmin(Eapp)]:.1f} K")
lk0 = -E_V/(R*T)
ax[0].plot(1000/T, lk0 + (lnk_iso(T[-1],CO2_ISOBARS[-1]*1e6,"CO2",ref) - lk0[-1]), "k--", lw=.5, label=r"Arrhenius, $E_\mathrm{v}$")
ax[1].axhline(E_V/1e3, color="k", ls="--", lw=.5)
ax[1].set_ylim(*EAPP_YLIM_B)
ax[0].set_xlabel(r"$10^3/T$ / K$^{-1}$")
ax[0].set_ylabel(r"$\ln\,(k_c/k_c^{\mathrm{ref}})$ (scCO$_2$)")
ax[1].set_xlabel("$T$ / K")
ax[1].set_ylabel(r"$E_\mathrm{app}$ / kJ mol$^{-1}$ (scCO$_2$)")
T = np.linspace(*H2O_T, N_T_H2O)
for P, c in zip(H2O_ISOBARS, cols):
    Eapp = (E_V - M_RHO * R * T**2 * PropsSI("isobaric_expansion_coefficient", "T", T, "P", P*1e6, "Water")) / 1e3
    ax[2].plot(T, Eapp, color=c, label=f"{P:.0f} MPa")
    print(f"H2O {P} MPa: min E_app = {Eapp.min():.0f} kJ/mol at {T[np.argmin(Eapp)]:.1f} K")
ax[2].axhline(E_V/1e3, color="k", ls="--", lw=.5)
ax[2].set_ylim(*EAPP_YLIM_C)
ax[2].set_xlabel(r"$T$ / K")
ax[2].set_ylabel(r"$E_\mathrm{app}$ / kJ mol$^{-1}$ (water)")
for a in ax:
    a.legend(frameon=False, fontsize=6)
for a, lab in zip(ax, "ABC"):
    a.text(-0.26, 0.975, lab, transform=a.transAxes, fontweight="bold", fontsize=7)
ax[0].set_xlim([2.4,3.4])
ax[0].set_ylim([-17,-12])
ax[1].set_xlim([298,420])
ax[2].set_xlim([563,793])

plt.savefig(os.path.join(HERE, SAVE_FIG_2), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG_2)
