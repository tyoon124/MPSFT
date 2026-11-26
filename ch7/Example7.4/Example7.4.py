"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 7 Reactions in Supercritical Fluids
Example 7.4
Piston effect in a closed cell of near-critical CO_2: In this example, we solve the one-dimensional
energy equation of a closed fluid layer, in which the pressure is uniform but changes with time, and
compare the thermal equilibration of near-critical CO_2 with that of a normal liquid. The results
are shown in Fig. 7.5.

No command-line arguments: edit the SETTINGS block below and run   python Example7.4.py
One-dimensional, low-Mach-number ("acoustically filtered") model of a closed cell of half-width L
(symmetry plane at z = L, wall at z = 0 held at T_w = T_0 + dT from t = 0, no gravity, no
convection):
rho c_p dT/dt = d/dz( lambda dT/dz ) + alpha_p T dp/dt ,      dp/dt = sum(alpha_p a) / sum(kappa_S)
(dz-weighted), where the uniform pressure p(t) follows from the constraint that the cell volume is
constant (Onuki, Hao, Ferrell 1990). All properties (rho, c_p, lambda with critical enhancement,
alpha_p, kappa_T) come from CoolProp at (T, p).
Output: time at which the cell center has reached 90 % of the wall step, with and without the piston
term, the conduction time L^2/alpha, gamma = c_p/c_v, and the estimate t_PE = t_D/(gamma-1)^2.
Requires: numpy, scipy, matplotlib, CoolProp.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
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
FLUID    = "CO2"
L        = 5.0e-3          # m, half-width of the cell (10 mm cell)
DT_WALL  = 0.01            # K, step of the wall temperature (small: linear regime)
N_CELLS  = 120             # grid cells, geometrically refined towards the wall
STRETCH  = 1.06            # ratio of successive cell widths
T_END    = 5.0e4           # s
RTOL     = 1.0e-8          # solver tolerances act on the scaled variables theta = (T-T0)/DT_WALL, pi = (p-p0)/dp_est
ATOL     = 1.0e-10
# states: label, T0 [K], pressure p0 [MPa] (None = critical density)
STATES   = [("$T_\\mathrm{crit}$ + 0.5 K, $\\rho_\\mathrm{crit}$", 0.5, None),
            ("$T_\\mathrm{crit}$ + 5 K, $\\rho_\\mathrm{crit}$",   5.0, None),
            ("liquid, 290 K, 10 MPa", None, 10.0)]
SAVE_FIG = "figure7.5.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
cm = 1/2.54
Tc, rhoc = PropsSI("Tcrit", FLUID), PropsSI("rhomass_critical", FLUID)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- make_props
- run
- t90
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def make_props(T0, p0):
    dp_est = PropsSI("isobaric_expansion_coefficient", "T", T0, "P", p0, FLUID) / PropsSI("isothermal_compressibility", "T", T0, "P", p0, FLUID) * DT_WALL
    Tg = np.linspace(T0 - 0.01, T0 + DT_WALL + 0.005, 40)
    pg = p0 + dp_est * np.linspace(-0.5, 2.0, 40)
    names = ["Dmass", "Cpmass", "conductivity", "isobaric_expansion_coefficient", "isothermal_compressibility"]
    tab = {n: np.array([[PropsSI(n, "T", t, "P", p, FLUID) for p in pg] for t in Tg]) for n in names}
    return Tg, pg, {n: RectBivariateSpline(Tg, pg, tab[n], kx=3, ky=3) for n in names}

def run(T0, p0, piston):
    Tg, pg, sp = make_props(T0, p0)
    dp_est = PropsSI("isobaric_expansion_coefficient", "T", T0, "P", p0, FLUID) / PropsSI("isothermal_compressibility", "T", T0, "P", p0, FLUID) * DT_WALL
    w = STRETCH ** np.arange(N_CELLS)
    w = w / w.sum() * L                    # cell widths (wall at z = 0)
    zc = np.cumsum(w) - 0.5 * w
    dzc = np.diff(zc)
    Tw = T0 + DT_WALL
    def f(t, y):
        # y = [theta_1..theta_N, pi]: scaled deviations, so that the tolerances are relative to the 10 mK step, not to 300 K
        T, p = np.clip(T0 + DT_WALL * y[:-1], Tg[0], Tg[-1]), np.clip(p0 + dp_est * y[-1], pg[0], pg[-1])
        rho, cp, lam, al, kt = (sp[n].ev(T, np.full_like(T, p)) for n in
                                ["Dmass", "Cpmass", "conductivity", "isobaric_expansion_coefficient", "isothermal_compressibility"])
        lam_f = 0.5 * (lam[1:] + lam[:-1])
        flux = np.concatenate([[lam[0] * (Tw - T[0]) / zc[0]], lam_f * (T[:-1] - T[1:]) / dzc, [0.0]])  # flux in at face i-1/2 .. i+1/2
        # flux[i] is the conductive heat flux entering cell i from the wall side; flux[i+1] leaves towards the center
        div = (flux[:-1] - np.concatenate([lam_f * (T[:-1] - T[1:]) / dzc, [0.0]])) / w
        a = div / (rho * cp)
        b = al * T / (rho * cp)
        if not piston:
            return np.concatenate([a / DT_WALL, [0.0]])
        ks = kt - al**2 * T / (rho * cp)                                     # kappa_S
        pdot = np.sum(al * a * w) / np.sum((ks) * w)
        return np.concatenate([(a + b * pdot) / DT_WALL, [pdot / dp_est]])
    y0 = np.zeros(N_CELLS + 1)
    ts = np.logspace(-3, np.log10(T_END), 300)
    sol = solve_ivp(f, (0, T_END), y0, t_eval=ts, method="LSODA", rtol=RTOL, atol=ATOL)
    if sol.status != 0 or sol.y.shape[1] != len(ts):
        print("solver:", sol.message, "reached t =", sol.t[-1])
    theta = sol.y[N_CELLS - 1]                                               # cell next to the symmetry plane
    return ts, theta, dp_est * sol.y[-1]

def t90(ts, th):
    i = np.argmax(th >= 0.9)
    return ts[i] if th.max() >= 0.9 else np.inf

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig, ax = plt.subplots(1, 2, figsize=(19*cm*0.75, 6.5*cm), gridspec_kw={'wspace':0.35})
cols = ["tab:red", "tab:orange", "tab:blue"]
print(f"{'state':45s} {'t_D/s':>9s} {'gamma':>7s} {'t_PE est/s':>11s} {'t90 PE/s':>10s} {'t90 cond/s':>11s} {'dp/kPa':>8s}")
for (lab, dT0, pMPa), c in zip(STATES, cols):
    if dT0 is not None:
        T0 = Tc + dT0
        p0 = PropsSI("P", "T", T0, "Dmass", rhoc, FLUID)
    else:
        T0 = 290.0
        p0 = pMPa * 1e6
    rho0, cp0, lam0 = (PropsSI(n, "T", T0, "P", p0, FLUID) for n in ["Dmass", "Cpmass", "conductivity"])
    cv0 = PropsSI("Cvmass", "T", T0, "P", p0, FLUID)
    gam = cp0 / cv0
    tD = L**2 / (lam0 / (rho0 * cp0))
    ts, th_pe, dp = run(T0, p0, True)
    ts, th_c, _ = run(T0, p0, False)
    print(f"{lab.replace('$','').replace(chr(92)+'mathrm','')[:44]:45s} {tD:9.3g} {gam:7.1f} {tD/(gam-1)**2:11.3g} {t90(ts,th_pe):10.3g} {t90(ts,th_c):11.3g} {dp[-1]/1e3:8.1f}")
    ax[0].plot(ts, th_pe, color=c, label=lab)
    ax[0].plot(ts, th_c, color=c, ls="--", lw=.5)
    ax[1].plot(ts / tD, th_pe, color=c)
    ax[1].plot(ts / tD, th_c, color=c, ls="--", lw=.5)
ax[0].set_xscale("log")
ax[1].set_xscale("log")
for a in ax:
    a.xaxis.set_minor_locator(mpl.ticker.NullLocator())   # no minor ticks on the log time axes
ax[0].set_xlabel("$t$ / s")
ax[0].set_ylabel(r"$(T_\mathrm{center}-T_0)/(T_\mathrm{wall}-T_0)$")
ax[1].set_xlabel(r"$t/t_\mathrm{D}$,  $t_\mathrm{D} = L^2/\alpha$")
ax[1].set_ylabel(r"$(T_\mathrm{center}-T_0)/(T_\mathrm{wall}-T_0)$")
ax[0].legend(frameon=False, fontsize=5.5, loc="upper left")
ax[0].set_ylim([-0.05,1.05])
ax[1].set_ylim([-0.05,1.05])
ax[0].set_xlim([5e-3,5e4])
ax[1].set_xlim([1e-5,50])
for a, lab in zip(ax, "AB"):
    a.text(-0.26, 0.975, lab, transform=a.transAxes, fontweight="bold", fontsize=7)
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
