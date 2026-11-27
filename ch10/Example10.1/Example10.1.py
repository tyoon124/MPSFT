"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 10 Particle Formation from Supercritical Solutions
Example 10.1
Isentropic RESS nozzle expansion of CO2: In this example, we follow the expansion of pure CO2 from
the pre-expansion state of Sec. 10.4.1 (35 MPa) along an isentrope with CoolProp, i.e. with the
Span-Wagner Equation of State as the closure of the flow equations (Eqs. 10.5-10.7 for an adiabatic
nozzle without friction). At every pressure the velocity follows from the energy balance,
u=sqrt(2(h_0-h)), and the mass flux is G=rho u. The sonic throat is the point of maximum G, where u
equals the equilibrium speed of sound, and A/A^*=G^*/G is the cross-section needed to pass the same
mass flow at the other pressures. The script is Example10.1.py in Examples/ch10; the temperatures
and pressures are set in its settings block. The results are shown in Fig. 10.3.

No command-line arguments: edit the SETTINGS block below and run   python Example10.1.py
CO2 is expanded from the pre-expansion state (P0, T0) along an isentrope (equilibrium, homogeneous
flow, no friction, no heat exchange, no solute). For every pressure P on the path the script
evaluates, with the Span-Wagner equation of state (CoolProp),
T, rho, vapor quality, the velocity u = sqrt(2 (h0 - h)) of the steady adiabatic flow, and the mass
flux G = rho u. The sonic throat is the point of maximum mass flux G* (dG/dP = 0), where u equals
the equilibrium speed of sound; the ratio A/A* = G*/G is the cross-section needed to pass the same
mass flow at the other pressures (converging-diverging nozzle). Marked on the path: the throat, the
crossing of the saturation curve (entry into the two-phase region), and T = Tc. Real RESS jets are
not in equilibrium (the fluid supercools and condensation is delayed, see the text of Example 10.1), so the two-phase
part of the path is the limit of an infinitely fast phase change.
Produces Fig. 10.3.   Requires: numpy, matplotlib, CoolProp.
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
FLUID    = "CO2"
P0       = 35e6                      # Pa, pre-expansion pressure
T0_LIST  = [313.15, 333.15, 353.15]  # K, pre-expansion temperatures
P_END    = 0.6e6                     # Pa, end of the path (just above the triple-point pressure, 0.52 MPa)
N_P      = 400                       # points along the path
SAVE_FIG = "figure10.3.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Set constants and grids
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
cm = 1/2.54
Tc, Pc = PropsSI("Tcrit", FLUID), PropsSI("pcrit", FLUID)
Tmin = PropsSI("Tmin", FLUID) + 1.0
Pgrid = np.logspace(np.log10(P0), np.log10(P_END), N_P)
cols = ["tab:blue", "tab:orange", "tab:red"]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define functions
- state_ps: (T, rho, h, q) at entropy s0 and pressure P. Avoids the CoolProp PS flash, which fails
near the critical point
- path
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def state_ps(s0, P):
    """(T, rho, h, q) at entropy s0 and pressure P. Avoids the CoolProp PS flash, which fails near the critical point in some versions:
    two-phase states from the saturated-liquid/vapor properties, single-phase states by bisection on S(T, P) = s0."""
    S = lambda T: PropsSI("Smass", "T", T, "P", P, FLUID)
    if P < Pc:
        Ts = PropsSI("T", "P", P, "Q", 0, FLUID)
        sL, sV = PropsSI("Smass", "P", P, "Q", 0, FLUID), PropsSI("Smass", "P", P, "Q", 1, FLUID)
        if sL <= s0 <= sV:
            q = (s0 - sL) / (sV - sL)
            f = lambda Q, p: PropsSI(p, "P", P, "Q", Q, FLUID)
            rho = 1.0 / ((1 - q) / f(0, "Dmass") + q / f(1, "Dmass"))
            h = f(0, "Hmass") + q * (f(1, "Hmass") - f(0, "Hmass"))
            return Ts, rho, h, q
        lo, hi = (Tmin, Ts - 1e-6) if s0 < sL else (Ts + 1e-6, 700.0)
    else:
        lo, hi = Tmin, 700.0
    for _ in range(80):                                   # S increases with T at constant P
        mid = 0.5 * (lo + hi)
        if S(mid) < s0:
            lo = mid
        else:
            hi = mid
    T = 0.5 * (lo + hi)
    return T, PropsSI("Dmass", "T", T, "P", P, FLUID), PropsSI("Hmass", "T", T, "P", P, FLUID), -1.0

def path(T0):
    s0, h0 = PropsSI("Smass", "T", T0, "P", P0, FLUID), PropsSI("Hmass", "T", T0, "P", P0, FLUID)
    out = []
    for P in Pgrid:
        T, rho, h, q = state_ps(s0, P)
        u = np.sqrt(max(2 * (h0 - h), 0.0))
        out.append((P, T, rho, q, u, rho * u))
    return np.array(out)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig, axs = plt.subplots(1, 3, figsize=(19 * cm, 7.0 * cm))
ps = np.linspace(0.6e6, Pc, 200)
Tsat = np.array([PropsSI("T", "P", p, "Q", 0, FLUID) for p in ps])
axs[0].plot(ps / 1e6, Tsat, color="0.5", ls="--", lw=.6)
axs[0].text(1.05, 238, "saturation curve", color="0.4", fontsize=5.5, rotation=34)
axs[0].axhline(Tc, color="0.7", lw=.4)
axs[0].text(24, Tc + 1.5, r"$T_\mathrm{crit}$", color="0.4")
print(f"{'T0/K':>6s} {'P*/MPa':>7s} {'T*/K':>7s} {'rho*':>7s} {'u*/m/s':>7s} {'G*/kg/m2/s':>11s} {'P_sat/MPa':>9s} {'T_sat/K':>8s} {'q at P_end':>10s}")
for T0, c in zip(T0_LIST, cols):
    d = path(T0)
    P, T, rho, q, u, G = d.T
    k = np.argmax(G)
    ent = np.where(q >= 0)[0]
    ke = ent[0] if len(ent) else None
    lab = f"{T0:.0f} K"
    axs[0].plot(P / 1e6, T, color=c, label=lab)
    axs[1].plot(P / 1e6, rho, color=c)
    axs[2].plot(P[1:] / 1e6, G[k] / G[1:], color=c)
    for a, y in zip(axs, (T, rho, G[k] / np.maximum(G, 1e-9))):
        a.plot(P[k] / 1e6, y[k], "o", ms=3, mfc=c, mec=c)                       # sonic throat
        if ke is not None:
            a.plot(P[ke] / 1e6, y[ke], "s", ms=3, mfc="w", mec=c, mew=.7)   # saturation curve crossing
    print(f"{T0:6.1f} {P[k]/1e6:7.2f} {T[k]:7.1f} {rho[k]:7.1f} {u[k]:7.1f} {G[k]:11.0f} {P[ke]/1e6 if ke is not None else np.nan:9.2f} {T[ke] if ke is not None else np.nan:8.1f} {q[-1]:10.3f}")

axs[0].set_ylabel("Temperature (K)")
axs[1].set_ylabel(r"Density (kg m$^{-3}$)")
axs[2].set_ylabel(r"Area ratio $A/A^{*}$")
axs[2].set_yscale("log")
axs[2].set_ylim(0.9, 60)
axs[2].set_ylim([0.96,20])
axs[0].set_xlim([30,0.7])
axs[1].set_xlim([30,0.7])
axs[2].set_xlim([30,0.7])
for a, L in zip(axs, "ABC"):
    a.set_xscale("log")
    a.set_xlim(P0 / 1e6 * 1.05, P_END / 1e6 * 0.95)
    a.set_xlabel("Pressure (MPa)")
    a.text(-0.2 if L == "A" else -0.2, 1.0, L, transform=a.transAxes, fontsize=8, fontweight="bold")
axs[0].legend(frameon=False, loc="lower left", title=r"$T_0$", title_fontsize=6)
axs[1].plot([], [], "o", ms=3, color="k", label="sonic throat")
axs[1].plot([], [], "s", ms=3, mfc="w", mec="k", label="enters two-phase region")
axs[1].legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(HERE, SAVE_FIG))
print("saved", SAVE_FIG)
