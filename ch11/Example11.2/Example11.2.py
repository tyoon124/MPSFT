"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 11 Nanomaterial and Thin-Film Synthesis
Example 11.2
Staying out of the two-phase region: The criterion that a drying path must never cross a
liquid-vapor boundary (Eq. 11.4 and Table 11.1) is a statement about the phase diagram of the fluid
that actually fills the pores, and this fluid changes in composition along the process. In this
example, we examine two drying paths of a gel filled with ethanol, the pure-CO2 route and the
mixture route, and the results are shown in Fig. 11.6.

No command-line arguments: edit the SETTINGS block below and run   python Example11.2.py
Panel A: p-T diagram of CO2 (CoolProp, Span-Wagner) with two drying paths of a gel whose pores are
filled with a liquid: (1) evaporation, which crosses the
liquid-vapor line, and (2) the CO2 route: liquid CO2 -> heating above Tc at constant p -> isothermal
depressurization, which never crosses it. Inset of panel A: critical temperature of CO2 confined in
a slit pore of width H from the Fisher-Nakanishi relation (Eq. 6.20 of Chapter 6),
   (Tc,bulk - Tc,pore)/Tc,bulk = A (sigma/H)^(1/nu),
with the amplitude A calibrated on the apparent critical temperatures of the three walls of
Example 6.4 (shifts of 10.1, 8.3, and 9.8 % at sigma/H = 1/7); the band spans the three walls.
Panel B: p-x-y diagram (two-phase region in the (z, p)
plane) of CO2 + ethanol at the drying temperature (dashed: a higher temperature), calculated with
PhasePy: Peng-Robinson equation of state with the Wong-Sandler mixing rule and the NRTL model for
the excess Gibbs energy, with the binary parameters
k12, g12, g21 regressed to CO2 + ethanol VLE data at 298-373 K by Kim, Yoo, Gee and Yoon, J. Chem.
Eng. Data 71 (2026) 1758 (their Table 4).
The bubble-point branch is obtained with bubblePy at low pressure and, near the mixture critical
point where bubblePy fails, by pseudo-arclength
continuation of the equal-fugacity equations (fugacity coefficients from PhasePy); the dew branch is
the vapor composition of the same states. Two depressurization paths of the same autoclave are
drawn: (i) at constant composition z, which enters the
two-phase region; (ii) after flushing with CO2 at a pressure above the critical region of the
mixture, which does not. CAUTION: the composition z of the autoclave content, the pressures and the
temperature of the drying step are illustrative; the model is accurate in the
range of the data it was fitted to, and near the mixture critical point (~8 MPa at 313 K) a cubic
equation of state is only approximate (Sec. 11.2.4).
Produces Fig. 11.6 (figure11.6.pdf).   Requires: numpy, matplotlib, CoolProp (panel A), phasepy
(panel B).
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from phasepy import component, mixture, preos
from phasepy.equilibrium import bubblePy

font = {'family':'Dejavu Sans','weight':'normal','size':6}
mpl.rc('font',**font)
for ax_ in ('xtick','ytick'):
    mpl.rcParams[ax_+'.major.size'] = 3
    mpl.rcParams[ax_+'.minor.size'] = 3
    mpl.rcParams[ax_+'.major.width'] = .5
    mpl.rcParams[ax_+'.minor.width'] = .5
mpl.rcParams['lines.linewidth']=.75
mpl.rcParams['axes.linewidth']=.5

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Settings
Edit the parameters below to change the calculation.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ============================== SETTINGS ====================================
# CO2 (1) + ethanol (2): critical constants and acentric factors (Table 3 of Kim et al. 2026); Tc [K], Pc [bar], omega
CO2     = dict(name="CO2",     Tc=304.2, Pc=73.83, w=0.225)
ETHANOL = dict(name="ethanol", Tc=513.9, Pc=61.4,  w=0.645)
K12        = 0.4006                  # Wong-Sandler binary interaction parameter k12 (Table 4 of Kim et al. 2026)
G12, G21   = 526.09, -61.72          # NRTL energy parameters g12, g21 [K]; tau_ij = g_ij / T (Table 4)
ALPHA      = 0.3                     # NRTL non-randomness parameter (fixed)
T_HOT      = 343.15                  # second temperature, shown as dashed outline [K]
P_MAX      = 16.0                    # upper pressure of panel B [MPa]
T_DRY      = 313.15                  # temperature of the drying step [K]
P_DRY      = 12.0                    # pressure of the drying step, above the critical pressure of the mixture at T_DRY [MPa]
Z_START    = 0.60                    # CO2 mole fraction of the autoclave content after pressurizing a gel filled with ethanol (illustrative)
P_END      = 0.1                     # pressure after depressurization [MPa]
LIQ_PATH_T = 290.0                   # temperature of the liquid-CO2 exchange of the CO2 route [K]
# Confinement (inset of panel A): Fisher-Nakanishi relation of Chapter 6 calibrated on Example 6.4
NU         = 0.630                   # correlation-length exponent
SHIFT_EX64 = [0.101, 0.083, 0.098]   # relative shifts of the apparent critical temperature in Example 6.4 (repulsive, attractive, strongly attractive wall)
SIGMAH_EX64 = 1/7                    # sigma/H of the pore of Example 6.4
SIGMA_CO2  = 0.38                    # molecular diameter of CO2 [nm] (Sec. 6.3.3)
H_PRINT    = [3, 5, 10, 20]          # pore widths [nm] for which the confined critical temperature is printed
SAVE_FIG   = "figure11.6.pdf"
# ============================================================================

R = 8.314462618
HERE = os.path.dirname(os.path.abspath(__file__))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
cm = 1/2.54
R = 8.314462618
# ---------------------------------------------------------------- PhasePy model
mix = mixture(component(**CO2), component(**ETHANOL))
mix.kij_ws(np.array([[0., K12], [K12, 0.]]))                       # Wong-Sandler binary parameter
alpha = ALPHA*(np.ones((2, 2)) - np.eye(2))
mix.NRTL(alpha, np.array([[0., G12], [G21, 0.]]))                  # NRTL, tau_ij = g_ij / T
eos = preos(mix, "ws_nrtl")                                        # Peng-Robinson + Wong-Sandler + NRTL

SC = np.array([1.0, 1.0, 0.1])                                     # scaling of (x1, y1, ln p) in the arclength of the continuation

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define functions
- residual: Equal fugacities of both components in the liquid (x1) and the vapor (y1) at T and p =
exp(u[2]) [bar]; NaN fo
- jacobian
- envelope: p-x-y curve of CO2 (1) + ethanol (2) at temperature T [K] up to the mixture critical
point.  Returns x1 (liqui
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def residual(u, T):
    """Equal fugacities of both components in the liquid (x1) and the vapor (y1) at T and p = exp(u[2]) [bar]; NaN for the trivial solution."""
    x1, y1, lnP = u
    if not (0 < x1 < 1 and 0 < y1 < 1):
        return np.full(2, np.nan)
    try:
        lnphi_L, vL = eos.logfugef(np.array([x1, 1 - x1]), T, np.exp(lnP), "L")
        lnphi_V, vV = eos.logfugef(np.array([y1, 1 - y1]), T, np.exp(lnP), "V")
    except Exception:
        return np.full(2, np.nan)
    if abs(vV - vL)/vL < 1e-3:
        return np.full(2, np.nan)             # both roots coincide: trivial solution x = y
    return np.array([np.log(x1) + lnphi_L[0] - np.log(y1) - lnphi_V[0], np.log(1 - x1) + lnphi_L[1] - np.log(1 - y1) - lnphi_V[1]])

def jacobian(u, T):
    J = np.zeros((2, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = 1e-7*max(1, abs(u[i]))
        J[:, i] = (residual(u + d, T) - residual(u - d, T))/(2*d[i])
    return J

def envelope(T, x_switch=0.05, ds=0.02, ds_min=1e-5):
    """p-x-y curve of CO2 (1) + ethanol (2) at temperature T [K] up to the mixture critical point.  Returns x1 (liquid), y1 (vapor), p [MPa] and the critical point.
    1) bubblePy of PhasePy from x1 = 1e-4 to x_switch (stable, the pressure rises monotonically with x1).
    2) Near the critical point the bubble pressure is not a monotonic function of x1 (the curve turns over at the nose of the envelope), and bubblePy fails.
       The curve (x1, y1, ln p) defined by the two equations 'residual = 0' is therefore followed with pseudo-arclength continuation (predictor along the tangent,
       Newton corrector on the two equations plus the arclength condition). The continuation passes through the critical point, where x1 = y1; the critical
       point is located by interpolation of the sign change of x1 - y1 and the curve beyond it is discarded."""
    xl = np.logspace(-4, np.log10(x_switch), 40)
    low = []
    y = np.array([0.01, 0.99])
    P = 0.3
    for x1 in xl:
        y, P = bubblePy(y, P, np.array([x1, 1 - x1]), T, eos)
        low.append((x1, y[0], P/10))
    u = np.array([xl[-1], y[0], np.log(P)])
    pts = [u.copy()]
    tprev = None
    step = ds
    while True:
        J = jacobian(u, T)/SC
        t = np.cross(J[0], J[1])
        t /= np.linalg.norm(t)           # tangent in the scaled variables
        if tprev is not None and t @ tprev < 0:
            t = -t
        ok = False
        while step > ds_min and not ok:
            v = u + step*t/SC
            for _ in range(20):                                       # Newton corrector: residual = 0 and t.(SC (v - u)) = step
                b = -np.r_[residual(v, T), t @ (SC*(v - u)) - step]
                if not np.all(np.isfinite(b)):
                    break
                dv = np.linalg.solve(np.vstack([jacobian(v, T), t*SC]), b)
                v = v + dv
                if np.max(abs(dv)) < 1e-10:
                    ok = True
                    break
            if not ok:
                step /= 2
        if not ok:
            break
        u = v
        tprev = t
        pts.append(u.copy())
        step = min(ds, 2*step)
        if len(pts) > 2 and pts[-1][0] > pts[-1][1]:
            break          # x1 > y1: the critical point has been passed
    c = np.array(pts)
    xc_, yc_, Pc_ = c[:, 0], c[:, 1], np.exp(c[:, 2])/10
    f = (yc_[-2] - xc_[-2])/((yc_[-2] - xc_[-2]) - (yc_[-1] - xc_[-1]))                      # interpolation of x1 = y1
    z_crit = xc_[-2] + f*(xc_[-1] - xc_[-2])
    p_crit = Pc_[-2] + f*(Pc_[-1] - Pc_[-2])
    low = np.array(low)
    x1 = np.r_[low[:-1, 0], xc_[:-1]]
    y1 = np.r_[low[:-1, 1], yc_[:-1]]
    p = np.r_[low[:-1, 2], Pc_[:-1]]
    return x1, y1, p, (z_crit, p_crit)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(19*cm, 8*cm), gridspec_kw={'wspace':0.32, 'width_ratios':[1, 1.08]})
# Panel A
Tcr, pcr = PropsSI("Tcrit", "CO2"), PropsSI("pcrit", "CO2")
Ts = np.linspace(216.6, Tcr, 120)
axA = ax[0]
axA.plot(Ts, [PropsSI("P", "T", t, "Q", 0, "CO2")/1e6 for t in Ts], color="k")
axA.plot(Tcr, pcr/1e6, "o", mfc="w", mec="k", ms=3.5, zorder=5)
axA.text(Tcr-3, pcr/1e6+0.2, "critical point\n304.13 K, 7.38 MPa", fontsize=5.2, ha="right", va="bottom")
# Widom line (maximum of kappa_T at constant p) as a guide to the eye
pw = np.linspace(pcr*1.02, 25e6, 40)
Tw_ = []
for p in pw:
    TT = np.linspace(Tcr*0.99, 1.4*Tcr, 400)
    k = [PropsSI("isothermal_compressibility", "T", t, "P", p, "CO2") for t in TT]
    Tw_.append(TT[int(np.argmax(k))])
axA.plot(Tw_, pw/1e6, color="0.55", ls="--", lw=.8)
axA.text(Tw_[22]+7, pw[22]/1e6, "Widom line", 
         fontsize=5.2, color="k", ha="left", va="center")
axA.text(225, 23, "Supercritical Fluid", fontsize=5.5, color="k")
axA.text(262, 1.0, "Vapor", fontsize=5.5, color="k")
axA.text(222, 4.0, "Liquid", fontsize=5.5, color="k")
# (1) evaporation of the pore liquid at ambient pressure along the liquid-vapor line (schematic: at the boiling line the meniscus exists)
axA.annotate("", xy=(Ts[60], PropsSI("P", "T", Ts[60], "Q", 0, "CO2")/1e6), xytext=(Ts[30], PropsSI("P", "T", Ts[30], "Q", 0, "CO2")/1e6), arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=1.2, shrinkA=0, shrinkB=0))
axA.text(222, 10.0, "(1) evaporation:\ncrosses the liquid-vapor line", color="#d62728", fontsize=5.2, ha="left")
# (2) CO2 route: liquid CO2 at LIQ_PATH_T, p = P_DRY ; heat at constant p to T_DRY ; depressurize at T_DRY to P_END
pts = [(LIQ_PATH_T, P_DRY), (T_DRY, P_DRY), (T_DRY, P_END)]
axA.plot([p[0] for p in pts], [p[1] for p in pts], color="#1f77b4", lw=1.2, marker="o", ms=2.5)
for (t, p), lab, dx, dy, ha in zip(pts, ["a", "b", "c"], (-2, 2, 2), (0.8, 0.8, 0.8), ("right", "left", "left")):
    axA.text(t+dx, p+dy, lab, fontsize=6, color="#1f77b4", ha=ha, fontweight="bold")
axA.text(LIQ_PATH_T+1, P_DRY+2.0, "(2) CO$_2$ route:  a $\\to$ b: heat at constant $p$\n          b $\\to$ c: vent at constant $T$", fontsize=5.2, color="#1f77b4", ha="left")
axA.set_xlabel("$T$ [K]")
axA.set_ylabel("$p$ [MPa]")
axA.set_xlim(216, 420)
axA.set_ylim(0, 25)
axA.text(-0.15, 1, "A", transform=axA.transAxes, fontweight="bold", fontsize=7)
# Inset: critical temperature of CO2 in a slit pore of width H (Fisher-Nakanishi, A from Example 6.4)
A_FN = np.array(SHIFT_EX64) / SIGMAH_EX64**(1/NU)
print(f"Fisher-Nakanishi amplitude from Example 6.4: A = {A_FN.min():.2f}-{A_FN.max():.2f} (mean {A_FN.mean():.2f})")
for Hp in H_PRINT:
    dT = Tcr * A_FN * (SIGMA_CO2 / Hp)**(1/NU)
    print(f"H = {Hp} nm: Tc,pore = {Tcr - dT.mean():.1f} K (shift {dT.min():.1f}-{dT.max():.1f} K)")
Hs = np.logspace(np.log10(3), np.log10(50), 100)
ins = axA.inset_axes([0.63, 0.09, 0.34, 0.29])
lo = Tcr * (1 - A_FN.max() * (SIGMA_CO2 / Hs)**(1/NU))
hi = Tcr * (1 - A_FN.min() * (SIGMA_CO2 / Hs)**(1/NU))
ins.fill_between(Hs, lo, hi, color="0.8", lw=0)
ins.plot(Hs, Tcr * (1 - A_FN.mean() * (SIGMA_CO2 / Hs)**(1/NU)), color="k", lw=.7)
ins.axhline(Tcr, color="k", lw=.4, ls=":")
ins.axhline(T_DRY, color="#1f77b4", lw=.6, ls="--")
ins.set_xscale("log")
ins.set_xlim(3, 50)
ins.set_ylim(270, 320)
ins.set_xticks([3, 5, 10, 20, 50])
ins.set_xticklabels(["3", "5", "10", "20", "50"])
ins.tick_params(labelsize=4.5, length=2, width=.4)
ins.set_xlabel("pore width $H$ [nm]", fontsize=4.8, labelpad=1)
ins.set_ylabel("$T_\\mathrm{crit,pore}$ [K]", fontsize=4.8, labelpad=1)
ins.text(3.3, T_DRY + 1.0, "vent, %.0f K" % T_DRY, fontsize=4.5, color="#1f77b4", va="bottom")
ins.text(48, Tcr - 1.5, "bulk", fontsize=4.5, ha="right", va="top")
for sp in ins.spines.values():
    sp.set_linewidth(.4)

# Panel B: p-x-y diagram of CO2 + ethanol (PhasePy: Peng-Robinson / Wong-Sandler / NRTL)
axB = ax[1]
crossing = None
for T, ls, lw in ((T_DRY, "-", .9), (T_HOT, "--", .7)):
    x1, y1, P, (z_crit, p_crit) = envelope(T)
    zc = np.r_[x1, z_crit, y1[::-1]]
    pc_ = np.r_[P, p_crit, P[::-1]]                    # closed outline of the two-phase region
    print(f"T = {T:.2f} K: mixture critical point at z = {z_crit:.3f}, p = {p_crit:.2f} MPa")
    if T == T_DRY:
        axB.fill(zc, pc_, fc="#d9d9d9", ec="none")
        P_in = np.interp(Z_START, x1, P)
        P_out = np.interp(Z_START, y1, P)
        print(f"constant-z path at z = {Z_START}: two-phase for p between {P_out:.2f} and {P_in:.2f} MPa (enters at the bubble pressure of x = z)")
    axB.plot(zc, pc_, color="#555555", ls=ls, lw=lw)
axB.plot([Z_START]*2, [P_DRY, P_END], color="#d62728", lw=1.2)
axB.plot([Z_START, 1.0, 1.0], [P_DRY, P_DRY, P_END], color="#1f77b4", lw=1.2)
axB.annotate("", xy=(Z_START, 1.5), xytext=(Z_START, 3.2), arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=1.0))
axB.annotate("", xy=(0.85, P_DRY), xytext=(Z_START, P_DRY), arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=1.0))
axB.annotate("", xy=(0.995, 4.6), xytext=(0.995, 7.6), arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=1.0))
axB.text(Z_START-0.02, 12, "(i) constant $z$:\nenters the two-phase\nregion", color="#d62728", fontsize=5.2, ha="right", va="top")
axB.text(0.63, P_DRY+0.4, "(ii) flush with CO$_2$ at\n$p$ = %.0f MPa, then vent" % P_DRY, color="#1f77b4", fontsize=5.2, ha="left", va="bottom")
axB.text(0.5, 3.0, "Two Phases\n(%.0f K)" % T_DRY, fontsize=5.5, color="k", ha="center", va="center")
axB.text(0.15, 8, "Single phase", fontsize=5.5, color="k", ha="center")
axB.text(0.05,13, "Dashed: %.0f K" % T_HOT, fontsize=5.2, color="k", va="top", ha="left")
axB.set_xlabel("CO$_2$ mole fraction $z$ of the autoclave content")
axB.set_ylabel("$p$ [MPa]")
axB.set_xlim(0, 1.0)
axB.set_ylim(0, 14)
axB.text(-0.15, 1, "B", transform=axB.transAxes, fontweight="bold", fontsize=7)
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
