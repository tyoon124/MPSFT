"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 8 Industrial Milestones: Historical Landmarks and Key Applications
Example 8.2
Position of the industrial milestones relative to the critical point: In this example, we test the
correspondence of Table 8.2 quantitatively: if a process exploits the compressibility anomaly of
Part I, its operating state must lie in the region where the compressibility number kappa_T p is
well above the dense-liquid value. With CoolProp, which uses the Span-Wagner and IAPWS-95 Equations
of State for CO_2 and water, we convert the operating states of ten processes to reduced variables
and evaluate kappa_T p. The results are shown in Fig. 8.3.

No command-line arguments: edit the SETTINGS block below and run   python Example8.2.py
For each process of Chapter 8 the operating state (T, p) of its working fluid is converted to
reduced variables (T/Tc, p/pc) and the compressibility number  Z_kappa = kappa_T * p  (= 1 for an
ideal gas, << 1 for a dense liquid, >> 1 near the critical point) is evaluated with CoolProp. Panel
A maps log10(Z_kappa) of CO2 in the (T/Tc, p/pc) plane with the Widom line (maximum of kappa_T at
constant p) of CO2 and water; Panel B gives Z_kappa of every process, which shows how strongly the
process exploits the critical anomaly of Part I. The operating states are taken from the
sources noted in the SETTINGS block (patents, papers, and technical data sheets).
Produces Fig. 8.3.   Requires: numpy, matplotlib, CoolProp.
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
# label, fluid, T [K], (p_low, p_high) [MPa], sourced (True), source of the state
PROCESSES = [
 ("Critical point drying (1951)",       "CO2",         308.15, (8.27, 8.27), True,  "EMS data sheet: about 35 C and 1200 psi"),
 ("Kaffee HAG decaffeination (1978)",   "CO2",         353.15, (19.3, 24.3), True,  "Zosel, US 4,247,570, examples: 80 C, 190-240 atm"),
 ("ROSE solvent recovery (1979)",       "n-Butane",    433.6,  (4.56, 4.56), True,  "Example 8.1: n-butane, T/Tc = 1.02, p/pc = 1.2"),
 ("SCORR resist removal (2000s)",       "CO2",         323.15, (7.6, 9.7),   True,  "Davenhall and Rubin, US 6,403,544, Example I: 50 C, pulses of 1100-1400 psi"),
 ("Adschiri hydrothermal synthesis",    "Water",       673.15, (30.0, 30.0), True,  "Hayashi and Hakuta (2010): 400 C, 30 MPa"),
 ("DyeCoo / Nike dyeing (2012)",        "CO2",         375.15, (25.0, 25.0), True,  "Textile World (2012): 216 F (102 C), 250 bar; 120 C is the maximum working temperature"),
 ("MODAR SCWO",                         "Water",       773.15, (24.1, 27.6), True,  "Modell, US 4,543,190, examples: about 500 C, 3500-4000 psi"),
 ("VERENA SCWG",                        "Water",       933.15, (28.0, 28.0), True,  "Boukis et al. (2007): 660 C, 280 bar"),
 ("SCW composite recycling",            "Water",       713.15, (30.0, 30.0), True,  "Bai et al. (2010): 440 C, 30 MPa"),
 ("HTL (continuous pilots)",            "Water",       623.15, (20.0, 25.0), True,  "Castello et al. (2018), Table 1: most continuous plants at 350 C and 200-250 bar (Cat-HTR conditions not published)"),
]
TR_RANGE   = (0.90, 1.60)     # T/Tc range of the map
PR_RANGE   = (0.30, 4.00)     # p/pc range of the map
N_T, N_P   = 160, 160         # map resolution
LOG_ZK_MIN, LOG_ZK_MAX = -1.0, 1.5   # color scale clip, log10(Z_kappa)
LABEL_OFFSET = {1: (-8, 2), 3: (-7, 4), 4: (5, -5), 5: (-3, 4), 10: (-10, -4)}   # offsets (points) of the process numbers in panel A; default (3, 3)
SAVE_FIG   = "figure8.3.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Set constants and grids
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
cm = 1/2.54
FL = {f: (PropsSI("Tcrit", f), PropsSI("pcrit", f)) for f in ("CO2", "Water", "n-Butane")}
# colors: hue = working fluid
FCOL = {("CO2", True): "#ff7f0e", ("Water", True): "#e7298a", ("n-Butane", True): "#4daf4a"}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define functions
- zk
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def zk(f, T, p):
    try:
        return PropsSI("isothermal_compressibility", "T", T, "P", p, f) * p
    except ValueError:
        return np.nan

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# --- Panel A: map of log10(kappa_T p) for CO2 ---
Tc, pc = FL["CO2"]
Tg = np.linspace(TR_RANGE[0], TR_RANGE[1], N_T)
pg = np.linspace(PR_RANGE[0], PR_RANGE[1], N_P)
Z = np.array([[zk("CO2", t * Tc, p * pc) for t in Tg] for p in pg])
with np.errstate(divide="ignore", invalid="ignore"):
    LZ = np.clip(np.log10(Z), LOG_ZK_MIN, LOG_ZK_MAX)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Define functions
- widom: locus of the maximum of kappa_T at constant p
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def widom(f, prs):                                  # locus of the maximum of kappa_T at constant p
    Tc_, pc_ = FL[f]
    out = []
    for pr in prs:
        T = np.linspace(0.97, 1.8, 500) * Tc_
        k = np.array([PropsSI("isothermal_compressibility", "T", t, "P", pr * pc_, f) for t in T])
        i = int(np.argmax(k))
        out.append((T[i] / Tc_, pr) if 0 < i < len(T) - 1 else (np.nan, pr))
    return np.array(out)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
7. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
prs = np.linspace(1.02, PR_RANGE[1], 60)
W = {f: widom(f, prs) for f in ("CO2", "Water")}

fig, ax = plt.subplots(1, 2, figsize=(19*cm, 7.6*cm), gridspec_kw={'wspace':1.15, 'width_ratios':[1.1, 1]})
cs = ax[0].pcolormesh(Tg, pg, LZ, vmin=LOG_ZK_MIN, vmax=LOG_ZK_MAX, cmap="viridis", shading="auto", rasterized=True)
cb = fig.colorbar(cs, ax=ax[0], pad=0.02, fraction=0.05)
cb.set_label(r"$\log_{10}(\kappa_\mathrm{T}\,p)$, CO$_2$")
cb.set_ticks([-1, 0, 1])
Ts_ = np.linspace(TR_RANGE[0], 1.0, 60)             # saturation curve (subcritical branch)
ax[0].plot(Ts_, [PropsSI("P", "T", t * Tc, "Q", 0, "CO2") / pc for t in Ts_], color="w", lw=.8)
ax[0].plot(1, 1, marker="o", mfc="w", mec="k", ms=3, zorder=5)
for f, ls in (("CO2", "-"), ("Water", "--")):
    ax[0].plot(W[f][:, 0], W[f][:, 1], color="w", ls="--" if f == "Water" else "-", lw=.6, label=f"Widom line, {'water' if f == 'Water' else 'CO$_2$'}")
for k, (lab, f, T, (p0, p1), q, note) in enumerate(PROCESSES, 1):
    Tc_, pc_ = FL[f]
    ax[0].annotate(str(k), (T / Tc_, 0.5 * (p0 + p1) * 1e6 / pc_), xytext=LABEL_OFFSET.get(k, (3, 3)), textcoords="offset points", fontsize=5.5, color="w", fontweight="bold", zorder=8)
    ax[0].plot([T / Tc_] * 2, [p0 * 1e6 / pc_, p1 * 1e6 / pc_], color="k", lw=.5, zorder=6)
    ax[0].plot(T / Tc_, 0.5 * (p0 + p1) * 1e6 / pc_, marker="o", ms=3.2, mfc=FCOL[(f, q)], mec="k", mew=.5, zorder=7, ls="none")
ax[0].set_xlabel("$T/T_\\mathrm{crit}$")
ax[0].set_ylabel("$p/p_\\mathrm{crit}$")
ax[0].set_xlim(TR_RANGE)
ax[0].set_ylim(PR_RANGE)
ax[0].legend(frameon=False, fontsize=5, loc="upper left", labelcolor="w")

res = []                                            # Panel B
print(f"{'process':36s} {'fluid':9s} {'T/Tc':>6s} {'p/pc':>10s} {'rho/rho_c':>10s} {'kappa_T p':>12s}")
for k, (lab, f, T, (p0, p1), q, note) in enumerate(PROCESSES, 1):
    Tc_, pc_ = FL[f]
    pm = np.sqrt(p0 * p1) * 1e6
    rho = PropsSI("Dmass", "T", T, "P", pm, f) / PropsSI("rhomass_critical", f)
    z0, z1, zm = zk(f, T, p0 * 1e6), zk(f, T, p1 * 1e6), zk(f, T, pm)
    res.append((f"{k}  {lab}", f, zm, min(z0, z1), max(z0, z1), q))
    print(f"{lab:36s} {f:9s} {T/Tc_:6.2f} {p0*1e6/pc_:4.2f}-{p1*1e6/pc_:4.2f} {rho:10.2f} {zm:12.3g}")
res.sort(key=lambda r: r[2])
y = np.arange(len(res))
for i, (lab, f, zm, zlo, zhi, q) in enumerate(res):
    ax[1].barh(i, zm, color=FCOL[(f, q)], edgecolor="k", lw=.4, height=.65)
    if zhi / zlo > 1.02:                                                  # pressure-range whisker with end caps (thin)
        ax[1].plot([zlo, zhi], [i, i], color="k", lw=.3, zorder=3)
        for zz in (zlo, zhi):
            ax[1].plot([zz, zz], [i - .13, i + .13], color="k", lw=.3, zorder=3)
ax[1].axvline(1, color="0.4", lw=.5, ls=":")
ax[1].text(1.08, len(res) - 0.45, "ideal gas", fontsize=5, color="0.3", va="bottom")
ax[1].set_xscale("log")
ax[1].set_xlim(0.1, 10)
ax[1].set_xticks([0.1, 1, 10])
ax[1].set_xticklabels(["0.1", "1", "10"])
ax[1].minorticks_off()
ax[1].set_yticks(y)
ax[1].set_yticklabels([r[0] for r in res], fontsize=5.2)
ax[1].set_xlabel(r"$\kappa_\mathrm{T}\,p$ at the operating state")
ax[1].set_ylim(-0.6, len(res) + 0.2)
from matplotlib.patches import Patch
ax[1].legend(handles=[Patch(fc=FCOL[("CO2", True)], ec="k", lw=.4, label="CO$_2$"), Patch(fc=FCOL[("Water", True)], ec="k", lw=.4, label="water"), Patch(fc=FCOL[("n-Butane", True)], ec="k", lw=.4, label="n-butane")], frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3, columnspacing=1.2, handlelength=1.2)
ax[0].text(-0.2, 1.02, "A", transform=ax[0].transAxes, fontweight="bold", fontsize=7)
fig.canvas.draw()                                                              # panel letter B at the left edge of the process labels
x0 = min(t.get_window_extent().x0 for t in ax[1].get_yticklabels())
xB = ax[1].transAxes.inverted().transform((x0, 0))[0]
ax[1].text(xB, 1.02, "B", transform=ax[1].transAxes, fontweight="bold", fontsize=7)
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
