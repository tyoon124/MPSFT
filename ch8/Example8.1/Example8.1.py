"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 8 Industrial Milestones: Historical Landmarks and Key Applications
Example 8.1
Solvent recovery by density collapse in a ROSE-type deasphalting loop: In this example, we quantify
with CoolProp the statement of Sec. 8.1.2 that the solvent of a ROSE unit is recovered above its
critical point instead of being evaporated. The solvents are propane, n-butane, and n-pentane, the
light paraffins named in that section, and the heat that has to be supplied by a hot utility per
kilogram of circulating solvent is compared for the two routes. The results are shown in Fig. 8.2.

No command-line arguments: edit the SETTINGS block below and run   python Example8.1.py
A light paraffinic solvent (propane, n-butane, n-pentane) holds the oil fraction in the dense state
at the extractor (T_ext < Tc, p_ext > pc). Two ways of returning the solvent to this state after the
separation are compared per kg of solvent:
(S) supercritical swing: heat at constant p_ext to T_sep > Tc, where the density collapses and the
oil separates; the lean
fluid is cooled back to T_ext. A counter-current exchanger with a minimum approach DT_MIN recovers
all the heat except
      the top-up h(T_sep, p) - h(T_sep - DT_MIN, p).
(E) evaporative recovery: flash to the saturation pressure at T_COND, condense at T_COND (cooling
water), pump to p_ext and
reheat to T_ext. The condensation heat leaves at T_COND < T_ext and cannot heat the feed without a
heat pump, so that
the hot-utility duty is the heat of the flash plus the reheat (pump work is printed separately).
Pure solvent, no solute, no pressure drop, ideal heat exchange: the numbers are the thermodynamic
duties, not a process simulation.
Produces Fig. 8.2.   Requires: numpy, matplotlib, CoolProp.
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
SOLVENTS   = [("Propane", "propane"), ("n-Butane", "n-butane"), ("n-Pentane", "n-pentane")]   # CoolProp name, label
P_RED      = 1.2            # p_ext / pc of the separation
TR_EXT     = 0.94           # T_ext / Tc (dense extraction state)
TR_SEP     = 1.02           # T_sep / Tc (density collapse)
DT_MIN     = 10.0           # K, minimum approach of the feed-effluent exchanger
T_COND     = 313.15         # K, condensation temperature of the evaporative route (cooling water)
P_SCAN     = [1.05, 1.2, 1.5]    # p/pc of the density curves of Panel A (propane)
SAVE_FIG   = "figure8.2.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
cm = 1/2.54
cols = ["tab:blue", "tab:orange", "tab:green"]
h = lambda f, T, p: PropsSI("Hmass", "T", T, "P", p, f)

rows = []
print(f"{'solvent':10s} {'Tc/K':>6s} {'pc/MPa':>7s} {'T_ext':>6s} {'T_sep':>6s} {'rho_ext':>8s} {'rho_sep':>8s} {'drho/%':>7s} "
      f"{'dh_sw':>7s} {'topup':>7s} {'q_flash':>8s} {'q_reheat':>9s} {'q_E':>6s} {'dHvap':>6s} {'w_pump':>7s}")
for f, lab in SOLVENTS:
    Tc, pc = PropsSI("Tcrit", f), PropsSI("pcrit", f)
    p = P_RED * pc
    Te, Ts = TR_EXT * Tc, TR_SEP * Tc
    r_e, r_s = PropsSI("Dmass", "T", Te, "P", p, f), PropsSI("Dmass", "T", Ts, "P", p, f)
    dh_sw = h(f, Ts, p) - h(f, Te, p)                                        # (S) gross heating of the swing, J/kg
    topup = h(f, Ts, p) - h(f, Ts - DT_MIN, p)                              # (S) hot utility after counter-current exchange
    hv = PropsSI("Hmass", "T", T_COND, "Q", 1, f)
    hl = PropsSI("Hmass", "T", T_COND, "Q", 0, f)
    q_flash = max(hv - h(f, Te, p), 0.0)                                     # (E) heat to bring the extract to saturated vapor at T_COND
    h2 = PropsSI("Hmass", "Smass", PropsSI("Smass", "T", T_COND, "Q", 0, f), "P", p, f)     # liquid after the isentropic pump
    w_pump = h2 - hl                                                         # (E) pump work, p_sat -> p_ext
    q_reheat = h(f, Te, p) - h2                                              # (E) reheat of the pumped liquid to T_ext
    qE = q_flash + q_reheat
    rows.append((lab, dh_sw / 1e3, topup / 1e3, qE / 1e3, (hv - hl) / 1e3))
    print(f"{lab:10s} {Tc:6.1f} {pc/1e6:7.2f} {Te:6.1f} {Ts:6.1f} {r_e:8.1f} {r_s:8.1f} {100*(1-r_s/r_e):7.1f} "
          f"{dh_sw/1e3:7.1f} {topup/1e3:7.1f} {q_flash/1e3:8.1f} {q_reheat/1e3:9.1f} {qE/1e3:6.1f} {(hv-hl)/1e3:6.1f} {w_pump/1e3:7.2f}")

fig, ax = plt.subplots(1, 2, figsize=(19*cm*0.8, 6.8*cm), gridspec_kw={'wspace':0.32, 'width_ratios':[1.15, 1]})
Tr = np.linspace(0.85, 1.25, 400)
for (f, lab), c in zip(SOLVENTS, cols):                                      # Panel A: density collapse in reduced variables
    Tc, pc, rc = PropsSI("Tcrit", f), PropsSI("pcrit", f), PropsSI("rhomass_critical", f)
    ax[0].plot(Tr, [PropsSI("Dmass", "T", t * Tc, "P", P_RED * pc, f) / rc for t in Tr], color=c, label=f"{lab}, $p/p_\\mathrm{{crit}}$ = {P_RED}")
f0 = SOLVENTS[0][0]
Tc, pc, rc = PropsSI("Tcrit", f0), PropsSI("pcrit", f0), PropsSI("rhomass_critical", f0)
for pr, ls in zip([P_SCAN[0], P_SCAN[2]], ["-.", "--"]):
    ax[0].plot(Tr, [PropsSI("Dmass", "T", t * Tc, "P", pr * pc, f0) / rc for t in Tr], color=cols[0], ls=ls, lw=.5, label=f"propane, $p/p_\\mathrm{{crit}}$ = {pr}")
ax[0].axvspan(TR_EXT, TR_SEP, color="0.9", lw=0)
ax[0].axvline(1, color="0.5", lw=.4, ls=":")
ax[0].text(TR_EXT + 0.003, 0.3, "swing", fontsize=5.5, rotation=90, va="bottom")
ax[0].text(1.003, 2.35, "$T_\\mathrm{crit}$", fontsize=5.5)
ax[0].set_xlabel("$T/T_\\mathrm{crit}$")
ax[0].set_ylabel(r"$\rho/\rho_\mathrm{crit}$")
ax[0].set_xlim(Tr[0], Tr[-1])
ax[0].legend(frameon=False, fontsize=5.2, loc="upper right")
x = np.arange(len(rows))
w = 0.26                                           # Panel B: hot-utility duty per kg of solvent
vals = [[r[4] for r in rows], [r[3] for r in rows], [r[1] for r in rows], [r[2] for r in rows]]
labs = ["evaporative: flash + reheat", "(heat of condensation at $T_\\mathrm{cond}$, rejected)", "supercritical swing, no recovery", "supercritical swing, $\\Delta T_\\mathrm{min}$ = %d K" % DT_MIN]
ax[1].bar(x - 1.5 * w * 0.8, vals[1], w * 0.8, color="tab:red", label=labs[0])
ax[1].bar(x - 0.5 * w * 0.8, vals[0], w * 0.8, color="none", edgecolor="tab:red", lw=.5, label=labs[1])
ax[1].bar(x + 0.5 * w * 0.8, vals[2], w * 0.8, color="tab:blue", label=labs[2])
ax[1].bar(x + 1.5 * w * 0.8, vals[3], w * 0.8, color="navy", label=labs[3])
ax[1].set_xticks(x)
ax[1].set_xticklabels([r[0] for r in rows])
ax[1].set_ylabel("duty / (kJ kg$^{-1}$ of solvent)")
ax[1].set_ylim(0, 1.45 * max(max(v) for v in vals))
ax[1].legend(frameon=False, fontsize=5, loc="upper left")
for a, lab in zip(ax, "AB"):
    a.text(-0.2, 1.0, lab, transform=a.transAxes, fontweight="bold", fontsize=7)
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
