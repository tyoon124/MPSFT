"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 9 Supercritical Fluid Extraction and Fractionation
Example 9.2
Chrastil correlation of two solid solubilities and the selectivity map: In this example, we fit the
Chrastil model, Eq. 9.5, to the solubilities of naphthalene and of glycyrrhizin (a triterpenoid
saponin, from the extraction of liquorice root) in CO2, and use the two correlations to compute the
separation factor alpha_AB = y_A/y_B that a solubility-controlled separation of the two solutes
would reach. The results are shown in Fig. 9.3. Caffeine and beta-carotene, which are the
usual choices for this exercise, are not used because their solubility data are not available in
tabulated, open form; the glycyrrhizin data of the settings block of the script come from the NIST
ThermoML archive and the naphthalene data are those of Example 9.1.

Produces Fig. 9.3: Chrastil fits (panels A, B) and map of the separation factor between the two solutes (panel C).
No command-line arguments: edit the SETTINGS block below and run   python Example9.2.py
Chrastil model, Eq. (9.5):  c = rho1^k exp(a/T + b)  with c = mass of solute per volume of CO2 [g/L]
and rho1 = density of pure CO2 [g/L] (CoolProp, Span-Wagner).  The measured mole fractions y2 are
converted to c = rho1 y2 M2 / ((1 - y2) M1).  ln c is regressed linearly on ln rho1, 1/T and 1
(ordinary least squares).  The separation factor of two solutes A and B dissolved independently from
their own solid phases is alpha_AB = y_A / y_B at the same (T, P); it is evaluated from the two
fitted correlations inside the range of the data (the correlations must not be extrapolated), and
ignores the mutual influence of the solutes on their solubilities. Data (T/K, P/kPa, y2), embedded:
naphthalene   Sako, Yamane, Negishi, Sato, Sekiyu Gakkaishi 37(3), 321-327 (1994) (the data set of
Example 9.1), 38 points at 308-338 K
glycyrrhizin  Jia, Zabihi, Gao, Zhao, J. Chem. Eng. Data 60, 1744 (2015), NIST ThermoML doi
10.1021/je5011328 (pure CO2, no cosolvent)
Requires: numpy, matplotlib, CoolProp.
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
M1 = 44.0095                                  # g/mol, CO2
SOLUTES = {                                   # name: (M2 in g/mol, nominal temperatures of the data in K, data)
 "naphthalene": (128.17, [308, 318, 328, 338],   # Sako, Yamane, Negishi, Sato, Sekiyu Gakkaishi 37(3), 321-327 (1994); P converted from bar to kPa
  [(308,8830,0.00975),(308,11470,0.01308),(308,13440,0.01457),(308,15790,0.01585),(308,18630,0.01623),(308,19710,0.01632),(308,22850,0.01662),(308,25100,0.01789),(308,28830,0.01829),
(318,8340,0.00111),(318,8920,0.00196),(318,9220,0.00571),(318,9710,0.00809),(318,10000,0.00995),(318,12230,0.01702),(318,15100,0.02069),(318,20400,0.02416),(318,22750,0.02664),(318,26580,0.02868),(318,30010,0.02935),
(328,7750,0.00093),(328,9810,0.00314),(328,10980,0.00761),(328,11770,0.01253),(328,13340,0.02236),(328,13830,0.02434),(328,15490,0.03012),(328,18140,0.03742),(328,21870,0.04398),(328,26870,0.05139),
(338,11770,0.00734),(338,12750,0.01082),(338,14710,0.02595),(338,16080,0.0335),(338,18730,0.04631),(338,19910,0.05104),(338,21870,0.05894),(338,22560,0.06394)]),
 "glycyrrhizin": (822.93, [308, 318, 328],
  [(308,9000,1.54e-6),(308,12000,2.06e-6),(308,15000,2.17e-6),(308,18000,2.49e-6),(308,21000,2.98e-6),
   (318,9000,2.27e-6),(318,12000,3.20e-6),(318,15000,3.85e-6),(318,18000,4.75e-6),(318,21000,5.72e-6),
   (328,9000,2.05e-6),(328,12000,3.85e-6),(328,15000,5.37e-6),(328,18000,6.95e-6),(328,21000,8.61e-6)]),
}
PAIR       = ("naphthalene", "glycyrrhizin")  # solutes A, B of the separation factor
MAP_T      = (308, 328)                       # K, temperature range of the selectivity map (common range of the data)
MAP_P      = (10.0, 20.0)                     # MPa, pressure range of the selectivity map
SAVE_FIG   = "figure9.3.pdf"
# ============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
cm = 1/2.54
rho = lambda T, P: PropsSI("Dmass", "T", T, "P", P, "CO2")                  # g/L = kg/m3

fits = {}
print(f"{'solute':13s} {'N':>3s} {'k':>6s} {'a/K':>8s} {'b':>7s} {'AARD/%':>7s}  {'dH=-aR/(kJ/mol)':>15s}")
for name, (M2, Tn, data) in SOLUTES.items():
    d = np.array(data)
    T, P, y = d[:, 0], d[:, 1] * 1e3, d[:, 2]
    r = np.array([rho(t, p) for t, p in zip(T, P)])
    c = r * y * M2 / ((1 - y) * M1)
    X = np.c_[np.log(r), 1 / T, np.ones(len(T))]
    (k, a, b), *_ = np.linalg.lstsq(X, np.log(c), rcond=None)
    aard = 100 * np.mean(np.abs(np.exp(X @ np.array([k, a, b])) / c - 1))
    fits[name] = dict(k=k, a=a, b=b, M2=M2, T=T, P=P, y=y, rho=r, c=c, Tn=Tn)
    print(f"{name:13s} {len(T):3d} {k:6.2f} {a:8.0f} {b:7.2f} {aard:7.1f}  {-a * 8.314462618 / 1e3:15.1f}")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define functions
- y_chrastil: mole fraction from the fitted correlation
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def y_chrastil(name, T, P):                                                  # mole fraction from the fitted correlation
    f = fits[name]
    r = rho(T, P)
    c = r**f["k"] * np.exp(f["a"] / T + f["b"])
    n2 = c / f["M2"]
    return n2 / (n2 + r / M1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ---- Fig. 9.3 A, B: ln c versus ln rho
cols = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
fig, axs = plt.subplots(1, 3, figsize=(19*cm, 6.2*cm), gridspec_kw={'wspace':0.45, 'width_ratios':[1, 1, 1.15]})
ax = axs[:2]
symbols = ["o",">","s","p"]
for a_, name, lab in zip(ax, SOLUTES, "AB"):
    f = fits[name]
    counter = 0
    for Tn, col in zip(f["Tn"], cols):
        m = np.abs(f["T"] - Tn) < 2.5
        a_.plot(f["rho"][m], f["c"][m], symbols[counter], ms=3, mfc="none", mec=col, mew=.6)
        rg = np.linspace(f["rho"][m].min() * 0.95, f["rho"][m].max() * 1.05, 40)
        a_.plot(rg, rg**f["k"] * np.exp(f["a"] / Tn + f["b"]), color=col, label=f"{Tn} K")
        counter += 1
    a_.set_xscale("log")
    a_.set_yscale("log")
    a_.set_xlabel(r"CO$_2$ density $\rho_1$ / (g L$^{-1}$)")
    a_.set_ylabel("solubility $c$ / (g L$^{-1}$)")
    a_.legend(frameon=False, fontsize=5.2, loc="upper left")
    a_.text(-0.3, 1.02, lab, transform=a_.transAxes, fontweight="bold", fontsize=7)
    a_.text(0.97, 0.05, f"{name}\n$k$ = {f['k']:.2f}, $a$ = {f['a']:.0f} K\n$b$ = {f['b']:.1f}", transform=a_.transAxes, ha="right", fontsize=5.3)
    a_.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    a_.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
    a_.set_xticks([300, 500, 700, 900])
    a_.xaxis.set_minor_locator(mpl.ticker.NullLocator())
ax[0].set_xlim([200,950])
ax[0].set_ylim([0.6,1.5e2])
ax[1].set_xlim([200,950])
ax[1].set_ylim([9e-3,1.3e-1])

# ---- Fig. 9.3 C: map of the separation factor alpha = y_A / y_B
A, B = PAIR
Tg = np.linspace(*MAP_T, 60)
Pg = np.linspace(*MAP_P, 60) * 1e6
Z = np.array([[np.log10(y_chrastil(A, t, p) / y_chrastil(B, t, p)) for t in Tg] for p in Pg])
ac = axs[2]
im = ac.pcolormesh(Tg, Pg / 1e6, Z, cmap="viridis", shading="auto")
cb = fig.colorbar(im, ax=ac, pad=0.02)
cb.set_label(r"$\log_{10}\,\alpha_{AB}$")
cs = ac.contour(Tg, Pg / 1e6, Z, colors="w", linewidths=.4, levels=np.arange(np.floor(Z.min() * 4) / 4, Z.max() + .25, .25))
ac.clabel(cs, fontsize=4.5, fmt="%.2f")
ac.set_xlabel("$T$ / K")
ac.set_ylabel("$p$ / MPa")
ac.set_title(f"A: {A}, B: {B}", fontsize=5.3)
ac.text(-0.27, 1.02, "C", transform=ac.transAxes, fontweight="bold", fontsize=7)
i, j = np.unravel_index(np.argmax(Z), Z.shape)
i2, j2 = np.unravel_index(np.argmin(Z), Z.shape)
print(f"alpha_AB: maximum {10**Z.max():.0f} at {Tg[j]:.0f} K and {Pg[i] / 1e6:.1f} MPa; minimum {10**Z.min():.0f} at {Tg[j2]:.0f} K and {Pg[i2] / 1e6:.1f} MPa")
for T_ in (308, 318, 328):
    print(T_, [f"{y_chrastil(A, T_, p) / y_chrastil(B, T_, p):.0f}" for p in (10e6, 12e6, 15e6, 20e6)])
ac.set_ylim([10,20])
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
