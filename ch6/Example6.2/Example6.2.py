"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 6 Interfacial Phenomena, Wetting, and Confinement
Example 6.2
Interfacial tension of the Lennard-Jones vapor-liquid interface from direct coexistence: In this
example, we perform MD simulations of a liquid slab of the Lennard-Jones (LJ) fluid in equilibrium
with its vapor and extract the coexisting densities, the interfacial width, and the interfacial
tension from the pressure tensor (Eq. 6.8). All quantities are in reduced units, T^* =
k_BT/epsilon, rho^* = rhosigma^3, and gamma^* = gammasigma^2/epsilon, and the results are shown in
Fig. 6.2.

The simulations are run with LAMMPS (one input file in.slab per temperature, in the folders
Example6.2/T<value>/); this script only analyzes and plots. No command-line arguments: edit the
SETTINGS block below and run
    python Example6.2.py
For every temperature folder it evaluates
* the density profile along x (bins of 0.5 sigma), re-centered on the slab in every output block
because the
    slab diffuses along x during the run,
* the coexisting densities (asymptotes) and the 10-90 % interfacial width from a fit of Eq. 6.1 to
the centered
profile (two interfaces, periodic images included); the plateau averages are printed for comparison,
* the interfacial tension from the pressure tensor, gamma = (Lx/2) [pxx - (pyy + pzz)/2] (Eq. 6.8),
    with the standard error from blocks of the production run, and from all temperatures
* the critical temperature and density (beta = 0.326 and a linear rectilinear diameter) of this
model,
* the amplitude of gamma = gamma0 t^p for p = 1.26 (3D Ising) and p = 1.5 (mean field), with Tc
fixed.
Requires: numpy, scipy, matplotlib.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt

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
TEMPERATURES = [0.90, 1.00, 1.10, 1.15, 1.20]  # reduced temperatures T* of the folders Example6.2/T<value>/
RHO_LATTICE  = 0.746                  # density of the initial fcc slab (sets the box, see in.slab)
NCELL        = (23, 5, 5)             # number of fcc cells along x, y, z (see in.slab)
BIN          = 0.5                    # bin width of the density profile (sigma, see in.slab)
CORE         = 4.0                    # liquid plateau: |x - x_slab| < CORE (sigma)
VAPOR_EDGE   = 5.0                    # vapor plateau: within VAPOR_EDGE (sigma) of the point opposite to the slab center
BETA, MU_GAMMA = 0.326, 1.26          # 3D Ising exponents: coexistence curve and surface tension
FIT_MIN_DRHO = 0.15                   # use only temperatures with rho_L - rho_V above this in the Tc fit
PROFILE_T    = [0.90, 1.00, 1.15]     # temperatures of the density profiles in Panel A
N_BLOCKS     = 10                     # blocks of the production run for the statistical error of gamma
SAVE_FIG     = "figure6.2.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
# the T<value>/ folders are next to this script (GitHub layout) or in the subfolder Example6.2/
DATA_DIR = os.path.join(HERE, "Example6.2") if os.path.isdir(os.path.join(HERE, "Example6.2")) else HERE
a_fcc = (4.0 / RHO_LATTICE) ** (1.0 / 3.0)
LX, LY, LZ = (n * a_fcc for n in NCELL)             # box (sigma)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- read_profile: Chunk file of fix ave/chunk -> bin edges-based centers, widths and counts[n_blocks,
n_bins].
- centered_profile: Density profile on a uniform periodic grid, each block shifted so that its slab
center is at Lx/2.
- slab_profile: Eq. 6.1 for a slab of half thickness h centered at Lx/2, with periodic images; d =
width parameter.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def read_profile(fn):
    """Chunk file of fix ave/chunk -> bin edges-based centers, widths and counts[n_blocks, n_bins]."""
    lines = open(fn).read().split("\n")[3:]
    blocks, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i].split()[1])
        rows = np.array([[float(v) for v in lines[i + 1 + k].split()] for k in range(n)])
        blocks.append(rows)
        i += n + 1
    lower = blocks[0][0, 1] - 0.5 * BIN
    edges = np.minimum(lower + BIN * np.arange(len(blocks[0]) + 1), LX)   # the last bin is cut at the box boundary
    xc, w = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    return xc, w, np.array([b[:, 2] for b in blocks])                  # counts (not density/number: last bin is partial)


def centered_profile(xc, w, counts, n_grid=80):
    """Density profile on a uniform periodic grid, each block shifted so that its slab center is at Lx/2."""
    xg = (np.arange(n_grid) + 0.5) * LX / n_grid
    prof = []
    for c in counts:
        rho = c / (w * LY * LZ)
        th = 2 * np.pi * xc / LX
        x0 = (np.arctan2((rho * w * np.sin(th)).sum(), (rho * w * np.cos(th)).sum()) / (2 * np.pi) * LX) % LX
        prof.append(np.interp(xg + x0 - 0.5 * LX, xc, rho, period=LX))
    return xg, np.array(prof)


def slab_profile(x, rl, rv, h, d):
    """Eq. 6.1 for a slab of half thickness h centered at Lx/2, with periodic images; d = width parameter."""
    u = x - 0.5 * LX
    s = sum(0.5 * (np.tanh(2 * (u - n * LX + h) / d) - np.tanh(2 * (u - n * LX - h) / d)) for n in (-1, 0, 1))
    return rv + (rl - rv) * s

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


results = {}
for T in TEMPERATURES:
    folder = os.path.join(DATA_DIR, f"T{T:.2f}")
    xc, w, counts = read_profile(os.path.join(folder, "profile.dat"))
    x, prof = centered_profile(xc, w, counts)
    mean = prof.mean(axis=0)
    u = np.abs(x - 0.5 * LX)
    rl = mean[u < CORE].mean()
    rv = mean[u > 0.5 * LX - VAPOR_EDGE].mean()
    p = np.loadtxt(os.path.join(folder, "pressure.dat"), skiprows=1)      # pxx pyy pzz every 100 steps
    gam_blocks = [0.5 * LX * (b[:, 0].mean() - 0.5 * (b[:, 1].mean() + b[:, 2].mean())) for b in np.array_split(p, N_BLOCKS)]
    gam, gam_err = np.mean(gam_blocks), np.std(gam_blocks, ddof=1) / np.sqrt(N_BLOCKS)
    popt, _ = curve_fit(slab_profile, x, mean, p0=[rl, rv, 10.0, 1.5])
    # coexisting densities = asymptotes of the fit of Eq. 6.1; the plateau averages are printed for comparison.
    # At high T the interface (up to ~8 sigma wide) reaches into the vapor region of the box, so the plateau
    # average overestimates rho_V, whereas the asymptote of the fit does not.
    results[T] = dict(x=x, profile=mean, rl=popt[0], rv=popt[1], rl_pl=rl, rv_pl=rv, gamma=gam, gamma_err=gam_err,
                      w=abs(popt[3]) * np.log(3), popt=popt)

print(" T*     rho_L*   rho_V*   (plateau L / V)    gamma*            w_10-90 (sigma)")
for T, r in results.items():
    print(f"{T:5.2f}  {r['rl']:7.4f}  {r['rv']:7.4f}  ({r['rl_pl']:.4f} / {r['rv_pl']:.4f})  "
          f"{r['gamma']:7.4f} +/- {r['gamma_err']:.4f}   {r['w']:6.2f}")

# ---------------- critical point from the coexistence curve --------------------------------------------
Ts = np.array([T for T in TEMPERATURES if results[T]["rl"] - results[T]["rv"] > FIT_MIN_DRHO])
rl_ = np.array([results[T]["rl"] for T in Ts])
rv_ = np.array([results[T]["rv"] for T in Ts])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Define functions
- coexistence: Law of rectilinear diameters + critical scaling: (rho_L + rho_V)/2 = rho_c + A (T -
Tc),
- residual_model: liquid and vapor branches stacked
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def coexistence(T, Tc, rho_c, A, B):
    """Law of rectilinear diameters + critical scaling: (rho_L + rho_V)/2 = rho_c + A (T - Tc),
    rho_L - rho_V = B (1 - T/Tc)^beta. Returns rho_L and rho_V."""
    dia = rho_c + A * (T - Tc)
    dr = B * np.clip(1 - T / Tc, 0, None) ** BETA
    return dia + 0.5 * dr, dia - 0.5 * dr


def residual_model(T2, Tc, rho_c, A, B):                                        # liquid and vapor branches stacked
    rl_m, rv_m = coexistence(T2[:len(Ts)], Tc, rho_c, A, B)
    return np.concatenate([rl_m, rv_m])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# nonlinear least-squares fit of both branches at once; starting values from the linearized forms
s0, i0 = np.polyfit(Ts, (rl_ - rv_) ** (1 / BETA), 1)
Tc0 = -i0 / s0
c0, e0 = np.polyfit(Ts, 0.5 * (rl_ + rv_), 1)
(Tc, rho_c, A_dia, B_amp), _ = curve_fit(residual_model, np.concatenate([Ts, Ts]), np.concatenate([rl_, rv_]),
                                         p0=[Tc0, c0 * Tc0 + e0, c0, (rl_[0] - rv_[0]) / (1 - Ts[0] / Tc0) ** BETA])
print(f"\nCritical point of the simulated model: Tc* = {Tc:.3f}, rho_c* = {rho_c:.3f}   "
      f"(fit over T* = {Ts.min():.2f}-{Ts.max():.2f}; full LJ: Tc* = 1.312, rho_c* = 0.316)")

# ---------------- gamma versus t ------------------------------------------------------------------------
# gamma = gamma0 t^p with p = 1.26 (Ising) and p = 1.5 (mean field) are compared by a weighted
# least-squares fit of the amplitude gamma0 (Tc fixed at the value from the coexistence curve). Because Tc of
# such a small box is uncertain, a free exponent is also fitted for Tc shifted by +/- 0.02 to show how
# strongly the result depends on it.
Tg = np.array([T for T in TEMPERATURES if T < Tc])
t = 1 - Tg / Tc
g = np.array([results[T]["gamma"] for T in Tg])
ge = np.array([results[T]["gamma_err"] for T in Tg])
fits = {}
for p_ in (MU_GAMMA, 1.5):
    f = t ** p_
    g0 = np.sum(f * g / ge ** 2) / np.sum(f * f / ge ** 2)
    chi2 = np.sum(((g - g0 * f) / ge) ** 2)
    fits[p_] = (g0, chi2)
    print(f"gamma = gamma0 t^{p_}:  gamma0 = {g0:.3f},  chi2 = {chi2:.2f}  ({len(g) - 1} degrees of freedom)")

# free exponent and its sensitivity to the uncertainty of Tc (weighted least squares, gamma0 and p free)
for dTc in (-0.02, 0.0, 0.02):
    tt_ = 1 - Tg / (Tc + dTc)
    po, pc = curve_fit(lambda t_, g0, p_: g0 * t_ ** p_, tt_, g, p0=[3.0, 1.3], sigma=ge, absolute_sigma=True)
    chi2 = np.sum(((g - po[0] * tt_ ** po[1]) / ge) ** 2)
    print(f"free fit with Tc* = {Tc + dTc:.3f}:  p = {po[1]:.2f} +/- {np.sqrt(pc[1, 1]):.2f},  chi2 = {chi2:.2f}")

# width exponent between the lowest temperature and T* = 1.15 (the width at 1.20 is limited by the box)
Tw = np.array([T for T in TEMPERATURES if T <= 1.15])
tw = 1 - Tw / Tc
ww = np.array([results[T]["w"] for T in Tw])
print(f"Exponent of the 10-90 width, T* = {Tw.min():.2f}-{Tw.max():.2f}: {np.polyfit(np.log(tw), np.log(ww), 1)[0]:.2f}"
      f"   (mean field -0.5, 3D Ising -0.63)")

# ------------------------------ figure ----------------------------------------
centimeter = 1 / 2.54
fig, ax = plt.subplots(1, 3,
                       figsize=(19*centimeter, 6.5*centimeter),
                       dpi=750,
                       gridspec_kw={'wspace':0.35})

colors = ["#1f77b4", "#2ca02c", "#d62728"]
for T, c in zip(PROFILE_T, colors):
    xf = np.linspace(0, LX, 600)                       # fitted tanh profile (Eq. 6.1), not the raw simulation data
    ax[0].plot(xf - 0.5 * LX, slab_profile(xf, *results[T]["popt"]), color=c, label=f"$T^*$ = {T:.2f}")
ax[0].set_xlabel(r"$x/\sigma$")
ax[0].set_ylabel(r"$\rho^*$")
ax[0].legend(frameon=False, fontsize=6)

Tall = np.array(TEMPERATURES)
ax[1].plot([results[T]["rl"] for T in Tall], Tall, "ro", ms=3, mew=.75, label="liquid")
ax[1].plot([results[T]["rv"] for T in Tall], Tall, "bs", ms=3, mew=.75, label="vapor")
Tf = np.linspace(Ts.min() - 0.02, Tc, 200)                     # one smooth curve through both branches
rl_f, rv_f = coexistence(Tf, Tc, rho_c, A_dia, B_amp)
ax[1].plot(np.concatenate([rl_f, rv_f[::-1]]), np.concatenate([Tf, Tf[::-1]]), "k-",zorder=0)
ax[1].plot(rho_c, Tc, "ko", ms=3, label=f"fit: $T_\\mathrm{{crit}}^*$ = {Tc:.2f}")
ax[1].set_xlabel(r"$\rho^*$")
ax[1].set_ylabel(r"$T^*$")
ax[1].legend(frameon=False, fontsize=6)
ax[1].set_xlim([0,0.8])
ax[1].set_ylim([0.88,1.25])
ax[2].errorbar(Tg, g, ge, fmt="ko", ms=3, mew=.75, capsize=2, elinewidth=.75)
Tf2 = np.linspace(Tg.min(), Tc, 80)
ax[2].plot(Tf2, fits[MU_GAMMA][0] * (1 - Tf2 / Tc) ** MU_GAMMA, "k:", label=r"$t^{1.26}$")
ax[2].plot(Tf2, fits[1.5][0] * (1 - Tf2 / Tc) ** 1.5, "r--", label=r"$t^{3/2}$")
ax[2].set_xlabel(r"$T^*$")
ax[2].set_ylabel(r"$\gamma^*$")
ax[2].legend(frameon=False, fontsize=6)
ax[2].set_xlim([0.88,1.25])
ax[2].set_ylim([0,0.6])

for a, lab in zip(ax, "ABC"):
    a.text(-0.26, 0.975, lab, transform=a.transAxes, fontweight="bold", fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
