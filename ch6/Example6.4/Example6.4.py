"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 6 Interfacial Phenomena, Wetting, and Confinement
Example 6.4
Attractive versus repulsive walls in a slit pore: In this example, we repeat the simulation of
Example 6.2 for a liquid slab confined between two structureless walls and compare the vapor-liquid
coexistence in the pore with that of the bulk. The wall acts on each atom through the integrated 9-3
potential of Eq. 6.21, shifted to zero at its cutoff.

The simulations are run with LAMMPS (one input file in.pore per wall type and temperature, in the
folders Example6.4/<wall>/T<value>/); this script only analyzes and plots.
No command-line arguments: edit the SETTINGS block below and run
    python Example6.4.py
For every wall type (repulsive, attractive, strongly attractive) and temperature it evaluates
* the coexisting liquid and vapor densities (asymptotes) and the interfacial width from a fit of Eq.
6.1 to the
density profile along the pore axis x, re-centered on the slab in every output block (as in Example
6.2),
* the density profile across the pore (z) in the liquid core, and from the temperatures of each wall
type
* the apparent critical temperature and density from the coexistence curve (beta = 0.326 and a
linear rectilinear
diameter, nonlinear fit of both branches), in the same way as for the bulk system of Example 6.2.
Densities in the pores are given per accessible volume: the number of atoms per box volume is
multiplied by H/(H - sigma). The bulk reference is evaluated from the folders of Example 6.2.
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
CASES = [("repulsive",  "repulsive wall"),
         ("attractive", "attractive wall"),
         ("strong",     "strongly attractive wall")]           # folder names of Example6.4/ and legend labels
COLORS       = {"repulsive": "#1f77b4", "attractive": "#2ca02c", "strong": "#d62728"}
TEMPERATURES = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10]           # temperatures of the pore runs
BULK_DIR     = "Example6.2"                                   # bulk reference (periodic slab of Example 6.2)
BULK_T       = [0.90, 1.00, 1.10, 1.15, 1.20]
RHO_LATTICE  = 0.746                                          # fcc density of the initial slab (box of in.slab / in.pore)
NCELL        = (23, 5, 5)                                     # fcc cells of the bulk box along x, y, z
H            = 7.0                                            # wall separation (sigma)
H_ACC        = H - 1.0                                        # accessible width used to convert densities (sigma)
BIN_X, BIN_Z = 0.5, 0.1                                       # bin widths of the profiles (sigma, see in.pore)
CORE_Z       = 8.0                                            # length of the liquid core region of the z profiles (sigma)
BETA         = 0.326
FIT_MIN_DRHO = 0.15                                           # minimum density difference used in the critical-point fits
T_PROFILE    = 0.90                                           # temperature of the density profiles across the pore (Panel A)
SAVE_FIG     = "figure6.5.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
# wall folders next to this script (GitHub layout) or in the subfolder Example6.4/; the bulk reference
# (Example 6.2) in the subfolder Example6.2/ or in the neighboring folder ../Example6.2/
PORE_DIR = os.path.join(HERE, "Example6.4") if os.path.isdir(os.path.join(HERE, "Example6.4")) else HERE
BULK_PATH = os.path.join(HERE, BULK_DIR) if os.path.isdir(os.path.join(HERE, BULK_DIR)) else os.path.join(HERE, "..", BULK_DIR)
a_fcc = (4.0 / RHO_LATTICE) ** (1.0 / 3.0)
LX, LY, LZ_BULK = (n * a_fcc for n in NCELL)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- read_blocks: fix ave/chunk file -> list of arrays [chunk, coord, Ncount, density/number], one per
output block.
- x_profiles: Density profiles along x (per box volume Ly*lz) from the atom counts; the last bin is
cut at the box boundary.
- centered_mean: Average profile on a uniform periodic grid; every block is shifted so that its slab
center is at Lx/2.
- slab_profile: Eq. 6.1 for a slab of half thickness h centered at Lx/2, with periodic images; d =
width parameter.
- coexisting: (rho_L, rho_V, w_10-90) per box volume from the fit of Eq. 6.1 to the centered mean
profile.
- coexistence: (rho_L + rho_V)/2 = rho_c + A (T - Tc);  rho_L - rho_V = B (1 - T/Tc)^beta.
- critical_fit: Nonlinear fit of both branches; only points with rho_L - rho_V > FIT_MIN_DRHO are
used.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def read_blocks(fn):
    """fix ave/chunk file -> list of arrays [chunk, coord, Ncount, density/number], one per output block."""
    lines = open(fn).read().split("\n")[3:]
    blocks, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i].split()[1])
        blocks.append(np.array([[float(v) for v in lines[i + 1 + k].split()] for k in range(n)]))
        i += n + 1
    return blocks


def x_profiles(folder, lz):
    """Density profiles along x (per box volume Ly*lz) from the atom counts; the last bin is cut at the box boundary."""
    blocks = read_blocks(os.path.join(folder, "profile_x.dat" if os.path.exists(os.path.join(folder, "profile_x.dat")) else "profile.dat"))
    edges = np.minimum(BIN_X * np.arange(len(blocks[0]) + 1), LX)
    xc, w = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    return xc, w, np.array([b[:, 2] for b in blocks]) / (w * LY * lz)


def centered_mean(xc, w, rho, n_grid=80):
    """Average profile on a uniform periodic grid; every block is shifted so that its slab center is at Lx/2."""
    xg = (np.arange(n_grid) + 0.5) * LX / n_grid
    th = 2 * np.pi * xc / LX
    out = []
    for r in rho:
        x0 = (np.arctan2((r * w * np.sin(th)).sum(), (r * w * np.cos(th)).sum()) / (2 * np.pi) * LX) % LX
        out.append(np.interp(xg + x0 - 0.5 * LX, xc, r, period=LX))
    return xg, np.mean(out, axis=0)


def slab_profile(x, rl, rv, h, d):
    """Eq. 6.1 for a slab of half thickness h centered at Lx/2, with periodic images; d = width parameter."""
    u = x - 0.5 * LX
    s = sum(0.5 * (np.tanh(2 * (u - n * LX + h) / d) - np.tanh(2 * (u - n * LX - h) / d)) for n in (-1, 0, 1))
    return rv + (rl - rv) * s


def coexisting(folder, lz):
    """(rho_L, rho_V, w_10-90) per box volume from the fit of Eq. 6.1 to the centered mean profile."""
    xc, w, rho = x_profiles(folder, lz)
    x, mean = centered_mean(xc, w, rho)
    popt, _ = curve_fit(slab_profile, x, mean, p0=[mean[len(x) // 2], mean[0], 10.0, 1.5])
    return popt[0], popt[1], abs(popt[3]) * np.log(3)


def coexistence(T, Tc, rho_c, A, B):
    """(rho_L + rho_V)/2 = rho_c + A (T - Tc);  rho_L - rho_V = B (1 - T/Tc)^beta."""
    dia = rho_c + A * (T - Tc)
    dr = B * np.clip(1 - T / Tc, 0, None) ** BETA
    return dia + 0.5 * dr, dia - 0.5 * dr


def critical_fit(T, rl, rv):
    """Nonlinear fit of both branches; only points with rho_L - rho_V > FIT_MIN_DRHO are used."""
    T, rl, rv = map(np.asarray, (T, rl, rv))
    ok = (rl - rv) > FIT_MIN_DRHO
    Ts, rl_, rv_ = T[ok], rl[ok], rv[ok]
    s0, i0 = np.polyfit(Ts, (rl_ - rv_) ** (1 / BETA), 1)
    Tc0 = -i0 / s0
    c0, e0 = np.polyfit(Ts, 0.5 * (rl_ + rv_), 1)

    def model(T2, Tc, rho_c, A, B):
        a, b = coexistence(T2[:len(Ts)], Tc, rho_c, A, B)
        return np.concatenate([a, b])

    p, pcov = curve_fit(model, np.concatenate([Ts, Ts]), np.concatenate([rl_, rv_]),
                     p0=[Tc0, c0 * Tc0 + e0, c0, (rl_[0] - rv_[0]) / (1 - Ts[0] / Tc0) ** BETA])
    return p, Ts.min(), np.sqrt(np.diag(pcov))             # parameters, lowest T used, standard errors

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate and plot the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# ---------------- bulk reference (same cutoff, same cross-section) ---------------------------------------
bulk = [coexisting(os.path.join(BULK_PATH, f"T{T:.2f}"), LZ_BULK) for T in BULK_T]
bl, bv = np.array([b[0] for b in bulk]), np.array([b[1] for b in bulk])
pb, Tmin_b, eb = critical_fit(BULK_T, bl, bv)
Tc_b, rc_b = pb[0], pb[1]
print(f"bulk (periodic):  Tc* = {Tc_b:.3f} +/- {eb[0]:.3f}, rho_c* = {rc_b:.3f} +/- {eb[1]:.3f}   (fit standard errors)")

# ---------------- pores -------------------------------------------------------------------------------------
scale = H / H_ACC                                                  # box-volume density -> density per accessible volume
res = {}
for name, label in CASES:
    out = []
    for T in TEMPERATURES:
        out.append(coexisting(os.path.join(PORE_DIR, name, f"T{T:.2f}"), H))
    rl = scale * np.array([o[0] for o in out])
    rv = scale * np.array([o[1] for o in out])
    wd = np.array([o[2] for o in out])
    p, Tmin, err = critical_fit(TEMPERATURES, rl, rv)
    res[name] = dict(rl=rl, rv=rv, w=wd, p=p, Tmin=Tmin, Tc=p[0], rc=p[1], eTc=err[0], erc=err[1])

print("\nwall type        Tc*     dTc/Tc_bulk   rho_c*(acc.)  d rho_c/rho_c,bulk")
print(f"{'bulk':12s}  {Tc_b:6.3f}   {'--':>9s}     {rc_b:7.3f}      {'--':>9s}")
for name, r in res.items():
    print(f"{name:12s}  {r['Tc']:6.3f}   {100*(r['Tc']/Tc_b-1):+8.1f} %    {r['rc']:7.3f}      {100*(r['rc']/rc_b-1):+8.1f} %"
          f"     (std. errors: Tc {r['eTc']:.3f}, rho_c {r['erc']:.3f})")
print("\nrho_L*, rho_V* (per accessible volume) and 10-90 width (sigma) at each temperature:")
print("  bulk        " + "  ".join(f"T={T:.2f}: {l:.3f}/{v:.3f}" for T, l, v in zip(BULK_T, bl, bv)))
for name, r in res.items():
    print(f"  {name:11s} " + "  ".join(f"T={T:.2f}: {l:.3f}/{v:.3f} ({w:.1f})" for T, l, v, w in zip(TEMPERATURES, r["rl"], r["rv"], r["w"])))

# ---------------- density profiles across the pore (liquid core) ------------------------------------------
zprof = {}
for name, _ in CASES:
    blocks = read_blocks(os.path.join(PORE_DIR, name, f"T{T_PROFILE:.2f}", "profile_z_liquid.dat"))
    z = blocks[0][:, 1]
    counts = np.mean([b[:, 2] for b in blocks], axis=0)
    zprof[name] = (z, counts / (BIN_Z * LY * CORE_Z))              # atoms per volume of the bin (not per accessible volume)
    pk = np.argmax(zprof[name][1][: len(z) // 2])
    print(f"{name:11s} T* = {T_PROFILE:.2f}: contact peak rho* = {zprof[name][1][pk]:.3f} at z = {z[pk]:.2f} sigma;"
          f" center rho* = {zprof[name][1][len(z)//2 - 2: len(z)//2 + 2].mean():.3f}")

# ------------------------------ figure ----------------------------------------
centimeter = 1 / 2.54
fig, ax = plt.subplots(1, 3,
                       figsize=(19*centimeter, 7*centimeter),
                       dpi=750,
                       gridspec_kw={'wspace':0.35})

for name, label in CASES:
    z, d = zprof[name]
    ax[0].plot(z, d, color=COLORS[name], label=label)
ax[0].axhline(bl[BULK_T.index(0.90)] if T_PROFILE == 0.90 else np.nan, color="k", ls=":", label="bulk liquid")
ax[0].set_xlabel(r"$z/\sigma$")
ax[0].set_ylabel(r"$\rho^*$ (liquid core, $T^*$ = %.2f)" % T_PROFILE)
ax[0].set_xlim(0, H)
ax[0].legend(frameon=False, fontsize=6)
ax[0].set_ylim([0,2])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Define functions
- smooth_curve
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def smooth_curve(p, Tmin, Tc, factor=1.0):
    Tf = np.linspace(Tmin - 0.02, Tc, 200)
    rl_f, rv_f = coexistence(Tf, *p)
    return factor * np.concatenate([rl_f, rv_f[::-1]]), np.concatenate([Tf, Tf[::-1]])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


x_, y_ = smooth_curve(pb, Tmin_b, Tc_b)
ax[1].plot(x_, y_, "k-")
ax[1].plot(np.concatenate([bl, bv]), BULK_T * 2, "ko", ms=3, mfc="none", mew=.75, label="bulk")
for name, label in CASES:
    r = res[name]
    x_, y_ = smooth_curve(r["p"], r["Tmin"], r["Tc"])
    ax[1].plot(x_, y_, "-", color=COLORS[name])
    ax[1].plot(np.concatenate([r["rl"], r["rv"]]), TEMPERATURES * 2, "o", ms=3, mew=.75, color=COLORS[name], label=label)
ax[1].set_xlabel(r"$\rho^*$ (per accessible volume)")
ax[1].set_ylabel(r"$T^*$")
ax[1].legend(frameon=False, fontsize=6, loc="lower center")

w_ = 0.35
xs = np.arange(len(CASES))
ax[2].bar(xs - w_ / 2, [100 * (res[n]["Tc"] / Tc_b - 1) for n, _ in CASES], w_,
          color=[COLORS[n] for n, _ in CASES], label=r"$\Delta T_\mathrm{crit}/T_\mathrm{crit}$")
ax[2].bar(xs + w_ / 2, [100 * (res[n]["rc"] / rc_b - 1) for n, _ in CASES], w_,
          color=[COLORS[n] for n, _ in CASES], alpha=0.6, label=r"$\Delta\rho_\mathrm{crit}/\rho_\mathrm{crit}$")
ax[2].axhline(0, color="k", lw=0.5)
ax[2].set_xticks(xs)
ax[2].set_xticklabels(["repulsive", "attractive", "strong"])
ax[2].set_ylabel("shift relative to bulk [%]")
ax[2].legend(frameon=False, fontsize=6)

for a, lab in zip(ax, "ABC"):
    a.text(-0.25, 0.975, lab, transform=a.transAxes, fontweight="bold", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
