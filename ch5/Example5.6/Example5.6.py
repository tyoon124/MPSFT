#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 5 Molecular Simulation Analyses
Example 5.6
Excess chemical potential and excess entropy of the Lennard-Jones fluid by the thermodynamic route
and Widom insertion: This example calculates mu_ex and s_ex of the LJ fluid on the supercritical
isotherm T^*=1.40 from the thermodynamic route and from Widom insertion, and compares them with the
reference Equation of State.

Widom insertion (Sections 3 and 4) reads traj.lammpstrj (id x y z, wrapped coordinates) and avg.txt
(columns: step, Z, U/N) in every rho_*/ folder, inserts test particles at random positions in each
frame, and estimates mu_ex = -kT ln < exp(-dU/kT) > + mu_tail, with dU the energy of the test
particle (LJ truncated at RC, minimum image) and mu_tail the analytic tail correction beyond RC
(matches `pair_modify tail yes` in the LAMMPS input). The results are stored in TABLE (columns: rho
Z Z_err U_per_atom mu_ex_Widom(+tail) mu_err ESS_fraction n_frames); the insertions are run only if
TABLE is missing, because they take long. The thermodynamic route is compared with Widom insertion
and with the reference equation of state of Thol et al. (2016) as implemented in teqp. Edit the
settings block, then run: python Example5.6.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline, PchipInterpolator

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Settings
Edit the parameters below to change the calculation.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ------------------------- SETTINGS -------------------------
TABLE        = "Example5.6_results.txt"
T_STAR       = 1.40
B2_STAR      = None        # None: computed here from the full LJ potential; or a number (e.g. from Ref. bell2020zero)
SMOOTH       = 1.0         # spline smoothing: sum of squared weighted residuals = SMOOTH * (number of points)
RHO_WIDOM_MAX = 0.60       # Widom points above this are drawn as open symbols (failed)
N_BOOT       = 300         # Monte Carlo resamplings of Z for the error band
USE_TEQP     = True        # reference EoS: Thol et al. (2016) LJ EoS from teqp (pip install teqp)
FIG_NAME     = "../figure5.6.pdf"
OUT_TABLE    = "Example5.6_thermo_route.txt"
SEED         = 1

# Widom insertion (run only if TABLE is missing)
RC            = 4.0         # LJ cutoff, same as in the LAMMPS input
N_ATOMS       = 500
N_INSERT      = 5000        # test insertions per frame
FRAME_STRIDE  = 1           # use every n-th frame (1 = all)
WIDOM_SEED    = 12345
# ------------------------------------------------------------

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- b2_lj
- mu_tail: Tail correction of mu_ex (energy units): 2 * u_tail, u_tail per atom beyond RC.
- read_frames
- insertion_energies: Energies dU of N_INSERT random test particles in one frame.
- widom
- fit_and_integrate: (Z-1)/rho anchored at B2 at rho=0 -> weighted smoothing spline -> beta*mu_ex,
Z, s+ on rho points.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def b2_lj(T):
    f = lambda r: (np.exp(-4*(r**-12 - r**-6)/T) - 1)*r*r
    return -2*np.pi*(quad(f, 0, 1, limit=200)[0] + quad(f, 1, 50, limit=400)[0])

def mu_tail(rho):
    """Tail correction of mu_ex (energy units): 2 * u_tail, u_tail per atom beyond RC."""
    return 2.0*(8.0/3.0)*np.pi*rho*((1.0/3.0)*RC**-9 - RC**-3)

def read_frames(fname, n_atoms):
    frames, boxes = [], []
    with open(fname) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("ITEM: BOX BOUNDS"):
                lo_hi = [list(map(float, f.readline().split()[:2])) for _ in range(3)]
                boxes.append(lo_hi[0][1] - lo_hi[0][0])        # cubic box
            if line.startswith("ITEM: ATOMS"):
                frames.append(np.array([f.readline().split()[1:4] for _ in range(n_atoms)], float))
    return frames, boxes

def insertion_energies(xyz, L, rng):
    """Energies dU of N_INSERT random test particles in one frame."""
    trial = rng.random((N_INSERT, 3))*L
    d = trial[:, None, :] - xyz[None, :, :]
    d -= L*np.round(d/L)                                         # minimum image
    r2 = (d**2).sum(-1)
    inv6 = np.where(r2 < RC**2, 1.0/np.maximum(r2, 1e-12)**3, 0.0)
    return (4.0*(inv6**2 - inv6)).sum(1)

def widom(frames, boxes, rng):
    boltz, dU_all = [], []
    for xyz, L in zip(frames, boxes):
        dU = insertion_energies(xyz, L, rng)
        boltz.append(np.exp(-dU/T_STAR).mean())
        dU_all.append(dU)
    boltz = np.array(boltz)
    mu = -T_STAR*np.log(boltz.mean())
    # jackknife over frames
    jk = np.array([-T_STAR*np.log(np.delete(boltz, i).mean()) for i in range(len(boltz))])
    err = np.sqrt((len(jk) - 1)/len(jk)*np.sum((jk - jk.mean())**2))
    # effective sample fraction of the Boltzmann weights (small -> estimate unreliable)
    w = np.exp(-np.concatenate(dU_all)/T_STAR)
    ess_fraction = w.sum()**2/(w**2).sum()/w.size
    return mu, err, ess_fraction

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Widom insertion (only if TABLE is missing)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

if not os.path.isfile(TABLE):
    rng = np.random.default_rng(WIDOM_SEED)
    rows = []
    print(f"{'rho':>6} {'frames':>6} {'Z':>8} {'U/N':>9} {'mu_ex':>9} {'err':>7} {'ESS/N':>7}")
    for folder in sorted(glob.glob("rho_*")):
        rho_f = float(os.path.basename(folder).split("_")[1])
        avg = np.loadtxt(os.path.join(folder, "avg.txt"), comments="#")
        Z_f = avg[:, 1].mean()
        Z_ferr = avg[:, 1].std(ddof=1)/np.sqrt(len(avg))
        u_f = avg[:, 2].mean()
        frames, boxes = read_frames(os.path.join(folder, "traj.lammpstrj"), N_ATOMS)
        frames, boxes = frames[::FRAME_STRIDE], boxes[::FRAME_STRIDE]
        mu, err, ess = widom(frames, boxes, rng)
        mu += mu_tail(rho_f)
        rows.append((rho_f, Z_f, Z_ferr, u_f, mu, err, ess, len(frames)))
        print(f"{rho_f:6.2f} {len(frames):6d} {Z_f:8.4f} {u_f:9.4f} {mu:9.4f} {err:7.4f} {ess:7.4f}")
    hdr = "rho  Z  Z_err  U_per_atom  mu_ex_Widom(+tail)  mu_err  ESS_fraction  n_frames"
    np.savetxt(TABLE, np.array(rows), header=hdr, fmt="%.5f")
    print("saved", TABLE)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Read the results of the simulations and Widom insertion
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

d = np.loadtxt(TABLE)
rho, Z, Ze, U, mu_w, mu_we = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5]
B2 = b2_lj(T_STAR) if B2_STAR is None else B2_STAR
print(f"B2* = {B2:.4f}")

def fit_and_integrate(Zvals, rng_grid):
    """(Z-1)/rho anchored at B2 at rho=0 -> weighted smoothing spline -> beta*mu_ex, Z, s+ on rho points."""
    x = np.r_[0.0, rho]
    y = np.r_[B2, (Zvals - 1)/rho]
    sig = np.r_[1e-3, Ze/rho]
    spl = UnivariateSpline(x, y, w=1/sig, k=3, s=SMOOTH*len(x))
    zfun = lambda r: 1 + r*spl(r)
    bmu = np.array([spl.integral(0, r) + zfun(r) - 1 for r in rng_grid])
    return spl, zfun, bmu

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Set constants and grids
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

grid = np.linspace(0.0, rho.max(), 200)
spl, zfun, bmu_pts = fit_and_integrate(Z, rho)
_, _, bmu_grid = fit_and_integrate(Z, grid)

# s_ex/k = (u_ex + T(Z-1) - mu_ex)/T ; u_ex/N from the simulations (interpolated)
u_int = PchipInterpolator(np.r_[0.0, rho], np.r_[0.0, U])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
7. Define functions
- splus
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def splus(r, bmu):
    return -((u_int(r) + T_STAR*(zfun(r) - 1) - T_STAR*bmu)/T_STAR)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
8. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
sp_pts, sp_grid = splus(rho, bmu_pts), splus(grid, bmu_grid)

# Monte Carlo error band (resample Z within its error, refit)
rng = np.random.default_rng(SEED)
bm_mc, sp_mc = [], []
for _ in range(N_BOOT):
    Zb = Z + rng.normal(0, Ze)
    sp_b, zf_b, bm = fit_and_integrate(Zb, grid)
    bm_mc.append(bm)
    sp_mc.append(-((u_int(grid) + T_STAR*(zf_b(grid) - 1) - T_STAR*bm)/T_STAR))
bm_lo, bm_hi = np.percentile(bm_mc, [16, 84], axis=0)
sp_lo, sp_hi = np.percentile(sp_mc, [16, 84], axis=0)

# ---- reference equation of state (Thol et al. 2016, teqp), reduced units ----
ref = None
if USE_TEQP:
    try:
        import teqp
        model = teqp.build_LJ126_TholJPCRD2016()
        z1 = np.array([1.0])
        def eos_ref(r):
            """Z, mu_ex/kT and -s_ex/k at (T_STAR, r) from the residual Helmholtz energy derivatives."""
            A00 = model.get_Arxy(0, 0, T_STAR, r, z1)
            A01 = model.get_Arxy(0, 1, T_STAR, r, z1)
            A10 = model.get_Arxy(1, 0, T_STAR, r, z1)
            return 1.0 + A01, A00 + A01, A00 - A10
        ref_grid = np.linspace(1e-4, rho.max(), 400)
        ref = np.array([eos_ref(r) for r in ref_grid])           # columns: Z, mu/kT, s+
        ref_pts = np.array([eos_ref(r) for r in rho])
    except ImportError:
        print("teqp not found: reference EoS skipped (pip install teqp)")

# ---- table ----
mu_th_pts = T_STAR*bmu_pts
cols = [rho, Z, bmu_pts, mu_w/T_STAR, mu_we/T_STAR, sp_pts]
hdr = "rho  Z(MD)  mu_ex/kT(thermo)  mu_ex/kT(Widom)  Widom_err  s+=-s_ex/k(thermo)"
if ref is not None:
    cols += [ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2]]
    hdr += "  Z(ref)  mu_ex/kT(ref)  s+(ref)"
out = np.column_stack(cols)
np.savetxt(OUT_TABLE, out, fmt="%.5f", header=hdr)
print("  ".join(hdr.split("  ")))
for row in out:
    print(" ".join(f"{x:9.4f}" for x in row))

# ---- figure ----
centimeter = 1/ 2.54
font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 7}
mpl.rc('font', **font)

for axis in ['x', 'y']:
    mpl.rcParams[f'{axis}tick.major.size'] = 3
    mpl.rcParams[f'{axis}tick.minor.size'] = 3
    mpl.rcParams[f'{axis}tick.major.width'] = .5
    mpl.rcParams[f'{axis}tick.minor.width'] = .5
    mpl.rcParams['lines.linewidth'] = .75

fig, ax = plt.subplots(1, 2, figsize=(19*centimeter, 8*centimeter),
                       dpi=750)
good = rho <= RHO_WIDOM_MAX
ax[0].plot(grid[::5], bmu_grid[::5], "o", ms=4, lw=.75, mew=.75,
           color="k", label="Thermodynamic route", mfc='w')
ax[0].errorbar(rho[good], mu_w[good]/T_STAR, 
               mu_we[good]/T_STAR, fmt="s", color="k", 
               ms=4, label="Widom insertion")
ax[0].errorbar(rho[~good], mu_w[~good]/T_STAR, 
               mu_we[~good]/T_STAR, fmt="s", mfc="k", color="k", ms=4)

if ref is not None:
    ax[0].plot(ref_grid, ref[:, 1], "r-", label="Thol EoS (LJ)")
ax[0].set_xlabel(r"$\rho^*$")
ax[0].set_ylabel(r"$\mu_\mathrm{ex}/k_\mathrm{B}T$")
ax[0].set_ylim(-2,1)       # keeps the failed Widom points from stretching the axis
ax[0].legend(frameon=False)
ax[0].text(-0.1, 1.0, "A", transform=ax[0].transAxes, fontweight="bold", va="top")

ax[1].plot(grid[::5], sp_grid[::5], ms=4, marker="o",
           color="k", ls='None', mew=0.75, mfc="w",
           label="Thermodynamic route")
if ref is not None:
    ax[1].plot(ref_grid, ref[:, 2], "r-", label="Thol EoS (LJ)")
ax[1].set_xlabel(r"$\rho^*$")
ax[1].set_ylabel(r"$-s_\mathrm{ex}/k_\mathrm{B}$")
ax[1].set_xlim([0,0.8])
ax[1].set_ylim([0,2.7])
ax[1].legend(frameon=False)
ax[1].text(-0.1, 1.0, "B", transform=ax[1].transAxes, fontweight="bold", va="top")
fig.tight_layout()
fig.savefig(FIG_NAME)
print("saved", FIG_NAME)