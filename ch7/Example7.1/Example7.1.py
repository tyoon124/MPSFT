"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 7 Reactions in Supercritical Fluids
Example 7.1
ReaxFF molecular dynamics of a radical-initiated oxidation at several fluid densities: In this
example, we follow the oxidation of methanol by OH radicals in dense water with the ReaxFF
parametrization of Chenoweth et al. and determine how the rate of the reaction depends on the
density of the fluid. The OH radicals are placed in the initial configuration to represent the
radical pool of supercritical water oxidation, which makes the reaction occur within tens of
picoseconds without an initiation barrier. The density is the control variable, and the results are
shown in Fig. 7.1.

The simulations are run with LAMMPS (one input file in.reax per density, in the folders
Example7.1/rho<value>/); this script only analyzes and plots. No command-line arguments: edit the
SETTINGS block below and run
    python Example7.1.py
For every density folder it
* reads the trajectory traj.lammpstrj (types: 1 = C, 2 = H, 3 = O) and builds the connectivity from
distance cutoffs; the counts below are saved to counts.dat in the same folder, and when the trajectory
is not present (as in the GitHub repository, which holds counts.dat instead of the 16 MB trajectories),
the script reads counts.dat,
* counts intact CH3OH molecules (a C with three H and one O, whose O carries one H), free OH
radicals
    (an O with one H and nothing else) and free H atoms as a function of time,
* fits the decay of the number of intact methanol molecules, N(t) = N0 exp(-k t), for the
first-order rate
constant k within the window FIT_TMIN ... FIT_TMAX (its error is a block bootstrap of the residuals,
because successive
frames are strongly correlated; the fit errors of the least-squares routine are 10-20 times too
small), and also in the first and
second half of the window, which shows whether the decay is a single exponential (it is not: the
initial radical burst is fast), (the first ps is excluded because the randomly inserted
configuration relaxes; the number of OH radicals is not constant, so k is an apparent constant of
this simulation),
* reads the average pressure from pressure.dat and compares it with the ideal-gas value N kT/V of
the N_MOL molecules
(Z = p/p_ideal), and measures the homogeneity of the fluid from the fraction of cells (4 x 4 x 4
grid) of the last
frame that hold less than 25 % of the mean number of atoms. A homogeneous fluid has a few such
cells; a phase-separated
one (ReaxFF water at 1500 K, 0.1-0.3 g/cm3) has 20-30 %, and its pressure no longer follows the
density. The density is
the control variable, so a state with a large empty fraction, or with fewer than MIN_EVENTS consumed
methanol molecules,
is flagged and left out of the power-law fit, and from all densities the local density exponent d ln
k / d ln rho and a power-law fit k ~ rho^m. Because the OH concentration is proportional to rho, the
exponent of the bimolecular constant k_c = k/[OH] is m - 1.
Requires: numpy, scipy, matplotlib.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os, glob
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter, FixedLocator

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
DENSITIES   = [0.05, 0.10, 0.20, 0.30]   # g/cm3, folders Example7.1/rho<value>/
TEMPERATURE = 2500.0                     # K (as in in.reax)
N_MOL       = 296                        # molecules at t = 0 (32 CH3OH + 64 OH + 200 H2O), for the ideal-gas pressure
DT_FRAME    = 0.05                       # ps between trajectory frames (200 steps x 0.25 fs)
R_CH, R_CO, R_OH = 1.35, 1.75, 1.15      # Angstrom, bond cutoffs of C-H, C-O, O-H
FIT_TMIN    = 1.0                        # ps, lower limit of the fit window (relaxation of the initial configuration)
FIT_TMAX    = None                       # ps, upper limit of the fit window (None = whole run)
MIN_EVENTS  = 10                         # minimum number of consumed methanol molecules in the window for a usable k
MAX_EMPTY   = 0.10                       # maximum fraction of nearly empty cells for a homogeneous fluid
N_BOOT      = 500                        # block-bootstrap resamples for the error of k
BLOCK_PS    = 5.0                        # ps, block length of the bootstrap (the misfit of a single exponential is correlated over several ps)
SNAPSHOT    = (0.30, 0.0)                # (density, time in ps) of the configuration printed to snapshot.xyz
SAVE_FIG    = "figure7.1.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
cm = 1/2.54
R_GAS, K_B = 8.314462618, 1.380649e-23

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- read_traj
- classify
- empty_fraction
- first_order: fit N(t) = Ninf + (N0 - Ninf) exp(-k (t - t0)) in the window (Ninf = 0 unless
plateau=True); returns k, its st
- bootstrap_k: standard error of k from a moving-block bootstrap of the residuals of the fit (blocks
of BLOCK_PS)
- second_order: k2 (per molecule and ps) of dN_M/dt = -k2 N_M N_OH from 1-ps block means: N_M(t) -
N_M(t1) = -k2 * integral(N_
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def read_traj(fn):
    with open(fn) as f:
        while True:
            line = f.readline()
            if not line:
                return
            assert line.startswith("ITEM: TIMESTEP")
            step = int(f.readline())
            f.readline()
            n = int(f.readline())
            f.readline()
            try:
                box = np.array([[float(x) for x in f.readline().split()[:2]] for _ in range(3)])
            except ValueError:
                return
            f.readline()
            try:
                data = np.array([f.readline().split() for _ in range(n)], dtype=float)
                if data.shape != (n, 5):
                    return           # last frame incomplete (run still going)
            except ValueError:
                return
            yield step, box, data[:, 1].astype(int), data[:, 2:5]

def classify(types, pos, box):
    L = box[:, 1] - box[:, 0]
    p = (pos - box[:, 0]) % L
    tree = cKDTree(p, boxsize=L)
    iC, iH, iO = (np.where(types == k)[0] for k in (1, 2, 3))
    def nbrs(idx, jdx, rc):                       # for each i in idx the list of neighbors within rc in jdx
        sub = cKDTree(p[jdx], boxsize=L)
        return [jdx[j] for j in sub.query_ball_point(p[idx], rc)]
    nH_of_C = nbrs(iC, iH, R_CH)
    nO_of_C = nbrs(iC, iO, R_CO)
    nC_of_C = nbrs(iC, iC, 1.8)
    nH_of_O = nbrs(iO, iH, R_OH)
    nO_of_O = nbrs(iO, iO, 1.6)
    nC_of_O = nbrs(iO, iC, R_CO)
    Hcount_O = dict(zip(iO, [len(x) for x in nH_of_O]))
    n_meoh = 0
    for k, c in enumerate(iC):
        if len(nH_of_C[k]) == 3 and len(nO_of_C[k]) == 1 and len(nC_of_C[k]) == 1 and Hcount_O[nO_of_C[k][0]] == 1:
            n_meoh += 1
    n_oh = sum(1 for k, o in enumerate(iO) if len(nH_of_O[k]) == 1 and len(nO_of_O[k]) == 1 and len(nC_of_O[k]) == 0)
    n_w = sum(1 for k, o in enumerate(iO) if len(nH_of_O[k]) == 2 and len(nO_of_O[k]) == 1 and len(nC_of_O[k]) == 0)
    n_h = 0
    if len(iH):
        near = [len(a) + len(b) for a, b in zip(nbrs(iH, iC, R_CH), nbrs(iH, iO, R_OH))]
        nH_H = nbrs(iH, iH, 0.9)
        n_h = sum(1 for k in range(len(iH)) if near[k] == 0 and len(nH_H[k]) == 1)   # only itself within 0.9 A
    return n_meoh, n_oh, n_w, n_h

def empty_fraction(pos, box, n=4):
    L = box[:, 1] - box[:, 0]
    idx = np.minimum((((pos - box[:, 0]) % L) / L * n).astype(int), n - 1)
    c = np.bincount(idx[:, 0] * n * n + idx[:, 1] * n + idx[:, 2], minlength=n**3)
    return np.mean(c < 0.25 * c.mean())

def first_order(t, n, tmin=0.0, tmax=None, plateau=False):
    """fit N(t) = Ninf + (N0 - Ninf) exp(-k (t - t0)) in the window (Ninf = 0 unless plateau=True); returns k, its standard error,
    the window mask and the fitted curve parameters"""
    m = t >= tmin
    if tmax is not None:
        m &= t <= tmax
    t0 = t[m][0]
    if plateau:
        f = lambda x, N0, k, Ninf: Ninf + (N0 - Ninf) * np.exp(-k * (x - t0))
        popt, pcov = curve_fit(f, t[m], n[m], p0=(n[m][0], 0.1, 0.5 * n[m][-1]), bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=20000)
    else:
        f = lambda x, N0, k: N0 * np.exp(-k * (x - t0))
        popt, pcov = curve_fit(f, t[m], n[m], p0=(n[m][0], 0.05), maxfev=10000)
    return popt[1], np.sqrt(pcov[1, 1]), m, (t0, popt[0], popt[2] if plateau else 0.0)

def bootstrap_k(t, n, tmin, tmax, curve, k):
    """standard error of k from a moving-block bootstrap of the residuals of the fit (blocks of BLOCK_PS)"""
    m = t >= tmin
    if tmax is not None:
        m &= t <= tmax
    tt, nn = t[m], n[m]
    t0, N0, _ = curve
    fit = N0 * np.exp(-k * (tt - t0))
    r = nn - fit
    nb = max(int(round(BLOCK_PS / DT_FRAME)), 1)
    rng = np.random.default_rng(1)
    ks = []
    for _ in range(N_BOOT):
        idx = np.concatenate([(j + np.arange(nb)) % len(r) for j in rng.integers(0, len(r), len(r) // nb + 1)])[:len(r)]
        try:
            ks.append(first_order(tt, np.maximum(fit + r[idx], 1e-6), tmin)[0])
        except RuntimeError:
            pass
    return np.std(ks)

def second_order(t, nM, nOH, tmin, tmax):
    """k2 (per molecule and ps) of dN_M/dt = -k2 N_M N_OH from 1-ps block means: N_M(t) - N_M(t1) = -k2 * integral(N_M N_OH dt)"""
    t1 = tmin
    tm = t.max() if tmax is None else tmax
    nbk = int((tm - t1) // 1.0)
    edges = [(t >= t1 + i) & (t < t1 + i + 1) for i in range(nbk)]
    Mb = np.array([nM[e].mean() for e in edges])
    Ob = np.array([nOH[e].mean() for e in edges])
    x = np.cumsum(np.r_[0, 0.5 * (Mb[1:] * Ob[1:] + Mb[:-1] * Ob[:-1])])
    y = Mb - Mb[0]
    return -(x @ y) / (x @ x)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

res = {}
for rho in DENSITIES:
    d = os.path.join(HERE, "Example7.1", f"rho{rho:.2f}")    # book layout (Examples/ch7/Example7.1/rho<value>/)
    if not os.path.isdir(d):
        d = os.path.join(HERE, f"rho{rho:.2f}")            # GitHub layout (data next to the script)
    fn = os.path.join(d, "traj.lammpstrj")
    fc = os.path.join(d, "counts.dat")
    if os.path.exists(fn):                       # full trajectory: count the species frame by frame and save the counts
        t, cnt = [], []
        for step, box, types, pos in read_traj(fn):
            t.append(step * 0.00025)
            cnt.append(classify(types, pos, box))
            if rho == SNAPSHOT[0] and abs(t[-1] - SNAPSHOT[1]) < 0.5 * DT_FRAME:
                L = box[:, 1] - box[:, 0]
                pw = (pos - box[:, 0]) % L
                el = {1: "C", 2: "H", 3: "O"}
                with open(os.path.join(HERE, "snapshot.xyz"), "w") as g:
                    g.write(f"{len(types)}\nrho={rho} t={t[-1]:.3f} ps Lattice=\"{L[0]} 0 0 0 {L[1]} 0 0 0 {L[2]}\"\n")
                    for ty, x in zip(types, pw):
                        g.write(f"{el[ty]} {x[0]:.4f} {x[1]:.4f} {x[2]:.4f}\n")
        t = np.array(t)
        cnt = np.array(cnt)
        V_A3 = np.prod(box[:, 1] - box[:, 0])    # box volume, A^3
        emp = empty_fraction(pos, box)           # homogeneity of the last frame
        np.savetxt(fc, np.column_stack([t, cnt]), fmt=["%.3f", "%d", "%d", "%d", "%d"],
                   header=f"{V_A3:.4f} {emp:.4f}\nfirst line: box volume (A^3) and fraction of nearly empty cells of the last frame\nt/ps  CH3OH  OH  H2O  H")
    elif os.path.exists(fc):                     # GitHub version: the counts saved from the trajectory
        with open(fc) as f:
            V_A3, emp = (float(x) for x in f.readline().lstrip("# ").split())
        data = np.loadtxt(fc)
        t = data[:, 0]
        cnt = data[:, 1:].astype(int)
    else:
        print("missing", fn, "and", fc)
        continue
    k, dk, m, curve = first_order(t, cnt[:, 0].astype(float), FIT_TMIN, FIT_TMAX)
    try:
        kp, dkp = first_order(t, cnt[:, 0].astype(float), FIT_TMIN, FIT_TMAX, plateau=True)[:2]
    except RuntimeError:
        kp, dkp = np.nan, np.nan
    dk = bootstrap_k(t, cnt[:, 0].astype(float), FIT_TMIN, FIT_TMAX, curve, k)
    tmid = 0.5 * (t[m][0] + t[m][-1])
    try:
        k_early = first_order(t, cnt[:, 0].astype(float), FIT_TMIN, tmid)[0]
        k_late = first_order(t, cnt[:, 0].astype(float), tmid, FIT_TMAX)[0]
    except RuntimeError:
        k_early = k_late = np.nan
    events = cnt[m][0, 0] - cnt[m][-1, 0]
    pr = np.loadtxt(os.path.join(d, "pressure.dat"), comments="#")
    prm = pr[pr[:, 0] * 0.00025 >= FIT_TMIN, 1].mean() * 0.101325       # atm -> MPa
    V = V_A3 * 1e-30                                                    # m3
    Z = prm * 1e6 / (N_MOL * K_B * TEMPERATURE / V)
    k2 = second_order(t, cnt[:, 0].astype(float), cnt[:, 1].astype(float), FIT_TMIN, FIT_TMAX)
    kc = k2 * V * 1e3 * 6.02214076e23                                     # L mol-1 ps-1 (V in m3 -> L)
    ok = events >= MIN_EVENTS and emp <= MAX_EMPTY and k > 0
    res[rho] = dict(t=t, n=cnt, k=k, dk=dk, kc=kc, ke=k_early, kl=k_late, p=prm, ok=ok, curve=curve, tmax=t[m][-1])
    print(f"rho = {rho:.2f} g/cm3: <p> = {prm:7.1f} MPa (Z = {Z:.2f}); empty cells {emp:.2f}; methanol {cnt[m][0,0]} -> {cnt[m][-1,0]} "
          f"({events} consumed) in the window; k = {k:.4f} +- {dk:.4f} 1/ps (block bootstrap; first/second half of the window: {k_early:.4f}/{k_late:.4f}; with a plateau: {kp:.4f} +- {dkp:.4f}); OH {cnt[0,1]} -> {cnt[-1,1]}, free H {cnt[0,3]} -> {cnt[-1,3]}"
          + ("" if ok else "   <-- NOT USABLE (too few events, inhomogeneous fluid, or k <= 0)"))

keys = sorted(res)
print("\nrho1-rho2   local density exponent d ln k / d ln rho  (m_c = this value - 1)")
for a, b in zip(keys[:-1], keys[1:]):
    if res[a]["ok"] and res[b]["ok"]:
        print(f"{a:.2f}-{b:.2f}   {np.log(res[b]['k'] / res[a]['k']) / np.log(b / a):6.2f}")
good = [r for r in keys if res[r]["ok"]]
if len(good) > 2:
    m_all, ln0 = np.polyfit(np.log(good), np.log([res[r]["k"] for r in good]), 1)
    print("bimolecular constant k_c = k_2 V N_A (L mol-1 ps-1) from d N_M/dt = -k_2 N_M N_OH:", {r: round(float(res[r]["kc"]), 3) for r in good},
          f" exponent {np.polyfit(np.log(good), np.log([res[r]['kc'] for r in good]), 1)[0]:.2f}")
    print(f"power-law fit over all densities: k ~ rho^{m_all:.2f}  (the OH concentration is proportional to rho, so the "
          f"density exponent of the bimolecular constant k_c = k/[OH] is m_c = {m_all - 1:.2f}, Eq. 7.12)")

fig, ax = plt.subplots(1, 3, figsize=(19*cm, 6.5*cm), gridspec_kw={'wspace':0.38})
cols = plt.cm.viridis(np.linspace(0, 0.9, len(keys)))
for c, rho in zip(cols, keys):
    r = res[rho]
    ax[0].plot(r["t"], r["n"][:, 0], color=c, label=rf"$\rho$ = {rho:.2f} g cm$^{{-3}}$")
    ax[1].plot(r["t"], r["n"][:, 1], color=c)
    ax[1].plot(r["t"], r["n"][:, 3], color=c, ls=":")
ax[0].set_xlabel("$t$ / ps")
ax[0].set_ylabel("intact CH$_3$OH molecules")
ax[0].set_xlim([0,30])
ax[0].legend(frameon=False, fontsize=6)
ax[1].set_xlabel("$t$ / ps")
ax[1].set_ylim([0,70])
ax[1].set_xlim([0,30])
ax[1].set_ylabel("free OH (solid), free H (dotted)")
rr = np.array([r for r in keys])
kk = np.array([res[r]["k"] for r in keys])
dk = np.array([res[r]["dk"] for r in keys])
okm = np.array([res[r]["ok"] for r in keys])
ax[2].errorbar(rr[okm], kk[okm], dk[okm], fmt="ko", ms=3, mew=.75, capsize=2, elinewidth=.75)
if (~okm).any():
    ax[2].plot(rr[~okm], np.abs(kk[~okm]), "o", color="gray", ms=3, mew=.75, mfc="none", label="not usable")
ax[2].set_xscale("log")
ax[2].set_yscale("log")
if (~okm).any():
    ax[2].legend(frameon=False, fontsize=6)
ax[2].set_xlabel(r"$\rho$ / g cm$^{-3}$")
ax[2].set_ylabel(r"$k$ / ps$^{-1}$")
for axis, ticks in ((ax[2].xaxis, [0.05, 0.1, 0.2, 0.3]), (ax[2].yaxis, [0.02, 0.03, 0.05, 0.1])):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())
ax[2].set_xlim(0.04, 0.37)
ax[2].set_ylim(0.015, 0.05)
for a, lab in zip(ax, "ABC"):
    a.text(-0.30, 0.975, lab, transform=a.transAxes, fontweight="bold", fontsize=7)
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
