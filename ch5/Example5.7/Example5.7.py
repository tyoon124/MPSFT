"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 5 Molecular Simulation Analyses
Example 5.7
Self-diffusion coefficient and shear viscosity of the LJ fluid: Green-Kubo versus Einstein: This
example produces the data of Fig. 5.7. Transport coefficients are expressed in the reduced units
D^*=Dsqrt(m/epsilon)/sigma and eta^*=etasigma^2/sqrt(mepsilon), which for argon correspond to
5.4x10^-8 m^2 s^-1 and 90 uPa s, respectively.

Green-Kubo versus Einstein  (produces Fig. 5.7)

ANALYSIS ONLY. The simulations are run beforehand with LAMMPS (one input per run folder); this script
scans the run subfolders next to it (N<N>_T<T>_r<rep>/), reads their in.lj_transport (N, rho, dt,
...) and output files (run.P.dat, run.trj), and computes

    D   from the Einstein relation (MSD slope / 6) and from the VACF integral
    eta from the Green-Kubo integral (five stress components)
Yeh-Hummer finite-size analysis, writes the numerical data of Fig. 5.7 to text files, and draws
Fig. 5.7.

Run directly:  python Example5.7.py      (edit the SETTINGS block below)

Reduced LJ units: m = epsilon = sigma = k_B = 1.
Output files (plain text, whitespace-separated, '#' header lines):
<OUT_PREFIX>A_finite_size.dat : Panel A  (T = T_AB, all N)  N, L, 1/L, D_L, D_VACF, eta (+ std.
errors)
<OUT_PREFIX>BC_vs_T.dat       : Panels B and C (N = N_C, all T)  T, T/Tc, eta, D_L, D_inf (+ std.
errors)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Settings
Edit the parameters below to change the calculation.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# =============================== SETTINGS ===============================
T_AB = 1.40                      # temperature of Panels A and B
N_C = 1000                       # number of atoms of Panel C

T_LAG = 20.0                     # maximum correlation time (reduced units)
FIT_MSD = (4.0, 15.0)            # time window of the linear MSD fit
T_VACF = 8.0                     # upper limit of the VACF integral
PLATEAU = (2.0, 4.0)             # time window over which the GK running integral is averaged

XI = 2.837297                    # Yeh-Hummer constant for a cubic periodic box
TCRIT = 1.08                     # T_c^* of the truncated and shifted LJ fluid (r_c = 2.5), used for T/Tc

OUT_PREFIX = ""
CORR_FILE = "corr.npz"            # correlation functions of each run (read instead of run.trj if present)
FIG_NAME = "../figure5.7.pdf"
# ========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- acf_fft: Multi-origin autocorrelation <x(0)x(t)>, averaged over all columns of x.
- msd_fft: Multi-origin mean-square displacement of unwrapped positions r (n, natoms, 3).
- trapz
- cumtrapz
- fit_slope
- read_settings: Read the 'variable <name> index <value>' lines of a LAMMPS input file.
- read_trj: Read a LAMMPS custom dump (id xu yu zu vx vy vz, sorted by id) -> (nframes, ntrack, 6).
- analyse_run: Correlation functions of one run folder (from corr.npz if present, otherwise from
  run.P.dat and run.trj, then stored in corr.npz) -> (result, None), or (None, reason) if
unusable.
- summarize
- make_figure: Fig. 5.7 (A: D_L versus 1/L with a linear fit; B: eta; C: D_L versus T)
- main
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# --------------------------- time-correlation tools ---------------------------
def acf_fft(x, nlag):
    """Multi-origin autocorrelation <x(0)x(t)>, averaged over all columns of x."""
    n = x.shape[0]
    x = x.reshape(n, -1)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    m = np.arange(nlag)
    out = np.zeros(nlag)
    for c0 in range(0, x.shape[1], 64):
        F = np.fft.rfft(x[:, c0:c0 + 64], n=nfft, axis=0)
        ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=0)[:nlag]
        out += (ac / (n - m)[:, None]).sum(axis=1)
    return out / x.shape[1]


def msd_fft(r, nlag):
    """Multi-origin mean-square displacement of unwrapped positions r (n, natoms, 3)."""
    n, na, _ = r.shape
    m = np.arange(nlag)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    out = np.zeros(nlag)
    for a0 in range(0, na, 32):
        rr = r[:, a0:a0 + 32, :]
        nb = rr.shape[1]
        F = np.fft.rfft(rr, n=nfft, axis=0)
        ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=0)[:nlag].sum(axis=2)
        ac /= (n - m)[:, None]
        cs = np.vstack([np.zeros((1, nb)), np.cumsum((rr ** 2).sum(axis=2), axis=0)])
        s1 = (cs[n - m] + cs[n][None, :] - cs[m]) / (n - m)[:, None]
        out += (s1 - 2.0 * ac).sum(axis=1)
    return out / na


def trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def cumtrapz(y, x):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def fit_slope(t, y, lo, hi):
    s = (t >= lo) & (t <= hi)
    return np.polyfit(t[s], y[s], 1)[0]


# ---------------------------- read LAMMPS output ----------------------------
def read_settings(fname):
    """Read the 'variable <name> index <value>' lines of a LAMMPS input file."""
    out = {}
    with open(fname) as f:
        for line in f:
            m = re.match(r"\s*variable\s+(\w+)\s+index\s+(\S+)", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def read_trj(fname, ntrack):
    """Read a LAMMPS custom dump (id xu yu zu vx vy vz, sorted by id) -> (nframes, ntrack, 6)."""
    with open(fname) as f:
        lines = f.read().splitlines()
    per = 9 + ntrack
    nfr = len(lines) // per
    body = np.array(lines[:nfr * per]).reshape(nfr, per)[:, 9:]
    data = np.fromstring("\n".join(body.ravel()), sep=" ").reshape(nfr, ntrack, 7)
    return data[:, :, 1:]


def analyse_run(d):
    """Correlation functions of one run folder -> (result, None), or (None, reason) if unusable."""
    inp = os.path.join(d, "in.lj_transport")
    if not os.path.exists(inp):
        return None, "no in.lj_transport"
    s = read_settings(inp)
    prefix = s.get("out", "run")
    pfile = os.path.join(d, prefix + ".P.dat")
    tfile = os.path.join(d, prefix + ".trj")
    N, T, rho = int(s["N"]), float(s["temp0"]), float(s["rho"])
    dt, stride, ntrack = float(s["dt"]), int(s["stride"]), int(s["ntrack"])
    nexp = int(s["nprod"]) // stride                      # expected number of samples
    dts = dt * stride
    nlag = int(T_LAG / dts)
    L = (N / rho) ** (1.0 / 3.0)

    # Correlation functions stored by an earlier run of this script (corr.npz, provided in the
    # GitHub repository because the trajectories are too large to upload)
    cfile = os.path.join(d, CORR_FILE)
    if os.path.exists(cfile):
        c = np.load(cfile)
        if len(c["t"]) >= nlag:
            return dict(N=N, T=T, L=L, V=L ** 3, Tmean=float(c["Tmean"]), t=c["t"][:nlag],
                        msd=c["msd"][:nlag], vacf=c["vacf"][:nlag], sacf=c["sacf"][:nlag]), None

    for f in (pfile, tfile):
        if not os.path.exists(f):
            return None, f"missing {os.path.basename(f)} and {CORR_FILE} (LAMMPS not run in this folder?)"
    P = np.loadtxt(pfile, comments="#", ndmin=2)[:, 1:]   # pxx pyy pzz pxy pxz pyz T
    if len(P) < nexp:
        return None, f"run unfinished ({len(P)} of {nexp} pressure samples)"
    trj = read_trj(tfile, ntrack)
    if len(trj) < nexp:
        return None, f"run unfinished ({len(trj)} of {nexp} trajectory frames)"
    r, v = trj[:, :, 0:3], trj[:, :, 3:6]
    # five independent stress components: xy, xz, yz, (xx-yy)/2, (yy-zz)/2
    S = np.column_stack([P[:, 3], P[:, 4], P[:, 5],
                         0.5 * (P[:, 0] - P[:, 1]), 0.5 * (P[:, 1] - P[:, 2])])
    S = S - S.mean(axis=0)
    res = dict(N=N, T=T, L=L, V=L ** 3, Tmean=P[:, 6].mean(), t=np.arange(nlag) * dts,
               msd=msd_fft(r, nlag), vacf=acf_fft(v, nlag), sacf=acf_fft(S, nlag))
    np.savez(cfile, t=res["t"], msd=res["msd"], vacf=res["vacf"], sacf=res["sacf"], Tmean=res["Tmean"])
    return res, None


# -------------------------------- analysis ---------------------------------
def summarize(rs):
    t = rs[0]["t"]
    Dm, Dv, eta, curves = [], [], [], []
    for r in rs:
        Dm.append(fit_slope(t, r["msd"], *FIT_MSD) / 6.0)
        s = t <= T_VACF
        Dv.append(trapz(r["vacf"][s], t[s]))
        c = r["V"] / r["Tmean"] * cumtrapz(r["sacf"], t)
        curves.append(c)
        w = (t >= PLATEAU[0]) & (t <= PLATEAU[1])
        eta.append(c[w].mean())
    Dm, Dv, eta, curves = map(np.array, (Dm, Dv, eta, curves))
    n = len(rs)
    se = (lambda x: x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)
    return dict(N=rs[0]["N"], T=rs[0]["T"], L=rs[0]["L"], t=t, n=n,
                Tmean=np.mean([r["Tmean"] for r in rs]),
                Dm=Dm, Dv=Dv, eta=eta, curves=curves,
                msd=np.mean([r["msd"] for r in rs], axis=0),
                D_msd=Dm.mean(), D_msd_se=se(Dm), D_vacf=Dv.mean(), D_vacf_se=se(Dv),
                eta_mean=eta.mean(), eta_se=se(eta),
                curve=curves.mean(axis=0),
                curve_se=(curves.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else 0 * curves[0]))


def make_figure(colsA, colsB):
    font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 7}
    mpl.rc('font', **font)
    for axis in ['x', 'y']:
        mpl.rcParams[f'{axis}tick.major.size'] = 3
        mpl.rcParams[f'{axis}tick.minor.size'] = 3
        mpl.rcParams[f'{axis}tick.major.width'] = .5
        mpl.rcParams[f'{axis}tick.minor.width'] = .5
    mpl.rcParams['lines.linewidth'] = .75
    centimeter = 1/2.54

    fig, ax = plt.subplots(1, 3, dpi=750, figsize=(19*centimeter, 7*centimeter),
                           gridspec_kw={'wspace': 0.4})
    # Panel A: columns N L 1/L D_L D_L_se D_VACF D_VACF_se ...
    ax[0].errorbar(colsA[:, 2], colsA[:, 3], yerr=colsA[:, 4], fmt='s',
                   elinewidth=.75, linewidth=.75, markersize=4, color='k', capsize=1.5,
                   mfc='w', mew=.75, label='MSD')
    ax[0].errorbar(colsA[:, 2], colsA[:, 5], yerr=colsA[:, 6], fmt='o', mfc='k',
                   elinewidth=.75, linewidth=.75, markersize=4, color='k', capsize=1.5,
                   markeredgewidth=.75, label='VACF')
    par = np.polyfit(colsA[:, 2], colsA[:, 3], deg=1)
    x = np.linspace(0, 0.15)
    ax[0].plot(x, np.polyval(par, x), color='k', lw=.5, label='linear fit')
    ax[0].set_xlim([0, 0.12])
    ax[0].set_ylim([0.54, 0.65])
    ax[0].set_xlabel(r'$1/L^*$')
    ax[0].set_ylabel(r'$D_\mathrm{L}^*$')
    ax[0].legend(frameon=False)
    ax[0].text(-0.3, 1.0, 'A', fontweight='bold', transform=ax[0].transAxes)

    # Panels B and C: columns T T/Tc eta eta_se D_L D_L_se ...
    ax[1].errorbar(colsB[:, 0], colsB[:, 2], yerr=colsB[:, 3],
                   fmt='s', elinewidth=.75, linewidth=.75, markersize=4, color='k', capsize=1.5,
                   mfc='w', mew=.75)
    ax[2].errorbar(colsB[:, 0], colsB[:, 4], yerr=colsB[:, 5],
                   fmt='s', elinewidth=.75, linewidth=.75, markersize=4, color='k', capsize=1.5,
                   mfc='w', mew=.75)
    for k in (1, 2):
        ax[k].text(0.05, 0.9, r'$\rho^*=0.316$', transform=ax[k].transAxes)
        ax[k].set_xlabel(r'$T^*$')
    ax[1].set_ylabel(r'$\eta^*$')
    ax[2].set_ylabel(r'$D^*$')
    ax[1].text(-0.3, 1.0, 'B', fontweight='bold', transform=ax[1].transAxes)
    ax[2].text(-0.3, 1.0, 'C', fontweight='bold', transform=ax[2].transAxes)
    plt.savefig(os.path.join(HERE, FIG_NAME), dpi=750, bbox_inches='tight')
    print("Saved", FIG_NAME)


def main():
    # ---- collect the finished runs and group the replicas of each (N, T) ----
    groups = {}
    for name in sorted(os.listdir(HERE)):
        d = os.path.join(HERE, name)
        if not (os.path.isdir(d) and re.match(r"N\d+_T[\d.]+_r\d+$", name)):
            continue
        r, reason = analyse_run(d)
        if r is None:
            print(f"skipping {name}: {reason}")
            continue
        groups.setdefault((r["N"], round(r["T"], 6)), []).append(r)
    S = {key: summarize(rs) for key, rs in groups.items()}
    print(f"{sum(s['n'] for s in S.values())} runs, {len(S)} state points")

    # ---- Panels A and B: all box sizes at T = T_AB ----
    ab = sorted([s for (N, T), s in S.items() if abs(T - T_AB) < 1e-6], key=lambda s: s["N"])
    if len(ab) < 2:
        raise SystemExit("Need at least two box sizes at T_AB for Panel B.")
    big = ab[-1]
    print(f"\nT*={T_AB}   (measured <T*> = {big['Tmean']:.4f} for N={big['N']})")
    print(f"{'N':>6} {'runs':>5} {'L':>7} {'D_Einstein':>16} {'D_VACF':>16} {'eta_GK':>16}")
    for s in ab:
        print(f"{s['N']:6d} {s['n']:5d} {s['L']:7.3f} {s['D_msd']:8.4f}+-{s['D_msd_se']:.4f} "
              f"{s['D_vacf']:8.4f}+-{s['D_vacf_se']:.4f} {s['eta_mean']:8.4f}+-{s['eta_se']:.4f}")
    invL = np.array([1.0 / s["L"] for s in ab])
    DL = np.array([s["D_msd"] for s in ab])
    DLse = np.array([s["D_msd_se"] for s in ab])
    slope, Dinf = np.polyfit(invL, DL, 1)
    eta_GK = big["eta_mean"]
    slope_YH = -T_AB * XI / (6.0 * np.pi * eta_GK)
    eta_from_slope = -T_AB * XI / (6.0 * np.pi * slope) if slope < 0 else np.nan
    print(f"\nD_inf (intercept)            = {Dinf:.4f}")
    print(f"slope of D_L vs 1/L          = {slope:.4f}   (Yeh-Hummer with eta_GK: {slope_YH:.4f})")
    print(f"eta from the fitted slope    = {eta_from_slope:.4f}   (Green-Kubo: {eta_GK:.4f})")

    # ---- Panels B and C: all temperatures at N = N_C ----
    cs = sorted([s for (N, T), s in S.items() if N == N_C], key=lambda s: s["T"])
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    print(f"\n{'T*':>6} {'T/Tc':>6} {'eta':>8} {'D_L':>8} {'YH corr [%]':>12} {'D_inf':>8}")
    Tc_, DLc, DLc_se, Dic, Dic_se, etac, etac_se = [], [], [], [], [], [], []
    for s in cs:
        T = s["T"]
        corr = T * XI / (6.0 * np.pi * s["eta"] * s["L"])      # Yeh-Hummer correction, same run's eta
        Dinf_r = s["Dm"] + corr
        Tc_.append(T)
        DLc.append(s["D_msd"])
        DLc_se.append(se(s["Dm"]))
        Dic.append(Dinf_r.mean())
        Dic_se.append(se(Dinf_r))
        etac.append(s["eta_mean"])
        etac_se.append(se(s["eta"]))
        print(f"{T:6.2f} {T / TCRIT:6.3f} {s['eta_mean']:8.4f} {s['D_msd']:8.4f} "
              f"{100 * np.mean(corr / s['Dm']):12.1f} {Dinf_r.mean():8.4f}")
    Tc_ = np.array(Tc_)

    # ---- write the data files ----
    Ls = np.array([s["L"] for s in ab])
    colsA = np.column_stack([
        [s["N"] for s in ab], Ls, 1.0 / Ls,
        DL, DLse,
        [s["D_vacf"] for s in ab], [s["D_vacf_se"] for s in ab],
        [s["eta_mean"] for s in ab], [s["eta_se"] for s in ab],
        [s["n"] for s in ab]])
    headA = (f"Example 5.7, Panel A: finite-size effect of the self-diffusion coefficient\n"
             f"T* = {T_AB}, rho* fixed; D_L from the Einstein relation (MSD slope/6); se = standard error over replicas\n"
             f"Linear fit D_L = D_inf + slope/L : D_inf = {Dinf:.5f}, slope = {slope:.5f}\n"
             f"Yeh-Hummer slope -k_B T xi/(6 pi eta_GK) = {slope_YH:.5f}  (xi = {XI}, eta_GK = {eta_GK:.5f} from N = {big['N']})\n"
             f"eta from the fitted slope = {eta_from_slope:.5f}\n"
             f"columns: N  L  1/L  D_L  D_L_se  D_VACF  D_VACF_se  eta_GK  eta_GK_se  n_replicas")
    np.savetxt(os.path.join(HERE, OUT_PREFIX + "A_finite_size.dat"), colsA, header=headA,
               fmt=["%d", "%.5f", "%.6f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%d"])

    nrep = [s["n"] for s in cs]
    corr_pct = [100 * np.mean(s["T"] * XI / (6.0 * np.pi * s["eta"] * s["L"]) / s["Dm"]) for s in cs]
    colsB = np.column_stack([Tc_, Tc_ / TCRIT, etac, etac_se, DLc, DLc_se, Dic, Dic_se, corr_pct, nrep])
    headB = (f"Example 5.7, Panels B and C: D* and eta* along the critical isochore\n"
             f"N = {N_C}, rho* fixed; se = standard error over replicas; T/Tc uses T_c* = {TCRIT}\n"
             f"D_L = box value (Einstein); D_inf = D_L + k_B T xi/(6 pi eta L) with eta of the same run (xi = {XI})\n"
             f"columns: T*  T/Tc  eta  eta_se  D_L  D_L_se  D_inf  D_inf_se  YH_correction_percent  n_replicas")
    np.savetxt(os.path.join(HERE, OUT_PREFIX + "BC_vs_T.dat"), colsB, header=headB,
               fmt=["%.3f", "%.4f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.2f", "%d"])
    print(f"\nSaved {OUT_PREFIX}A_finite_size.dat and {OUT_PREFIX}BC_vs_T.dat")
    make_figure(colsA, colsB)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Run the calculation
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


if __name__ == "__main__":
    main()