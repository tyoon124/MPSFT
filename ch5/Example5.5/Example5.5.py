#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 5 Molecular Simulation Analyses
Example 5.5
Response functions of the Lennard-Jones fluid near the critical point: fluctuation versus finite
difference: This example calculates C_p, kappa_T, and alpha_p of the LJ fluid along an isobar
slightly above the critical pressure with the fluctuation formulas and with finite differences, and
compares both with the reference Equation of State of Thol et al. (J. Phys. Chem. Ref. Data 45,
023101, 2016), which is evaluated with the teqp package (pip install teqp).

fluctuation route versus finite-difference route, for several system sizes. (reduced LJ units, k_B =
1)

Reads  <DATA_DIR>/<folder>/run_series.dat   (columns: step V H pe P T, H = E + p*V) for every data
set listed in DATASETS, and
  - computes C_p, kappa_T, alpha_p from fluctuations (delete-one-block jackknife errors),
  - computes C_p and alpha_p by central finite differences between neighboring
    temperatures (at the temperatures given in FD_CENTERS),
  - prints a table with sampling diagnostics, writes a CSV, and draws the 3-panel figure.

Edit the SETTINGS block and run:  python Example5.5.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import glob
import os
import re

import numpy as np
import teqp
from scipy.optimize import brentq
import matplotlib as mpl
import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Settings
Edit the parameters below to change the calculation.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ============================== SETTINGS ==============================
font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 7}
mpl.rc('font', **font)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

for axis in ['x', 'y']:
    mpl.rcParams[f'{axis}tick.major.size'] = 3
    mpl.rcParams[f'{axis}tick.minor.size'] = 3
    mpl.rcParams[f'{axis}tick.major.width'] = .5
    mpl.rcParams[f'{axis}tick.minor.width'] = .5
    mpl.rcParams['lines.linewidth'] = .75
    
DATA_DIR = ""        # folder containing the run folders ("" = folder of this script)
P_STAR   = 0.15      # target pressure p*
DT_MD    = 0.005     # MD time step (reduced units)
DT_FD    = 0.02      # temperature step of the central finite differences

# One entry per system size.
#   glob    : pattern of the run folders;  regex : extracts T* from the folder name
#   fd_at   : temperatures at which the finite-difference route is evaluated
#             (needs the folders at T* +- DT_FD)
#   blocks  : number of jackknife blocks
DATASETS = [
    dict(label="N = 500",  N=500,  glob="N500_T*_p0.15", regex=r"^N500_T(\d+\.\d+)_p0.15$",
         fd_at=[1.36], blocks=20, color="C0", marker="o"),
    dict(label="N = 2000", N=2000, glob="N2000_T*_p0.15", regex=r"^N2000_T(\d+\.\d+)_p0.15$",
         fd_at=[1.38, 1.40], blocks=50, color="C3", marker="s"),
]

DISCARD_FRAC   = 0.0   # fraction of each series discarded as extra equilibration
MIN_RUN_OVER_TAU = 200 # warn if run length / volume correlation time is smaller than this

# Reference curve from the EoS of Thol et al. (teqp)
REF_T_MIN    = 1.28    # temperature range of the reference curve
REF_T_MAX    = 1.50
REF_N_POINTS = 301     # use a fine grid: the curves have sharp maxima
# ======================================================================

KB = 1.0
model = teqp.build_LJ126_TholJPCRD2016()
Z_PURE = np.array([1.0])             # pure fluid
CV_IDEAL = 1.5                       # monatomic ideal gas, per atom (k_B = 1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define functions
- base_dir
- find_runs: Return {T*: folder} for a data set.
- load_series
- block_sums
- fluct_estimators: C_p, kappa_T, alpha_p from summed block statistics (fluctuation formulas).
- jackknife
- mean_err
- tau_int: Integrated autocorrelation time in samples (Sokal window).
- analyse
- report
- save_csv
- make_figure
- ref_pressure, ref_dpdrho_T, ref_dpdT_rho: p, (dp/drho)_T and (dp/dT)_rho of the reference EoS
  (teqp: Ar_xy = tau^x delta^y d^(x+y) alpha_r / dtau^x ddelta^y, tau = Tr/T)
- ref_density: Density on the isobar (single root for p* above the critical pressure)
- ref_response: rho*, C_p/(N k_B), kappa_T and alpha_p of the reference EoS
- reference_curve: Columns T*, C_p/(N k_B), kappa_T, alpha_p, rho* along the isobar
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# ------------------------------------------------------------------ reference EoS
def ref_pressure(T, rho):
    return rho * T * (1.0 + model.get_Ar01(T, rho, Z_PURE))


def ref_dpdrho_T(T, rho):
    return T * (1.0 + 2.0 * model.get_Ar01(T, rho, Z_PURE) + model.get_Ar02(T, rho, Z_PURE))


def ref_dpdT_rho(T, rho):
    return rho * (1.0 + model.get_Ar01(T, rho, Z_PURE) - model.get_Ar11(T, rho, Z_PURE))


def ref_density(T, p):
    f = lambda r: ref_pressure(T, r) - p
    return brentq(f, 1e-6, 1.2, xtol=1e-14, rtol=1e-13)


def ref_response(T, p):
    rho = ref_density(T, p)
    dpr = ref_dpdrho_T(T, rho)
    dpT = ref_dpdT_rho(T, rho)
    cv = CV_IDEAL - model.get_Ar20(T, rho, Z_PURE)
    cp = cv + T * dpT**2 / (rho**2 * dpr)
    kappa = 1.0 / (rho * dpr)
    alpha = dpT / (rho * dpr)
    return rho, cp, kappa, alpha


def reference_curve():
    rows = []
    for T in np.linspace(REF_T_MIN, REF_T_MAX, REF_N_POINTS):
        rho, cp, kappa, alpha = ref_response(T, P_STAR)
        rows.append([T, cp, kappa, alpha, rho])
    return np.array(rows)


# ------------------------------------------------------------------ I/O
def base_dir():
    return DATA_DIR if DATA_DIR else os.path.dirname(os.path.abspath(__file__))


def find_runs(ds):
    """Return {T*: folder} for a data set."""
    runs = {}
    for path in glob.glob(os.path.join(base_dir(), ds["glob"])):
        name = os.path.basename(path)
        m = re.match(ds["regex"], name)
        if m and os.path.exists(os.path.join(path, "run_series.dat")):
            runs[round(float(m.group(1)), 4)] = path
    return dict(sorted(runs.items()))


def load_series(folder):
    d = np.loadtxt(os.path.join(folder, "run_series.dat"))
    d = d[int(DISCARD_FRAC * len(d)):]
    return dict(step=d[:, 0], V=d[:, 1], H=d[:, 2], P=d[:, 4], T=d[:, 5])


# --------------------------------------------- block sums and jackknife
def block_sums(s, nb):
    n = (len(s["V"]) // nb) * nb
    V = s["V"][:n].reshape(nb, -1)
    H = s["H"][:n].reshape(nb, -1)
    return dict(n=np.full(nb, V.shape[1], dtype=float), V=V.sum(1), H=H.sum(1),
                VV=(V * V).sum(1), HH=(H * H).sum(1), VH=(V * H).sum(1))


def fluct_estimators(S, T):
    """C_p, kappa_T, alpha_p from summed block statistics (fluctuation formulas)."""
    n = S["n"]
    mV, mH = S["V"] / n, S["H"] / n
    varV = S["VV"] / n - mV**2
    varH = S["HH"] / n - mH**2
    cov = S["VH"] / n - mV * mH
    return varH / (KB * T**2), varV / (mV * KB * T), cov / (mV * KB * T**2)


def jackknife(S, T):
    nb = len(S["n"])
    tot = {k: v.sum(keepdims=True) for k, v in S.items()}
    theta = np.array(fluct_estimators(tot, T)).ravel()
    loo = np.array([np.array(fluct_estimators(
        {k: (v.sum() - v[i]) * np.ones(1) for k, v in S.items()}, T)).ravel() for i in range(nb)])
    err = np.sqrt((nb - 1) / nb * ((loo - loo.mean(0)) ** 2).sum(0))
    return theta, err


def mean_err(x, nb):
    n = (len(x) // nb) * nb
    b = x[:n].reshape(nb, -1).mean(1)
    return b.mean(), b.std(ddof=1) / np.sqrt(nb)


def tau_int(x, c=5.0):
    """Integrated autocorrelation time in samples (Sokal window)."""
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 1.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * acf[1:m + 1].sum()
        if m >= c * tau:
            break
    return tau


# --------------------------------------------------------- analysis
def analyse(ds):
    runs = find_runs(ds)
    if not runs:
        print(f"WARNING: no runs found for {ds['label']} (glob {ds['glob']})")
        return None
    N, nb = ds["N"], ds["blocks"]
    res = {}
    for T, folder in runs.items():
        s = load_series(folder)
        dts = (s["step"][1] - s["step"][0]) * DT_MD
        tauV = tau_int(s["V"]) * dts
        Ttot = (s["step"][-1] - s["step"][0]) * DT_MD
        (Cp, kT, ap), (eCp, ekT, eap) = jackknife(block_sums(s, nb), T)
        mV, eV = mean_err(s["V"], nb)
        mH, eH = mean_err(s["H"], nb)
        res[T] = dict(Cp=Cp / N, eCp=eCp / N, kT=kT, ekT=ekT, ap=ap, eap=eap,
                      V=mV, eV=eV, H=mH, eH=eH, pmean=s["P"].mean(), Tmean=s["T"].mean(),
                      tauV=tauV, ratio=Ttot / tauV, nsamp=len(s["V"]))
    # finite differences (per atom for C_p)
    fd = {}
    for Tc in ds["fd_at"]:
        Tc = round(Tc, 4)
        lo, hi = round(Tc - DT_FD, 4), round(Tc + DT_FD, 4)
        if not all(t in res for t in (lo, Tc, hi)):
            print(f"  [{ds['label']}] finite difference at T*={Tc}: need runs at {lo}, {Tc}, {hi}")
            continue
        a, c, b = res[lo], res[Tc], res[hi]
        Cp = (b["H"] - a["H"]) / (2 * DT_FD) / N
        eCp = np.hypot(b["eH"], a["eH"]) / (2 * DT_FD) / N
        ap = (b["V"] - a["V"]) / (2 * DT_FD * c["V"])
        eap = abs(ap) * np.sqrt((np.hypot(b["eV"], a["eV"]) / (b["V"] - a["V"])) ** 2
                                + (c["eV"] / c["V"]) ** 2)
        fd[Tc] = dict(Cp=Cp, eCp=eCp, ap=ap, eap=eap)
    return dict(ds=ds, res=res, fd=fd)


def report(out):
    ds, res, fd = out["ds"], out["res"], out["fd"]
    print("=" * 92)
    print(f"{ds['label']}   (p* = {P_STAR}, {ds['blocks']} jackknife blocks)")
    print("=" * 92)
    print(f"{'T*':>6} {'<p>':>7} {'<V>':>8} {'tau_V':>7} {'run/tau':>8} "
          f"{'Cp/(N kB)':>15} {'kappa_T':>15} {'alpha_p':>15}")
    for T, r in res.items():
        flag = "  <-- short run" if r["ratio"] < MIN_RUN_OVER_TAU else ""
        print(f"{T:6.3f} {r['pmean']:7.4f} {r['V']:8.1f} {r['tauV']:7.1f} {r['ratio']:8.0f} "
              f"{r['Cp']:7.2f}+-{r['eCp']:5.2f} {r['kT']:7.2f}+-{r['ekT']:5.2f} "
              f"{r['ap']:7.3f}+-{r['eap']:5.3f}{flag}")
    for Tc, f in fd.items():
        r = res[Tc]
        print(f"  finite difference at T*={Tc:.2f} (dT*={DT_FD}): "
              f"Cp/(N kB) = {f['Cp']:.2f}+-{f['eCp']:.2f} (fluct. {r['Cp']:.2f}+-{r['eCp']:.2f}),  "
              f"alpha_p = {f['ap']:.3f}+-{f['eap']:.3f} (fluct. {r['ap']:.3f}+-{r['eap']:.3f})")


def save_csv(outs):
    with open(f"Example5.5_summary.csv", "w") as fh:
        fh.write("# label N T Cp/(N kB) err kappa_T err alpha_p err run/tau_V\n")
        for o in outs:
            for T, r in o["res"].items():
                fh.write(f"{o['ds']['label'].replace(' ', '')} {o['ds']['N']} {T:.4f} "
                         f"{r['Cp']:.5f} {r['eCp']:.5f} {r['kT']:.5f} {r['ekT']:.5f} "
                         f"{r['ap']:.5f} {r['eap']:.5f} {r['ratio']:.0f}\n")
            for Tc, f in o["fd"].items():
                fh.write(f"# FD {o['ds']['label'].replace(' ', '')} T={Tc:.4f} "
                         f"Cp/(N kB)={f['Cp']:.5f}+-{f['eCp']:.5f} "
                         f"alpha_p={f['ap']:.5f}+-{f['eap']:.5f}\n")


# ------------------------------------------------------------ figure
def make_figure(outs):
    centimeter = 1 / 2.54
    plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.6,
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, 3, figsize=(19*centimeter, 10*centimeter), dpi=750)
    panels = [("Cp", "eCp", r"$C_\mathrm{p}/(Nk_\mathrm{B})$"),
              ("kT", "ekT", r"$\kappa_\mathrm{T}\ (\sigma^3/\epsilon)$"),
              ("ap", "eap", r"$\alpha_\mathrm{p}\ (k_\mathrm{B}/\epsilon)$")]
    
    ref = reference_curve()
    for k, (ax, (key, ekey, ylabel)) in enumerate(zip(axes, panels)):
        for o in outs:
            ds, res, fd = o["ds"], o["res"], o["fd"]
            T = np.array(list(res))
            y = np.array([res[t][key] for t in res])
            e = np.array([res[t][ekey] for t in res])
            ax.errorbar(T, y, e, fmt=ds["marker"], ms=4, color='b', capsize=1.5,
                        mew=.75,
                        lw=0.5, mfc="w", label=f"fluctuation, {ds['label']}",elinewidth=.5)
            if key in ("Cp", "ap") and fd:
                Tc = np.array(list(fd))
                ax.errorbar(Tc, [fd[t][key] for t in fd], [fd[t][ekey] for t in fd],
                            fmt="3", ms=6, mew=.75, color='r', capsize=1.5, lw=0.5,                            
                            elinewidth=.5, label=f"finite difference, {ds['label']}")
        if ref is not None:
            ax.plot(ref[:, 0], ref[:, 1 + k], "-", color="0.3", lw=1.0, label="Thol EoS (LJ)")
        ax.set_xlabel(r"$T^*$")
        ax.set_ylabel(ylabel)
        ax.text(-0.15, 1, "ABC"[k], fontweight="bold", va="top",
                transform=ax.transAxes,fontsize=7)
    # one shared legend below the panels (handles from the panel that has all entries)
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, frameon=False, loc="lower center", ncol=len(l))
    fig.tight_layout(pad=0.5, w_pad=1.0, rect=(0, 0.09, 1, 1))
    plt.savefig('../figure5.5.pdf',dpi=750,bbox_inches='tight')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


outputs = []
for ds in DATASETS:
    o = analyse(ds)
    if o:
        report(o)
        outputs.append(o)
if not outputs:
    raise SystemExit("No data found. Check DATA_DIR and DATASETS.")
save_csv(outputs)
make_figure(outputs)