"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 7 Reactions in Supercritical Fluids
Example 7.3
Solvent clustering around an attractive solute in a near-critical Lennard-Jones fluid: In this
example, we perform MD simulations of a dilute attractive solute in an LJ solvent on the critical
isochore and quantify the clustering from the radial distribution function and Eq. 7.15. All
quantities are in reduced units. The results are shown in Fig. 7.4.

The simulations are run with LAMMPS (one input file in.solute per temperature, in the folders
Example7.3/T<value>/); this script only analyzes and plots. No command-line arguments: edit the
SETTINGS block below and run
    python Example7.3.py
For every temperature it reads the solvent-solute radial distribution function g_12(r) (30 blocks of
500 time units, rdf.dat) and evaluates
* the first-shell coordination number n_1 (solvent atoms within R_SHELL of a solute atom) and the
augmented density
rho_loc/rho of the first shell, rho_loc = n_1 / (4 pi R_SHELL^3 / 3); a common R_SHELL is used for
all temperatures,
because the position of the first minimum of g_12 is only known to the bin width (0.05 sigma) and
would otherwise
    make n_1 jump from one temperature to the next,
  * the excess number of solvent molecules around the solute,
        N_exc(R) = rho * 4 pi * int_0^R [g_12(r) - 1] r^2 dr ,
    which is related to the infinite-dilution partial molar volume by Kirkwood-Buff theory,
        V_2 = k_B T kappa_T - N_exc(inf)/rho ,
with the standard error from the blocks. In a closed (canonical) box, N_exc(R) has to return to zero
at large R and
the plateau is biased low by about the fraction of the box inside the sphere of radius R; the
finite-size
correction of Ganguly and van der Vegt (J. Chem. Theory Comput. 2013), applied to g_12 before the
integration,
        g_12 -> g_12 N f / (N f - Delta N(r)),   f = 1 - 4 pi r^3 / (3 V),   Delta N(r) = N_exc(r),
(N = number of solvent atoms, V = box volume) is printed and shown as open symbols. Near the
critical point the
corrected N_exc(R) does not reach a plateau within the box, which means that the box is too small to
converge
    N_exc(inf); the plateau value of the uncorrected curve is a lower estimate,
* a smooth fit of g_12(r) with the simplified radial distribution function of Matteoli and Mansoori
(J. Chem. Phys. 103,
4672 (1995), Eqs. 6a and 6b, pure-fluid form with seven parameters d, m, g(d), lambda, alpha, beta,
theta), fitted by
least squares to the block-averaged g_12 in FIT_RANGE, with the residuals weighted by r^2 as in the
integrand of N_exc, so
that the fitted N_exc follows the simulated one. It is the curve shown in Panel A (the noisy
simulated g_12 is not plotted) and gives the height of
the first peak and the position of the first minimum without the 0.05 sigma bin-width limit.
N_exc(R) of Panel B is
integrated from the fitted g_12 (noise-free), but only up to R_MAX: the fitted tail is not tied to
the canonical-box
value (the simulated g_12 lies slightly below one at large r), and the weight r^2 in the integral
turns this small
difference into a large deviation beyond about 7 sigma, where the fitted N_exc keeps growing. The
values in Panel C and
in the printout are integrated from the simulated g_12 (the fitted value at R_MAX is printed for
comparison),
* a check of the infinite-dilution limit: the fraction of the box filled by the spheres within which
the excess of
solvent around the solutes accumulates, phi = N_SOLUTE (4 pi/3) R_p^3 / V, where R_p is the radius
at which N_exc(R)
reaches 95 % of its maximum. For phi > PHI_MAX the solutes sit in each other's clusters, g_12 does
not return to one
    for the right reason, and N_exc is biased; the temperature is flagged.
Requires: numpy, scipy, matplotlib.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
from scipy.optimize import least_squares
import matplotlib as mpl
import matplotlib.pyplot as plt

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
TEMPERATURES = [1.30, 1.40, 1.50, 1.70, 2.00]   # reduced temperatures T* of the folders Example7.3/T<value>/
RHO          = 0.32                             # total number density (see in.solute)
N_TOTAL, N_SOLUTE = 4000, 3                     # atoms in the box (see in.solute)
TC           = 1.24                             # apparent critical temperature of the truncated LJ model (Example 6.2)
R_SHELL      = 1.8                              # sigma, common radius of the first shell (first minimum of g_12, 1.77-1.82)
R_MAX        = 7.0                              # sigma, upper limit of N_exc(R) in Panel C (the box is 23.2 sigma, g_12 is sampled to 10 sigma)
PHI_MAX      = 0.25                             # largest acceptable volume fraction of the excess spheres of all solutes
FIT_RANGE    = (0.9, 7.0)                       # sigma, range of g_12(r) used for the Matteoli-Mansoori fit (the core r < 0.9 is empty); upper limit = R_MAX
FIT_WEIGHT   = 2                                # residuals of the fit are multiplied by r**FIT_WEIGHT (2: weight of g_12 - 1 in N_exc; 0: plain fit of g_12)
PROFILE_T    = [1.30, 1.50, 2.00]               # temperatures shown in Panels A and B
SAVE_FIG     = "figure7.4.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
cm = 1/2.54
N_SOLV = N_TOTAL - N_SOLUTE
V_BOX = N_TOTAL / RHO
rho_v = N_SOLV / V_BOX                                     # solvent number density

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- read_rdf: Blocks 'step nrows' followed by rows (bin, r, g12, n12, g11, n11) -> arrays [n_blocks,
n_bins].
- mm_gr: Matteoli-Mansoori RDF, Eqs. 6a,b; p = (d, m, g(d), lambda, alpha, beta, theta); y = r/d.
- fit_mm: least-squares fit of mm_gr to g(r) in FIT_RANGE with residuals weighted by r**FIT_WEIGHT;
several starting val
- corrected: finite-size correction of g_12 (see docstring)
- nexc: cumulative excess number, g: [n_blocks, n_bins]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def read_rdf(fn):
    """Blocks 'step nrows' followed by rows (bin, r, g12, n12, g11, n11) -> arrays [n_blocks, n_bins]."""
    lines = [l for l in open(fn).read().split("\n")[3:] if l.strip()]
    blocks, i = [], 0
    while i < len(lines):
        nrow = int(lines[i].split()[1])
        blocks.append(np.array([l.split() for l in lines[i+1:i+1+nrow]], float))
        i += 1 + nrow
    b = np.array(blocks)
    return b[0, :, 1], b[:, :, 2], b[:, :, 3] * (N_TOTAL - N_SOLUTE) / N_SOLUTE   # coordination: solvent atoms around a solute

def mm_gr(r, p):
    """Matteoli-Mansoori RDF, Eqs. 6a,b; p = (d, m, g(d), lambda, alpha, beta, theta); y = r/d."""
    d, m, gd, lam, al, be, th = p
    r = np.asarray(r, float)
    y = r / d
    g = np.empty_like(r)
    a = y < 1
    g[a] = gd * np.exp(-th * (y[a] - 1)**2)
    y = y[~a]
    g[~a] = 1 + y**-m * (gd - 1 - lam) + (y - 1 + lam) / y * np.exp(-al * (y - 1)) * np.cos(be * (y - 1))
    return g

def fit_mm(r, g, d0, gpk):
    """least-squares fit of mm_gr to g(r) in FIT_RANGE with residuals weighted by r**FIT_WEIGHT; several starting values of m and alpha, the best fit is kept."""
    w = (r >= FIT_RANGE[0]) & (r <= FIT_RANGE[1])
    best = None
    lo = [0.9, 1, 1, -5, 0.01, 0.5, 1]
    hi = [1.5, 30, 10, 10, 10, 12, 2000]            # bounds on d, m, g(d), lambda, alpha, beta, theta
    for m0 in (1.5, 5, 12):
        for al0 in (0.3, 1, 2):
            f = least_squares(lambda p: (mm_gr(r[w], p) - g[w]) * r[w]**FIT_WEIGHT, [d0, m0, gpk, 0.5, al0, 6, 50], bounds=(lo, hi))
            if best is None or f.cost < best.cost:
                best = f
    return best.x, np.sqrt(np.mean((mm_gr(r[w], best.x) - g[w])**2))

def corrected(r, g):                                       # finite-size correction of g_12 (see docstring)
    f = 1 - 4 * np.pi * r**3 / (3 * V_BOX)
    dn = nexc(r, g)
    return g * N_SOLV * f / (N_SOLV * f - dn)

def nexc(r, g):                                            # cumulative excess number, g: [n_blocks, n_bins]
    dr = r[1] - r[0]
    return rho_v * np.cumsum(4 * np.pi * r**2 * (g - 1.0), axis=-1) * dr

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

res = {}
for T in TEMPERATURES:
    fn = os.path.join(HERE, "Example7.3", f"T{T:.2f}", "rdf.dat")    # book layout (Examples/ch7/Example7.3/T<value>/)
    if not os.path.exists(fn):
        fn = os.path.join(HERE, f"T{T:.2f}", "rdf.dat")             # GitHub layout (data next to the script)
    if not os.path.exists(fn):
        print("missing", fn)
        continue
    r, g, n = read_rdf(fn)
    gm = g.mean(0)
    i1 = np.argmax(gm)
    imin = i1 + np.argmin(gm[i1:i1 + 60])   # first minimum after the first peak (for information)
    pf, rms = fit_mm(r, gm, r[i1], gm[i1])
    gfit = mm_gr(r, pf)
    rf = np.linspace(pf[0], 4.0, 3001)
    gf = mm_gr(rf, pf)
    jp = np.argmax(gf)
    jm = jp + np.argmin(gf[jp:jp + 1000])   # peak and first minimum of the fit
    ne = nexc(r, g)
    nec = nexc(r, corrected(r, g))
    iR = np.argmin(abs(r - R_MAX))
    n1 = np.array([np.interp(R_SHELL, r, nb_) for nb_ in n])
    rl = n1 / (4 * np.pi * R_SHELL**3 / 3) / rho_v
    nb = len(g)
    se = lambda x: x.std(0, ddof=1) / np.sqrt(nb)
    res[T] = dict(r=r, g=gm, gfit=gfit, pfit=pf, nefit=nexc(r, gfit), ne=ne.mean(0), rmin=r[imin], gpeak=gm[i1], n1=(n1.mean(), se(n1)), aug=(rl.mean(), se(rl)),
                  neR=(ne[:, iR].mean(), se(ne[:, iR])), neRc=(nec[:, iR].mean(), se(nec[:, iR])))
    me = ne.mean(0)
    Rp = r[np.argmax(me >= 0.95 * me[r <= R_MAX].max())]
    phi = N_SOLUTE * 4 / 3 * np.pi * Rp**3 / V_BOX
    res[T]["phi"] = phi
    x = res[T]
    print(f"T* = {T:.2f}: Matteoli-Mansoori fit, rms deviation {rms:.3f}; d = {pf[0]:.3f}, m = {pf[1]:.2f}, g(d) = {pf[2]:.3f}, lambda = {pf[3]:.3f}, "
          f"alpha = {pf[4]:.2f}, beta = {pf[5]:.2f}, theta = {pf[6]:.1f}; first peak g = {gf[jp]:.2f} at r = {rf[jp]:.3f}, first minimum at r = {rf[jm]:.3f}; "
          f"N_exc({R_MAX:.0f}) from the fit = {res[T]['nefit'][iR]:.2f}")
    print(f"T* = {T:.2f} (T/Tc = {T/TC:.2f}): g_peak = {x['gpeak']:.2f}, r_min = {x['rmin']:.2f}, n1(R={R_SHELL}) = {x['n1'][0]:.2f} +- {x['n1'][1]:.2f}, "
          f"rho_loc/rho = {x['aug'][0]:.2f} +- {x['aug'][1]:.2f}, N_exc({R_MAX:.0f}) = {x['neR'][0]:.2f} +- {x['neR'][1]:.2f} "
          f"(finite-size corrected: {x['neRc'][0]:.2f} +- {x['neRc'][1]:.2f}); "
          f"excess sphere R_p = {Rp:.1f} sigma fills {phi:.2f} of the box" + ("" if phi <= PHI_MAX else "   <-- caution: excess spheres fill a sizeable part of the box"))

fig, ax = plt.subplots(1, 3, figsize=(19*cm, 6.5*cm), gridspec_kw={'wspace':0.38})
cols = {1.30: "tab:red", 1.50: "tab:orange", 1.70: "tab:blue", 2.00: "tab:green"}
for T in PROFILE_T:
    if T in res:
        c = cols.get(T, None)
        ax[0].plot(res[T]["r"], res[T]["gfit"], color=c, label=rf"$T^*$ = {T:.0f}" if T == int(T) else rf"$T^*$ = {T:.2f}")
        ax[1].plot(res[T]["r"], res[T]["nefit"], color=c)
#ax[0].axhline(1, color="k", lw=.5, ls="-"); 
ax[1].axhline(0, color="k", lw=.5, ls="-")
ax[0].set_xlim(0, 4)
ax[1].set_xlim(0, R_MAX)
ax[0].set_ylim([0,3])
ax[0].set_xlabel(r"$r/\sigma$")
ax[0].set_ylabel(r"$g_{12}(r)$")
ax[0].legend(frameon=False, fontsize=6)
ax[1].set_xlabel(r"$R/\sigma$")
ax[1].set_ylabel(r"$N_\mathrm{exc}(R)$")
ax[1].set_xlim([0,6])
ax[1].set_ylim([-2,14])
Ts = [T for T in TEMPERATURES if T in res]
tt = np.array(Ts) / TC
ax[2].errorbar(tt, [res[T]["aug"][0] for T in Ts], [res[T]["aug"][1] for T in Ts], fmt="ko", ms=3, mew=.75, capsize=2, elinewidth=.75, label=r"$\rho_\mathrm{loc}/\rho$ (first shell)")
ax[2].set_xlabel(r"$T/T_\mathrm{crit}$")
ax[2].set_ylabel(r"$\rho_\mathrm{loc}/\rho$")
a2 = ax[2].twinx()
a2.errorbar(tt, [res[T]["neR"][0] for T in Ts], [res[T]["neR"][1] for T in Ts], fmt="rs", ms=3, mew=.75, capsize=2, elinewidth=.75)
a2.errorbar(tt, [res[T]["neRc"][0] for T in Ts], [res[T]["neRc"][1] for T in Ts], fmt="rs", mfc="none", ms=3, mew=.75, capsize=2, elinewidth=.75)
a2.set_ylabel(rf"$N_\mathrm{{exc}}({R_MAX:.0f}\sigma)$", color="r")
a2.tick_params(axis="y", colors="r")
for a, lab in zip(ax, "ABC"):
    a.text(-0.26, 0.975, lab, transform=a.transAxes, fontweight="bold", fontsize=7)
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
