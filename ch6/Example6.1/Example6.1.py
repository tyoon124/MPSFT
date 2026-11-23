"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 6 Interfacial Phenomena, Wetting, and Confinement
Example 6.1
Gibbs adsorption at the CO_2/n-hexane interface: This example produces the data of Fig. 6.1. For the
binary vapor-liquid interface, the squared-gradient theory of Sec. 6.2.1 is extended to two
components with the Peng-Robinson (PR) Equation of State and a binary interaction parameter k_ij =
0.11. The influence parameters are taken as constants and the cross influence parameter as the
geometric mean (i.e., beta_ij = 0), which reduces the two-component density path to a
one-dimensional problem: for each rho_C_6 between the two bulk values, rho_CO_2 follows from (mu_i -
mu_i^0)/sqrt(c_i) being the same for both components, where mu_i^0 is the equilibrium chemical
potential and c_i = 2kappa_i. The influence parameter of CO_2 is taken from Example 6.3 of Sec.
6.2.2, and that of n-hexane is fitted to its interfacial tension at 313.15 K.

No command-line arguments: edit the SETTINGS block below and run   python Example6.1.py
For each (T, p) on the coexistence curve the script
  (i)  solves the bulk VLE with the Peng-Robinson (PR) EoS,
  (ii) solves the squared-gradient-theory (SGT) interface (beta_ij = 0, reference-component path),
  (iii) evaluates gamma and the relative adsorption Gamma_CO2(hexane) from the density profiles, and
  (iv) compares Gamma with the Gibbs adsorption equation  Gamma = -(d gamma / d mu_CO2)_T
       evaluated by finite differences of gamma(p) along the coexistence curve.
Produces the data of Fig. 6.1.   Requires: numpy, scipy, matplotlib, CoolProp (reference gamma of
hexane only).
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
from scipy.optimize import fsolve, brentq
from scipy.integrate import quad, trapezoid
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
TEMPERATURES = [313.15, 333.15, 353.15]          # K
P_START, P_STEP = 0.5e6, 0.5e6                   # Pa, first pressure and step of the coarse pressure march
N_APPROACH   = 8                                 # extra points approaching the mixture critical pressure
KIJ          = 0.11                              # PR binary interaction parameter CO2-hexane
#              Tc [K]    pc [Pa]   acentric factor
COMP = {"CO2": (304.1282, 7.3773e6, 0.22394), "C6": (507.82, 3.034e6, 0.30)}
KAPPA_CO2    = 1.668e-20                         # J m^5 mol^-2, from Example 6.3 (pure CO2, fitted at T/Tc = 0.9)
T_FIT_HEX    = 313.15                            # K, n-hexane kappa is fitted to its gamma at this temperature
N_GRID, EPS  = 300, 2e-4                         # grid points along the SGT path, margin from the bulk endpoints
PROFILE_CASE = (313.15, 5.0e6)                   # (T, p) of the density profiles in Panel C
SAVE_FIG     = "figure6.1.pdf"
# ============================================================================

R, S2 = 8.314462618, np.sqrt(2.0)
NAMES = ["CO2", "C6"]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- pr_pure
- ab
- f_helmholtz: Helmholtz energy density, J/m3 (rho: mol/m3, array of 2)
- mus: chemical potentials by complex-step differentiation
- pressure
- ln_phi_Z
- vle: returns (x1, y1), liquid and vapor densities (mol/m3)
- kappa_hexane: fit of the n-hexane influence parameter (pure-component SGT)
- interface: SGT path (beta_ij = 0): rho_CO2 follows from rho_hexane through (mu_i-mu_i0)/sqrt(c_i)
= common value.
- gamma_excess: relative adsorption Gamma_CO2(hexane) from the profile, mol/m2
- pressure_grid: Coarse pressure march, bisection for the PR mixture critical pressure, geometric
approach to it.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def pr_pure(T, Tc, pc, w):
    a = 0.45724 * R**2 * Tc**2 / pc
    b = 0.07780 * R * Tc / pc
    m = 0.37464 + 1.54226 * w - 0.26992 * w**2
    return a * (1 + m * (1 - np.sqrt(T / Tc)))**2, b

def ab(T):
    a, b = zip(*[pr_pure(T, *COMP[n]) for n in NAMES])
    a, b = np.array(a), np.array(b)
    return np.sqrt(np.outer(a, a)) * (1 - KIJ * (1 - np.eye(2))), b

def f_helmholtz(rho, T):                         # Helmholtz energy density, J/m3 (rho: mol/m3, array of 2)
    aij, b = ab(T)
    rt = rho.sum()
    Bn = (rho * b).sum()
    An = rho @ aij @ rho
    return ((rho * R * T * (np.log(rho) - 1)).sum() - R * T * rt * np.log(1 - Bn)
            - An / (2 * S2 * Bn) * np.log((1 + (1 + S2) * Bn) / (1 + (1 - S2) * Bn)))

def mus(rho, T, h=1e-20):                        # chemical potentials by complex-step differentiation
    out = []
    for i in range(2):
        r = rho.astype(complex)
        r[i] += 1j * h * rho[i]
        out.append(np.imag(f_helmholtz(r, T)) / (h * rho[i]))
    return np.array(out)

def pressure(rho, T):
    return (rho * mus(rho, T)).sum() - f_helmholtz(rho.astype(float), T)

def ln_phi_Z(x, T, P, phase):
    aij, b = ab(T)
    a = x @ aij @ x
    bm = x @ b
    A = a * P / (R * T)**2
    B = bm * P / (R * T)
    rts = np.roots([1, -(1 - B), A - 3 * B**2 - 2 * B, -(A * B - B**2 - B**3)])
    rts = np.real(rts[abs(np.imag(rts)) < 1e-9])
    rts = rts[rts > B]
    Z = rts.max() if phase == "V" else rts.min()
    lp = (b / bm) * (Z - 1) - np.log(Z - B) - A / (2 * S2 * B) * (2 * (aij @ x) / a - b / bm) * np.log((Z + (1 + S2) * B) / (Z + (1 - S2) * B))
    return lp, Z

def vle(T, P, guess):                            # returns (x1, y1), liquid and vapor densities (mol/m3)
    def eq(v):
        x1, y1 = np.clip(v, 1e-8, 1 - 1e-8)
        x = np.array([x1, 1 - x1])
        y = np.array([y1, 1 - y1])
        lx, _ = ln_phi_Z(x, T, P, "L")
        ly, _ = ln_phi_Z(y, T, P, "V")
        return [np.log(x1) + lx[0] - np.log(y1) - ly[0], np.log(1 - x1) + lx[1] - np.log(1 - y1) - ly[1]]
    try:
        sol = fsolve(eq, guess, xtol=1e-13)
        x1, y1 = sol
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not (0 < x1 < 1 and 0 < y1 < 1) or abs(y1 - x1) < 1e-3 or max(abs(np.array(eq(sol)))) > 1e-9:
        return None
    x = np.array([x1, 1 - x1])
    y = np.array([y1, 1 - y1])
    _, Zl = ln_phi_Z(x, T, P, "L")
    _, Zv = ln_phi_Z(y, T, P, "V")
    return sol, x * P / (Zl * R * T), y * P / (Zv * R * T)

def kappa_hexane():                              # fit of the n-hexane influence parameter (pure-component SGT)
    import CoolProp.CoolProp as CP
    T = T_FIT_HEX
    a, b = pr_pure(T, *COMP["C6"])
    p_ = lambda r: R * T / (1 / r - b) - a / ((1 / r)**2 + 2 * b / r - b * b)
    fr = lambda r: r * (-R * T * np.log(1 - b * r) - a / (2 * S2 * b) * np.log((1 + (1 + S2) * b * r) / (1 + (1 - S2) * b * r)))
    f0 = lambda r: r * R * T * (np.log(r) - 1) + fr(r)
    mu_ = lambda r, h=1e-6: (f0(r * (1 + h)) - f0(r * (1 - h))) / (2 * r * h)
    g0 = [CP.PropsSI("Dmolar", "T", T, "Q", q, "n-Hexane") for q in (0, 1)]
    sol = fsolve(lambda x: [(p_(np.exp(x[0])) - p_(np.exp(x[1]))) / 1e5, (mu_(np.exp(x[0])) - mu_(np.exp(x[1]))) / (R * T)], np.log(g0), xtol=1e-13)
    rl, rg = np.exp(sol)
    mu0, p0 = mu_(rl), p_(rl)
    I = quad(lambda r: np.sqrt(max(f0(r) - r * mu0 + p0, 0)), rg, rl, epsrel=1e-8, limit=400)[0]
    return (CP.PropsSI("I", "T", T, "Q", 0, "n-Hexane") / (2 * I))**2

KAPPA = np.array([KAPPA_CO2, kappa_hexane()])
SC = np.sqrt(2 * KAPPA)                          # sqrt(c_i), c_i = 2 kappa_i  (F = int f0 + 1/2 sum c_ij grad rho_i grad rho_j)
print(f"kappa(CO2) = {KAPPA[0]:.3e}, kappa(n-hexane) = {KAPPA[1]:.3e} J m^5 mol^-2")

def interface(T, P, bulk, n=N_GRID, eps=EPS):
    """SGT path (beta_ij = 0): rho_CO2 follows from rho_hexane through (mu_i-mu_i0)/sqrt(c_i) = common value."""
    sol, rl, rg = bulk
    mu0, p0 = mus(rl, T), pressure(rl, T)
    dH = rl[1] - rg[1]
    rH = np.linspace(rg[1] + eps * dH, rl[1] - eps * dH, n)
    rC = np.empty(n)
    guess = rg[0] + (rH[0] - rg[1]) * (rl[0] - rg[0]) / dH
    for k in range(n):
        def f(x):
            m = mus(np.array([x[0] * 1e3, rH[k]]), T)
            return [(m[0] - mu0[0]) / SC[0] - (m[1] - mu0[1]) / SC[1]]
        rC[k] = fsolve(f, [guess / 1e3], xtol=1e-13)[0] * 1e3
        guess = rC[k]
    rho = np.column_stack([rC, rH])
    dw = np.array([(r * (mus(r, T) - mu0)).sum() - (pressure(r, T) - p0) for r in rho])
    phi = rho @ SC                                # d phi = sum sqrt(c_i) d rho_i
    gam = trapezoid(np.sqrt(2 * np.maximum(dw, 0)), phi)
    z = np.concatenate([[0], np.cumsum(np.diff(phi) / np.sqrt(2 * np.maximum(0.5 * (dw[1:] + dw[:-1]), 1e-30)))])
    z = -(z - np.interp(0.5 * (rl[1] + rg[1]), rH, z))   # z = 0 at the midpoint of the hexane step, liquid at z < 0
    return dict(gamma=abs(gam), rho=rho, z=z, mu0=mu0, rl=rl, rg=rg, x1=sol[0], y1=sol[1])

def gamma_excess(res):                           # relative adsorption Gamma_CO2(hexane) from the profile, mol/m2
    z, rho, rl, rg = res["z"], res["rho"], res["rl"], res["rg"]
    order = np.argsort(z)
    z, rho = z[order], rho[order]    # z increases from the liquid to the vapor side
    i0 = np.argmin(abs(z))                        # dividing surface z0 = 0 (result independent of z0)
    E = []
    for i in range(2):
        dev = np.where(z < 0, rho[:, i] - rl[i], rho[:, i] - rg[i])
        E.append(trapezoid(dev[:i0 + 1], z[:i0 + 1]) + trapezoid(dev[i0:], z[i0:]))
    return E[0] - (rl[0] - rg[0]) / (rl[1] - rg[1]) * E[1]

def pressure_grid(T):
    """Coarse pressure march, bisection for the PR mixture critical pressure, geometric approach to it."""
    guess, coarse, P = [0.03, 0.8], [], P_START
    while True:
        b = vle(T, P, guess)
        if b is None and not coarse:                  # try other starting guesses at the first pressure
            for g in ([0.1, 0.95], [0.05, 0.9], [0.02, 0.6]):
                b = vle(T, P, g)
                if b is not None:
                    break
        if b is None:
            break
        guess = b[0]
        coarse.append(P)
        P += P_STEP
    lo, hi, g = coarse[-1], P, guess
    for _ in range(18):                               # bisection (continuation guess from the last success)
        mid = 0.5 * (lo + hi)
        b = vle(T, mid, g)
        if b is None:
            hi = mid
        else:
            lo, g = mid, b[0]
    approach = lo - np.geomspace(0.5 * P_STEP, 2e3, N_APPROACH)
    return sorted([p for p in coarse if p < lo - 0.5 * P_STEP] + list(approach)), lo

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

results = {}
for T in TEMPERATURES:
    Ps, Pc_est = pressure_grid(T)
    print(f"\nT = {T:.2f} K: mixture critical pressure of the PR model ~ {Pc_est/1e6:.3f} MPa")
    rows, guess = [], [0.03, 0.8]
    for P in Ps:
        b = vle(T, P, guess)
        if b is None:
            continue
        guess = b[0]
        res = interface(T, P, b)
        rows.append(dict(P=P, mu=res["mu0"][0], gamma=res["gamma"], Gam=gamma_excess(res), x1=res["x1"], y1=res["y1"], res=res))
    # Gibbs adsorption equation from finite differences: Gamma_Gibbs = -d gamma / d mu_CO2 at constant T
    mu = np.array([r["mu"] for r in rows])
    gam = np.array([r["gamma"] for r in rows])
    dg = np.gradient(gam, mu, edge_order=2)
    for r, d in zip(rows, dg):
        r["Gam_G"] = -d
    results[T] = rows
    print("  p[MPa]   x_CO2   y_CO2   gamma[mN/m]   Gamma_profile[umol/m2]   Gamma_Gibbs[umol/m2]")
    for r in rows:
        print(f"  {r['P']/1e6:6.3f}  {r['x1']:6.3f}  {r['y1']:6.3f}  {1e3*r['gamma']:10.3f}  {1e6*r['Gam']:14.3f}  {1e6*r['Gam_G']:14.3f}")

# ------------------------------ figure ----------------------------------------
centimeter = 1 / 2.54
fig, ax = plt.subplots(1, 3, 
                       figsize=(19*centimeter, 8*centimeter),
                       dpi=750)

cols = ["#1f77b4", "#2ca02c", "#d62728"]

for T, c in zip(TEMPERATURES, cols):
    rows = results[T]
    P = np.array([r["P"] for r in rows]) / 1e6
    ax[0].plot(P, [1e3 * r["gamma"] for r in rows], "o", ms=3, color=c, mew=.75,
               label=f"{T:.0f} K")
    ax[1].plot(P, [1e6 * r["Gam"] for r in rows], "o", ms=3, color=c, mew=.75,
               label=f"{T:.0f} K (density profile)")
    ax[1].plot(P, [1e6 * r["Gam_G"] for r in rows], ">", ms=4, color=c, mew=.75, 
               label=f"{T:.0f} K (Gibbs equation)")

ax[0].set_xlabel("$p$ [MPa]")
ax[0].set_ylabel(r"$\gamma$ [mN m$^{-1}$]")
ax[0].set_xlim([0,11])
ax[0].set_ylim([0,16])
ax[0].legend(frameon=False, fontsize=6)
ax[1].set_xlabel("$p$ [MPa]")
ax[1].set_ylabel(r"$\Gamma_{\mathrm{CO_2(C_6)}}$ [$\mu$mol m$^{-2}$]")
ax[1].legend(frameon=False, fontsize=6)
Tp, Pp = PROFILE_CASE
row = min(results[Tp], key=lambda r: abs(r["P"] - Pp))
res = row["res"]
ax[1].set_ylim([0,12])
ax[1].set_yticks([0,3,6,9,12])
ax[1].set_xlim([0,11])
ax[2].plot(res["z"] * 1e9, res["rho"][:, 0] / 1e3, "b-", label=r"CO$_2$")
ax[2].plot(res["z"] * 1e9, res["rho"][:, 1] / 1e3, "k-", label=r"$n$-hexane")
ax[2].set_xlim(-3, 3)
ax[2].set_xlabel("$z$ [nm]")
ax[2].set_ylabel(r"$\rho_i$ [mol L$^{-1}$]")
ax[2].legend(frameon=False,fontsize=6)
ax[2].text(0.05,0.05,f"{Tp:.2f} K, {row['P']/1e6:.1f} MPa", fontsize=8, fontweight='bold',
           transform=ax[2].transAxes)
ax[2].set_xlim([-3,2])
ax[2].set_ylim([0,8])

for a, lab in zip(ax, "ABC"): 
    a.text(-0.16, 1, lab, transform=a.transAxes, fontweight="bold", fontsize=8)

plt.tight_layout()
plt.savefig(SAVE_FIG)
print("saved", SAVE_FIG)
