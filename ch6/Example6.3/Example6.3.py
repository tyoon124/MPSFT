"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 6 Interfacial Phenomena, Wetting, and Confinement
Example 6.3
Squared-gradient theory of the CO_2 vapor-liquid interface: This example evaluates Eqs. 6.12 and
6.13 for CO_2 with the Peng-Robinson (PR) Equation of State (Eq. 2.7) and produces the data of Fig.
6.3. The critical constants and acentric factor are T_crit = 304.13 K, p_crit = 7.377 MPa, and omega
= 0.2239.

Peng-Robinson (PR) EoS + constant influence parameter, no command-line arguments: edit the SETTINGS
block below and run   python Example6.3.py
Produces the data of Fig. 6.3 (density profiles, interfacial thickness, gamma scaling).
Requires: numpy, scipy, matplotlib, CoolProp (CoolProp is used only for the reference gamma(T)).
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
from scipy.integrate import quad
import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------------------------ plotting style (same as Example 6.1) ----------
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
TC, PC, OMEGA = 304.1282, 7.3773e6, 0.22394   # CO2: K, Pa, acentric factor
MW            = 44.0098e-3                    # kg/mol
TR_PROFILES   = [0.80, 0.90, 0.95]            # T/Tc for the density profiles (Panel A)
TR_FIT        = 0.90                          # kappa is fitted to the reference gamma at this T/Tc
TR_SCAN       = np.concatenate([np.linspace(0.60, 0.90, 7), 1 - np.logspace(-1.3, -3.0, 16)])  # Panels B, C
T_WINDOW      = (1e-3, 1e-2)                  # reduced-temperature window t = 1 - T/Tc for exponent fits
USE_COOLPROP_REFERENCE = True                 # reference gamma(T): CoolProp surface-tension correlation
Z_RANGE       = 7.5                           # nm, half-width of the z axis in Panel A (profiles are padded with the bulk values)
SAVE_FIG      = "figure6.3.pdf"
# ============================================================================

R = 8.314462618
A0 = 0.45724 * R**2 * TC**2 / PC
B  = 0.07780 * R * TC / PC
M  = 0.37464 + 1.54226 * OMEGA - 0.26992 * OMEGA**2
S2 = np.sqrt(2.0)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- alpha
- pressure: PR EoS, rho in mol/m3
- a_res: residual Helmholtz energy per mole
- da_res: d a_res / d rho
- mu: chemical potential, J/mol
- dp_drho: analytic (dp/drho)_T of the PR EoS
- coexistence: Maxwell construction: bracket the coexistence pressure between the spinodal
pressures.
- delta_omega: grand-potential density relative to coexistence, J/m3
- sgt
- width_10_90: distance between 10 % and 90 % of the density step
- profile
- gamma_reference
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def alpha(T):
    return (1 + M * (1 - np.sqrt(T / TC)))**2

def pressure(rho, T):                       # PR EoS, rho in mol/m3
    v = 1.0 / rho
    return R * T / (v - B) - A0 * alpha(T) / (v * v + 2 * B * v - B * B)

def a_res(rho, T):                          # residual Helmholtz energy per mole
    return (-R * T * np.log(1 - B * rho)
            - A0 * alpha(T) / (2 * S2 * B) * np.log((1 + (1 + S2) * B * rho) / (1 + (1 - S2) * B * rho)))

def da_res(rho, T):                         # d a_res / d rho
    return (R * T * B / (1 - B * rho)
            - A0 * alpha(T) / (2 * S2 * B) * ((1 + S2) * B / (1 + (1 + S2) * B * rho)
                                              - (1 - S2) * B / (1 + (1 - S2) * B * rho)))

def mu(rho, T):                             # chemical potential, J/mol
    return R * T * np.log(rho) + a_res(rho, T) + rho * da_res(rho, T)

def dp_drho(rho, T):                        # analytic (dp/drho)_T of the PR EoS
    q = 1 + 2 * B * rho - B * B * rho**2
    return R * T / (1 - B * rho)**2 - A0 * alpha(T) * (2 * rho * q - rho**2 * (2 * B - 2 * B * B * rho)) / q**2

def coexistence(T, guess=None):
    """Maxwell construction: bracket the coexistence pressure between the spinodal pressures."""
    from scipy.optimize import brentq
    grid = np.linspace(1e-3, 0.999 / B, 20000)
    d = dp_drho(grid, T)
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    ra = brentq(lambda r: dp_drho(r, T), grid[idx[0]], grid[idx[0] + 1], xtol=1e-12)   # vapor spinodal
    rb = brentq(lambda r: dp_drho(r, T), grid[idx[1]], grid[idx[1] + 1], xtol=1e-12)   # liquid spinodal
    plo, phi = max(pressure(rb, T), 1.0), pressure(ra, T)
    def roots(p):
        rg = brentq(lambda r: pressure(r, T) - p, 1e-3, ra, xtol=1e-12)
        rl = brentq(lambda r: pressure(r, T) - p, rb, 0.999 / B, xtol=1e-12)
        return rl, rg
    def f(p):
        rl, rg = roots(p)
        return mu(rl, T) - mu(rg, T)
    p = brentq(f, plo + 1e-6 * (phi - plo), phi - 1e-6 * (phi - plo), xtol=1e-6, rtol=1e-14)
    return np.array(roots(p))

def delta_omega(rho, T, mu0, p0):           # grand-potential density relative to coexistence, J/m3
    return rho * (mu(rho, T) - mu0) - (pressure(rho, T) - p0)

def sgt(T, kappa):
    rl, rg = coexistence(T)
    mu0, p0 = mu(rl, T), pressure(rl, T)
    dw = lambda r: delta_omega(r, T, mu0, p0)
    # gamma = 2 sqrt(kappa) * int sqrt(Delta omega) d rho
    I = quad(lambda r: np.sqrt(max(dw(r), 0.0)), rg, rl, epsabs=0, epsrel=1e-9, limit=400)[0]
    return rl, rg, dw, 2 * np.sqrt(kappa) * I

def width_10_90(rl, rg, dw, kappa):         # distance between 10 % and 90 % of the density step
    r1, r2 = rg + 0.1 * (rl - rg), rg + 0.9 * (rl - rg)
    return quad(lambda r: np.sqrt(kappa / dw(r)), r1, r2, limit=400)[0]

def profile(rl, rg, dw, kappa, n=500, eps=1e-4):
    rmid = 0.5 * (rl + rg)
    rho = np.linspace(rg + eps * (rl - rg), rl - eps * (rl - rg), n)
    z = np.array([-quad(lambda r: np.sqrt(kappa / dw(r)), rmid, r, limit=400)[0] for r in rho])
    return z, rho                             # liquid at z < 0, vapor at z > 0

def gamma_reference(T):
    import CoolProp.CoolProp as CP
    return CP.PropsSI('I', 'T', T, 'Q', 0, 'CO2')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ------------------------------ calibration ---------------------------------
T_fit = TR_FIT * TC
_, _, _, g1 = sgt(T_fit, 1.0)
KAPPA = (gamma_reference(T_fit) / g1)**2
print(f"Influence parameter kappa = {KAPPA:.4e} J m^5 mol^-2 (fitted at T/Tc = {TR_FIT})")

# ------------------------------ scan -----------------------------------------
rows = []
for tr in TR_SCAN:
    T = tr * TC
    rl, rg, dw, g = sgt(T, KAPPA)
    rows.append((tr, 1 - tr, g, gamma_reference(T), width_10_90(rl, rg, dw, KAPPA), (rl - rg) * MW))
rows = np.array(rows)
print("\n T/Tc      t        gamma_SGT   gamma_ref   w_10-90   (rhoL-rhoG)")
print("                      [mN/m]      [mN/m]       [nm]     [kg/m3]")
for r in rows:
    print(f"{r[0]:6.4f}  {r[1]:8.2e}  {1e3*r[2]:9.4f}  {1e3*r[3]:9.4f}  {1e9*r[4]:7.3f}  {r[5]:8.2f}")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Define functions
- slope
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def slope(x, y, lo, hi):
    m = (x >= lo) & (x <= hi)
    return np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
t_, g_, gr_, w_ = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
print(f"\nExponent of gamma in t in the window {T_WINDOW}:  SGT-PR = {slope(t_, g_, *T_WINDOW):.3f},"
      f"  reference = {slope(t_, gr_, *T_WINDOW):.3f}   (mean field 1.5, 3D Ising 1.26)")
print(f"Exponent of the 10-90 width in the same window:    {slope(t_, w_, *T_WINDOW):.3f}   (mean field -0.5, 3D Ising -0.63)")

# ------------------------------ figure ----------------------------------------
centimeter = 1 / 2.54
fig, ax = plt.subplots(1, 3,
                       figsize=(19*centimeter, 7*centimeter),
                       dpi=750,
                       gridspec_kw={'wspace':0.35})

colors = ["#1f77b4", "#2ca02c", "#d62728"]
for tr, c in zip(TR_PROFILES, colors):
    T = tr * TC
    rl, rg, dw, g = sgt(T, KAPPA)
    z, rho = profile(rl, rg, dw, KAPPA)
    # the profile is computed only until rho is within 1e-4 of the bulk values; beyond that it is flat,
    # so it is extended with rho_G (z > 0) and rho_L (z < 0) over the whole plotted range
    z_plot = np.concatenate([[Z_RANGE * 1e-9], z, [-Z_RANGE * 1e-9]])
    rho_plot = np.concatenate([[rg], rho, [rl]])
    ax[0].plot(z_plot * 1e9, rho_plot * MW, color=c, label=f"$T/T_\\mathrm{{crit}}$ = {tr:.2f}")
    print(f"T/Tc = {tr:.2f}: rho_L = {rl*MW:7.2f}, rho_G = {rg*MW:7.2f} kg/m3, gamma = {1e3*g:6.3f} mN/m, w10-90 = {1e9*width_10_90(rl, rg, dw, KAPPA):.3f} nm")
ax[0].set_xlim(-Z_RANGE, Z_RANGE)
ax[0].set_xlabel("$z$ [nm]")
ax[0].set_ylabel(r"$\rho$ [kg m$^{-3}$]")
ax[0].legend(frameon=False, fontsize=6)

tt = np.array([1e-3, 1e-1])
ax[1].loglog(t_, 1e3 * gr_, "ko", ms=3, mfc="none", mew=.75, label="reference correlation")
ax[1].loglog(t_, 1e3 * g_, "r-", label="SGT / PR")
ax[1].loglog(tt, 1e3 * gr_[np.argmin(abs(t_ - 1e-2))] * (tt / 1e-2)**1.26, "k:", label=r"$t^{1.26}$")
ax[1].loglog(tt, 1e3 * g_[np.argmin(abs(t_ - 1e-2))] * (tt / 1e-2)**1.5, "r:", label=r"$t^{1.5}$")
ax[1].set_xlabel(r"$t = 1 - T/T_\mathrm{crit}$")
ax[1].set_ylabel(r"$\gamma$ [mN m$^{-1}$]")
ax[1].legend(frameon=False, fontsize=6)
ax[1].set_xlim([1e-3,0.4])

ax[2].loglog(t_, 1e9 * w_, "ro", ms=3, mew=.75)
ax[2].loglog(tt, 1e9 * w_[np.argmin(abs(t_ - 1e-2))] * (tt / 1e-2)**-0.5, "k:", label=r"$t^{-1/2}$")
ax[2].set_xlabel(r"$t = 1 - T/T_\mathrm{crit}$")
ax[2].set_ylabel(r"$w_{10\!-\!90}$ [nm]")
ax[2].legend(frameon=False, fontsize=6)
ax[2].set_xlim([1e-3,0.4])

for a, lab in zip(ax, "ABC"):
    a.text(-0.25, 0.975, lab, transform=a.transAxes, fontweight="bold", fontsize=8)

plt.tight_layout()
plt.savefig(SAVE_FIG,bbox_inches='tight',dpi=750)
print("saved", SAVE_FIG)