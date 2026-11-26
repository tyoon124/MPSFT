"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 9 Supercritical Fluid Extraction and Fractionation
Example 9.4
Extraction curve of a packed bed with the broken-and-intact-cells model: In this example, we
integrate the balances of Eqs. 9.14 and 9.15 for a packed bed of milled solid and show the three
periods of the extraction curve (Fig. 9.6A, B), the shrinking-core curve fitted to it, and the effect of
the particle diameter (Fig. 9.6C, D). All parameters are illustrative, and no experimental extraction
curve is fitted, except that the fluid-side (film) mass-transfer coefficient is now calculated from
the transport properties of Chapter 3 instead of being chosen.

comparison with the shrinking-core (SC) model.   Produces Fig. 9.6 (panels A-D).
No command-line arguments: edit the SETTINGS block below and run   python Example9.4.py
Model (Sovova, Chem. Eng. Sci. 49, 409 (1994)), bed of length H, plug flow of solvent at mass flux
G, no axial dispersion, fluid hold-up neglected:
   G dy/dz = J,        (1 - eps) rho_s dx/dt = -J,
J = K_f(...)*(y* - y)                       while x > x_k   (broken-cell solute at the particle
surface, fluid-side resistance)
J = K_s (1 - eps) rho_s x (1 - y/y*)        when  x <= x_k  (solute inside intact cells, solid-side
resistance) x = solute / insoluble solid [kg/kg], y = solute / solvent [kg/kg], x_k = x0 (1 - r) is
the solute in intact cells and r the accessible (broken-cell) fraction.  The fluid-side term is
written with a number of transfer units NTU_f = k_f a H / u_s, with a = 6 (1 - eps) / d_p the
particle surface area per bed volume. k_f is NOT chosen: it follows from the chain
   residual entropy s+ = -s_res/R of CO2 (Span-Wagner, CoolProp)
   -> viscosity eta from the residual-entropy master curve of Example 3.3 (Eq. 3.30, coefficients
      fitted to CO2 in Example 3.3 and copied into the settings block)
   -> self-diffusion coefficient of CO2 from entropy scaling, D+ = 0.5016 exp(-0.2031 s+)
      (Saric et al., Sci. Rep. 13, 16107 (2023), Eqs. 2-3)
   -> infinite-dilution diffusion coefficient of the solute, D12 = 0.1276 M2^-0.6124 D_CO2
      (M2 in kg/mol; Saric et al. 2023, Eq. 4)
   -> Re, Sc, Sh = 2 + 1.1 Re^0.6 Sc^(1/3) (Wakao-Kaguei, Eq. 9.18) -> k_f = Sh D12 / d_p.
Each size class has its own d_p, hence its own k_f and a. The balances are
discretised in z (upwind, N_CELLS cells, y of each cell solved algebraically) and integrated in time
with an explicit fourth-order Runge-Kutta scheme. The SC curve is Eq. (9.17), t = tau [1 -
3(1-X)^(2/3) + 2(1-X)], with tau fitted by least squares to the BIC yield curve. ALL OTHER PARAMETERS
ARE ILLUSTRATIVE (no experimental extraction curve was fitted). The solute is modeled as triolein
(M2 = 0.885 kg/mol), which lies outside the range of the hydrocarbons (up to naphthalene) to which
Eq. 4 of Saric et al. was fitted, so D12 is an extrapolated estimate.  The particle-size scalings used for
Fig. 9.6C, D are assumptions: the accessible fraction r grows as 1/d (more cells are broken by finer
grinding) and the intact-cell rate constant K_s as 1/d^2 (diffusion). CO2 density at the extraction
state (CoolProp) gives the superficial velocity u_s = G / rho.
Requires: numpy, scipy (brentq, minimize_scalar), matplotlib, CoolProp; numba is optional (about 5
times faster time loop; without it the script still runs, only more slowly).
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
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
T_EXT, P_EXT = 313.15, 30e6      # K, Pa, extraction state
H          = 0.30                # m, bed length
EPS        = 0.40                # bed voidage
RHO_S      = 1300.0              # kg/m3, density of the insoluble solid
G          = 0.80                # kg/(m2 s), solvent mass flux
X0         = 0.12                # kg/kg, initial solute content (oil) of the solid, per kg of insoluble solid
Y_STAR     = 0.008               # kg/kg, equilibrium solubility of the solute in CO2 at the extraction state
M_SOLUTE   = 0.885               # kg/mol, molar mass of the solute (triolein as a model vegetable oil)
# Residual-entropy master curve of the reduced viscosity of CO2, Eq. 3.30 (fitted in Example 3.3):
# eta_tilde = A_ETA s+^(-2/3) + B_ETA s+^C_ETA exp(-D_ETA s+),  eta_tilde = eta / [rho_N^(2/3) (m kB T)^(1/2)]
A_ETA, B_ETA, C_ETA, D_ETA = 0.26746, 0.19160, 0.73735, -0.35007
# Entropy scaling of the self-diffusion coefficient of CO2 and the solute relation of Saric et al. (2023)
A_D, B_D   = 0.5016, 0.2031      # D+ = D (rho_N)^(1/3) (M1/(R T))^(1/2) s+^(2/3) = A_D exp(-B_D s+)
C_12, E_12 = 0.1276, -0.6124     # D12 = C_12 M2^E_12 D_CO2, M2 in kg/mol
D_REF      = 0.5                 # mm, reference particle diameter
R_REF      = 0.30                # accessible (broken-cell) fraction at D_REF
KS_REF     = 4.0e-4              # 1/s, rate constant of the intact-cell step at D_REF
SIGMA_G    = 2.0                 # geometric standard deviation of the (lognormal, mass-based) particle-size distribution of Fig. 9.6A, B; D_REF is the median
N_CLASS    = 7                   # size classes (Gauss-Hermite nodes in ln d) used for the distribution
D_SIZES    = [0.25, 0.5, 1.0]    # mm, particle diameters of Fig. 9.6C, D
N_CELLS    = 80                  # axial cells
DT         = 3.0                 # s, time step of the explicit RK4 integration
T_END_S    = 9000.0              # s, simulated time
SF_MAX     = 40.0                # kg solvent per kg insoluble solid, upper end of Fig. 9.6C
SAVE_FIG   = "figure9.6.pdf"
# ============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
cm = 1/2.54
m_s = (1 - EPS) * RHO_S * H                                                  # kg insoluble solid per m2 of bed cross-section

try:                                                                              # optional speed-up: the cell loop is sequential and slow in pure Python
    from numba import njit
except ImportError:
    def njit(f):
        return f

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define functions
- transport: viscosity, self- and solute diffusion coefficients, Re, Sc, Sh, k_f and NTU_f of a
particle of diameter d_p (the Chapter 3 -> Eq. 9.18 chain)
- _integrate: RK4 time loop; x[class, cell]; returns the output arrays (time index, cumulative
yield, y_out/y*)
- bic: explicit RK4 integration for a bed of particles of several size classes, classes = [(r, Ks,
w), ...] (w = mass
- size_classes: lognormal mass distribution of the diameter -> n (r, Ks, w) classes; r and Ks scale
with d as assumed in Fig.
- sc_time
- sc_yield
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

kB = 1.380649e-23
NA = 6.02214076e23
RGAS = kB * NA
M_CO2 = PropsSI("molar_mass", "CO2")                                        # kg/mol
RHO = PropsSI("Dmass", "T", T_EXT, "P", P_EXT, "CO2")                       # kg/m3
U_S = G / RHO                                                                # m/s, superficial velocity

def transport(d_mm):
    """Chapter 3 -> Eq. 9.18 chain for a particle of diameter d_mm [mm]; returns a dict"""
    rho_n = PropsSI("Dmolar", "T", T_EXT, "P", P_EXT, "CO2") * NA            # molecules per m3
    s_plus = -PropsSI("Smolar_residual", "T", T_EXT, "P", P_EXT, "CO2") / RGAS
    m = M_CO2 / NA
    eta_t = A_ETA * s_plus**(-2/3) + B_ETA * s_plus**C_ETA * np.exp(-D_ETA * s_plus)
    eta = eta_t * rho_n**(2/3) * np.sqrt(m * kB * T_EXT)                      # Pa s
    D11 = A_D * np.exp(-B_D * s_plus) / s_plus**(2/3) / (rho_n**(1/3) * np.sqrt(M_CO2 / (RGAS * T_EXT)))
    D12 = C_12 * M_SOLUTE**E_12 * D11
    d = d_mm * 1e-3
    Re = RHO * U_S * d / eta
    Sc = eta / (RHO * D12)
    Sh = 2 + 1.1 * Re**0.6 * Sc**(1/3)
    kf = Sh * D12 / d
    a = 6 * (1 - EPS) / d                                                     # m2 of particle surface per m3 of bed
    return dict(s_plus=s_plus, eta=eta, eta_ref=PropsSI("viscosity", "T", T_EXT, "P", P_EXT, "CO2"),
                D11=D11, D12=D12, Re=Re, Sc=Sc, Sh=Sh, kf=kf, kfa=kf * a, NTU=kf * a * H / U_S)

@njit
def _integrate(x, xk, gs, w, bf, cf, dt, n, every, y_star, x0):
    # bf[c]: fluid-side transfer units per cell of size class c
    """RK4 time loop; x[class, cell]; returns the output arrays (time index, cumulative yield, y_out/y*)"""
    nc, nz = x.shape
    n_out = n // every + 1
    Yc = np.zeros(n_out)
    Yo = np.zeros(n_out)
    T = np.zeros(n_out)
    j = 1
    k1 = np.empty_like(x)
    k2 = np.empty_like(x)
    k3 = np.empty_like(x)
    k4 = np.empty_like(x)
    xs = np.empty_like(x)
    for k in range(1, n + 1):
        for stage in range(4):
            if stage == 0:
                xs[:] = x
            elif stage == 1:
                xs[:] = x + 0.5 * dt * k1
            elif stage == 2:
                xs[:] = x + 0.5 * dt * k2
            else:
                xs[:] = x + dt * k3
            y = 0.0
            for i in range(nz):
                sa = 0.0
                sb = 0.0
                for c in range(nc):                                              # per class: J = al - be*y_i
                    if xs[c, i] > xk[c]:
                        al = bf[c] * y_star
                        be = bf[c]
                    else:
                        al = gs[c] * xs[c, i]
                        be = gs[c] * xs[c, i] / y_star
                    sa += w[c] * al
                    sb += w[c] * be
                yi = (y + sa) / (1.0 + sb)                                       # fluid balance of cell i (mixed classes), solved for y_i
                for c in range(nc):
                    if xs[c, i] > xk[c]:
                        al = bf[c] * y_star
                        be = bf[c]
                    else:
                        al = gs[c] * xs[c, i]
                        be = gs[c] * xs[c, i] / y_star
                    d = -cf * (al - be * yi)
                    if stage == 0:
                        k1[c, i] = d
                    elif stage == 1:
                        k2[c, i] = d
                    elif stage == 2:
                        k3[c, i] = d
                    else:
                        k4[c, i] = d
                y = yi
            yl = y
        x[:] = np.maximum(x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0)
        if k % every == 0:
            s = 0.0
            for c in range(nc):
                s += w[c] * x[c].mean()
            T[j] = k * dt
            Yc[j] = 1.0 - s / x0
            Yo[j] = yl / y_star
            j += 1
    return T[:j], Yc[:j], Yo[:j]

def bic(classes, t_end, n_out=400):
    """explicit RK4 integration for a bed of particles of several size classes, classes = [(r, Ks, w, d_mm), ...] (w = mass fraction of the solid);
    returns t, cumulative yield (fraction of the initial solute) and y_out/y* at n_out+1 equally spaced times"""
    r_, K_, w_, d_ = (np.array(v, float) for v in zip(*classes))
    w_ = w_ / w_.sum()
    dz = H / N_CELLS
    bf = np.array([transport(d)["NTU"] for d in d_]) / N_CELLS                 # per size class
    gs = K_ * (1 - EPS) * RHO_S * dz / G
    cf = G / (dz * (1 - EPS) * RHO_S)
    n = int(np.ceil(t_end / DT))
    every = max(n // n_out, 1)
    return _integrate(np.full((len(w_), N_CELLS), X0), X0 * (1 - r_), gs, w_, bf, cf, DT, n, every, Y_STAR, X0)

def size_classes(d_med, sigma_g, n):
    """lognormal mass distribution of the diameter -> n (r, Ks, w, d) classes; r and Ks scale with d as assumed in Fig. 9.6C, D"""
    if sigma_g <= 1.0 or n == 1:
        dl, wl = [d_med], [1.0]
    else:
        z, w = np.polynomial.hermite_e.hermegauss(n)
        dl = d_med * np.exp(np.log(sigma_g) * z)
        wl = w / w.sum()
    return [(min(R_REF * D_REF / d, 1.0), KS_REF * (D_REF / d)**2, wi, d) for d, wi in zip(dl, wl)]

def sc_time(X, tau):
    return tau * (1 - 3 * (1 - X)**(2 / 3) + 2 * (1 - X))   # Eq. (9.17)
def sc_yield(t, tau):
    X = np.zeros_like(t)
    for i, ti in enumerate(t):
        X[i] = 1.0 if ti >= tau else brentq(lambda x: sc_time(x, tau) - ti, 0.0, 1.0)
    return X

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

rho = PropsSI("Dmass", "T", T_EXT, "P", P_EXT, "CO2")
print(f"CO2 at {T_EXT:.0f} K, {P_EXT / 1e6:.0f} MPa: {rho:.0f} kg/m3; superficial velocity u = G/rho = {1e3 * G / rho:.2f} mm/s; solvent-to-feed ratio of the run = {G * T_END_S / m_s:.1f} kg/kg")
for d in D_SIZES:
    tp = transport(d)
    print(f"d_p = {d} mm: s+ = {tp['s_plus']:.3f}; eta = {tp['eta']*1e6:.1f} uPa s (reference correlation {tp['eta_ref']*1e6:.1f}); "
          f"D_CO2 = {tp['D11']:.2e} m2/s; D12 = {tp['D12']:.2e} m2/s; Re = {tp['Re']:.2f}; Sc = {tp['Sc']:.1f}; Sh = {tp['Sh']:.2f}; "
          f"k_f = {tp['kf']:.2e} m/s; k_f a = {tp['kfa']:.3f} 1/s; NTU_f = {tp['NTU']:.0f}")

# ---- Fig. 9.6 A, B: reference particle, BIC and SC curves
r0 = R_REF
t, Y, yo = bic(size_classes(D_REF, SIGMA_G, N_CLASS), T_END_S)
Y_cer = Y_STAR * G * t / (m_s * X0)                                          # Eq. (9.16): Y = Q y* t (as a fraction of the initial solute)
i_up = np.argmax(yo >= 0.95)
i_c = i_up + np.argmax(yo[i_up:] < 0.95)
i_d = np.argmax(Y > 0.9 * Y[-1]) if Y[-1] > 0 else len(t) - 1
f = minimize_scalar(lambda lt: np.sum((sc_yield(t, np.exp(lt)) - Y)**2), bounds=(np.log(200), np.log(40000)), method="bounded")
tau = np.exp(f.x)
Xsc = sc_yield(t, tau)
print(f"BIC (d = {D_REF} mm): CER (y_out >= 0.95 y*) ends at t = {t[i_c] / 60:.0f} min, yield {Y[i_c]:.2f} (broken-cell fraction r = {r0:.2f}); 90% of the final yield at {t[i_d] / 60:.0f} min")
i_10 = i_c + np.argmax(yo[i_c:] < 0.10)
print(f"outlet concentration falls to 10% of y* at t = {t[i_10] / 60:.0f} min")
print(f"SC fit: tau = {tau:.0f} s;  maximum deviation from BIC = {np.max(np.abs(Xsc - Y)):.3f}")
fig, axs = plt.subplots(2, 2, figsize=(19*cm*0.8, 12.8*cm), gridspec_kw={'wspace':0.3, 'hspace':0.4})
ax = axs[0]
tm = t / 60
ax[0].plot(tm[::15], Y[::15], "o", ms=3, color="tab:blue", label="BIC")
ax[0].plot(tm, Xsc, "--", color="tab:red", label="SC (fitted $\\tau$)")
ax[0].plot(tm, np.minimum(Y_cer, 1.0), ":", color="k", lw=.5, label=r"CER line, $Q\,y^*t$")
for lab, (a_, b_) in zip(("CER", "FER", "DC"), ((0, tm[i_c]), (tm[i_c], tm[np.argmax(Y > 0.8 * Y[-1])]), (tm[np.argmax(Y > 0.8 * Y[-1])], tm[-1]))):
    ax[0].axvspan(a_, b_, color={"CER": "0.92", "FER": "0.97", "DC": "0.92"}[lab], lw=0)
    ax[0].text((a_ + b_) / 2, 0.03, lab, ha="center", fontsize=5.5)
ax[0].set_xlabel("time / min")
ax[0].set_ylabel("cumulative yield / initial solute")
ax[0].set_xlim(0, tm[-1])
ax[0].set_ylim(0, 1.02)
ax[0].legend(frameon=False, fontsize=5.2, loc="center right")
ax[1].plot(tm[::15], yo[::15], "o", ms=3, color="tab:blue")
ax[1].axhline(1, color="k", lw=.4, ls=":")
ax[1].axhline(0.95, color="k", lw=.4, ls=":")
ax[1].set_xlabel("time / min")
ax[1].set_ylabel(r"outlet concentration $y_\mathrm{out}/y^*$")
ax[1].set_xlim(0, tm[-1])
ax[1].set_ylim(0, 1.1)
for a_, lab in zip(ax, "AB"):
    a_.text(-0.16, 1.02, lab, transform=a_.transAxes, fontweight="bold", fontsize=7)

# ---- Fig. 9.6 C, D: particle size (one size per curve, no distribution)
cols = ["tab:blue", "tab:orange", "tab:green"]
ax = axs[1]
T_FIN = SF_MAX * m_s / G
t80 = []
for d, c in zip(D_SIZES, cols):
    r, Ks, _, _ = size_classes(d, 1.0, 1)[0]
    ts, Yd, _ = bic([(r, Ks, 1.0, d)], T_FIN, 250)
    sf = G * ts / m_s
    ax[0].plot(sf, Yd, color=c, label=f"$d_\\mathrm{{p}}$ = {d:g} mm ($r$ = {r:.2f}, $K_\\mathrm{{s}}$ = {Ks * 1e4:.1f}$\\times 10^{{-4}}$ s$^{{-1}}$)")
    k80 = np.argmax(Yd > 0.8) if Yd.max() > 0.8 else None
    t80.append(ts[k80] / 60 if k80 else np.nan)
    print(f"d = {d} mm: r = {r:.2f}, K_s = {Ks:.2e} 1/s, S/F for 50% / 80% yield = {sf[np.argmax(Yd > .5)]:.1f} / {sf[k80] if k80 else float('nan'):.1f}, final yield {Yd[-1]:.2f}")
ax[0].set_xlabel("solvent-to-feed ratio $S/F$ / (kg kg$^{-1}$)")
ax[0].set_ylabel("cumulative yield / initial solute")
ax[0].set_xlim(0, SF_MAX)
ax[0].set_ylim(0, 1.02)
ax[0].legend(frameon=False, fontsize=4.8, loc="lower right")
dd = np.linspace(0.2, 1.2, 12)
tt = []
for d in dd:
    r, Ks, _, _ = size_classes(d, 1.0, 1)[0]
    ts, Yd, _ = bic([(r, Ks, 1.0, d)], T_FIN, 250)
    tt.append(ts[np.argmax(Yd > 0.8)] / 60 if Yd.max() > 0.8 else np.nan)
ax[1].plot(dd, tt, "o", ms=3, color="k")
ax[1].set_xlabel("particle diameter $d_\\mathrm{p}$ / mm")
ax[1].set_ylabel("time to 80% yield / min")
ax[1].set_ylim([50,170])
for a_, lab in zip(ax, "CD"):
    a_.text(-0.16, 1.02, lab, transform=a_.transAxes, fontweight="bold", fontsize=7)
print("time to 80% yield / min:", ", ".join(f"{d} mm: {x:.0f}" for d, x in zip(D_SIZES, t80)))
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
