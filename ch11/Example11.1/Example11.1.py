"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 11 Nanomaterial and Thin-Film Synthesis
Example 11.1
Why does zero-order deposition fill deep features? Sec. 11.3.3 attributes the near-100% step
coverage of SFD to the zero-order plateau of Eq. 11.7. This example quantifies the statement with
the steady diffusion-reaction balance of Eq. 11.5 in a trench of width w and depth L, D w C"(z)=2
r(C) with C(0)=C_0 and C'(L)=0, where r=r_maxKC/(1+KC) is the molar deposition rate per sidewall
area and the bottom face is neglected.

No command-line arguments: edit the SETTINGS block below and run   python Example11.1.py
Part 1 (printed): density and viscosity of CO2 at the SFD state (CoolProp, Span-Wagner),
self-diffusivity of CO2 and binary diffusivity of the precursor in CO2, each from the Chapman-Enskog
dilute-gas value with the Enskog dense-fluid factor (contact value of the hard-sphere mixture for the
unlike pair), mean free path, and the Knudsen number in a trench of width w
for the supercritical fluid and for a
low-pressure CVD gas. Part 2 (Fig. 11.4A, B): A) growth rate G(C0) of Eq. (11.7) for non-dissociative
(order 1) and dissociative (order 1/2) Langmuir adsorption at the
plateau rate of Cu SFD quoted in the text;  B) step coverage (growth rate at the trench bottom / at
the opening) from the steady
diffusion-reaction balance of Eq. (11.5) in a trench, w D C'' = 2 r(C), for CVD (Knudsen diffusion,
first order) and SFD (molecular diffusion,
saturated surface) at the SAME growth rate G at the opening. Part 3 (Fig. 11.4C, D): film profiles in a
trench of aspect ratio AR from the same model, with the trench narrowing as the film grows
(quasi-steady C(z) at each time). CAUTION: ONLY the plateau rate (30 nm/min) and the
precursor-molecule type (Cu(hfac)2) come from Chapter 11; the solubility C0, the Langmuir constant
K, the CVD partial pressure and temperature, and the sticking-free sidewall model are ILLUSTRATIVE
choices to be replaced by data of your precursor. The bottom face of the trench is neglected, so the
model shows whether the sidewalls close uniformly (seam) or at the top first (void), not bottom-up
fill.
Produces Fig. 11.4 (figure11.4.pdf). The SFD state (473 K, 15 MPa) is the substrate temperature, which the fluid in the trench takes on.   Requires: numpy, scipy, matplotlib, CoolProp.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os, warnings

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
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
3. Settings
Edit the parameters below to change the calculation.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ============================== SETTINGS ====================================
# --- supercritical fluid state (SFD) ---
T_SCF, P_SCF = 473.15, 15.0e6        # K, Pa: fluid inside the trench, at the substrate temperature
SIGMA, EPS_K, M_CO2 = 0.3941e-9, 195.2, 44.0095e-3   # Lennard-Jones size [m], well depth/kB [K], molar mass [kg/mol] of CO2 (Bird et al.)
C0_SCF   = 5.0                       # precursor concentration at the trench opening [mol/m3]  (illustrative solubility, ~5 mM)
BETA_SCF = 10.0                      # K*C0 of the SFD process (surface 91 % covered)
# --- low-pressure CVD gas ---
T_CVD, P_PREC, P_TOT = 450.0, 10.0, 133.0   # K, precursor partial pressure [Pa], total pressure [Pa] (1 Torr)
M_PREC   = 0.477                     # kg/mol, Cu(hfac)2
# --- precursor Lennard-Jones parameters (ESTIMATES; replace with data of your precursor) ---
SIGMA_PREC = 0.77e-9                 # m: 1.166 V^(1/3) (Bird et al.) with V ~ 283 cm3/mol = twice the molar volume of liquid Hhfac (208.06 g/mol, 1.47 g/cm3)
EPS_K_PREC = 710.0                   # K: 1.92 T_m (Bird et al.) with T_m ~ 370 K for anhydrous Cu(hfac)2
# --- film growth ---
G_PLATEAU = 30e-9/60                 # m/s: Cu plateau rate quoted in the text (Romang and Watkins), 30 nm/min
V_M       = 7.1e-6                   # m3/mol, molar volume of Cu
W0        = 10e-9                    # trench width [m]
AR_LIST   = np.logspace(0, 3.3, 60)  # aspect ratios of the step-coverage curves
AR_PROFILE = 40                      # aspect ratio of the profile simulation (Fig. 11.4C, D)
N_Z       = 100                      # grid points along the trench in the profile simulation
W_PINCH   = 0.3e-9                   # width at which a trench segment is regarded as closed [m]
SAVE_FIG = "figure11.4.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
cm = 1/2.54
kB, R = 1.380649e-23, 8.314462618

# ---------------- Part 1: transport properties ----------------
rho = PropsSI("Dmass", "T", T_SCF, "P", P_SCF, "CO2")
eta = PropsSI("V", "T", T_SCF, "P", P_SCF, "CO2")
n = rho/M_CO2*6.02214076e23
m = M_CO2/6.02214076e23
Ts = T_SCF/EPS_K
Om11 = 1.06036*Ts**-0.15610 + 0.19300*np.exp(-0.47635*Ts) + 1.03587*np.exp(-1.52996*Ts) + 1.76474*np.exp(-3.89411*Ts)   # Neufeld collision integral
D_CE = 3/8/(n*SIGMA**2*Om11)*np.sqrt(kB*T_SCF/(np.pi*m))                           # Chapman-Enskog self-diffusion at this density
eta_p = np.pi/6*n*SIGMA**3
g_c = (1 - eta_p/2)/(1 - eta_p)**3                      # hard-sphere contact value (Enskog factor)
D_SELF = D_CE/g_c                                       # self-diffusivity of CO2
# binary diffusivity of the dilute precursor (2) in CO2 (1): Chapman-Enskog with the Lorentz-Berthelot pair parameters,
# divided by the contact value g12 of the hard-sphere mixture at infinite dilution (Boublik-Mansoori-Carnahan-Starling-Leland)
s12 = 0.5*(SIGMA + SIGMA_PREC)
e12 = np.sqrt(EPS_K*EPS_K_PREC)
mu12 = M_CO2*M_PREC/(M_CO2 + M_PREC)/6.02214076e23
Ts12 = T_SCF/e12
Om11_12 = 1.06036*Ts12**-0.15610 + 0.19300*np.exp(-0.47635*Ts12) + 1.03587*np.exp(-1.52996*Ts12) + 1.76474*np.exp(-3.89411*Ts12)
xi2 = np.pi/6*n*SIGMA**2
dd = SIGMA*SIGMA_PREC/(SIGMA + SIGMA_PREC)
g12 = 1/(1 - eta_p) + 3*dd*xi2/(1 - eta_p)**2 + 2*dd**2*xi2**2/(1 - eta_p)**3
D_SCF = 3/16/(n*s12**2*Om11_12)*np.sqrt(2*kB*T_SCF/(np.pi*mu12))/g12      # used for the precursor transport below
lam_SCF = 1/(np.sqrt(2)*np.pi*SIGMA**2*n)/g_c
n_cvd = P_TOT/(kB*T_CVD)
lam_CVD = 1/(np.sqrt(2)*np.pi*SIGMA**2*n_cvd)
vbar = np.sqrt(8*R*T_CVD/(np.pi*M_PREC))
D_K = lambda w: w*vbar/3
print(f"scCO2 {T_SCF:.0f} K, {P_SCF/1e6:.0f} MPa: rho = {rho:.0f} kg/m3, eta = {eta*1e6:.1f} uPa s, D_self(CO2, Enskog) = {D_SELF:.2e} m2/s, D_12(Cu(hfac)2 in CO2) = {D_SCF:.2e} m2/s, mean free path = {lam_SCF*1e9:.2f} nm")
print(f"  Knudsen number in a {W0*1e9:.0f} nm trench: SCF {lam_SCF/W0:.3f},  low-pressure CVD gas ({P_TOT:.0f} Pa, {T_CVD:.0f} K): {lam_CVD/W0:.0f}   (lambda = {lam_CVD*1e6:.0f} um)")
print(f"  precursor mean speed in CVD {vbar:.0f} m/s -> Knudsen diffusivity {D_K(W0):.2e} m2/s (precursor molecular diffusivity in scCO2: {D_SCF:.2e} m2/s, ratio {D_K(W0)/D_SCF:.1f})")

# ---------------- Part 2: kinetic law and step coverage ----------------
C0_CVD = P_PREC/(R*T_CVD)
r_top = G_PLATEAU/V_M                                    # molar deposition rate at the opening [mol m-2 s-1], same for both processes
K_SCF = BETA_SCF/C0_SCF
rmax = r_top*(1 + BETA_SCF)/BETA_SCF       # saturated rate of the SFD chemistry
k1_CVD = r_top/C0_CVD                                    # first-order rate constant [m/s] needed in CVD for the same rate at its low concentration
print(f"same growth rate {G_PLATEAU*6e10:.0f} nm/min at the opening:  CVD C0 = {C0_CVD:.2e} mol/m3, k1 = {k1_CVD:.3f} m/s (sticking coefficient {4*k1_CVD/vbar:.4f});  "
      f"SFD C0 = {C0_SCF:.1f} mol/m3, K C0 = {BETA_SCF:.0f}")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Define functions
- step_coverage: steady trench, c = C/C0, s = z/L:  c'' = theta*beta*c/(1 + beta*c),  c(0) = 1,
c'(1) = 0,  theta = 2 L^2 rmax/
- sc_curve
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def step_coverage(theta, beta):
    """steady trench, c = C/C0, s = z/L:  c'' = theta*beta*c/(1 + beta*c),  c(0) = 1, c'(1) = 0,  theta = 2 L^2 rmax/(D w C0).
    The concentration at the closed end is found by shooting backward from s = 1 (robust for strong depletion)."""
    def c_top(lncL):
        sol = solve_ivp(lambda s, y: [y[1], theta*beta*y[0]/(1 + beta*y[0])], [1, 0], [np.exp(lncL), 0.0], rtol=1e-9, atol=1e-300)
        return sol.y[0, -1]
    lo, hi = -700.0, 0.0
    if c_top(hi) - 1.0 < 1e-10:
        cL = 1.0                      # no depletion
    else:
        cL = np.exp(brentq(lambda u: np.log(c_top(u)) - 0.0, lo, hi, xtol=1e-10))
    rate = lambda c: beta*c/(1 + beta*c)
    return rate(cL)/rate(1.0)

def sc_curve(D_of_w, C0, beta, rmax_):
    return np.array([step_coverage(2*(ar*W0)**2*rmax_/(D_of_w*W0*C0), beta) for ar in AR_LIST])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Set constants and grids
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# first-order CVD: r = k1 C  ==  Langmuir limit beta -> 0 with rmax*beta/C0 = k1
BETA_CVD = 1e-6
rmax_cvd = k1_CVD*C0_CVD/BETA_CVD
SC = {"CVD": sc_curve(D_K(W0), C0_CVD, BETA_CVD, rmax_cvd),
      "SFD": sc_curve(D_SCF, C0_SCF, BETA_SCF, rmax),
      "SFD_1": sc_curve(D_SCF, C0_SCF/BETA_SCF*1.0, 1.0, rmax),
      "SFD_01": sc_curve(D_SCF, C0_SCF/BETA_SCF*0.1, 0.1, rmax)}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
7. Define functions
- ar50
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def ar50(sc):
    return np.exp(np.interp(0.5, sc[::-1], np.log(AR_LIST)[::-1]))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
8. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print(f"aspect ratio at which the bottom grows at half the rate of the opening:  CVD {ar50(SC['CVD']):.0f},  SFD (K C0 = {BETA_SCF:.0f}) {ar50(SC['SFD']):.0f}, "
      f"K C0 = 1: {ar50(SC['SFD_1']):.0f}, K C0 = 0.1: {ar50(SC['SFD_01']):.0f}")

fig, axs = plt.subplots(2, 2, figsize=(19*cm, 14.5*cm), gridspec_kw={'wspace':0.3, 'hspace':0.35, 'height_ratios':[1, 1.15]})
ax = axs[0]
C = np.logspace(-3, 2, 200)
for order, ls, lab in ((1.0, "-", "order 1 (non-dissociative)"), (0.5, "--", "order 1/2 (dissociative)")):
    x = (K_SCF*C)**order
    ax[0].plot(C, G_PLATEAU*6e10*rmax/rmax*x/(1 + x), ls=ls, color="#1f77b4", label=lab)
ax[0].plot(C0_SCF, G_PLATEAU*6e10, "o", mfc="#ff7f0e", mec="k", mew=.4, ms=3.5, zorder=5)
ax[0].plot(C0_CVD, G_PLATEAU*6e10, "s", mfc="#2ca02c", mec="k", mew=.4, ms=3.5, zorder=5)
ax[0].annotate("SFD", (C0_SCF, 30), xytext=(C0_SCF*1.5, 12), fontsize=5.5, arrowprops=dict(arrowstyle="-", lw=.4))
ax[0].annotate("CVD (needs a first-order\nconstant %.0f times larger)" % (k1_CVD/(rmax*K_SCF)), (C0_CVD, 30), xytext=(0.15, 0.7), fontsize=5.5, arrowprops=dict(arrowstyle="-", lw=.4))
ax[0].axhline(G_PLATEAU*6e10*(1 + BETA_SCF)/BETA_SCF, color="0.5", lw=.5, ls=":")
ax[0].text(1.5e-3, G_PLATEAU*6e10*(1 + BETA_SCF)/BETA_SCF*1.04, "plateau (zero order)", fontsize=5.2, color="0.35")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlim(1e-3, 1e2)
ax[0].set_ylim(1e-1, 1e2)
ax[0].set_xlabel("precursor concentration $C_0$ [mol m$^{-3}$]")
ax[0].set_ylabel("growth rate [nm min$^{-1}$]")
ax[0].legend(frameon=False, fontsize=5.2, loc="lower right")
ax[0].text(-0.2, 1.02, "A", transform=ax[0].transAxes, fontweight="bold", fontsize=7)
ax[1].plot(AR_LIST, SC["CVD"], color="#2ca02c", label="CVD (Knudsen, first order)")
ax[1].plot(AR_LIST, SC["SFD"], color="#ff7f0e", label=f"SFD, $KC_0$ = {BETA_SCF:.0f}")
ax[1].plot(AR_LIST, SC["SFD_1"], color="#ff7f0e", ls="--", label="SFD, $KC_0$ = 1")
ax[1].plot(AR_LIST, SC["SFD_01"], color="#ff7f0e", ls=":", label="SFD, $KC_0$ = 0.1")
ax[1].axhline(0.5, color="0.6", lw=.4)
ax[1].set_xscale("log")
ax[1].set_xlim(1, AR_LIST.max())
ax[1].set_ylim(0, 1.05)
ax[1].set_xlabel("aspect ratio $L/w$ ($w$ = %.0f nm)" % (W0*1e9))
ax[1].set_ylabel("step coverage  $r_\\mathrm{bottom}/r_\\mathrm{top}$")
ax[1].legend(frameon=False, fontsize=5.2, loc="lower left")
ax[1].text(-0.2, 1.02, "B", transform=ax[1].transAxes, fontweight="bold", fontsize=7)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
9. Define functions
- profile_run
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ---------------- Part 3: film profiles in a narrowing trench ----------------
def profile_run(kind, snaps=(0.3, 0.6, 0.9)):
    L = AR_PROFILE*W0
    z = np.linspace(0, L, N_Z)
    dz = z[1] - z[0]
    h = np.zeros(N_Z)
    closed = np.zeros(N_Z, bool)
    if kind == "CVD":
        C0, beta, rm, Dfun = C0_CVD, BETA_CVD, rmax_cvd, (lambda w: D_K(w))
    else:
        C0, beta, rm, Dfun = C0_SCF, BETA_SCF, rmax, (lambda w: D_SCF + 0*w)
    K = beta/C0
    out = []
    t = 0.0
    c = np.ones(N_Z)*C0
    targets = list(snaps)
    rate = lambda c_: rm*K*c_/(1 + K*c_)
    while True:
        w = np.maximum(W0 - 2*h, 1e-12)
        wface = 0.5*(w[1:] + w[:-1])
        Dw = Dfun(wface)*wface
        # first closed segment cuts the supply to everything below it
        idx = np.where(closed)[0]
        ncut = idx.min() if len(idx) else N_Z
        nn = ncut                                           # active nodes 0..nn-1 (node 0 = opening)
        cc = c[:nn].copy()
        cc[0] = C0
        for _ in range(60):                                 # Newton iteration of the finite-volume balance (tridiagonal, dense solve)
            F = np.zeros(nn)
            J = np.zeros((nn, nn))
            F[0] = cc[0] - C0
            J[0, 0] = 1
            for i in range(1, nn):
                fl = Dw[i-1]/dz*(cc[i-1] - cc[i])
                fr = Dw[i]/dz*(cc[i] - cc[i+1]) if i < nn-1 else 0.0
                F[i] = fl - fr - 2*rate(cc[i])*dz*(0.5 if i == nn-1 else 1.0)
                J[i, i-1] = Dw[i-1]/dz
                J[i, i] = -Dw[i-1]/dz - (Dw[i]/dz if i < nn-1 else 0) - 2*rm*K/(1 + K*cc[i])**2*dz*(0.5 if i == nn-1 else 1.0)
                if i < nn-1:
                    J[i, i+1] = Dw[i]/dz
            dc = np.linalg.solve(J, -F)
            cc = np.maximum(cc + dc, 0)
            if np.max(abs(dc)) < 1e-10*C0:
                break
        c[:nn] = cc
        r = np.zeros(N_Z)
        r[:nn] = rate(cc)
        dt = 0.02*W0/2/(V_M*r.max())
        h += V_M*r*dt
        t += dt
        w = W0 - 2*h
        closed |= (w < W_PINCH) & (np.arange(N_Z) < ncut + 1)
        topw = w[0]/W0
        while targets and topw <= 1 - targets[0]:
            out.append((t, np.maximum(h, 0).copy(), closed.copy()))
            targets.pop(0)
        if closed[0] or closed.all() or t > 1e4:
            out.append((t, np.maximum(h, 0).copy(), closed.copy()))
            break
    return z, out, L

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
10. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

ax = axs[1]
for a_, kind, lab, letter in zip(ax, ("CVD", "SFD"), ("CVD (Knudsen diffusion, first order)", "SFD (molecular diffusion, saturated surface)"), "CD"):
    z, out, L = profile_run(kind)
    print(f"{kind}: closure of the top at t = {out[-1][0]:.0f} s; film thickness at opening / mid-depth / bottom = "
          f"{out[-1][1][0]*1e9:.1f} / {out[-1][1][N_Z//2]*1e9:.1f} / {out[-1][1][-1]*1e9:.1f} nm")
    a_.add_patch(plt.Rectangle((-W0*1e9*0.5-6, 0), 6, L*1e9, fc="0.6", ec="none"))
    a_.add_patch(plt.Rectangle((W0*1e9*0.5, 0), 6, L*1e9, fc="0.6", ec="none"))
    cols = ["#9ecae1", "#6baed6", "#3182bd", "#08306b"]
    hf = out[-1][1]*1e9
    a_.fill_betweenx(z*1e9, -W0*1e9/2, -W0*1e9/2 + hf, color="#fdd9a8", lw=0)
    a_.fill_betweenx(z*1e9, W0*1e9/2 - hf, W0*1e9/2, color="#fdd9a8", lw=0)
    a_.text(0, L*1e9*0.45, "film", fontsize=5.5, ha="center", color="0.3") if False else None
    for k, (t, h, closed) in enumerate(out):
        xl = -W0*1e9/2 + h*1e9
        xr = W0*1e9/2 - h*1e9
        a_.plot(xl, z*1e9, color=cols[min(k, 3)], lw=.8)
        a_.plot(xr, z*1e9, color=cols[min(k, 3)], lw=.8, label=f"{t:.0f} s")
    a_.set_ylim(L*1e9, 0)
    a_.set_xlim(-W0*1e9/2-5, W0*1e9/2+5)
    a_.set_xlabel("$x$ [nm]")
    a_.set_title(lab, fontsize=6)
    a_.legend(frameon=False, fontsize=5.2, loc="lower center", ncol=2, handlelength=1.2, title="film surfaces at", title_fontsize=5.2)
    a_.text(-0.2 if letter == "C" else -0.05, 1.05, letter, transform=a_.transAxes, fontweight="bold", fontsize=7)
ax[0].set_ylabel("depth $z$ [nm]")
ax[1].set_yticklabels([])
plt.savefig(os.path.join(HERE, SAVE_FIG), bbox_inches='tight', dpi=750)
print("saved", SAVE_FIG)
