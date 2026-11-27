"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 10 Particle Formation from Supercritical Solutions
Example 10.4
Jet break-up and dynamic surface tension vanishing in SAS: the model of Marra et al.: In this
example, we code the complete calculation of Marra, De Marco, and Reverchon for YAc in DMSO,
injected into CO2 through a 100 mum nozzle at 1 mL/min (jet velocity u=2.12 m/s). We calibrate it on
their results, which are for 40 ^degC, and apply it at 60 ^degC. The reason for the change of
temperature is the pseudocritical (Widom) line of CO2. At 40 ^degC, CO2 crosses it between about 85
and 95 bar, and its density doubles within 20 bar (280 to 560 kg/m^3 between 80 and 100 bar). This
puts a shoulder into tau_gamma(P) that belongs to the antisolvent, not to the jet.

F. Marra, I. De Marco, E. Reverchon, Chem. Eng. Sci. 71, 39-45 (2012).  System: yttrium acetate
(YAc) / DMSO / CO2 (acetone and acetone/DMSO feeds: Example 10.5), 100 um nozzle, 1 mL/min liquid
flow (their Table 1).  The paper works at 40 C; the model is calibrated on its 40 C results and then
applied at 60 C (T_K), where CO2 crosses its pseudocritical (Widom) line gradually instead of
within a few bar.
No command-line arguments: edit the SETTINGS block below and run   python Example10.4.py
Produces Fig. 10.5 and the data file Example10.4_data.npz (all arrays drawn in the figure).  With
RECOMPUTE = False the script only reads the data file and redraws the figure (a few seconds), so the
plotting style can be changed without repeating the calculation.
Requires: numpy, scipy, matplotlib, phasepy, CoolProp (CO2 viscosity only).

The calculation chain of the paper is coded in full: (1) Phase behavior and densities: Peng-Robinson
EoS (phasepy) with the van der Waals one-fluid mixing rules and two binary parameters,
a12 = (1-k12) sqrt(a1 a2) and b12 = (1-l12)(b1+b2)/2.  CO2-DMSO: k12, l12 of the two-parameter PR
correlation of Rajasingam et al.
(J. Supercrit. Fluids 31, 227 (2004)) with their critical constants (mean of 308 and 318 K for 40 C;
the 318 K values, the
highest temperature of the fit, for 60 C); it reproduces their CO2 solubilities at 308 and 318 K to
about 1 %.  (Their one-
parameter value, k12 = 0.08, under-predicts the bubble pressure by about 25 % at low CO2 content and
puts the 40 C mixture
critical point (MCP) near 240 bar.)  The molar density C(x) of the mixture along the CO2-solvent
composition axis enters the
gradient-theory integral; the binary MCP is located where Gamma = 1 + x dln(phi_CO2)/dx first
touches zero. (2) Jet break-up time T = ln(a/eta0)/beta* (Eq. 10.16): beta* is the largest root of
the Sterling-Sleicher dispersion relation
beta^2 F1 + beta F3 = F4 + 0.175 F5 (Eqs. 2-7 of the paper; modified Bessel functions), with the jet
density, viscosity and
surface tension, the density of the CO2-rich phase (PR) and the jet velocity u = Q/(pi a^2). (3)
Dynamic surface tension: gamma(t) = int sum_ij kappa_ij dC_i/dr dC_j/dr dr (Eq. 10.17, Cahn-Hilliard)
evaluated on the transient
interdiffusion profile x(r,t) of CO2 and solvent (Fick's law, Eq. 21 of the paper).  The profile
between the jet liquid (x = 0)
and the CO2 (x = 1) is self-similar, x = f(r/sqrt(t)), so that gamma(t) = S/sqrt(t) with S from ONE
similarity solution per
state point (no time stepping).  D(x) interpolates (Vignes) between the Wilke-Chang values of CO2 in
the liquid and of the solvent
in CO2.  Eq. 10.17 alone decays as t^(-1/2) and never reaches zero, so "vanished" means sigma <
SIGMA_ZERO. (4) Cross-over: T = t; T < t -> jet break-up first (droplets, microparticles), T > t ->
DSTV first (gas-like mixing, nanoparticles).

What the paper does not report, and how it is fixed from its 40 C results (calibration at T_CAL):
- ln(a/eta0), fitted by the authors to measured jet lengths: fitted to their pure-DMSO break-up
times (Table 2, c = 0) at 80, 120
    and 160 bar;
- the diffusion coefficient: D = phi * D_Vignes(x), phi fitted to their pure-DMSO DSTV times (Table
3, c = 0);
- the viscosity and surface tension of the YAc solutions (from Braeuer et al. 2011 in the paper):
one pair per concentration, inferred
by least squares from Tables 2 and 3 (c > 0), with extra weight on ln(t/T); smooth (quadratic)
functions of c are used between. ln(a/eta0) and phi are carried to 60 C as functions of the CO2
density (ln y = A + B exp(rho/rho0) through the three calibration points, smooth and monotone), the
quantity through which the dense gas enters both the disturbance of the jet and the interfacial
profile.  At 60 C, 100-240 bar spans 290-760 kg/m3, inside the density range of the calibration
(80-160 bar at 40 C: 280-770 kg/m3).  The liquid properties are the values of Table 4 of the paper
(20 C), as in the paper. Simplifications: the bulk flow caused by the change of total molar density
during mixing (Eq. 22 of the paper) is neglected in the transport (it is kept in sigma through
C(x)); the solute is not a component of the EoS; plain PR (no volume translation).
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os, warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import ive, kve
from scipy.optimize import brentq, minimize_scalar, least_squares
from scipy.linalg import solve_banded

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")       # numpy >= 2.0 / older numpy

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
RECOMPUTE  = True                   # False: skip the calculation, redraw the figure from DATA_FILE
DATA_FILE  = "Example10.4_data.npz"
T_K        = 333.15                 # K (60 C), temperature of the calculation
T_CAL      = 313.15                 # K (40 C), temperature of the results of Marra et al. used for the calibration
P_GRID     = np.arange(100., 241., 5.)   # bar (at 60 C: 290-760 kg/m3, inside the CO2 density range of the calibration)
C_MAP      = np.linspace(0., 50., 11)    # mg/mL, concentrations of the regime map (Panel B)
C_PLOT     = [0., 25., 50.]              # mg/mL, concentrations drawn in Panel A
D_NOZZLE   = 100e-6                 # m, injector diameter (Table 1 of the paper)
Q_LIQ      = 1.0                    # mL/min, liquid flow rate (Table 1)
C_AERO     = 0.175                  # aerodynamic reduction factor of Sterling and Sleicher (Eq. 2 of the paper)
# pure components: Tc/K, Pc/bar, acentric factor, M/(g/mol); liquids: density kg/m3, viscosity Pa s, surface tension N/m (Table 4
# of the paper, in air at 20 C), Le Bas molar volume cm3/mol (for Wilke-Chang).  CO2 and DMSO constants as in Rajasingam et al. (2004)
CO2  = dict(Tc=304.12, Pc=73.74, w=0.225, M=44.01, VA=34.0)
LIQ  = {"DMSO":    dict(Tc=726.0, Pc=56.5, w=0.2094, M=78.13, rho=1104., mu=1.996e-3, sig=43.54e-3, VA=84.8),
        "acetone": dict(Tc=508.1, Pc=47.0, w=0.307, M=58.08, rho=793.,  mu=0.3075e-3, sig=25.20e-3, VA=74.0)}
# PR binary parameters with CO2 (k12, l12).  DMSO: two-parameter fit of Rajasingam et al. (2004); one-parameter alternative (0.08, 0.0)
K12_CAL    = {"DMSO": (0.0505, -0.0472), "acetone": (0.005, 0.0)}   # 40 C: mean of the 308 and 318 K values
K12        = {"DMSO": (0.0496, -0.0484), "acetone": (0.005, 0.0)}   # 60 C: 318 K values (highest temperature of the fit)
# influence parameters, J m5 mol-2: CO2, DMSO and CO2-DMSO as given by Marra et al.; acetone from the Miqueu correlation (None), its
# cross parameter with CO2 reduced from the geometric mean by the same factor as the CO2-DMSO value of the paper
KAPPA      = {"CO2": 4.44e-21, "DMSO": 3.74e-19, "acetone": None, ("CO2", "DMSO"): 2.33e-20}
SIGMA_ZERO = 1e-6                   # N/m (1e-3 mN/m): dynamic tension regarded as vanished
# results of Marra et al. at 40 C, used for the calibration
P_TAB      = np.array([80., 120., 160.])                  # bar
C_TAB      = np.array([0., 10., 25., 50.])                # mg/mL YAc
T_JBU_TAB  = np.array([[7.8, 8.4, 9.2, 10.3], [5.0, 5.3, 5.7, 6.0], [3.2, 3.6, 4.0, 4.2]])       # ms, their Table 2
T_DSTV_TAB = np.array([[22., 23., 26.3, 34.], [5.1, 5.7, 7.2, 8.9], [2.9, 3.3, 3.7, 4.3]])      # ms, their Table 3
EXPERIMENTS = [(150., 15., "nano"), (150., 50., "micro")]  # 40 C: (P/bar, c/(mg/mL), observed morphology), their Fig. 1
W_RATIO    = 3.0                    # extra weight of ln(t/T) in the inference of the YAc-solution properties
N_X        = 121                    # composition grid of C(x) and Q(x)
N_XI       = 1500                   # grid points of the similarity solution
SAVE_FIG   = "figure10.5.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
cm = 1/2.54
R = 8.314462618

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define functions
- model: PR (phasepy), Sterling-Sleicher and Cahn-Hilliard/interdiffusion at temperature TK; returns
a namespace of fun
- compute
- crossover: first pressure at which t falls below T, and T there (nan: no cross-over in P)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def model(TK, k12):
    """PR (phasepy), Sterling-Sleicher and Cahn-Hilliard/interdiffusion at temperature TK; returns a namespace of functions"""
    from phasepy import component, mixture, preos
    from CoolProp.CoolProp import PropsSI
    warnings.filterwarnings("ignore")
    a = D_NOZZLE / 2
    u = Q_LIQ * 1e-6 / 60 / (np.pi * a**2)
    def comp(name, d):
        return component(name=name, Tc=d["Tc"], Pc=d["Pc"], w=d["w"], Mw=d["M"])
    co2 = comp("CO2", CO2)
    eos_co2 = preos(co2)
    rho_co2 = lambda P: eos_co2.density(TK, P, "V") * CO2["M"] * 1e3      # kg/m3 (phasepy: mol/cm3, P in bar)
    def miqueu(d):
        """influence parameter of the Miqueu et al. (2003) correlation for the PR EoS, J m5 mol-2"""
        Tc, Pc, w = d["Tc"], d["Pc"] * 1e5, d["w"]
        k = 0.37464 + 1.54226 * w - 0.26992 * w * w
        aa = 0.45724 * (R * Tc)**2 / Pc * (1 + k * (1 - np.sqrt(TK / Tc)))**2
        b = 0.07780 * R * Tc / Pc
        return (-1e-16 / (1.2326 + 1.3757 * w) * (1 - TK / Tc) + 1e-16 / (0.9051 + 1.5410 * w)) * aa * b**(2 / 3)
    XS = np.linspace(1e-4, 1 - 1e-4, N_X)

    class Binary:
        """CO2 (x) + liquid (1-x): PR EoS with k12 and l12, molar density C(x), influence-parameter matrix"""
        def __init__(self, name):
            d = LIQ[name]
            mix = mixture(co2, comp(name, d))
            k, l = k12[name]
            mix.kij_cubic(np.array([[0, k], [k, 0]]))
            eos = preos(mix)
            lij = np.array([[0, l], [l, 0]])
            def qmr_l12(X, RT, ai, bi, order, Kij):                        # phasepy's qmr with b12 = (1-l12)(b1+b2)/2
                ax = np.sqrt(np.outer(ai, ai)) * (1 - Kij) * X
                D_ = np.sum(ax.T * X)
                bij = 0.5 * (bi[:, None] + bi[None, :]) * (1 - lij)
                B = X @ bij @ X
                if order == 0:
                    return D_, B
                return D_, 2 * np.sum(ax, axis=1), B, 2 * (bij @ X) - B
            eos.mixrule = qmr_l12
            self.eos = eos
            self.d = d
            k1 = KAPPA["CO2"]
            k2 = KAPPA[name] if KAPPA.get(name) else miqueu(d)
            red = KAPPA[("CO2", "DMSO")] / np.sqrt(KAPPA["CO2"] * KAPPA["DMSO"])
            self.kap = np.array([[k1, red * np.sqrt(k1 * k2)], [red * np.sqrt(k1 * k2), k2]])
            self.cq = {}
        def stable(self, x, P):
            """(sum x ln phi, molar volume cm3/mol) of the PR root of lowest Gibbs energy at x_CO2 = x"""
            X = np.array([x, 1 - x])
            best = None
            for st in ("L", "V"):
                try:
                    lnphi, v = self.eos.logfugef(X, TK, P, st)
                    g = np.sum(X * lnphi)
                    if best is None or g < best[0]:
                        best = (g, v)
                except Exception:
                    pass
            return best
        def Q(self, P):
            """gradient-theory weight Q(x) = sum_ij kappa_ij dC_i/dx dC_j/dx in J/m (cached)"""
            if P not in self.cq:
                C = np.array([1e6 / self.stable(x, P)[1] for x in XS])
                Ci = np.column_stack([XS * C, (1 - XS) * C])
                dCi = np.gradient(Ci, XS, axis=0)
                self.cq[P] = np.einsum("ki,ij,kj->k", dCi, self.kap, dCi)
            return self.cq[P]
        def gamma_min(self, P, xs=np.linspace(0.5, 0.995, 60), h=1e-3):
            gr = lambda x: self.stable(x, P)[0]
            return min(1 + x * (1 - x) * (gr(x + h) - 2 * gr(x) + gr(x - h)) / h**2 for x in xs)
        def mcp(self):
            return brentq(self.gamma_min, 70., 350., xtol=0.2)

    I01 = lambda z: ive(0, z) / ive(1, z)
    K01 = lambda z: kve(0, z) / kve(1, z)
    def growth_rate(eps, rL, mu, sig, rV):
        F1 = eps * I01(eps) / 2 + rV * eps * K01(eps) / (2 * rL)
        F4 = sig / (2 * rL * a**3) * (1 - eps**2) * eps**2
        F5 = u**2 * rV * eps**3 * K01(eps) / (2 * rL * a**2)
        rhs = F4 + C_AERO * F5
        def g(b):
            e1 = np.sqrt(eps**2 + b * a**2 * rL / mu)
            F3 = mu * eps**2 / (rL * a**2) * (2 * eps * I01(eps) - 1 + 2 * eps**2 / (e1**2 - eps**2) * (eps * I01(eps) - e1 * I01(e1)))
            return b * b * F1 + b * F3 - rhs
        bmax = 1.01 * np.sqrt(rhs / F1)
        return brentq(g, 1e-10 * bmax, bmax)
    def beta_star(P, rL, mu, sig):
        rV = rho_co2(P)
        r = minimize_scalar(lambda e: -growth_rate(e, rL, mu, sig, rV), bounds=(0.02, 0.999), method="bounded", options={"xatol": 1e-5})
        return -r.fun                                                         # 1/s
    wilke_chang = lambda MB, muB, VA: 7.4e-8 * np.sqrt(MB) * TK / (muB * 1e3 * VA**0.6) * 1e-4      # m2/s
    def S_factor(binary, P, mu_liq):
        """S in gamma(t) = S/sqrt(t) (J m-2 s^1/2) for D = D_Vignes(x) (phi = 1)"""
        d = binary.d
        Q = binary.Q(P)
        Dl = wilke_chang(d["M"], mu_liq, CO2["VA"])                           # CO2 in the jet liquid
        Dg = wilke_chang(CO2["M"], PropsSI("V", "T", TK, "P", P * 1e5, "CO2"), d["VA"])   # solvent in CO2
        D = lambda x: Dl**(1 - x) * Dg**x
        xi = np.linspace(-10 * np.sqrt(Dl), 10 * np.sqrt(Dg), N_XI)
        h = xi[1] - xi[0]
        x = (xi > 0).astype(float)
        dtau = 0.05                               # pseudo-time march of (D x')' + (xi/2) x' = 0
        for it in range(5000):
            Df = D(0.5 * (x[1:] + x[:-1]))
            A = Df[:-1] / h**2
            B = Df[1:] / h**2
            v = xi[1:-1] / (4 * h)
            ab = np.zeros((3, N_XI))
            ab[1] = 1.0
            ab[1, 1:-1] = 1 + dtau * (A + B)
            ab[0, 2:] = -dtau * (B + v)
            ab[2, :-2] = -dtau * (A - v)
            rhs = x.copy()
            rhs[0], rhs[-1] = 0.0, 1.0
            xn = solve_banded((1, 1), ab, rhs)
            if np.max(abs(xn - x)) < 1e-9:
                x = xn
                break
            x = xn
            dtau = min(1.2 * dtau, 50.)
        return trapz(np.interp(x, XS, Q) * np.gradient(x, xi)**2, xi)
    class M:
        pass
    m = M()
    m.rho_co2, m.Binary, m.beta_star, m.S_factor = rho_co2, Binary, beta_star, S_factor
    return m

def compute():
    # ---------------- calibration at 40 C on the results of Marra et al. ----------------
    mc = model(T_CAL, K12_CAL)
    dmso_c = mc.Binary("DMSO")
    L = LIQ["DMSO"]
    rho_tab = np.array([mc.rho_co2(P) for P in P_TAB])
    Lam_tab = T_JBU_TAB[:, 0] * 1e-3 * np.array([mc.beta_star(P, L["rho"], L["mu"], L["sig"]) for P in P_TAB])
    phi_tab = (np.array([mc.S_factor(dmso_c, P, L["mu"]) for P in P_TAB]) / SIGMA_ZERO)**2 / (T_DSTV_TAB[:, 0] * 1e-3)
    def of_rho(ytab):
        """ln y = A + B exp(rho/rho0) through the three calibration points: smooth and monotone in the CO2 density"""
        r, ly = rho_tab, np.log(ytab)
        q = (ly[2] - ly[1]) / (ly[1] - ly[0])
        g = lambda r0: (np.exp(r[2] / r0) - np.exp(r[1] / r0)) / (np.exp(r[1] / r0) - np.exp(r[0] / r0)) - q
        r0 = brentq(g, 20., 1e6)
        B = (ly[1] - ly[0]) / (np.exp(r[1] / r0) - np.exp(r[0] / r0))
        A = ly[0] - B * np.exp(r[0] / r0)
        return lambda rho: float(np.exp(A + B * np.exp(rho / r0)))
    Lam, phi = of_rho(Lam_tab), of_rho(phi_tab)
    print(f"calibration at {T_CAL - 273.15:.0f} C, P = {P_TAB} bar, rho_CO2 = {np.round(rho_tab)} kg/m3:")
    print("  ln(a/eta0) =", np.round(Lam_tab, 1), "  phi =", np.round(phi_tab, 2))
    Tc_ = lambda m, P, liq, mu, sig: Lam(m.rho_co2(P)) / m.beta_star(P, liq["rho"], mu, sig) * 1e3       # ms
    tc_ = lambda m, b, P, mu: (m.S_factor(b, P, mu) / SIGMA_ZERO)**2 / phi(m.rho_co2(P)) * 1e3            # ms
    MU_C, SIG_C = [L["mu"]], [L["sig"]]                                       # YAc solutions from Tables 2 and 3
    for j in range(1, len(C_TAB)):
        def res(p):
            lT = np.array([np.log(Tc_(mc, P, L, np.exp(p[0]), np.exp(p[1])) / T_JBU_TAB[i, j]) for i, P in enumerate(P_TAB)])
            lt = np.array([np.log(tc_(mc, dmso_c, P, np.exp(p[0])) / T_DSTV_TAB[i, j]) for i, P in enumerate(P_TAB)])
            return np.concatenate([lT, lt, W_RATIO * (lt - lT)])
        p = least_squares(res, [np.log(MU_C[-1]), np.log(SIG_C[-1])], diff_step=1e-3, xtol=1e-6).x
        MU_C.append(np.exp(p[0]))
        SIG_C.append(np.exp(p[1]))
    MU_C, SIG_C = np.array(MU_C), np.array(SIG_C)
    pm, ps = np.polyfit(C_TAB, np.log(MU_C), 2), np.polyfit(C_TAB, np.log(SIG_C), 2)
    mu_of_c = lambda c: np.exp(np.polyval(pm, c))
    sig_of_c = lambda c: np.exp(np.polyval(ps, c))
    print("  inferred YAc-solution viscosity / mPa s:", np.round(MU_C * 1e3, 2), " surface tension / (mN/m):", np.round(SIG_C * 1e3, 1))
    check = np.array([[[Tc_(mc, P, L, mu_of_c(c), sig_of_c(c)), tc_(mc, dmso_c, P, mu_of_c(c))] for P in P_TAB] for c in C_TAB])
    print("  check at 40 C (model / paper), T and t in ms:")
    for j, c in enumerate(C_TAB):
        print(f"   {c:4.0f} mg/mL: " + "  ".join(f"{P:.0f} bar T {check[j, i, 0]:5.2f}/{T_JBU_TAB[i, j]:4.1f} t {check[j, i, 1]:5.2f}/{T_DSTV_TAB[i, j]:4.1f}" for i, P in enumerate(P_TAB)))
    for P, c, obs in EXPERIMENTS:
        pr = "micro" if Tc_(mc, P, L, mu_of_c(c), sig_of_c(c)) < tc_(mc, dmso_c, P, mu_of_c(c)) else "nano"
        print(f"   experiment {P:.0f} bar, {c:.0f} mg/mL (40 C): observed {obs}, predicted {pr}")

    # ---------------- calculation at T_K ----------------
    m = model(T_K, K12)
    dmso = m.Binary("DMSO")
    P_mcp = dmso.mcp()
    C_RUN = np.array(sorted(set(list(C_PLOT) + list(C_MAP))))
    TT = np.array([[Tc_(m, P, L, mu_of_c(c), sig_of_c(c)) for P in P_GRID] for c in C_RUN])
    tt = np.array([[tc_(m, dmso, P, mu_of_c(c)) for P in P_GRID] for c in C_RUN])
    rho = np.array([m.rho_co2(P) for P in P_GRID])
    return dict(P_GRID=P_GRID, rho=rho, C_RUN=C_RUN, TT=TT, tt=tt, P_mcp=P_mcp,
                Lam_tab=Lam_tab, phi_tab=phi_tab, rho_tab=rho_tab, MU_C=MU_C, SIG_C=SIG_C, check40=check)

def crossover(P, T, t):
    """first pressure at which t falls below T, and T there (nan: no cross-over in P)"""
    dd = np.log(t) - np.log(T)
    i = np.where((dd[:-1] > 0) & (dd[1:] <= 0))[0]
    if not len(i):
        return np.nan, np.nan
    i = i[0]
    f = dd[i] / (dd[i] - dd[i + 1])
    Px = P[i] + f * (P[i + 1] - P[i])
    return Px, np.exp(np.interp(Px, P, np.log(T)))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

if RECOMPUTE:
    data = compute()
    np.savez(os.path.join(HERE, DATA_FILE), **data)
    print("saved", DATA_FILE)
else:
    data = dict(np.load(os.path.join(HERE, DATA_FILE)))
Pg, C_RUN, TT, tt, Pm = data["P_GRID"], data["C_RUN"], data["TT"], data["tt"], float(data["P_mcp"])
row = lambda c: int(np.argmin(abs(C_RUN - c)))
print(f"\nat {T_K - 273.15:.0f} C: binary MCP (PR) DMSO-CO2 {Pm:.1f} bar")
print("  c/(mg/mL)   P_x/bar   t_x/ms")
for c in C_PLOT:
    Px, tx = crossover(Pg, TT[row(c)], tt[row(c)])
    print(f"  {c:6.0f}   {Px:8.1f}   {tx:6.2f}")
for P in (150., 200., 240.):
    k = np.argmin(abs(Pg - P))
    print(f"  {P:.0f} bar: DMSO T {TT[0][k]:.2f} t {tt[0][k]:.2f} ms")

# ---------------- figure (uses only the arrays in data) ----------------
TC = f"{T_K:.0f} K"
fig, axs = plt.subplots(1, 2, figsize=(16 * cm, 8 * cm))
cols = {0.: "k", 10.: "tab:blue", 25.: "tab:green", 50.: "tab:red"}
ax = axs[0]
for c in C_PLOT:
    ax.plot(Pg, TT[row(c)], "-", c=cols[c], label=f"{c:.0f} mg mL$^{{-1}}$")
    ax.plot(Pg, tt[row(c)], "--", c=cols[c])
    Px, tx = crossover(Pg, TT[row(c)], tt[row(c)])
    if np.isfinite(Px):
        ax.plot(Px, tx, "o", c=cols[c], ms=4, zorder=5)
ax.plot([], [], "k-", label=r"$\tau_\mathrm{j}$ (jet break-up)")
ax.plot([], [], "k--", label=r"$\tau_\gamma$ (DSTV)")
ax.axvline(Pm, c="gray", lw=.4, ls=":")
#ax.set_yscale("log"); ax.set_ylim(1.5, 40); ax.set_xlim(Pg[0], Pg[-1])
ax.set_yticks([2, 5, 10, 20])
ax.set_yticklabels(["2", "5", "10", "20"])
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_xlabel("$p$ / bar")
ax.set_ylabel("characteristic time / ms")
ax.legend(frameon=False, fontsize=5, loc="upper right")
ax.set_xlim([140,240])
ax.set_ylim([2,15])
ax.text(0.03, 0.95, f"YAc/DMSO, {TC}; circles:"+r" $\tau_\mathrm{j}=\tau_\gamma$",
        transform=ax.transAxes, fontsize=7)
ax.text(Pm-1,ax.get_ylim()[0]+0.5,"MCP",
        horizontalalignment='right',
        fontsize=7,fontweight='bold')
ax = axs[1]                                                                   # regime map: t/T on a (P, c) grid
rm = np.array([row(c) for c in C_MAP])
lnr = np.log(tt[rm] / TT[rm])
from scipy.interpolate import RectBivariateSpline                            # bicubic interpolation of ln(t/T) for smooth contours
Pf, Cf = np.linspace(Pg[0], Pg[-1], 300), np.linspace(C_MAP[0], C_MAP[-1], 300)
ratio = np.exp(RectBivariateSpline(C_MAP, Pg, lnr, kx=3, ky=3)(Cf, Pf))
ax.contourf(Pf, Cf, ratio, levels=[0, 1.0, 1e3], colors=["#cfe3f5", "#f7e0c0"])          # blue: t < T (DSTV), orange: t > T (jet break-up)
ax.contour(Pf, Cf, ratio, levels=[1.0], colors="k", linewidths=.75)
ax.axvline(Pm, c="gray", lw=.4, ls=":")
ax.text(Pm-1,ax.get_ylim()[0]+2.5,"MCP",
        horizontalalignment='right',
        fontsize=7,fontweight='bold')
ax.plot([], [], "k-", label=r"$\tau_\mathrm{j}=\tau_\gamma$")
#ax.text(Pg[0] + 4, 44, "$T<t$: jet break-up\nmicroparticles", fontsize=5); ax.text(Pg[-1] - 4, 3, "$T>t$: DSTV\nnanoparticles", fontsize=5, ha="right")
#ax.text(Pg[0] + 4, 36, "gray: $T$ and $t$ within 5 %", fontsize=5, color="dimgray")
ax.set_xlim(Pg[0], Pg[-1])
ax.set_ylim(0, 50)
ax.set_xlabel("$p$ / bar")
ax.set_ylabel("$c$ (YAc) / (mg mL$^{-1}$)")
ax.text(0.05,0.9,r"$\tau_\mathrm{j}<\tau_\gamma$ (Microparticles)",
        transform=ax.transAxes)
ax.text(0.95,0.05,r"$\tau_\mathrm{j}>\tau_\gamma$ (Nanoparticles)",
        horizontalalignment='right',
        transform=ax.transAxes)
ax.legend(frameon=False, fontsize=5, loc="upper right")
ax.set_xlim([140,240])
for a_, Lb in zip(axs, "AB"):
    a_.text(-0.1, 0.95, Lb, transform=a_.transAxes, weight="bold", fontsize=8, ha="right")
fig.tight_layout()
plt.show()
fig.savefig(os.path.join(HERE, SAVE_FIG), dpi=300)
print("saved", SAVE_FIG)
