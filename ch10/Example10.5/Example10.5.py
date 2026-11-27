"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 10 Particle Formation from Supercritical Solutions
Example 10.5
Feed-solvent composition and the cross-over: acetone/DMSO mixtures: In this example, we use the
model and the calibration of Example 10.4 to see how the cross-over moves at 60 ^degC when part of
the DMSO in the feed solution is replaced by acetone, without solute. The feed is specified by the
volume fraction v_AC of acetone in the solvent. Its density follows from ideal volume mixing, its
viscosity from =_i x_i_i, and its surface tension is taken as linear in the mole fractions. The
ternary CO2-DMSO-acetone is followed along the dilution line of the feed, i.e. the two solvents are
assumed to interdiffuse with CO2 alike, so that the composition of the interfacial region is x_CO2=s
with the solvents in the feed ratio.

No command-line arguments: edit the SETTINGS block below and run   python Example10.5.py
Produces Fig. 10.6 and the data file Example10.5_data.npz (all arrays drawn in the figure).  With
RECOMPUTE = False the script only reads the data file and redraws the figure, so the plotting style
can be changed without repeating the calculation.
Requires: numpy, scipy, matplotlib, phasepy, CoolProp (CO2 viscosity only).

The model is the one of Example 10.4 (Marra, De Marco, Reverchon, Chem. Eng. Sci. 71, 39 (2012)) at
60 C, with the same calibration: ln(a/eta0) and the effective-diffusivity factor phi are fixed by
the pure-DMSO times of Tables 2 and 3 of the paper at 40 C, carried to 60 C as functions of the CO2
density, and kept for every feed (the first is a property of the disturbance by the dense gas, the
second of the transport model).  What changes with the volume fraction v of acetone in the feed
solvent (no solute) is computed:
- jet: density (ideal volume mixing), viscosity (ln mu = sum x_i ln mu_i) and surface tension
(linear in the mole fractions) of the
    mixed solvent enter the Sterling-Sleicher dispersion relation;
- interface: the ternary CO2 + DMSO + acetone is treated along the dilution line of the feed (x_CO2
= s, the two solvents in the feed
ratio), i.e. the two solvents are assumed to interdiffuse with CO2 alike.  The Peng-Robinson EoS
(phasepy; k_ij and l_ij, CO2-DMSO
from the two-parameter fit of Rajasingam et al., J. Supercrit. Fluids 31, 227 (2004)) gives the
molar density C(s) for the
Cahn-Hilliard weight Q(s) = sum_ij kappa_ij dC_i/ds dC_j/ds (3x3 influence matrix); D(s)
interpolates (Vignes) between Wilke-Chang
    values of CO2 in the mixed solvent and of the solvents in CO2.
- mixture critical point of the feed: the lowest pressure at which every composition on the dilution
line is outside the spinodal
of the ternary (Hessian of the Gibbs energy of mixing positive definite), an estimate of the ternary
MCP for this feed.  Below it
the jet keeps an equilibrium interface, the tension never vanishes, and break-up wins, so the
physical cross-over pressure is
max(P_x, P_MCP). Influence parameters: CO2, DMSO and CO2-DMSO as in Marra et al.; acetone from the
Miqueu correlation; CO2-acetone reduced from the geometric mean as CO2-DMSO; DMSO-acetone geometric
mean.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os, warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import ive, kve
from scipy.optimize import brentq, minimize_scalar
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
DATA_FILE  = "Example10.5_data.npz"
T_K        = 333.15                 # K (60 C), temperature of the calculation
T_CAL      = 313.15                 # K (40 C), temperature of the results of Marra et al. used for the calibration
D_NOZZLE   = 100e-6                 # m, injector diameter
Q_LIQ      = 1.0                    # mL/min, liquid flow rate
C_AERO     = 0.175                  # aerodynamic reduction factor of Sterling and Sleicher
P_GRID     = np.arange(100., 241., 5.)                 # bar (at 60 C: 290-760 kg/m3, inside the CO2 density range of the calibration)
V_CURVES   = [0.0, 0.25, 0.5, 0.75, 1.0]               # acetone volume fraction in the feed solvent, curves of Panel A
V_MAP      = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])                # ... further feeds for the regime map (Panel B)
P_PRINT    = [140., 180., 220.]                        # bar, pressures of the printed table of tau_j and tau_gamma against v_AC
P_REF      = 180.                                      # bar, pressure of the printed comparison of the feeds
CO2  = dict(Tc=304.12, Pc=73.74, w=0.225, M=44.01, VA=34.0)
LIQ  = {"DMSO":    dict(Tc=726.0, Pc=56.5, w=0.2094, M=78.13, rho=1104., mu=1.996e-3, sig=43.54e-3, VA=84.8),
        "acetone": dict(Tc=508.1, Pc=47.0, w=0.307, M=58.08, rho=793.,  mu=0.3075e-3, sig=25.20e-3, VA=74.0)}
SOLVENTS   = ["DMSO", "acetone"]
# PR binary parameters (k_ij, l_ij); CO2-DMSO: Rajasingam et al. (2004), two-parameter fit (40 C: mean of 308 and 318 K; 60 C: 318 K)
KIJ_CAL    = {("CO2", "DMSO"): (0.0505, -0.0472), ("CO2", "acetone"): (0.005, 0.0), ("DMSO", "acetone"): (0.0, 0.0)}
KIJ        = {("CO2", "DMSO"): (0.0496, -0.0484), ("CO2", "acetone"): (0.005, 0.0), ("DMSO", "acetone"): (0.0, 0.0)}
KAPPA      = {"CO2": 4.44e-21, "DMSO": 3.74e-19, "acetone": None, ("CO2", "DMSO"): 2.33e-20}   # J m5 mol-2 (None: Miqueu)
SIGMA_ZERO = 1e-6                   # N/m, dynamic tension regarded as vanished (as in Example 10.4)
P_TAB      = np.array([80., 120., 160.])                # calibration at 40 C: pure-DMSO times of Marra et al. (Tables 2, 3; c = 0)
T_JBU_DMSO = np.array([7.8, 5.0, 3.2])                  # ms
T_DSTV_DMSO = np.array([22., 5.1, 2.9])                 # ms
AC_FIG7_T  = [8.8, 4.8, 2.6]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Set constants and grids
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
AC_FIG7_t = [9.6, 2.6, 1.4]   # ms, pure acetone at 80, 120, 160 bar, read from Fig. 7 of the paper (check only)
N_X        = 121                    # composition grid of C(s) and Q(s)
N_XI       = 1500                   # grid points of the similarity solution
SAVE_FIG   = "figure10.6.pdf"
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
cm = 1/2.54
R = 8.314462618
V_ALL = np.array(sorted(set(list(V_MAP) + list(V_CURVES))))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Define functions
- model: PR (ternary, phasepy), Sterling-Sleicher and Cahn-Hilliard/interdiffusion at temperature TK
- compute
- crossover
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def model(TK, KIJ_T):
    """PR (ternary, phasepy), Sterling-Sleicher and Cahn-Hilliard/interdiffusion at temperature TK"""
    from phasepy import component, mixture, preos
    from CoolProp.CoolProp import PropsSI
    warnings.filterwarnings("ignore")
    a = D_NOZZLE / 2
    u = Q_LIQ * 1e-6 / 60 / (np.pi * a**2)
    NAMES = ["CO2"] + SOLVENTS
    PROPS = [CO2] + [LIQ[s] for s in SOLVENTS]

    # ---------------- Peng-Robinson (phasepy), ternary CO2 + DMSO + acetone, quadratic covolume ----------------
    comps = [component(name=n, Tc=d["Tc"], Pc=d["Pc"], w=d["w"], Mw=d["M"]) for n, d in zip(NAMES, PROPS)]
    mix = mixture(comps[0], comps[1])
    mix.add_component(comps[2])
    K = np.zeros((3, 3))
    Lij = np.zeros((3, 3))
    for (i, j), (k, l) in KIJ_T.items():
        i, j = NAMES.index(i), NAMES.index(j)
        K[i, j] = K[j, i] = k
        Lij[i, j] = Lij[j, i] = l
    mix.kij_cubic(K)
    eos = preos(mix)
    def qmr_l12(X, RT, ai, bi, order, Kij):                                   # phasepy's qmr with b_ij = (1-l_ij)(b_i+b_j)/2
        ax = np.sqrt(np.outer(ai, ai)) * (1 - Kij) * X
        D_ = np.sum(ax.T * X)
        bij = 0.5 * (bi[:, None] + bi[None, :]) * (1 - Lij)
        B = X @ bij @ X
        if order == 0:
            return D_, B
        return D_, 2 * np.sum(ax, axis=1), B, 2 * (bij @ X) - B
    eos.mixrule = qmr_l12
    eos_co2 = preos(comps[0])
    rho_co2 = lambda P: eos_co2.density(TK, P, "V") * CO2["M"] * 1e3     # kg/m3
    def miqueu(d):
        Tc, Pc, w = d["Tc"], d["Pc"] * 1e5, d["w"]
        k = 0.37464 + 1.54226 * w - 0.26992 * w * w
        aa = 0.45724 * (R * Tc)**2 / Pc * (1 + k * (1 - np.sqrt(TK / Tc)))**2
        b = 0.07780 * R * Tc / Pc
        return (-1e-16 / (1.2326 + 1.3757 * w) * (1 - TK / Tc) + 1e-16 / (0.9051 + 1.5410 * w)) * aa * b**(2 / 3)
    kp = np.array([KAPPA[n] if KAPPA.get(n) else miqueu(d) for n, d in zip(NAMES, PROPS)])
    KAP = np.sqrt(np.outer(kp, kp))
    red = KAPPA[("CO2", "DMSO")] / np.sqrt(KAPPA["CO2"] * KAPPA["DMSO"])
    KAP[0, 1:] *= red
    KAP[1:, 0] *= red
    SS = np.linspace(1e-4, 1 - 1e-4, N_X)

    class Feed:
        """feed solvent of acetone volume fraction v; s = x_CO2 along the dilution line"""
        def __init__(self, v):
            d1, d2 = LIQ["DMSO"], LIQ["acetone"]
            n = np.array([(1 - v) * d1["rho"] / d1["M"], v * d2["rho"] / d2["M"]])
            z = n / n.sum()
            self.z = z
            self.v = v
            self.M = z @ [d1["M"], d2["M"]]
            self.rho = self.M / (z[0] * d1["M"] / d1["rho"] + z[1] * d2["M"] / d2["rho"])   # ideal volume mixing
            self.mu = np.exp(z @ np.log([d1["mu"], d2["mu"]]))
            self.sig = z @ [d1["sig"], d2["sig"]]
            self.cq = {}
        def X(self, s):
            return np.concatenate([[s], (1 - s) * self.z])
        def gres(self, X, P):
            """(sum x ln phi, molar volume cm3/mol) of the PR root of lowest Gibbs energy"""
            best = None
            for st in ("L", "V"):
                try:
                    lnphi, vm = eos.logfugef(X, TK, P, st)
                    g = np.sum(X * lnphi)
                    if best is None or g < best[0]:
                        best = (g, vm)
                except Exception:
                    pass
            return best
        def Q(self, P):
            if P not in self.cq:
                C = np.array([1e6 / self.gres(self.X(s), P)[1] for s in SS])
                Ci = np.array([self.X(s) for s in SS]) * C[:, None]
                dCi = np.gradient(Ci, SS, axis=0)
                self.cq[P] = np.einsum("ki,ij,kj->k", dCi, KAP, dCi)
            return self.cq[P]
        def stability_min(self, P, ss=np.linspace(0.6, 0.99, 30), h=1e-3):
            """smallest value along the dilution line of det(H) x1 x2 x3, H = Hessian of g/RT in (x_CO2, x_AC): 1 for an ideal
            mixture, 0 on the spinodal.  For a binary feed (v = 0 or 1) it is the thermodynamic factor Gamma."""
            out = []
            G0 = lambda X: self.gres(X, P)[0]
            for s in ss:
                if self.v in (0.0, 1.0):
                    gr = lambda q: G0(self.X(q))
                    out.append(1 + s * (1 - s) * (gr(s + h) - 2 * gr(s) + gr(s - h)) / h**2)
                    continue
                x1, x3, x2 = self.X(s)                                        # CO2, DMSO, acetone
                h1, h2 = min(h, 0.3 * x1, 0.3 * x3), min(h, 0.3 * x2, 0.3 * x3)
                G = lambda d1, d2: G0(np.array([x1 + d1, x3 - d1 - d2, x2 + d2]))
                g0 = G(0, 0)
                H11 = (G(h1, 0) - 2 * g0 + G(-h1, 0)) / h1**2
                H22 = (G(0, h2) - 2 * g0 + G(0, -h2)) / h2**2
                H12 = (G(h1, h2) - G(h1, -h2) - G(-h1, h2) + G(-h1, -h2)) / (4 * h1 * h2)
                H11 += 1 / x1 + 1 / x3
                H22 += 1 / x2 + 1 / x3
                H12 += 1 / x3                # ideal-mixing part
                out.append((H11 * H22 - H12**2) * x1 * x2 * x3)
            return min(out)
        def mcp(self):
            return brentq(self.stability_min, 70., 350., xtol=0.2)

    # ---------------- jet break-up (Sterling-Sleicher) ----------------
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
    def beta_star(P, f):
        rV = rho_co2(P)
        r = minimize_scalar(lambda e: -growth_rate(e, f.rho, f.mu, f.sig, rV), bounds=(0.02, 0.999), method="bounded", options={"xatol": 1e-5})
        return -r.fun

    # ---------------- dynamic surface tension (self-similar interdiffusion + Cahn-Hilliard) ----------------
    wilke_chang = lambda MB, muB, VA: 7.4e-8 * np.sqrt(MB) * TK / (muB * 1e3 * VA**0.6) * 1e-4      # m2/s
    def S_factor(f, P):
        Q = f.Q(P)
        Dl = wilke_chang(f.M, f.mu, CO2["VA"])
        muc = PropsSI("V", "T", TK, "P", P * 1e5, "CO2")
        Dg = f.z @ [wilke_chang(CO2["M"], muc, LIQ[s]["VA"]) for s in SOLVENTS]
        D = lambda s: Dl**(1 - s) * Dg**s
        xi = np.linspace(-10 * np.sqrt(Dl), 10 * np.sqrt(Dg), N_XI)
        h = xi[1] - xi[0]
        x = (xi > 0).astype(float)
        dtau = 0.05
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
        return trapz(np.interp(x, SS, Q) * np.gradient(x, xi)**2, xi)

    class M:
        pass
    m = M()
    m.Feed, m.beta_star, m.S_factor, m.rho_co2 = Feed, beta_star, S_factor, rho_co2
    return m

def compute():
    # ---------------- calibration at 40 C on the pure-DMSO results of Marra et al. (as in Example 10.4) ----------------
    mc = model(T_CAL, KIJ_CAL)
    f0 = mc.Feed(0.0)
    rho_tab = np.array([mc.rho_co2(P) for P in P_TAB])
    def of_rho(ytab):
        """ln y = A + B exp(rho/rho0) through the three calibration points: smooth and monotone in the CO2 density"""
        r, ly = rho_tab, np.log(ytab)
        q = (ly[2] - ly[1]) / (ly[1] - ly[0])
        g = lambda r0: (np.exp(r[2] / r0) - np.exp(r[1] / r0)) / (np.exp(r[1] / r0) - np.exp(r[0] / r0)) - q
        r0 = brentq(g, 20., 1e6)
        B = (ly[1] - ly[0]) / (np.exp(r[1] / r0) - np.exp(r[0] / r0))
        A = ly[0] - B * np.exp(r[0] / r0)
        return lambda rho: float(np.exp(A + B * np.exp(rho / r0)))
    Lam = of_rho(T_JBU_DMSO * 1e-3 * np.array([mc.beta_star(P, f0) for P in P_TAB]))
    phi = of_rho((np.array([mc.S_factor(f0, P) for P in P_TAB]) / SIGMA_ZERO)**2 / (T_DSTV_DMSO * 1e-3))
    fa = mc.Feed(1.0)                                                         # check: pure acetone at 40 C against Fig. 7 of the paper
    check_ac = np.array([[Lam(mc.rho_co2(P)) / mc.beta_star(P, fa) * 1e3, (mc.S_factor(fa, P) / SIGMA_ZERO)**2 / phi(mc.rho_co2(P)) * 1e3] for P in P_TAB])
    print("pure acetone at 40 C, model / Marra et al. Fig. 7 (read), ms:")
    for P, (Tm, tm), Tp, tp in zip(P_TAB, check_ac, AC_FIG7_T, AC_FIG7_t):
        print(f"  {P:.0f} bar   T {Tm:5.2f} / {Tp:4.1f}   t {tm:5.2f} / {tp:4.1f}")

    # ---------------- feeds at T_K ----------------
    m = model(T_K, KIJ)
    feeds = {v: m.Feed(v) for v in V_ALL}
    T = np.array([[Lam(m.rho_co2(P)) / m.beta_star(P, feeds[v]) * 1e3 for P in P_GRID] for v in V_ALL])                  # ms
    t = np.array([[(m.S_factor(feeds[v], P) / SIGMA_ZERO)**2 / phi(m.rho_co2(P)) * 1e3 for P in P_GRID] for v in V_ALL])  # ms
    P_mcp = np.array([feeds[v].mcp() for v in V_ALL])
    props = np.array([[feeds[v].z[1], feeds[v].rho, feeds[v].mu, feeds[v].sig] for v in V_ALL])
    return dict(V_ALL=V_ALL, P_GRID=P_GRID, T=T, t=t, P_mcp=P_mcp, props=props, check_ac40=check_ac)

def crossover(P, T, t):
    dd = np.log(t) - np.log(T)
    if dd[0] <= 0:
        return P[0]                                                # t < T already at the lowest pressure
    i = np.where((dd[:-1] > 0) & (dd[1:] <= 0))[0]
    if not len(i):
        return np.nan
    i = i[0]
    return P[i] + dd[i] / (dd[i] - dd[i + 1]) * (P[i + 1] - P[i])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

if RECOMPUTE:
    data = compute()
    np.savez(os.path.join(HERE, DATA_FILE), **data)
    print("saved", DATA_FILE)
else:
    data = dict(np.load(os.path.join(HERE, DATA_FILE)))
VA, Pg, TT, tt, Pm = data["V_ALL"], data["P_GRID"], data["T"], data["t"], data["P_mcp"]
row = lambda v: int(np.argmin(abs(VA - v)))
Px = np.array([crossover(Pg, TT[k], tt[k]) for k in range(len(VA))])
Pphys = np.fmax(Px, Pm)

print(" v_AC  x_AC   rho     mu/mPa s  sigma/(mN/m)   P_MCP/bar   P_x(model)/bar   P_x = max(P_x, P_MCP)   t/T at P_REF")
k120 = np.argmin(abs(Pg - P_REF))
for k, v in enumerate(VA):
    zA, rho, mu, sig = data["props"][k]
    print(f"{v:5.2f} {zA:5.3f} {rho:6.0f} {mu*1e3:8.3f} {sig*1e3:10.1f} {Pm[k]:12.1f} {Px[k]:14.1f} {Pphys[k]:18.1f} {tt[k][k120]/TT[k][k120]:16.2f}")
print("\n contributions at P_REF, relative to pure DMSO:  v   T/T_DMSO   t/t_DMSO")
for v in V_CURVES:
    print(f"   {v:4.2f}   {TT[row(v)][k120]/TT[0][k120]:6.2f}   {tt[row(v)][k120]/tt[0][k120]:6.2f}")
print("\n tau_j / tau_gamma in ms against v_AC:")
for P in P_PRINT:
    k = np.argmin(abs(Pg - P))
    print(f"  {P:.0f} bar: " + "  ".join(f"v={v:.2f}: {TT[row(v)][k]:.2f}/{tt[row(v)][k]:.2f}" for v in V_CURVES))

# ---------------- figure (uses only the arrays in data) ----------------
fig, axs = plt.subplots(1, 2, figsize=(16 * cm, 8 * cm))
cmap = plt.get_cmap("viridis")
col = lambda v: cmap(0.85 * v)
ax = axs[0]
for v in V_CURVES:
    k = row(v)
    ax.plot(Pg, TT[k], "-", c=col(v), label=f"$v_\\mathrm{{AC}}$ = {v:.2f}")
    ax.plot(Pg, tt[k], "--", c=col(v))
    if np.isfinite(Px[k]) and Px[k] > Pg[0]:
        ax.plot(Px[k], np.exp(np.interp(Px[k], Pg, np.log(TT[k]))), "o", c=col(v), 
                mew=.4, ms=4, zorder=5)
ax.plot([], [], "k-", label=r"$\tau_\mathrm{j}$")
ax.plot([], [], "k--", label=r"$\tau_\gamma$")
ax.set_yscale("log")
ax.set_ylim(0.5, 30)
ax.set_xlim(Pg[0], 200)
ax.set_yticks([0.5, 1, 2, 5, 10, 20])
ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_xlabel("$p$ / bar")
ax.set_ylabel("Characteristic time / ms")
ax.legend(frameon=False, fontsize=5, loc="upper right", ncol=2)
ax.text(0.03, 0.04, r"circles: $\tau_\mathrm{j}=\tau_\gamma$", transform=ax.transAxes, fontsize=5)
ax = axs[1]                                                                   # regime map, smooth: bicubic spline of ln(tau_gamma/tau_j)
from scipy.interpolate import RectBivariateSpline, PchipInterpolator
vf, Pf = np.linspace(0, 1, 300), np.linspace(Pg[0], Pg[-1], 300)
lnr = RectBivariateSpline(VA, Pg, np.log(tt / TT), kx=3, ky=3)(vf, Pf)       # > 0: tau_j < tau_gamma (jet break-up)
Pm_f = PchipInterpolator(VA, Pm)(vf)                                          # MCP of the feed, smooth in v_AC
lnr_x = np.where(Pf[None, :] >= Pm_f[:, None], lnr, np.nan)                  # tau_j = tau_gamma line, drawn only above the MCP
lnr = np.maximum(lnr, (Pm_f[:, None] - Pf[None, :]) / 10.0)                   # below the MCP the interface cannot vanish: jet break-up
ax.contourf(vf, Pf, lnr.T, levels=[-1e3, 0, 1e3], colors=["#cfe3f5", "#f7e0c0"])
ax.contour(vf, Pf, lnr_x.T, levels=[0.0], colors="k", linewidths=.75)
#ax.fill_between(vf, Pg[0], Pm_f, facecolor="gray", alpha=0.25, lw=0)
ax.plot(vf, Pm_f, "--", c="r", lw=.75,label='MCP')
ax.plot([], [], "k-", label=r"$\tau_\mathrm{j}=\tau_\gamma$")
ax.text(0.95, 0.75, "$\\tau_\\mathrm{j}>\\tau_\\gamma$\nNanoparticles", fontsize=6,
        horizontalalignment='right',transform=ax.transAxes)
ax.text(0.05, 0.1, "$\\tau_\\mathrm{j}<\\tau_\\gamma$\nMicroparticles", fontsize=6,
        transform=ax.transAxes)
ax.set_xlim(0, 1)
ax.set_ylim(Pg[0], Pg[-1])
ax.set_xlabel("Acetone volume fraction in the feed, $v_\\mathrm{AC}$")
ax.set_ylabel("$p$ / bar")
ax.legend(frameon=False, fontsize=5, loc="upper right")
for a_, L in zip(axs, "AB"):
    a_.text(-0.15, 0.95, L, transform=a_.transAxes, weight="bold", fontsize=8, ha="right")
fig.tight_layout()
plt.show()
fig.savefig(os.path.join(HERE, SAVE_FIG), dpi=300)
plt.close(fig)
print("saved", SAVE_FIG)
