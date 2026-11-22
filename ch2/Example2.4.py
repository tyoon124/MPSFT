#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 2 Phase Behavior and Thermodynamic Anomalies
Example 2.4
Compares the classical PR EoS with the crossover PR EoS of Feyzi et al. (2010) for CO2 near its critical point: (a) coexistence curve, (b) C_p along the critical isobar, against the Span-Wagner reference EoS.

Model
-----
The crossover Peng-Robinson EoS of Feyzi, Seydi & Alavi, Fluid Phase Equilib.
293 (2010) 251-260, which applies Kiselev's renormalisation procedure
[S.B. Kiselev, Fluid Phase Equilib. 147 (1998) 7-23; S.B. Kiselev & D.G. Friend,
Fluid Phase Equilib. 162 (1999) 51-82] to the classical PR EoS.

  Classical part (Eqs. 3, 9):
      dA(dT, dV) = -ln(dV/b1 + 1) + (T0c/T) 2.077873 a(dT) ln[(dV/b2+1)/(dV/b3+1)]
                   + dV/b1 - (T0c/T) 0.45724 a(dT) dV / (Z0c b2 b3)
      A/RT       = dA - (V/V0c) P0(T) + mu0(T)                                (7,10)

  Renormalisation (Eqs. 11-13):
      tau_bar = tau Y^(-alpha/2D1) + (1+tau) dtau_c Y^(2(2-alpha)/3D1)
      eta_bar = eta Y^((gamma-2beta)/4D1) + (1+eta) deta_c Y^((2-alpha)/2D1)
      Y = [q'/(1+q')]^(2D1),     q' = q exp(|12.2 - 12.2 T/Tc|)

  Parametric distance q from the critical point (Eq. 14), largest real root of
      (q^2 - tau/Gi)[1 - p^2/4b^2 (1 - tau/(q^2 Gi))]
          = b^2 { [eta(1 + v1 exp(-10 eta)) + d1 tau] / (m0 Gi^beta) }^2 Y^((1-2beta)/D1)

  All thermodynamic properties are obtained from A(T,V) by differentiation:
  first derivatives by the complex-step method (machine precision, the implicit
  q(T,V) dependence is handled by one complex Newton correction of Eq. 14),
  second derivatives by central differences of the complex-step derivatives.

Outputs (figure2.5.pdf and console tables)
-----------------------------------------------
  (a) Vapour-liquid coexistence curve T-rho: NIST vs PR vs CPR (flat top)
      + log-log plot of (rho_L - rho_V) vs |t|  -> effective exponent beta
  (b) Isobaric heat capacity C_p along each model's own critical isobar
  (c) Isothermal compressibility kappa_T along the critical isochore (T > Tc)
      -> effective exponent gamma
  (d) Relative deviation of C_p from NIST vs reduced temperature t = T/Tc - 1

Note on the critical point of the CPR model
------------------------------------------
Eq. (14) puts the critical point at (Tc, Vc).  Because Z0c in Table 1 is a fitted
value (0.30464, not the PR value 0.30740), V0c is not exactly the classical PR
critical volume, and the model's own isotherms show a vanishingly small loop
(amplitude ~1 Pa) for |t| < ~3e-5.  The coexistence curve is therefore computed
for |t| >= 1e-4, where this is irrelevant.  The |12.2 - 12.2T/Tc| term in Eq. (13)
also makes Y non-analytic on the critical isotherm; temperature derivatives are
taken one-sided when T is within a few mK of Tc.

Reference data
--------------
"NIST" = the Span-Wagner reference EoS for CO2 (J. Phys. Chem. Ref. Data 25
(1996) 1509), i.e. the formulation behind the NIST Chemistry WebBook / REFPROP,
evaluated through CoolProp (pip install CoolProp).

Requirements: numpy, scipy, matplotlib, CoolProp
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and make figure settings.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import cmath
import math
import warnings

import numpy as np
from scipy.optimize import brentq, root
import matplotlib.pyplot as plt
import matplotlib as mpl

font = {'family':'Dejavu Sans',
        'weight':'normal',
        'size':7}
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

centimeter = 1/2.54
fig, ax = plt.subplots(1, 2, 
                       figsize=(19*centimeter, 8*centimeter),
                       dpi=500, gridspec_kw = {'hspace':0.25,'wspace':0.25})

try:
    import CoolProp.CoolProp as CP
    HAVE_COOLPROP = True
except ImportError:                                    # pragma: no cover
    HAVE_COOLPROP = False
    warnings.warn("CoolProp not found - NIST reference curves will be skipped. "
                  "Install with: pip install CoolProp")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Define thermophysical constants
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

R = 8.314462618          # J/(mol K)
M_CO2 = 44.0098e-3       # kg/mol

# Universal non-classical exponents used by Feyzi et al. (2010)
D1 = 0.51
ALPHA = 0.11
BETA = 0.325
GAMMA = 2.0 - ALPHA - 2.0 * BETA      # = 1.24
B2_LM = 1.359                         # b^2 = p^2 = b_LM^2   ->  p^2/(4 b^2) = 1/4


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Ideal-gas heat capacity of CO2 (Span & Wagner 1996, Table 27)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
_SW_TC = 304.1282
_SW_A = (1.99427042, 0.62105248, 0.41195293, 1.04028922, 0.08327678)
_SW_TH = (3.15163, 6.11190, 6.77708, 11.32384, 27.08792)


def cv0_CO2(T):
    """Ideal-gas isochoric heat capacity of CO2, J/(mol K)."""
    tau = _SW_TC / T
    s = 2.5
    for a, th in zip(_SW_A, _SW_TH):
        x = th * tau
        ex = math.exp(-x)
        s += a * x * x * ex / (1.0 - ex) ** 2
    return R * s


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define crossover / classical Peng-Robinson EoS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class CrossoverPR:
    """
    Peng-Robinson EoS in Kiselev form; crossover=True gives the CPR EoS of
    Feyzi et al. (2010), crossover=False gives the original PR EoS.
    """

    def __init__(self, name, T0c, P0c, Z0c, omega,
                 deta_c=0.0, dtau_c=0.0,
                 Gi=None, d1=None, v1=None, m0=None,
                 crossover=True, new_crossover_function=True, cv0=cv0_CO2):
        self.name = name
        self.T0c, self.P0c, self.Z0c = T0c, P0c, Z0c
        self.kap = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
        self.b1 = 1.0 - 0.0778 / Z0c
        self.b2 = 1.0 - 0.032226 / Z0c
        self.b3 = 1.0 + 0.187826 / Z0c
        self.V0c = Z0c * R * T0c / P0c
        self.b = 0.0778 * R * T0c / P0c                     # PR co-volume
        self.dtau_c, self.deta_c = dtau_c, deta_c
        self.Tc = T0c * (1.0 + dtau_c)                      # real critical T
        self.Vc = self.V0c * (1.0 + deta_c)                 # real critical V
        self.Gi, self.d1, self.v1, self.m0 = Gi, d1, v1, m0
        self.crossover = crossover
        self.new_Y = new_crossover_function
        self.cv0 = cv0
        # exponents
        self.eY_tau = -ALPHA / (2 * D1)
        self.eY_tauc = 2 * (2 - ALPHA) / (3 * D1)
        self.eY_eta = (GAMMA - 2 * BETA) / (4 * D1)
        self.eY_etac = (2 - ALPHA) / (2 * D1)
        self.eU_rhs = 2 * (1 - 2 * BETA)        # Y^((1-2b)/D1) = u^(2(1-2b))
        self.Gi_beta = Gi ** BETA if Gi else None
        # real critical pressure: dA/d(eta) = 0 at (Tc, Vc)  ->  P = RT P0/V0c
        self.Pc = R * self.Tc * self.P0bar(self.Tc).real / self.V0c

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Classical pieces
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def alpha_fn(self, Tr):
        return (1.0 + self.kap * (1.0 - cmath.sqrt(Tr))) ** 2

    def P0bar(self, T):
        """P(T, V0c) V0c / RT from the classical EoS (Eq. 3 at dV = 0)."""
        a = self.alpha_fn(T / self.T0c)
        return 1.0 / self.b1 - (self.T0c / T) * 0.45724 / self.Z0c * a / (self.b2 * self.b3)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # parametric variable q (Eq. 14)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def _s_factor(self, T):
        """q'/q = exp(|12.2 - 12.2 T/Tc|) (new crossover function, Eq. 13)."""
        if not self.new_Y:
            return 1.0
        x = 12.2 * (1.0 - T / self.Tc)
        if x.real < 0:
            x = -x
        return cmath.exp(x) if isinstance(x, complex) else math.exp(x)

    def _q_terms(self, tau, eta, s):
        t = tau / self.Gi
        g = (eta * (1.0 + self.v1 * cmath.exp(-10.0 * eta)) + self.d1 * tau) / (self.m0 * self.Gi_beta)
        c2 = B2_LM * g * g
        return t, c2

    def _f_q(self, q, t, c2, s):
        u = s * q / (1.0 + s * q)
        return 0.75 * q * q - 0.5 * t - t * t / (4.0 * q * q) - c2 * u ** self.eU_rhs

    def _df_q(self, q, t, c2, s):
        u = s * q / (1.0 + s * q)
        du = s / (1.0 + s * q) ** 2
        return 1.5 * q + t * t / (2.0 * q ** 3) - c2 * self.eU_rhs * u ** (self.eU_rhs - 1.0) * du

    def solve_q(self, tau, eta, s):
        """Largest positive real root of Eq. 14 (complex-step aware)."""
        tr, er, sr = tau.real, eta.real, s.real
        t, c2 = self._q_terms(tr, er, sr)
        t, c2 = t.real if isinstance(t, complex) else t, c2.real
        if t == 0.0 and c2 == 0.0:
            return 0.0
        f = lambda q: self._f_q(q, t, c2, sr)
        qhi = 2.0 * math.sqrt(abs(t) + c2) + 1e-14
        qlo = qhi
        flo = f(qlo)
        while flo > 0.0:                    # scan down to first sign change
            qhi = qlo
            qlo *= 0.7
            flo = f(qlo)
            if qlo < 1e-300:
                return 0.0
        q = brentq(f, qlo, qhi, xtol=1e-300, rtol=1e-15, maxiter=200)
        if isinstance(tau, complex) or isinstance(eta, complex) or isinstance(s, complex):
            tc, c2c = self._q_terms(tau, eta, s)
            q = q - self._f_q(q, tc, c2c, s) / self._df_q(q, t, c2, sr)
        return q

    def crossover_Y(self, T, V):
        tau = T / self.Tc - 1.0
        eta = V / self.Vc - 1.0
        s = self._s_factor(T)
        q = self.solve_q(tau, eta, s)
        qp = s * q
        return (qp / (1.0 + qp)) ** (2 * D1)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Helmholtz free energy
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def abar(self, T, V):
        """
        A/RT without the ideal-gas temperature function A0(T)/RT.
        Works for real or complex (complex-step) T and V.
        """
        T0c, b1, b2, b3, Z0c = self.T0c, self.b1, self.b2, self.b3, self.Z0c
        if self.crossover:
            tau = T / self.Tc - 1.0
            eta = V / self.Vc - 1.0
            s = self._s_factor(T)
            q = self.solve_q(tau, eta, s)
            qp = s * q
            Y = (qp / (1.0 + qp)) ** (2 * D1)
            tbar = tau * Y ** self.eY_tau + (1.0 + tau) * self.dtau_c * Y ** self.eY_tauc
            ebar = eta * Y ** self.eY_eta + (1.0 + eta) * self.deta_c * Y ** self.eY_etac
        else:
            tbar = T / T0c - 1.0
            ebar = V / self.V0c - 1.0
        a = self.alpha_fn(tbar + 1.0)
        dA = (-cmath.log(ebar / b1 + 1.0)
              + (T0c / T) * 2.077873 * a * cmath.log((ebar / b2 + 1.0) / (ebar / b3 + 1.0))
              + ebar / b1
              - (T0c / T) * 0.45724 * a * ebar / (Z0c * b2 * b3))
        a0 = self.alpha_fn(T / T0c)
        P0 = self.P0bar(T)
        mu0 = P0 - math.log(b1) + (T0c / T) * 2.077873 * a0 * math.log(b2 / b3)
        return dA - (V / self.V0c) * P0 + mu0

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # First derivatives by complex step
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
    def pressure(self, T, V):
        h = 1e-20 * V
        return -R * T * self.abar(T, complex(V, h)).imag / h

    def _dAres_dT(self, T, V):
        """d[RT*abar]/dT at constant V (= -S_res-like quantity)."""
        h = 1e-20 * T
        Tc_ = complex(T, h)
        return (R * Tc_ * self.abar(Tc_, V)).imag / h

    def mu(self, T, V):
        """Chemical potential minus its ideal-gas temperature function, J/mol."""
        P = self.pressure(T, V)
        return R * T * self.abar(T, V).real + P * V
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Second derivatives
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def _hT(self, T):
        d = abs(T - self.Tc)
        return max(min(1e-2, 0.05 * d), 1e-7)

    def _dT(self, fun, T):
        """d fun/dT; one-sided near Tc so as not to straddle the |T-Tc| kink."""
        h = self._hT(T)
        if self.crossover and self.new_Y and abs(T - self.Tc) < 2.5 * h:
            sgn = 1.0 if T >= self.Tc else -1.0
            h *= sgn
            return (-3 * fun(T) + 4 * fun(T + h) - fun(T + 2 * h)) / (2 * h)
        return (fun(T + h) - fun(T - h)) / (2 * h)

    def dPdV(self, T, V):
        h = 1e-6 * V
        return (self.pressure(T, V + h) - self.pressure(T, V - h)) / (2 * h)

    def dPdT(self, T, V):
        return self._dT(lambda x: self.pressure(x, V), T)

    def cv(self, T, V):
        d2 = self._dT(lambda x: self._dAres_dT(x, V), T)
        return self.cv0(T) - T * d2

    def cp(self, T, V):
        dpdt = self.dPdT(T, V)
        return self.cv(T, V) - T * dpdt ** 2 / self.dPdV(T, V)

    def kappa_T(self, T, V):
        return -1.0 / (V * self.dPdV(T, V))

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # State solvers
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def volume_TP(self, T, P, Vmin_fac=1.05, Vmax_fac=60.0, n=250):
        """Stable molar volume at (T, P): all roots on a grid, lowest Gibbs energy."""
        Vg = np.geomspace(Vmin_fac * self.b, Vmax_fac * self.Vc, n)
        F = np.array([self.pressure(T, v) - P for v in Vg])
        roots = []
        for i in range(n - 1):
            if F[i] == 0.0:
                roots.append(Vg[i])
            elif F[i] * F[i + 1] < 0:
                roots.append(brentq(lambda v: self.pressure(T, v) - P, Vg[i], Vg[i + 1],
                                    xtol=1e-16, rtol=1e-14))
        if not roots:
            raise RuntimeError(f"{self.name}: no volume root at T={T}, P={P}")
        if len(roots) == 1:
            return roots[0]
        G = [self.mu(T, v) for v in roots]        # mu = G/n (same T-function)
        return roots[int(np.argmin(G))]

    def saturation(self, T, VL0, VV0):
        """Solve P_L = P_V, mu_L = mu_V for (VL, VV); returns (Psat, VL, VV)."""
        RT = R * T

        def res(x):
            VL, VV = math.exp(x[0]), math.exp(x[1])
            PL, PV = self.pressure(T, VL), self.pressure(T, VV)
            Ps = self.Pc
            return [(PL - PV) / Ps, (self.mu(T, VL) - self.mu(T, VV)) / RT]

        def jac(x):
            VL, VV = math.exp(x[0]), math.exp(x[1])
            dL, dV = self.dPdV(T, VL), self.dPdV(T, VV)
            return [[dL * VL / self.Pc, -dV * VV / self.Pc],
                    [VL * dL * VL / RT, -VV * dV * VV / RT]]

        sol = root(res, [math.log(VL0), math.log(VV0)], jac=jac, method="hybr",
                   options={"xtol": 1e-13})
        VL, VV = math.exp(sol.x[0]), math.exp(sol.x[1])
        ok = sol.success and abs(VV / VL - 1) > 1e-6 and max(map(abs, res(sol.x))) < 1e-9
        return (self.pressure(T, VL) if ok else np.nan), VL, VV, ok


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define the models
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
OMEGA_CO2 = 0.22394

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Crossover PR EoS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
cpr = CrossoverPR("CPR", T0c=304.10, P0c=7.3700e6, Z0c=0.304640, omega=OMEGA_CO2,
                  deta_c=-0.10077, Gi=0.23136, d1=9.1179, v1=0.02437, m0=1.0459,
                  crossover=True)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Original PR EoS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
pr = CrossoverPR("PR", T0c=304.1282, P0c=7.3773e6, Z0c=0.30740, omega=OMEGA_CO2,
                 crossover=False)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
NIST (Span-Wagner via CoolProp) helpers
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if HAVE_COOLPROP:
    AS = CP.AbstractState("HEOS", "CO2")
    NIST_Tc = CP.PropsSI("Tcrit", "CO2")
    NIST_Pc = CP.PropsSI("pcrit", "CO2")
    NIST_rhoc = CP.PropsSI("rhomolar_critical", "CO2")

    def nist_sat(T):
        return (CP.PropsSI("P", "T", T, "Q", 0, "CO2"),
                CP.PropsSI("Dmolar", "T", T, "Q", 0, "CO2"),
                CP.PropsSI("Dmolar", "T", T, "Q", 1, "CO2"))

    def nist_rho_TP(T, P):
        def f(r):
            AS.update(CP.DmolarT_INPUTS, r, T)
            return AS.p() - P
        return brentq(f, 2000.0, 22000.0, xtol=1e-9, rtol=1e-14)

    def nist_props_rhoT(rho, T):
        AS.update(CP.DmolarT_INPUTS, rho, T)
        return AS.cpmolar(), AS.isothermal_compressibility()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Define functions for the calculation.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def coexistence(model, t_grid, guess_fn):
    """March towards Tc along t_grid (negative, increasing); returns arrays."""
    out = []
    VL, VV = guess_fn(model.Tc * (1 + t_grid[0]))
    prev = []
    for t in t_grid:
        T = model.Tc * (1 + t)
        if len(prev) >= 2:                  # extrapolate ln(V/Vc) in ln|t|
            (t1, l1, v1), (t2, l2, v2) = prev[-2], prev[-1]
            w = (math.log(-t) - math.log(-t2)) / (math.log(-t2) - math.log(-t1))
            VL = model.Vc * math.exp(l2 + w * (l2 - l1))
            VV = model.Vc * math.exp(v2 + w * (v2 - v1))
        Ps, VL_, VV_, ok = model.saturation(T, VL, VV)
        if not ok:
            continue
        VL, VV = VL_, VV_
        prev.append((t, math.log(VL / model.Vc), math.log(VV / model.Vc)))
        out.append((T, Ps, 1.0 / VL, 1.0 / VV))
    return np.array(out)


def nist_guess(T):
    T = min(T, 304.0)
    _, rl, rv = nist_sat(T)
    return 1.0 / rl, 1.0 / rv


def main():
    if not HAVE_COOLPROP:
        raise SystemExit("This example needs CoolProp for the NIST reference curves.")

    print("=" * 74)
    print("Example 2.4  Crossover PR EoS for CO2 (Feyzi et al., FPE 293 (2010) 251)")
    print("=" * 74)
    for m in (pr, cpr):
        print(f"{m.name:4s}: Tc = {m.Tc:9.4f} K   Pc = {m.Pc/1e6:7.4f} MPa   "
              f"rho_c = {M_CO2/m.Vc:7.2f} kg/m3   Zc = {m.Pc*m.Vc/(R*m.Tc):.4f}")
    print(f"NIST: Tc = {NIST_Tc:9.4f} K   Pc = {NIST_Pc/1e6:7.4f} MPa   "
          f"rho_c = {NIST_rhoc*M_CO2:7.2f} kg/m3   Zc = {NIST_Pc/(NIST_rhoc*R*NIST_Tc):.4f}")

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    6. (1) Calculate the coexistence curves
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    t_far = -np.linspace(0.22, 0.02, 21)
    t_near = -np.geomspace(0.02, 1e-4, 40)[1:]
    t_coex = np.concatenate([t_far, t_near])
    coex = {m.name: coexistence(m, t_coex, nist_guess) for m in (pr, cpr)}
    nist_coex = np.array([(NIST_Tc * (1 + t), *nist_sat(NIST_Tc * (1 + t))) for t in t_coex])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Validation
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print("\nSaturated properties, AAD% vs NIST for 240 K <= T <= 304 K "
          "(cf. Table 2 of Feyzi et al.)")
    print(f"{'':6s}{'Psat':>10s}{'rho_L':>10s}{'rho_V':>10s}")
    Tval = np.arange(240.0, 304.01, 2.0)
    for m in (pr, cpr):
        dP, dL, dV = [], [], []
        VL, VV = nist_guess(Tval[0])
        for T in Tval:
            Ps, VL_, VV_, ok = m.saturation(T, VL, VV)
            if not ok:
                continue
            VL, VV = VL_, VV_
            Pn, rln, rvn = nist_sat(T)
            dP.append(abs(Ps / Pn - 1))
            dL.append(abs(1 / VL / rln - 1))
            dV.append(abs(1 / VV / rvn - 1))
        print(f"{m.name:6s}{100*np.mean(dP):10.3f}{100*np.mean(dL):10.3f}{100*np.mean(dV):10.3f}")

    # effective exponent beta from rho_L - rho_V ~ |t|^beta
    def beta_fit(t, drho, lo=3e-4, hi=1e-2):
        msk = (np.abs(t) > lo) & (np.abs(t) < hi)
        return np.polyfit(np.log(np.abs(t[msk])), np.log(drho[msk]), 1)[0]

    print("\nEffective exponent beta  (rho_L - rho_V ~ |t|^beta, 3e-4 < |t| < 1e-2)")
    b_eff = {}
    for name, c in coex.items():
        m = pr if name == "PR" else cpr
        b_eff[name] = beta_fit(c[:, 0] / m.Tc - 1, c[:, 2] - c[:, 3])
        print(f"  {name:5s}: beta_eff = {b_eff[name]:.3f}")
    b_eff["NIST"] = beta_fit(nist_coex[:, 0] / NIST_Tc - 1, nist_coex[:, 2] - nist_coex[:, 3])
    print(f"  NIST : beta_eff = {b_eff['NIST']:.3f}   (Ising value 0.326)")

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    6. (2) Calculate the heat capacity and compare the results.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    t_iso = np.concatenate([-np.geomspace(0.05, 2e-4, 45), np.geomspace(2e-4, 0.05, 45)])
    cp_curves = {}
    for m in (pr, cpr):
        rows = []
        for t in t_iso:
            T = m.Tc * (1 + t)
            V = m.volume_TP(T, m.Pc)
            rows.append((t, T, 1 / V, m.cp(T, V)))
        cp_curves[m.name] = np.array(rows)
    rows = []
    for t in t_iso:
        T = NIST_Tc * (1 + t)
        rho = nist_rho_TP(T, NIST_Pc)
        rows.append((t, T, rho, nist_props_rhoT(rho, T)[0]))
    cp_curves["NIST"] = np.array(rows)

    print("\nC_p along each model's critical isobar, J/(mol K), and deviation from NIST")
    print(f"{'t':>9s}{'NIST':>10s}{'PR':>10s}{'CPR':>10s}{'dev PR %':>10s}{'dev CPR %':>11s}")
    t_tab = [-0.02, -0.01, -0.005, -0.002, -0.001, 0.001, 0.002, 0.005, 0.01, 0.02]
    interp = lambda name, t: np.interp(t, cp_curves[name][:, 0], cp_curves[name][:, 3])
    for t in t_tab:
        cn, cpp, cc = interp("NIST", t), interp("PR", t), interp("CPR", t)
        print(f"{t:9.3f}{cn:10.1f}{cpp:10.1f}{cc:10.1f}{100*(cpp/cn-1):10.1f}{100*(cc/cn-1):11.1f}")

    msk = (np.abs(t_iso) >= 1e-3) & (np.abs(t_iso) <= 0.02)
    cpN = cp_curves["NIST"][:, 3]
    aad_pr = 100 * np.mean(np.abs(cp_curves["PR"][msk, 3] / cpN[msk] - 1))
    aad_cpr = 100 * np.mean(np.abs(cp_curves["CPR"][msk, 3] / cpN[msk] - 1))
    print(f"\nAAD of C_p for 1e-3 <= |t| <= 0.02 :  PR = {aad_pr:.1f} %,  CPR = {aad_cpr:.1f} %"
          f"  -> reduction of {aad_pr - aad_cpr:.1f} percentage points")

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    6. (3) Visualize the results.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    col = {"NIST": "#222222", "PR": "#1f77b4", "CPR": "#d62728"}
    sty = {"NIST": dict(ls="none", marker="o", ms=3.5, mfc="none"),
           "PR": dict(ls="--", lw=1), "CPR": dict(ls="-", lw=1)}
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Coexistence curve
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    a = ax[0]
    a.plot(nist_coex[:, 2] * M_CO2, nist_coex[:, 0], color=col["NIST"], 
           label="NIST (Span-Wagner)", **sty["NIST"])
    a.plot(nist_coex[:, 3] * M_CO2, nist_coex[:, 0], color=col["NIST"], 
           **sty["NIST"])
    
    for name, c in coex.items():
        rho = np.concatenate([c[:, 2], c[::-1, 3]]) * M_CO2
        T = np.concatenate([c[:, 0], c[::-1, 0]])
        a.plot(rho, T, color=col[name], label=name, **sty[name])
    a.set_xlabel(r"$\rho$ [kg m$^{-3}$]")
    a.set_ylabel("$T$ [K]")
    a.set_ylim(255, 306)
    a.set_xlim(0, 1100)
    a.legend(loc="lower right")
    a.legend(frameon=False)
    a.text(-0.175,1.0,'A',fontweight='bold',fontsize=8,transform=a.transAxes)
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Isobaric heat capacity
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    b = ax[1]
    for name in ("NIST", "PR", "CPR"):
        c = cp_curves[name]
        b.semilogy(c[:, 0], c[:, 3], color=col[name], label=name, **sty[name])
    b.set_xlabel(r"$t = T/T_\mathrm{crit} - 1$")
    b.set_ylabel(r"$C_\mathrm{p}$ [J mol$^{-1}$ K$^{-1}$]")
    b.set_xlim(-0.05, 0.05)
    b.text(-0.175,1.0,'B',fontweight='bold',fontsize=8,transform=b.transAxes)
    
    
    fig.tight_layout()
    fig.savefig("figure2.5.pdf", dpi=500)
    print("\nFigure saved to figure2.5.pdf")
    plt.show()


if __name__ == "__main__":
    main()