# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 2 Phase Behavior and Thermodynamic Anomalies
Example 2.3 
The SAFT-$\gamma$ Mie EoS is illustrated here for the binary system CO2--C6H14 (carbon dioxide--$n$-hexane), a prototype system for supercritical extraction with hydrocarbon solutes. Using the SGTPy open-source library (https://github.com/gustavochm/sgtpy) with published SAFT-$\gamma$ Mie group parameters for \ce{CO2} and the methylene/methyl groups of $n$-hexane, we compute (a) vapor--liquid equilibria at $T=298$~K and $T=313$~K, and (b) the densities of the coexisting phases. The SAFT-$\gamma$ Mie (dashed lines) and PR (solid lines) results are compared with experimental bubble and dew points (markers) compiled in c6h14.xlsx. Python code is provided in \texttt{Example2.3.py} on the GitHub repository.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and make basic plot setting.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
import pandas as pd
from sgtpy import component, mixture, saftgammamie
from sgtpy.equilibrium import bubblePy
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Define the system.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
c6h14 = component(GC={'CH3':2,'CH2':4})
co2   = component(GC={'CO2':1})
mix   = mixture(co2,c6h14)
mix.saftgammamie()
eos   = saftgammamie(mix)
Mw    = eos.Mw

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Load the experimental data.
The measured bubble (x1) and dew (y1) points of CO2 (1) + n-hexane (2) are read from c6h14.xlsx,
which has the same columns as c2oh.xlsx: x1, y1, T(K), P(bar), u, Reference.
Data within T_TOL of each isotherm are drawn. If the file is not found, only the EoS results are drawn.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TinKs    = [298, 313, 353, 393]
T_TOL    = 0.5                      # K, tolerance for matching the data to an isotherm
EXP_FILE = 'c6h14.xlsx'
if os.path.exists(EXP_FILE):
    expdata = pd.read_excel(EXP_FILE)
else:
    expdata = None
    print(EXP_FILE + ' not found: the experimental data are not drawn.')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define the functions for the VLE continuation.
4-(1) SAFT-gamma Mie (from VLEcontinuation-SAFT.py): _phase, vle_equations, _tangent, trace_vle_saft
4-(2) Peng-Robinson (from VLEcontinuation.py): trace_bubble_arc, close_curve
Calling bubblePy at fixed x1 fails close to the mixture critical point, because the liquid and vapor
density roots merge there; the curves then stop short of the critical point and stay open.
Instead, each phase is described by its composition and molar density, and u = [x1, y1, ln rhoL, ln rhoV]
is followed by pseudo-arclength continuation. The equations are mu_i(x, rhoL) = mu_i(y, rhoV) (i = 1, 2)
and P(x, rhoL) = P(y, rhoV). Near the critical point the gap d = y1 - x1 is reduced step by step, and the
curves are extrapolated to d = 0, which gives the mixture critical point (or the vapor pressure of CO2
below its critical temperature). The bubble and dew curves therefore meet and form a closed loop.
For the PR EoS, (x1, ln y_i, ln P) are followed by pseudo-arclength continuation with the fugacity
equations, as in Example 2.2. The continuation passes through the critical point (y1 - x1 changes sign),
so the curve is cut there and the critical point is added by interpolation in y1 - x1.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def _phase(eos, z1, rho, temp_aux):
    """mu/RT, d(mu/RT)/d(z1, ln rho), P/RT and its derivatives for one phase."""
    rhoi = rho*np.array([z1, 1.0 - z1])
    mu, H, _ = eos.dmuad_aux(rhoi, temp_aux)
    P, _ = eos.pressure_aux(np.array([z1, 1.0 - z1]), rho, temp_aux)
    drho_dz = np.array([rho, -rho])          # d rho_i / d z1 at fixed rho
    drho_dlnrho = rhoi                       # d rho_i / d ln rho at fixed z1
    dmu = np.column_stack([H@drho_dz, H@drho_dlnrho])
    g = H@rhoi                               # d(P/RT)/d rho_i  (Gibbs-Duhem)
    dP = np.array([g@drho_dz, g@drho_dlnrho])
    return mu, dmu, P, dP


def vle_equations(u, eos, temp_aux, RT, rho_ref=1.0e3):
    """Residuals F (3) and Jacobian J (3 x 4) for u = [x1, y1, ln rhoL, ln rhoV]."""
    x1, y1 = u[0], u[1]
    rhoL, rhoV = np.exp(u[2]), np.exp(u[3])
    muL, dmuL, PL, dPL = _phase(eos, x1, rhoL, temp_aux)
    muV, dmuV, PV, dPV = _phase(eos, y1, rhoV, temp_aux)
    F = np.r_[muL - muV, (PL - PV)/(RT*rho_ref)]
    J = np.zeros((3, 4))
    J[:2, [0, 2]] = dmuL
    J[:2, [1, 3]] = -dmuV
    J[2, [0, 2]] = dPL/rho_ref
    J[2, [1, 3]] = -dPV/rho_ref
    return F, J, 0.5*(PL + PV)


def _tangent(J, prev=None):
    t = np.linalg.svd(J)[2][-1]
    if prev is None:
        return t if t[0] > 0 else -t
    return -t if t@prev < 0 else t


def trace_vle_saft(eos, T, x_start=0.001, y_guess=0.99, P_guess=1.0e5,
                   ds0=0.02, ds_min=1e-7, ds_max=0.05, d_switch=0.1, d_min=1e-3,
                   n_close=25, tol=1e-9, nmax=3000, verbose=False):
    """Trace the bubble and dew curves of a binary mixture at temperature T [K].

    Returns a dict with arrays x1, y1, P [Pa], rhoL, rhoV [mol/m^3]; the last entry is the
    extrapolated end point (mixture critical point, or pure component 1 saturation),
    'reason' (why the tracing stopped) and 'critical' (x1, P, rho at the end point).
    """
    R = 8.314462618
    RT = R*T
    temp_aux = eos.temperature_aux(T)

    # starting point from the standard bubble-point solver
    res = bubblePy(np.array([y_guess, 1 - y_guess]), P_guess,
                   np.array([x_start, 1 - x_start]), T, eos, full_output=True)
    u = np.array([x_start, res.Y[0], -np.log(res.v1), -np.log(res.v2)])

    def correct(up, extra_row, extra_res):
        """Newton iterations on F(u) = 0 plus one extra linear condition."""
        for _ in range(15):
            F, J, P = vle_equations(up, eos, temp_aux, RT)
            A = np.vstack([J, extra_row])
            b = -np.r_[F, extra_res(up)]
            du = np.linalg.solve(A, b)
            up = up + du
            if not np.all(np.isfinite(up)) or np.linalg.norm(du) > 1.0:
                return None, None
            if np.linalg.norm(du) < 1e-11:
                break
        F, J, P = vle_equations(up, eos, temp_aux, RT)
        if np.linalg.norm(F) > tol or not (0.0 < up[0] < 1.0 and 0.0 < up[1] < 1.0):
            return None, None
        return up, P

    # polish the starting point at fixed x1
    e0 = np.array([1.0, 0.0, 0.0, 0.0])
    u, P = correct(u, e0, lambda v: v[0] - x_start)
    if u is None:
        raise RuntimeError('Starting bubble point did not converge.')
    pts = [u.copy()]
    Ps = [P]

    """Stage 1: pseudo-arclength continuation"""
    F, J, _ = vle_equations(u, eos, temp_aux, RT)
    t = _tangent(J)
    ds = ds0
    dmax_seen = u[1] - u[0]
    reason = 'max steps'
    for _ in range(nmax):
        u0 = u.copy()
        up, P = correct(u + ds*t, t, lambda v: t@(v - u0) - ds)
        if up is not None:
            F, J, _ = vle_equations(up, eos, temp_aux, RT)
            tn = _tangent(J, t)
            if t@tn < 0.95 and ds > 10*ds_min:
                up = None
        if up is None:
            ds /= 2
            if ds < ds_min:
                reason = 'step underflow'
                break
            continue
        u = up
        t = tn
        pts.append(u.copy())
        Ps.append(P)
        ds = min(1.5*ds, ds_max)
        d = u[1] - u[0]
        dmax_seen = max(dmax_seen, d)
        if dmax_seen > 2*d_switch and d < d_switch:
            reason = 'critical region'
            break
        if d <= 0.0:
            reason = 'trivial solution'
            break
    if verbose:
        print(f'T = {T} K: stage 1 ended ({reason}) at x1 = {u[0]:.4f}, '
              f'y1 = {u[1]:.4f}, P = {P/1e5:.2f} bar')

    """Stage 2: prescribed gap d = y1 - x1 down to d_min"""
    if reason == 'critical region':
        d_now = u[1] - u[0]
        targets = np.geomspace(d_now, d_min, n_close)[1:]
        ed = np.array([-1.0, 1.0, 0.0, 0.0])
        prev = np.array(pts[-2])
        for dt in targets:
            # secant predictor in the gap d
            dd_prev = (u[1] - u[0]) - (prev[1] - prev[0])
            slope = (u - prev)/dd_prev if abs(dd_prev) > 0 else np.zeros(4)
            guess = u + slope*(dt - (u[1] - u[0]))
            up, P = correct(guess, ed, lambda v, dt=dt: (v[1] - v[0]) - dt)
            if up is None:
                if verbose:
                    print(f'  closing stopped at d = {u[1] - u[0]:.2e}')
                break
            prev = u
            u = up
            pts.append(u.copy())
            Ps.append(P)

    pts = np.array(pts)
    Ps = np.array(Ps)

    """Stage 3: extrapolate to d = 0 (end point of the two curves)"""
    d = pts[:, 1] - pts[:, 0]
    end = None
    if reason == 'critical region' or d[-1] < 0.02:
        k = min(6, len(d))
        dd = d[-k:]
        cols = [pts[-k:, 0], pts[-k:, 1], np.log(Ps[-k:]), pts[-k:, 2], pts[-k:, 3]]
        vals = [np.polyval(np.polyfit(dd, c, 2), 0.0) for c in cols]
        xc = 0.5*(vals[0] + vals[1])
        Pc = np.exp(vals[2])
        rhoc = np.exp(0.5*(vals[3] + vals[4]))
        end = (xc, Pc, rhoc)
        pts = np.vstack([pts, [xc, xc, np.log(rhoc), np.log(rhoc)]])
        Ps = np.r_[Ps, Pc]

    return {'x1': pts[:, 0], 'y1': pts[:, 1], 'P': Ps,
            'rhoL': np.exp(pts[:, 2]), 'rhoV': np.exp(pts[:, 3]),
            'reason': reason, 'critical': end}


def trace_bubble_arc(Xof, T, model, y_guess, x_start=0.001, ds0=0.05, ds_min=1e-6,
                     ds_max=0.05, lnK_stop=0.0, tol=1e-10, nmax=4000):
    """Xof(x1) -> liquid composition vector (length nc). Returns x1, Y (npts, nc), P, reason.
    P is in the units of the model (bar)."""
    temp_aux = model.temperature_aux(T)
    nc = len(y_guess)

    def F(u):
        X = Xof(u[0])
        Y = np.exp(u[1:1+nc])
        P = np.exp(u[-1])
        lphiL, _ = model.logfugef_aux(X, temp_aux, P, 'L')
        lphiV, _ = model.logfugef_aux(Y, temp_aux, P, 'V')
        return np.r_[lphiL + np.log(X) - lphiV - u[1:1+nc], Y.sum() - 1.0]

    def J(u):
        Jm = np.empty((nc+1, nc+2))
        for k in range(nc+2):
            h = 1e-6*max(1, abs(u[k]))
            up = u.copy()
            um = u.copy()
            up[k] += h
            um[k] -= h
            Jm[:, k] = (F(up) - F(um))/(2*h)
        return Jm

    # Starting point at a tiny x1 from the phasepy bubble-point solver, then polish it
    Y0, P0 = bubblePyPR(np.asarray(y_guess, float), 1.0, Xof(x_start), T, model)
    u = np.r_[x_start, np.log(Y0), np.log(P0)]
    for _ in range(20):
        u = u + np.linalg.lstsq(J(u), -F(u), rcond=None)[0]
        if np.linalg.norm(F(u)) < tol:
            break

    def tangent(u, prev=None):
        t = np.linalg.svd(J(u))[2][-1]
        if prev is None:
            return t if t[0] > 0 else -t
        return -t if t@prev < 0 else t

    t = tangent(u)
    ds = ds0
    pts = [u.copy()]
    reason = 'max steps'
    for _ in range(nmax):
        up = u + ds*t
        ok = False
        try:
            for _k in range(12):
                Fa = np.r_[F(up), t@(up-u) - ds]
                du = np.linalg.solve(np.vstack([J(up), t]), -Fa)
                up = up + du
                if np.linalg.norm(du) < 1e-10 and np.linalg.norm(F(up)) < tol:
                    ok = True
                    break
                if np.linalg.norm(du) > 1.0:
                    break
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            ok = False
        if ok:
            tn = tangent(up, t)
            if t@tn < 0.9 and ds > ds_min*10:      # too curved -> smaller step
                ok = False
        if not ok:
            ds /= 2
            if ds < ds_min:
                reason = 'step underflow'
                break
            continue
        u = up
        t = tn
        pts.append(u.copy())
        ds = min(ds*1.5, ds_max)
        X = Xof(u[0])
        if np.max(np.abs(u[1:1+nc] - np.log(X))) < lnK_stop:
            reason = 'trivial solution'
            break
        if u[0] > 0.999:
            reason = 'x1 limit'
            break
    pts = np.array(pts)
    return pts[:, 0], np.exp(pts[:, 1:1+nc]), np.exp(pts[:, -1]), reason


def close_curve(x1, y1, P, n_fit=3):
    """Cut the traced curve where it passes the mixture critical point (y1 - x1 changes sign;
    beyond it the continuation follows the same curve with x1 and y1 swapped) and add the
    critical point itself, found by interpolating x1, y1 and ln P in d = y1 - x1 to d = 0 with
    n_fit points on each side. Returns x1, y1, P; the last entry is the critical point
    (x1 = y1). If there is no sign change (e.g. T below Tc of component 1, where the curve
    ends at x1 -> 1), the curve is returned as traced."""
    d = y1 - x1
    k = np.where(d <= 0)[0]
    k = k[k > 0]
    if len(k) == 0:
        return x1, y1, P
    k = k[0]
    sl = slice(max(k - n_fit, 0), min(k + n_fit, len(d)))
    xc1 = np.polyval(np.polyfit(d[sl], x1[sl], 2), 0.0)
    xc2 = np.polyval(np.polyfit(d[sl], y1[sl], 2), 0.0)
    Pc = np.exp(np.polyval(np.polyfit(d[sl], np.log(P[sl]), 2), 0.0))
    xc = 0.5*(xc1 + xc2)
    return np.r_[x1[:k], xc], np.r_[y1[:k], xc], np.r_[P[:k], Pc]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. SAFT-gamma Mie EoS (dashed lines)
5-(1) Trace the bubble and dew curves by continuation up to the mixture critical point
5-(2) Calculate the densities of the coexisting liquid and vapor phases
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
centimeter = 1 / 2.54
fig, axes = plt.subplots(1,2,
                         figsize=(19*centimeter, 8*centimeter),
                         dpi=500, gridspec_kw = {'hspace':0.25,'wspace':0.25}
                         )

colors = sns.color_palette('viridis',n_colors=len(TinKs))

for i,TinK in enumerate(TinKs):
    out = trace_vle_saft(eos, TinK)
    x1s = out['x1']
    y1s = out['y1']
    Pbubls = out['P']
    srL = np.dot(np.column_stack([x1s, 1-x1s]), Mw) * out['rhoL'] / 1000
    srG = np.dot(np.column_stack([y1s, 1-y1s]), Mw) * out['rhoV'] / 1000

    # Closed loop: bubble curve up to the critical point, then the dew curve back down
    x_loop = np.r_[x1s, y1s[::-1]]
    P_loop = np.r_[Pbubls, Pbubls[::-1]]
    sr_loop = np.r_[srL, srG[::-1]]
    axes[0].plot(x_loop, P_loop/1e6, '--', color=colors[i])
    axes[1].plot(x_loop, sr_loop, '--', color=colors[i])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Peng-Robinson EoS (solid lines)
6-(1) Trace the bubble and dew curves by continuation up to the mixture critical point
6-(2) Calculate the densities of the coexisting liquid and vapor phases
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from phasepy import component, mixture, preos
from phasepy.equilibrium import bubblePy as bubblePyPR

co2   = component(name = 'Carbon dioxide', Mw = 44.010,
                  Tc = 304.2, Pc = 73.83, w = 0.224)
c6h14 = component(name = "n-hexane", Mw = 86.177, 
                  Tc = 507.6, Pc = 30.25, w = 0.301)
mix = mixture(co2,c6h14)

model = preos(mix, mixrule='qmr')

Xof = lambda x1: np.array([x1, 1-x1])
for i,TinK in enumerate(TinKs):
    x1s, Ys, Pbubls, reason = trace_bubble_arc(Xof, TinK, model, [0.9, 0.1])
    x1s, y1s, Pbubls = close_curve(x1s, Ys[:, 0], Pbubls)

    # densities of the coexisting phases (phasepy volumes in cm3/mol)
    temp_aux = model.temperature_aux(TinK)
    srL, srG = [], []
    for x1, y1, P in zip(x1s, y1s, Pbubls):
        _, vL = model.logfugef_aux(Xof(x1), temp_aux, P, 'L')
        _, vV = model.logfugef_aux(Xof(y1), temp_aux, P, 'V')
        srL.append(np.dot(Xof(x1), Mw) / vL * 1000)
        srG.append(np.dot(Xof(y1), Mw) / vV * 1000)
    if y1s[-1] == x1s[-1]:
        # at the critical point both phases have the same density
        srL[-1] = 0.5*(srL[-1] + srG[-1])
        srG[-1] = srL[-1]

    # Closed loop: bubble curve up to the critical point, then the dew curve back down
    x_loop = np.r_[x1s, y1s[::-1]]
    P_loop = np.r_[Pbubls, Pbubls[::-1]]
    sr_loop = np.r_[srL, srG[::-1]]
    axes[0].plot(x_loop, P_loop/10, color=colors[i], label='%d K' % TinK)
    axes[1].plot(x_loop, sr_loop, color=colors[i])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
7. Draw the experimental data and finish the figure.
Filled markers: bubble points (x1), open markers: dew points (y1).
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if expdata is not None:
    for i,TinK in enumerate(TinKs):
        sel = expdata[np.abs(expdata['T(K)'] - TinK) <= T_TOL]
        axes[0].plot(sel['x1'], sel['P(bar)']/10, 'o', markersize=2.5, color=colors[i],
                     markeredgewidth=0.4, zorder=3)
        axes[0].plot(sel['y1'], sel['P(bar)']/10, 'o', markersize=2.5, markerfacecolor='w',
                     markeredgecolor=colors[i], markeredgewidth=0.4, zorder=3)

axes[0].plot([], [], '-', color='k', label='PR')
axes[0].plot([], [], '--', color='k', label=r'SAFT-$\gamma$ Mie')
if expdata is not None:
    axes[0].plot([], [], 'o', markersize=2.5, color='k', label='Exp. (bubble)')
    axes[0].plot([], [], 'o', markersize=2.5, markerfacecolor='w', markeredgecolor='k',
                 markeredgewidth=0.4, label='Exp. (dew)')
axes[0].legend(frameon=False)
axes[0].set_xlim([0,1])
axes[0].set_ylim([0,12])
axes[0].text(0.975,0.95,r'CO$_\mathbf{2}$/C$_\mathbf{6}$H$_\mathbf{14}$',fontweight='bold',
             horizontalalignment='right',
             transform=axes[0].transAxes)
axes[0].set_xlabel(r'$x_1,~y_1$')
axes[0].set_ylabel(r'$p$ [MPa]')
axes[1].set_xlabel(r'$x_1,~y_1$')
axes[1].set_ylabel(r'$\rho$ [kg/m$^3$]')
axes[1].set_xlim([0,1])
axes[1].set_ylim([0,850])

axes[0].text(-0.15,1.0,'A',fontweight='bold',transform=axes[0].transAxes)
axes[1].text(-0.15,1.0,'B',fontweight='bold',transform=axes[1].transAxes)

plt.savefig('figure2.4.pdf',dpi=500,bbox_inches='tight')


