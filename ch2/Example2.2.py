# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 2 Phase Behavior and Thermodynamic Anomalies
Example 2.2 
CEoS models have been extensively used to describe the phase behavior between CO2 and solutes or co-solvents. Here we demonstrate how the PhasePy package (https://github.com/gustavochm/phasepy) can be used to calculate the vapor--liquid equilibria of the \ce{CO2}--\ce{C2H5OH} binary system using PR EoS parameters in conjunction with the excess Gibbs free energy (non-random two-liquid, NRTL) model from Kim et al. Panel A shows the calculated isotherms from 313 to 373 K, and panel B the phase envelopes at constant composition with the pure-component vapor pressure curves (van Konynenburg--Scott Type I). Panels C and D compare the bubble pressures and vapor compositions calculated at the measured temperature and liquid composition with the literature data compiled in c2oh.xlsx (pressure residuals and parity plot of the ethanol mole fraction in the vapor, y2).
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and make basic plot setting.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import pandas as pd
from phasepy import component, mixture, preos
from phasepy.equilibrium import bubblePy
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.interpolate import splprep, splev

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
co2 = component(name = 'Carbon dioxide', Mw = 44.010,
                Tc = 304.2, Pc = 73.83, w = 0.224)
c2oh = component(name = "Ethanol", Mw = 46.069, 
                 Tc = 513.92, Pc = 61.48, w = 0.645)
mix = mixture(co2,c2oh)
mixkij = mix.copy()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define the NRTL parameters and the CEoS-GE model.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
NRTLpars = [0.40, 526.09, -61.72]
kij, g12T, g21T = NRTLpars
alpha = 0.30
alpha = np.array([[0,alpha],
                  [alpha,0]])
gT = np.array([[0,g12T],
               [g21T,0]])
mixkij.NRTL(alpha,gT)
mixkij.kij_ws(np.array([[0,kij],
                        [kij,0]]))

model = preos(mixkij, mixrule='ws_nrtl')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define the functions for the VLE continuation (from VLEcontinuation.py).
Stepping x1 and calling bubblePy at every point becomes ill-conditioned near the mixture critical
point and can jump to another solution branch. Instead, (x1, y_i, P) are followed along one curve by
pseudo-arclength continuation, so that the bubble and dew curves meet at the mixture critical point
and form a closed loop. Unknowns u = [x1, ln y_1..ln y_nc, ln P]; the equations are
ln(phi_i^L x_i) = ln(phi_i^V y_i) (nc equations) and sum(y) = 1.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def trace_bubble_arc(Xof, T, model, y_guess, x_start=0.001, P_guess=1.0, ds0=0.05, ds_min=1e-6,
                     ds_max=0.05, lnK_stop=0.0, tol=1e-10, nmax=4000):
    """Xof(x1) -> liquid composition vector (length nc). Returns x1, Y (npts, nc), P, reason.
    P and P_guess (initial guess of the bubble pressure at x_start) are in the units of the model (bar)."""
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

    # Starting point at a tiny x1 from the standard bubble-point solver, then polish it
    Y0, P0 = bubblePy(np.asarray(y_guess, float), P_guess, Xof(x_start), T, model)
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


def truncate_curve(x, Y, P, Xof, lnK_crit=0.3):
    """Keep the physical part of the curve: stop at the mixture critical point (minimum of
    max|ln K| once < lnK_crit) or at the first fold in x1, whichever comes first."""
    X = np.array([Xof(xi) for xi in x])
    m = np.max(np.abs(np.log(Y/X)), axis=1)
    n = len(x)
    fold = np.where(np.diff(x) <= 0)[0]
    if len(fold):
        n = fold[0] + 1
    k = np.argmin(m[:n])
    if m[k] < lnK_crit:
        n = k + 1
    return x[:n], Y[:n], P[:n]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Settings and experimental data.
The measured bubble (x1) and dew (y1) points of CO2 (1) + ethanol (2) are read from c2oh.xlsx
(columns x1, y1, T(K), P(bar), Reference). Panel A shows only the calculated isotherms.
Panels C and D compare the model with the data measured between T_MIN and T_MAX.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TinKs   = np.arange(313.15, 374.15, 10)   # K, isotherms drawn in panel A
T_MIN   = 313.0                          # K, lower end of the data compared in panels C and D
T_MAX   = 374.0                          # K, upper end of the data compared in panels C and D
expdata = pd.read_excel('c2oh.xlsx')
expdata = expdata.dropna(subset=['P(bar)'])
expdata = expdata[(expdata['T(K)'] >= T_MIN) & (expdata['T(K)'] <= T_MAX)]
expdata = expdata[(expdata['x1'] > 0) & (expdata['x1'] < 1)].reset_index(drop=True)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Bubble-point calculation at each measured state.
For each data point, the bubble pressure p_cal and vapor composition y_cal are calculated at the
measured temperature and liquid composition (T_lit, x_lit). The measured pressure is used only as
the initial guess. Points where no initial guess leads to a non-trivial solution (y1 != x1) are skipped.
The pressure residual is %dp = 100 (p_lit - p_cal) / p_lit.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
p_lit, p_cal, y_lit, y_cal, T_pts, x_pts = [], [], [], [], [], []
n_fail = 0
y1_guesses = [0.95, 0.99, 0.995, 0.90, 0.98, 0.80]   # initial guesses of y1 tried in turn
P_factors  = [1.0, 0.9, 1.1, 0.8]                    # initial guesses of P as multiples of p_lit

def bubble_point(x1, TinK, Plit):
    """Bubble point at (x1, T); returns (Y, P) or None. Several initial guesses are tried
    because a single guess can converge to the trivial solution y = x near the critical region."""
    for y1g in y1_guesses:
        for fP in P_factors:
            try:
                Ycal, Pcal = bubblePy([y1g, 1-y1g], fP*Plit, [x1, 1-x1], TinK, model)
            except Exception:
                continue
            if np.isfinite(Pcal) and abs(Ycal[0] - x1) > 1e-3 and 0 < Ycal[0] < 1:
                return Ycal, Pcal
    return None

for _, row in expdata.iterrows():
    x1   = row['x1']
    TinK = row['T(K)']
    Plit = row['P(bar)']
    sol  = bubble_point(x1, TinK, Plit)
    if sol is None:
        n_fail += 1
        continue
    Ycal, Pcal = sol
    p_lit.append(Plit)
    p_cal.append(Pcal)
    y_lit.append(row['y1'])
    y_cal.append(Ycal[0])
    T_pts.append(TinK)
    x_pts.append(x1)

p_lit = np.array(p_lit)
p_cal = np.array(p_cal)
y_lit = np.array(y_lit, dtype=float)
y_cal = np.array(y_cal)
T_pts = np.array(T_pts)
x_pts = np.array(x_pts)
dP    = 100 * (p_lit - p_cal) / p_lit
has_y = np.isfinite(y_lit)

print('Data points compared: %d (skipped: %d)' % (len(dP), n_fail))
print('AAD of p: %.2f %%,  bias: %.2f %%' % (np.mean(np.abs(dP)), np.mean(dP)))
print('Points with y1: %d,  AAD of y1: %.4f' % (has_y.sum(), np.mean(np.abs(y_lit[has_y] - y_cal[has_y]))))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
7. Phase envelopes at constant composition (panel B).
At each temperature of T_env_grid, the bubble curve is traced by continuation; each point gives a
bubble point (T, p, x1) and the coexisting dew point (T, p, y1). Where the curve reaches the mixture
critical point (y1 = x1), the critical pressure and composition are found by extrapolating x1 and ln p
of the last points to y1 - x1 = 0. For each composition z in Z_ENV, the bubble pressure (x1 = z) and
the dew pressure(s) (y1 = z) are then interpolated at every temperature, and the two curves are joined
at the critical point of that composition. The vapor pressures of the pure components (z = 1 and 0)
are calculated with the PR EoS (same Tc, pc, and omega).
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Xof = lambda x1: np.array([x1, 1-x1])
T_env_grid = np.arange(270.0, 512.0, 3.0)    # K
Z_ENV      = [0.1, 0.3, 0.5, 0.7, 0.9]       # CO2 mole fractions of the envelopes drawn in panel B

def trace_isotherm(TinK, n_fit=6):
    """Bubble branch at TinK up to the critical point: x1, y1, P [bar], and the critical point
    (Pc, xc) or None. Several starting guesses are tried, because at high temperature the bubble
    pressure at x1 = 0.001 is close to the vapor pressure of ethanol."""
    for P0 in [1.0, 20.0, 40.0, 55.0]:
        for yg in ([0.9, 0.1], [0.2, 0.8]):
            try:
                x1s, Ys, Ps, reason = trace_bubble_arc(Xof, TinK, model, yg, P_guess=P0)
            except Exception:
                continue
            lnK = np.log(Ys[:, 0]/x1s)
            if lnK[0] < 0.05:                       # trivial start (y = x)
                continue
            neg = np.where(lnK <= 0)[0]
            n = neg[0] if len(neg) else len(lnK)
            if n < n_fit + 2:
                continue
            x, y, P = x1s[:n], Ys[:n, 0], Ps[:n]
            crit = None
            if len(neg) or (y[-1] - x[-1]) < 0.02:  # the curve reached the critical region
                d  = y - x
                sl = slice(n - n_fit, n)
                xc = np.polyval(np.polyfit(d[sl], x[sl], 2), 0.0)
                Pc = np.exp(np.polyval(np.polyfit(d[sl], np.log(P[sl]), 2), 0.0))
                crit = (Pc, xc)
            return x, y, P, crit
    return None

def crossings(c, P, z):
    """Pressures where the composition array c crosses z (linear interpolation)."""
    out = []
    for k in range(len(c) - 1):
        if (c[k] - z)*(c[k+1] - z) <= 0 and c[k] != c[k+1]:
            w = (z - c[k])/(c[k+1] - c[k])
            out.append(P[k] + w*(P[k+1] - P[k]))
    return out

isotherms = {}
for TinK in T_env_grid:
    res = trace_isotherm(TinK)
    if res is not None:
        isotherms[TinK] = res
print('Phase envelopes: %d of %d temperatures traced' % (len(isotherms), len(T_env_grid)))

envelopes = {}
for z in Z_ENV:
    bub, dew_lo, dew_hi, crit_T = [], [], [], []
    for TinK, (x, y, P, crit) in isotherms.items():
        xb = np.r_[x, crit[1]] if crit else x
        yb = np.r_[y, crit[1]] if crit else y
        Pb = np.r_[P, crit[0]] if crit else P
        pb = crossings(xb, Pb, z)
        if pb:
            bub.append((TinK, pb[0]))
        pd_ = crossings(yb, Pb, z)
        if len(pd_) >= 1:
            dew_lo.append((TinK, pd_[0]))
        if len(pd_) >= 2:
            dew_hi.append((TinK, pd_[-1]))
        if crit:
            crit_T.append((TinK, crit[1], crit[0]))
    # critical point of composition z: where xc(T) crosses z
    zc = None
    ct = np.array(crit_T)
    for k in range(len(ct) - 1):
        if (ct[k, 1] - z)*(ct[k+1, 1] - z) <= 0:
            w = (z - ct[k, 1])/(ct[k+1, 1] - ct[k, 1])
            zc = (ct[k, 0] + w*(ct[k+1, 0] - ct[k, 0]), ct[k, 2] + w*(ct[k+1, 2] - ct[k, 2]))
            break
    envelopes[z] = dict(bub=np.array(bub), dew_lo=np.array(dew_lo),
                        dew_hi=np.array(dew_hi), crit=zc)

def psat_pr(comp, TinK, P0):
    """Vapor pressure [bar] of a pure component from the PR EoS (successive substitution)."""
    R  = 83.14462618                                   # cm3 bar mol^-1 K^-1
    kappa = 0.37464 + 1.54226*comp.w - 0.26992*comp.w**2
    a  = 0.45724*R**2*comp.Tc**2/comp.Pc*(1 + kappa*(1 - np.sqrt(TinK/comp.Tc)))**2
    b  = 0.07780*R*comp.Tc/comp.Pc
    P  = P0
    for _ in range(500):
        A = a*P/(R*TinK)**2
        B = b*P/(R*TinK)
        Z = np.roots([1, -(1 - B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)])
        Z = np.sort(Z[np.abs(Z.imag) < 1e-10].real)
        if len(Z) < 2:
            # one real root only: move P toward the two-phase region
            P = P*0.5 if Z[0] < 0.3 else P*1.5
            continue
        lnphi = lambda z: (z - 1 - np.log(z - B)
                           - A/(2*np.sqrt(2)*B)*np.log((z + (1 + np.sqrt(2))*B)/(z + (1 - np.sqrt(2))*B)))
        ratio = np.exp(lnphi(Z[0]) - lnphi(Z[-1]))
        P = P*ratio
        if abs(ratio - 1) < 1e-10:
            break
    return P

def psat_curve(comp, T_lo, n=200):
    Ts = np.linspace(T_lo, comp.Tc*0.999, n)
    Ps, P0 = [], comp.Pc*0.01
    for TinK in Ts:
        P0 = psat_pr(comp, TinK, P0)
        Ps.append(P0)
    return Ts, np.array(Ps)

T_co2,  P_co2  = psat_curve(co2, 250.0)
T_c2oh, P_c2oh = psat_curve(c2oh, 300.0)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
8. Draw the figure.
(A) Calculated vapor-liquid coexistence curves: the bubble curve is traced by continuation and the
loop is closed with the dew curve. (B) Phase envelopes at constant composition and vapor pressure curves.
(C) Pressure residuals. (D) Parity plot of y2 = 1 - y1 (log-log).
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
centimeter = 1 / 2.54

fig, axs = plt.subplots(2,2,
                        figsize=(19*centimeter, 16*centimeter),
                        dpi=500, gridspec_kw = {'wspace':0.35, 'hspace':0.35}
                        )
axes = axs.ravel()

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=TinKs[0], vmax=TinKs[-1])
for TinK in TinKs:
    x1s, Ys, Pbubls, reason = trace_bubble_arc(Xof, TinK, model, [0.9, 0.1])
    x1s, Ys, Pbubls = truncate_curve(x1s, Ys, Pbubls, Xof)
    y1s = Ys[:, 0]
    # Closed loop: bubble curve up to the mixture critical point, then the dew curve back down
    x_loop = np.r_[x1s, y1s[::-1]]
    P_loop = np.r_[Pbubls, Pbubls[::-1]]
    axes[0].plot(x_loop, P_loop/10, color=cmap(norm(TinK)))
axes[0].set_xlim([0,1])
axes[0].set_ylim([0,18])
axes[0].text(0.05,0.9,r'CO$_\mathbf{2}$/C$_\mathbf{2}$H$_\mathbf{5}$OH',fontweight='bold',
             horizontalalignment='left',
             transform=axes[0].transAxes)
axes[0].set_xlabel(r'$x_1,~y_1$')
axes[0].set_ylabel(r'$p$ [MPa]')

# Panel B: phase envelopes at constant composition (blue: bubble, green: dew) and pure-component vapor pressures
c_bub, c_dew = 'tab:blue', 'tab:green'
for z, env in envelopes.items():
    bub, lo, hi, zc = env['bub'], env['dew_lo'], env['dew_hi'], env['crit']
    # One path along the envelope: lower dew branch (increasing T), upper (retrograde) dew branch,
    # critical point, bubble branch (decreasing T). A smoothing spline through this path removes
    # the kinks of the 3 K temperature grid.
    dew = lo[np.argsort(lo[:, 0])]
    if len(hi):
        dew = np.vstack([dew, hi[np.argsort(hi[:, 0])[::-1]]])
    bub = bub[np.argsort(bub[:, 0])[::-1]]
    if zc is not None:
        path = np.vstack([dew, zc, bub])
        n_dew = len(dew)                                  # index of the critical point in path
    else:
        path = np.vstack([dew, bub[::-1]])
        n_dew = len(dew) - 1
    Ts, Ps = path[:, 0], path[:, 1]/10
    sc = np.array([Ts.max() - Ts.min(), Ps.max() - Ps.min()])
    tck, u = splprep([Ts/sc[0], Ps/sc[1]], s=len(Ts)*2e-5, k=3)
    uu_d = np.linspace(0, u[n_dew], 300)
    uu_b = np.linspace(u[n_dew], 1, 300)
    Td, Pd = splev(uu_d, tck)
    Tb, Pb = splev(uu_b, tck)
    axes[1].plot(Td*sc[0], Pd*sc[1], color=c_dew)
    axes[1].plot(Tb*sc[0], Pb*sc[1], color=c_bub)
    k_top = np.argmax(Ps)                                 # label each envelope at its highest pressure
    axes[1].text(Ts[k_top], Ps[k_top] + 0.3, r'$z_1=%.1f$' % z, ha='center', va='bottom', fontsize=6)
axes[1].plot(T_co2,  P_co2/10,  color='k')
axes[1].plot(T_c2oh, P_c2oh/10, color='k')
axes[1].plot([co2.Tc, c2oh.Tc], [co2.Pc/10, c2oh.Pc/10], 'o', ms=3, color='k', zorder=3)
axes[1].text(co2.Tc - 3, co2.Pc/10 + 0.4, r'CO$_2$', ha='right')
axes[1].text(c2oh.Tc + 3, c2oh.Pc/10 - 0.3, r'C$_2$H$_5$OH', ha='left', va='top', fontsize=6)
axes[1].plot([], [], color=c_bub, label=r'Bubble point ($x_1$ fixed)')
axes[1].plot([], [], color=c_dew, label=r'Dew point ($y_1$ fixed)')
axes[1].plot([], [], color='k', label='Vapor pressure')
axes[1].set_xlim([250, 550])
axes[1].set_ylim([0, 24])
axes[1].set_xlabel(r'$T$ [K]')
axes[1].set_ylabel(r'$p$ [MPa]')
axes[1].legend(frameon=False, loc='upper left', fontsize=6)

# Panel C: pressure residuals against the liquid composition
axes[2].scatter(x_pts, dP, c=T_pts, cmap=cmap, norm=norm, s=3, linewidths=0, zorder=3)
axes[2].axhline(0, color='k', lw=.5, zorder=1)
axes[2].set_xlim([0,1])
axes[2].set_xlabel(r'$x_1$')
axes[2].set_ylabel(r'$\%\Delta p$')

# Panel D: parity plot of the ethanol mole fraction in the vapor, y2 = 1 - y1 (log-log)
y2_lit = 1 - y_lit
y2_cal = 1 - y_cal
ok = has_y & (y2_lit > 0) & (y2_cal > 0)
axes[3].scatter(y2_lit[ok], y2_cal[ok], c=T_pts[ok], cmap=cmap, norm=norm,
                s=3, linewidths=0, zorder=3)
lo = 10**np.floor(np.log10(min(y2_lit[ok].min(), y2_cal[ok].min())))
hi = 10**np.ceil(np.log10(max(y2_lit[ok].max(), y2_cal[ok].max())))
axes[3].plot([lo, hi], [lo, hi], color='k', lw=.5, zorder=1)
axes[3].set_xscale('log')
axes[3].set_yscale('log')
axes[3].set_xlim([lo, hi])
axes[3].set_ylim([lo, hi])
axes[3].set_xlabel(r'$y_{2,\mathrm{lit}}$')
axes[3].set_ylabel(r'$y_{2,\mathrm{cal}}$')
print('Points in panel D: %d (y2_lit = 0 excluded: %d)' % (ok.sum(), (has_y & ~ok).sum()))

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.03, shrink=0.5)
cbar.set_label(r'$T$ [K]')
cbar.ax.tick_params(width=.5, length=3)
cbar.outline.set_linewidth(.5)

for k, panel in enumerate(['A', 'B', 'C', 'D']):
    axes[k].text(-0.18, 1.0, panel, fontweight='bold', fontsize=8, transform=axes[k].transAxes)

plt.savefig('figure2.3.pdf',dpi=500,bbox_inches='tight')
