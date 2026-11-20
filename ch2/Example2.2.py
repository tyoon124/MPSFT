# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 2 Phase Behavior and Thermodynamic Anomalies
Example 2.2 
CEoS models have been extensively used to describe the phase behavior between CO2 and solutes or co-solvents. Here we demonstrate how the PhasePy package (https://github.com/gustavochm/phasepy) can be used to calculate the vapor--liquid equilibria of the \ce{CO2}--\ce{C2H5OH} binary system using PR EoS parameters in conjunction with the excess Gibbs free energy (non-random two-liquid, NRTL) model from Kim et al. The calculated isotherms are compared with experimental bubble and dew points compiled in c2oh.xlsx.
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

    # Starting point at a tiny x1 from the standard bubble-point solver, then polish it
    Y0, P0 = bubblePy(np.asarray(y_guess, float), 1.0, Xof(x_start), T, model)
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
5. Load the experimental data.
The measured bubble (x1) and dew (y1) points of CO2 (1) + ethanol (2) are read from c2oh.xlsx
(columns x1, y1, T(K), P(bar), u, Reference). Data within T_TOL of each isotherm are drawn.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TinKs   = [313.15, 333.15, 353.15, 373.15]
T_TOL   = 0.5                       # K, tolerance for matching the data to an isotherm
expdata = pd.read_excel('c2oh.xlsx')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. 
(1) Trace the bubble curve by continuation and close the loop with the dew curve to construct the vapor-liquid coexistence curves 
(2) Draw the Pxy diagram with the experimental data.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
centimeter = 1 / 2.54

fig, ax = plt.subplots(1,1,
                       figsize=(9*centimeter, 8*centimeter),
                       dpi=500, gridspec_kw = {'hspace':0.25,'wspace':0.25}
                       )

colors = sns.color_palette('viridis',n_colors=len(TinKs))
Xof = lambda x1: np.array([x1, 1-x1])
for i,TinK in enumerate(TinKs):
    x1s, Ys, Pbubls, reason = trace_bubble_arc(Xof, TinK, model, [0.9, 0.1])
    x1s, Ys, Pbubls = truncate_curve(x1s, Ys, Pbubls, Xof)
    y1s = Ys[:, 0]
    
    # Closed loop: bubble curve up to the mixture critical point, then the dew curve back down
    x_loop = np.r_[x1s, y1s[::-1]]
    P_loop = np.r_[Pbubls, Pbubls[::-1]]
    ax.plot(x_loop, P_loop/10, color=colors[i], label='%.2f K' % TinK)
    
    # Experimental data at this temperature: filled = bubble points, open = dew points
    sel = expdata[np.abs(expdata['T(K)'] - TinK) <= T_TOL]
    ax.plot(sel['x1'], sel['P(bar)']/10, 'o', markersize=2.5, color=colors[i],
            markeredgewidth=0.4, zorder=3)
    ax.plot(sel['y1'], sel['P(bar)']/10, 'o', markersize=2.5, markerfacecolor='w',
            markeredgecolor=colors[i], markeredgewidth=0.4, zorder=3)
ax.plot([], [], 'o', markersize=2.5, color='k', label='Exp. (bubble)')
ax.plot([], [], 'o', markersize=2.5, markerfacecolor='w', markeredgecolor='k',
        markeredgewidth=0.4, label='Exp. (dew)')
ax.legend(frameon=False)
ax.set_xlim([0,1])
ax.set_ylim([0,18])
ax.text(0.975,0.95,r'CO$_\mathbf{2}$/C$_\mathbf{2}$H$_\mathbf{5}$OH',fontweight='bold',
        horizontalalignment='right',
        transform=ax.transAxes)
ax.set_xlabel(r'$x_1,~y_1$')
ax.set_ylabel(r'$p$ [MPa]')

plt.savefig('figure2.3.pdf',dpi=500,bbox_inches='tight')