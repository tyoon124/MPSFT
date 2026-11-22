# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 3 Transport and Dynamic Anomalies
Example 3.2
The critical enhancement of the transport properties is illustrated here for pure CO2 using the
reference correlation of Vesovic et al. Both the viscosity eta and the thermal conductivity lambda
are written as the sum of three contributions, X(rho,T) = X_0(T) + X_exc(rho,T) + X_crit(rho,T),
i.e., the dilute-gas term, the excess (background) term, and the critical enhancement arising from
long-range density fluctuations. The critical enhancement is evaluated with the mode-coupling
crossover model of Olchowy and Sengers, in which the correlation length is obtained from the
symmetrized compressibility chi. The required thermodynamic properties (C_p, C_v, (drho/d P)_T) are
taken from the Span-Wagner EoS via CoolProp (https://github.com/CoolProp/CoolProp) after removing
its non-analytic terms, which otherwise produce spurious double peaks around rho_crit.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and make basic plot setting.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import json
import numpy as np
from scipy.special import expit
import CoolProp.CoolProp as CP
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
2. Define the constants of the correlation (Vesovic et al., 1990).
   Units: T [K], rho [kg/m3], eta [uPa s], lambda [mW m^-1 K^-1].
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
kB     = 1.380649e-23          # Boltzmann constant [J/K]
Tc     = 304.107               # critical temperature [K]       (Table 5)
rhoc   = 467.69                # critical density [kg/m3]       (Table 5)
Pc     = 7.3721e6              # critical pressure [Pa]         (Table 5)

epsk   = 251.196               # energy scaling parameter [K]   (Eq. 6)
Ramp   = 1.01                  # universal amplitude R          (Eq. 37)
zexp   = 0.06                  # universal exponent z           (Eq. 37)
nu     = 0.630                 # critical exponents             (Eq. 43)
gam    = 1.2415
xi0    = 1.5e-10               # critical amplitudes [m], [-]   (Eq. 44)
Gam0   = 0.052
qDinv  = 2.3e-10               # cutoff wavenumber^-1 [m]       (Table 5)
qDinvS = 4.0e-10               # for the simplified equation    (Eq. 61)
Tref   = 450.0                 # reference temperature [K]      (Eq. 48)

# Table 1: zero-density viscosity, thermal conductivity, ideal-gas c_int
a_eta = np.array([0.235156, -0.491266, 5.211155e-2, 5.347906e-2, -1.537102e-2])
b_lam = np.array([0.4226159, 0.6280115, -0.5387661, 0.6735941, 0.0, 0.0,
                  -0.4362677, 0.2255388])
c_int = np.array([2.387869e-2, 4.350794, -10.33404, 7.981590, -1.940558])

# Table 8: excess thermal conductivity (i = 1-4) and gas-phase viscosity
d_lam = np.array([2.447164e-2, 8.705605e-5, -6.547950e-8, 6.594919e-11])
e_eta = {1: 3.6350734e-3, 2: 7.209997e-5, 7: 3.00306e-20}

# Blending function between gas- and liquid-phase viscosity (Eq. 74)
rhos, Ts, Zbl = 467.689, 302.0, 1.0

# Gauss-Legendre nodes for the crossover integral
xGL, wGL = np.polynomial.legendre.leggauss(64)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define the thermodynamic model.
   The Span-Wagner EoS (CoolProp) is copied without its three non-analytic
   terms. These terms make chi~ and c_p non-monotonic around rho_c within a
   few K of Tc and produce spurious double peaks in the enhancement.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def register_smooth_co2(name='CO2_noNA'):
    if name in CP.get_global_param_string('FluidsList').split(','):
        return name
    j = json.loads(CP.get_fluid_param_string('CO2','JSON'))
    j = j[0] if isinstance(j, list) else j
    j['EOS'][0]['alphar'] = [t for t in j['EOS'][0]['alphar']
                             if t['type'] != 'ResidualHelmholtzNonAnalytic']
    j['INFO'].update(NAME=name, ALIASES=[], CAS='124-38-9-'+name,
                     REFPROP_NAME=name)
    CP.add_fluids_as_JSON('HEOS', json.dumps([j]))
    return name

fluid = register_smooth_co2()
AS    = CP.AbstractState('HEOS', fluid)

def chi_tilde(rho, T):
    """Symmetrized compressibility, Eq. (40)."""
    try:
        AS.update(CP.DmassT_INPUTS, float(rho), float(T))
        return Pc*rho/rhoc**2*AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
    except ValueError:
        return np.nan

def thermo(rho, T):
    """c_p, c_v [J/kg/K]; NaN inside the two-phase dome."""
    try:
        AS.update(CP.DmassT_INPUTS, float(rho), float(T))
        if AS.phase() == CP.iphase_twophase:
            return np.nan, np.nan
        return AS.cpmass(), AS.cvmass()
    except ValueError:
        return np.nan, np.nan

def correlation_length(rho, T):
    """Correlation length xi [m], Eqs. (46)-(49)."""
    if T > 445.0:
        return correlation_length(rho, 445.0)*np.exp(-(T-445.0)/10.0)
    dchi = chi_tilde(rho, T) - chi_tilde(rho, Tref)*Tref/T
    if not np.isfinite(dchi) or dchi <= 0.0:
        return 0.0
    return xi0*(dchi/Gam0)**(nu/gam)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Define the zero-density and excess (background) contributions.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def eta0(T):
    """Dilute-gas viscosity [uPa s], Eqs. (3)-(6)."""
    lnTs = np.log(np.asarray(T, float)/epsk)
    return 1.00697*np.sqrt(T)/np.exp(sum(a*lnTs**i for i, a in enumerate(a_eta)))

def lam0(T):
    """Dilute-gas thermal conductivity [mW m^-1 K^-1], Eqs. (29)-(31)."""
    T  = np.asarray(T, float)
    cint = 1.0 + np.exp(-183.5/T)*sum(c*(T/100.0)**(2-i)
                                      for i, c in enumerate(c_int, start=1))
    S  = sum(b/(T/epsk)**i for i, b in enumerate(b_lam))
    return 0.475598*np.sqrt(T)*(1.0 + 2.0*cint/5.0)/S

def dlam_excess(rho):
    """Excess thermal conductivity [mW m^-1 K^-1], Eq. (63)."""
    return sum(d*np.asarray(rho, float)**i for i, d in enumerate(d_lam, start=1))

def deta_gas(rho):
    """Excess viscosity of the gas phase [uPa s], Eq. (67)."""
    return sum(e*np.asarray(rho, float)**i for i, e in e_eta.items())

def eta_liquid(rho, T):
    """Liquid-phase background viscosity [uPa s], Eqs. (70)-(72)."""
    B   = 18.56 + 0.014*T
    V0  = 7.41e-4 - 3.3e-7*T
    flu = B*(1.0/np.asarray(rho, float) - V0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(flu > 0, 1.0/flu, np.inf)

def eta_background(rho, T):
    """Background viscosity: Eq. (76) without the critical term."""
    rho, T = np.asarray(rho, float), np.asarray(T, float)
    e0, deg, el = eta0(T), deta_gas(rho), eta_liquid(rho, T)
    wTg, wTl = expit(Zbl*(T-Ts)), expit(-Zbl*(T-Ts))
    wrg, wrl = expit(-Zbl*(rho-rhos)), expit(Zbl*(rho-rhos))
    liq = np.where(np.isfinite(el), (el-e0)*wTl*wrl, 0.0)
    return e0 + deg*wTg + deg*wTl*wrg + liq

def lam_background(rho, T):
    """Background thermal conductivity, Eq. (52)."""
    return lam0(T) + dlam_excess(rho)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Define the critical enhancements (Olchowy-Sengers crossover model).
   Omega (Table 3) is evaluated as the real integral equivalent to its
   partial-fraction form; H (Table 4) is evaluated with the cubic roots.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def F(r, s):
    """F(r,s) = int_0^s dx/(r + cos x), complex-safe."""
    r  = complex(r)
    sq = np.sqrt(1.0 - r*r + 0j)
    t  = np.tan(s/2.0)
    return np.log((1 + r + sq*t)/(1 + r - sq*t))/sq

def H_viscosity(yD, yd, ya, yb, yg):
    """Crossover function H of Table 4."""
    yeta, ynu = (yd + yb/ya)/yD, yg*yd/yD
    vi  = -np.roots([1.0, yeta, yg, ynu]).astype(complex)
    den = yeta*yg - ynu
    p   = np.array([
        yeta**4 - 4*yeta**2*yg - 2*yeta**2 + 2*yeta*ynu + 3*yg**2 + 4*yg + 1,
        (yeta*yg - ynu)*(yeta**2 - 3*yg - 2) + yg**2*(1 + yg)**2/den,
        yeta*ynu*(yeta**2 - 3*yg - 2) + ynu**2 + ynu*yg*(1 + yg)**2/den])
    M = np.empty((3,3), complex)
    for i in range(3):
        M[0,i], M[1,i], M[2,i] = 1.0, vi.sum() - vi[i], np.prod(vi)/vi[i]
    N = np.linalg.solve(M, p)
    h = ((3*yg*yeta + 1.5*yeta - yeta**3 - ynu)*yD
         + (yeta**2 - 2*yg - 1.25)*np.sin(yD)
         - yeta*np.sin(2*yD)/4.0 + np.sin(3*yD)/12.0
         + (yg*(1 + yg))**1.5/(ynu - yg*yeta)
         *np.arctan(np.sqrt(yg/(1 + yg))*np.tan(yD)))
    return float(np.real(h + sum(N[i]*F(vi[i], yD) for i in range(3))))

def critical_enhancement(rho, T, eta_bar, lam_bar, simplified=False):
    """Return (Delta_c lambda [mW m^-1 K^-1], Delta_c eta [uPa s]).
    Full model: Eqs. (38),(39). simplified=True: Eqs. (58)-(61), Delta_c eta=0."""
    cp, cv = thermo(rho, T)
    if not np.isfinite(cp):
        return np.nan, np.nan
    xi = correlation_length(rho, T)
    if xi <= 0.0:
        return 0.0, 0.0
    eta  = eta_bar*1e-6                         # Pa s
    lam  = lam_bar*1e-3                         # W/(m$\cdot$K)
    pref = rho*cp*Ramp*kB*T/(6.0*np.pi*eta*xi)  # W/(m$\cdot$K)

    if simplified:
        q   = xi/qDinvS
        om  = 2/np.pi*((cp - cv)/cp*np.arctan(q) + cv/cp*q)
        om0 = -2/np.pi*np.expm1(-1.0/(1.0/q + (q*rhoc/rho)**2/3.0))
        return pref*(om - om0)*1e3, 0.0

    q   = xi/qDinv
    sq  = np.sqrt(1.0 + q*q)
    yD  = np.arctan(q)
    yd  = (np.arctan(q/sq) - yD)/sq
    ya  = rho*kB*T/(8.0*np.pi*eta**2*xi)
    yb  = lam/(eta*(cp - cv))
    yg  = cv/(cp - cv)
    # Omega: 2/pi/(1+yg) * int_0^yD (cos^2 x + yg)^2 / D(cos x) dx
    x   = 0.5*yD*(xGL + 1.0)
    c   = np.cos(x)
    D   = c**4 + yD*ya*c**3 + (yg + yb + ya*yd)*c**2 + yg*ya*yD*c + yg*ya*yd
    om  = 2/np.pi/(1 + yg)*0.5*yD*np.sum(wGL*(c*c + yg)**2/D)
    om0 = -np.expm1(-1.0/(1.0/q + (q*rhoc/rho)**2/3.0)) \
          /((1 + ya*(yD + yd) + yb/(1 + yg))*np.pi/2)
    dlam = pref*(om - om0)*1e3
    if q < 1e-4:                                 # H -> 0 as xi -> 0
        return dlam, 0.0
    deta = eta_bar*(np.exp(zexp*H_viscosity(yD, yd, ya, yb, yg)) - 1.0)
    return dlam, deta

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Define the overall correlation, Eqs. (75) and (76), and check it against
   the values given at the end of Appendix IV for checking computer code.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def transport(rho, T, simplified=False):
    """Return a dict with all contributions; rho and T broadcast."""
    rho, T = np.broadcast_arrays(np.asarray(rho, float), np.asarray(T, float))
    eb, lb = eta_background(rho, T), lam_background(rho, T)
    dl, de = np.empty_like(rho), np.empty_like(rho)
    for idx in np.ndindex(rho.shape):
        dl[idx], de[idx] = critical_enhancement(rho[idx], T[idx],
                                                eb[idx], lb[idx], simplified)
    return dict(eta=eb + de, lam=lb + dl, eta0=eta0(T), lam0=lam0(T),
                eta_bar=eb, lam_bar=lb, dc_eta=de, dc_lam=dl)

def density_TP(T, P):
    return np.vectorize(lambda t, p: CP.PropsSI('Dmass','T',t,'P',p,fluid))(T, P)

check = [  # T [K], P [MPa], rho [kg/m3], lambda [mW m^-1 K^-1], eta [uPa s]
    (220,  0.1,    2.440,  10.90,  11.06),
    (300,  0.1,    1.773,  16.77,  15.02),
    (800,  0.1,    0.662,  56.65,  35.09),
    (304,  7.0,  254.3205, 42.52,  20.89),
    (220, 15.0, 1194.86,  187.50, 274.22),
    (300, 50.0, 1029.27,  137.61, 133.15),
    (800, 75.0,  407.828,  78.47,  48.62)]

print('  T/K  P/MPa   rho/kg m-3 |  lam(ref)  lam(calc)  dev% |  eta(ref)  eta(calc)  dev%')
for TinK, PinMPa, rho, lam_ref, eta_ref in check:
    res = transport(rho, TinK)
    lam, eta = float(res['lam']), float(res['eta'])
    print('%5d %6.1f %12.4f | %8.2f %10.2f %5.2f | %8.2f %10.2f %5.2f'
          % (TinK, PinMPa, rho, lam_ref, lam, 100*(lam/lam_ref - 1),
             eta_ref, eta, 100*(eta/eta_ref - 1)))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
7. Perform the calculations and plot.
Fig.3.2: 
    A: Viscosity along isotherms
    B: critical enhancement of viscosities along some isotherms.
    C: three-term decomposition of eta along a supercritical isobar
Fig.3.3:
    A: Thermal conducitivity along isotherms
    B: critical enhancement of thermal conductivity along some isotherms.
    C: three-term decomposition of lambda along a supercritical isobar
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
centimeter = 1 / 2.54
TinKs  = [304.65, 305.25, 307.95, 313.15, 323.15, 348.15]
colors = sns.color_palette('viridis',n_colors=len(TinKs))
rhos_plot = np.linspace(20, 900, 300)
PinPa = 7.5e6
TinKs_iso = np.concatenate([np.linspace(304.4, 312, 1000), np.linspace(312.2, 360, 200)])

# 7-(1) Transport properties along the isotherms (shared by Figs. 3.2 and 3.3)
res_T = [transport(rhos_plot, TinK) for TinK in TinKs]

# 7-(2) Simplified critical enhancement of lambda, Eq. (58), along one isotherm
TinK_s = 307.95
dl_s = [critical_enhancement(r, TinK_s, float(eta_background(r, TinK_s)),
                             float(lam_background(r, TinK_s)), simplified=True)[0]
        for r in rhos_plot]

# 7-(3) Transport properties along the supercritical isobar
rho_iso = density_TP(TinKs_iso, PinPa)
res_P   = transport(rho_iso, TinKs_iso)
decomp  = sns.color_palette('viridis', n_colors=3)

# 7-(4) Fig. 3.2: viscosity
fig32, ax32 = plt.subplots(1, 3, figsize=(19*centimeter, 6.5*centimeter), dpi=500,
                           gridspec_kw={'wspace':0.35})
for i, TinK in enumerate(TinKs):
    ax32[0].plot(rhos_plot, res_T[i]['eta'], color=colors[i], label='%.2f K' % TinK)
    ax32[1].plot(rhos_plot, 100*res_T[i]['dc_eta']/res_T[i]['eta_bar'], color=colors[i])
for TinK, ls in [(TinKs[0],'--'), (TinKs[-1],':')]:
    ax32[0].plot(rhos_plot, eta_background(rhos_plot, TinK), 'k', ls=ls,
                 label=r'$(\eta_0+\eta_\mathrm{exc})$ (%.0f K)' % TinK)
ax32[2].stackplot(TinKs_iso, res_P['eta0'], res_P['eta_bar'] - res_P['eta0'],
                  res_P['dc_eta'], colors=decomp, alpha=.8,
                  labels=[r'$\eta_0(T)$', r'$\eta_\mathrm{exc}(\rho,T)$',
                          r'$\eta_\mathrm{crit}(\rho,T)$'])


# 7-(5) Fig. 3.3: thermal conductivity
fig33, ax33 = plt.subplots(1, 3, figsize=(19*centimeter, 6.5*centimeter), dpi=500,
                           gridspec_kw={'wspace':0.35})
for i, TinK in enumerate(TinKs):
    ax33[0].plot(rhos_plot, res_T[i]['lam'], color=colors[i], label='%.2f K' % TinK)
    ax33[1].plot(rhos_plot, res_T[i]['dc_lam'], color=colors[i])
for TinK, ls in [(TinKs[0],'--'), (TinKs[-1],':')]:
    ax33[0].plot(rhos_plot, lam_background(rhos_plot, TinK), 'k', ls=ls,
                 label=r'$\lambda_0+\lambda_\mathrm{exc}$ (%.0f K)' % TinK)
ax33[2].stackplot(TinKs_iso, res_P['lam0'], res_P['lam_bar'] - res_P['lam0'],
                  res_P['dc_lam'], colors=decomp, alpha=.8,
                  labels=[r'$\lambda_0(T)$', r'$\lambda_\mathrm{exc}(\rho,T)$',
                          r'$\lambda_\mathrm{crit}(\rho,T)$'])


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
8. Decorate the figures and save.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
for axes in [ax32, ax33]:
    for ax in axes[:2]:
        ax.axvline(rhoc, color='gray', lw=.5, ls='-.')
        ax.set_xlabel(r'$\rho$ [kg m$^{-3}$]')
        ax.set_xlim([0,900])
    axes[2].set_xlabel(r'$T$ [K]')
    axes[2].set_xlim([TinKs_iso[0], TinKs_iso[-1]])
    axes[0].legend(frameon=False, fontsize=5, ncol=2)
    axes[2].legend(frameon=False)
    axes[0].text(0.975,0.05,r'CO$_\mathbf{2}$',fontweight='bold',
                 horizontalalignment='right', transform=axes[0].transAxes)
    axes[2].text(0.975,0.05,r'$p$ = %.1f MPa' % (PinPa/1e6),
                 horizontalalignment='right', transform=axes[2].transAxes)
    for ax, label in zip(axes, 'ABC'):
        ax.text(-0.25,1.0,label,fontweight='bold',transform=ax.transAxes)

ax32[0].set_xlim([20,800])
ax33[0].set_xlim([20,800])
ax32[0].set_ylabel(r'$\eta$ [$\mu$Pa s]')
ax32[1].set_ylabel(r'$\eta_\mathrm{crit}/(\eta_0+\eta_\mathrm{exc})$ [%]')
ax32[2].set_ylabel(r'$\eta$ [$\mu$Pa s]')
ax33[0].set_ylabel(r'$\lambda$ [mW m$^{-1}$ K$^{-1}$]')
ax33[1].set_ylabel(r'$\lambda_\mathrm{crit}$ [mW m$^{-1}$ K$^{-1}$]')
ax33[1].legend(frameon=False, fontsize=5)
ax33[2].set_ylabel(r'$\lambda$ [mW m$^{-1}$ K$^{-1}$]')

fig32.savefig('figure3.2.pdf',dpi=500,bbox_inches='tight')
fig33.savefig('figure3.3.pdf',dpi=500,bbox_inches='tight')