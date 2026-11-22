# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 3 Transport and Dynamic Anomalies
Example 3.3
This example computes the Rosenfeld reduced viscosity and residual entropy for CO2, CH4, and H2O
from CoolProp across a grid of supercritical (T,rho) states and produces the master curve for each
substance. The master curve for the Rosenfeld reduced viscosity eta is expressed using the empirical
formula:
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and make figure settings.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
import CoolProp as CP
import matplotlib as mpl
from scipy.optimize import curve_fit

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
fig, ax = plt.subplots(1, 3,
                       figsize=(19*centimeter, 7*centimeter),
                       dpi=500)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Constants and state-point grid.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
kB = 1.380649e-23        # Boltzmann constant [J/K]
NA = 6.02214076e23       # Avogadro constant [1/mol]
R = kB*NA                # Gas constant [J/(mol K)]

fluids  = ['CO2', 'CH4', 'Water']
labels  = [r'CO$_\mathbf{2}$', r'CH$_\mathbf{4}$', r'H$_\mathbf{2}$O']

TrRange = np.linspace(1.1, 2, 20)     # T/Tcrit
DrRange = np.linspace(0.1, 2, 1000)      # D/Dcrit
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=TrRange.min(), vmax=TrRange.max())
colors = cmap(norm(TrRange))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Calculate the reduced viscosity and the residual entropy using CoolProp.
   eta_tilde = eta / [rho^(2/3) (m kB T)^(1/2)]   (rho: number density)
   s_res/kB  = S_res,molar / R
   Also, we can fit an empirical equation to the collapsed data.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def fit(spls,a,b,c,d):
    return a*spls**(-2/3) + b*spls**c*np.exp(-d*spls)

for i,fluid in enumerate(fluids):
    Tcrit = CP.CoolProp.PropsSI('Tcrit', fluid)
    Pcrit = CP.CoolProp.PropsSI('Pcrit', fluid)
    Dcrit = CP.CoolProp.PropsSI('RHOMASS_CRITICAL', fluid)
    m = CP.CoolProp.PropsSI('molar_mass', fluid) / NA     # molecular mass [kg]

    Tr, Dr = np.meshgrid(TrRange, DrRange)
    TinKs = Tr.flatten() * Tcrit
    Dinkgm3s = Dr.flatten() * Dcrit

    eta = CP.CoolProp.PropsSI('viscosity','T',TinKs,'D',Dinkgm3s,fluid)
    rhoN = CP.CoolProp.PropsSI('Dmolar','T',TinKs,'D',Dinkgm3s,fluid) * NA
    sres = CP.CoolProp.PropsSI('Smolar_residual','T',TinKs,'D',Dinkgm3s,fluid) / R
    spls = -sres
    eta_tilde = eta / (rhoN**(2/3) * np.sqrt(m*kB*TinKs))
    
    for j,TinK in enumerate(np.unique(TinKs)):
        spls1 = spls[TinKs==TinK]
        eta_tilde1 = eta_tilde[TinKs==TinK]
        ax[i].plot(spls1,eta_tilde1,lw=1,color=colors[j],alpha=0.5)
        
    popt,pcov = curve_fit(fit,
                          spls,eta_tilde)
    splsINTERP = np.linspace(np.min(spls),np.max(spls))
    ax[i].plot(splsINTERP,fit(splsINTERP,*popt),
               color='k',ls='--',label='Fit')
    ax[i].set_xlabel(r'$-s^\mathrm{res}/k_\mathrm{B}$')
    ax[0].set_ylabel(r'$\tilde{\eta}=\eta \rho^{-2/3}(mk_\mathrm{B}T)^{-1/2}$')
    ax[i].set_xlim([0,1.8])
    ax[i].set_ylim([0.4,1.6])
    ax[i].legend(frameon=False)
    ax[i].text(-0.15,1.05,'ABC'[i],fontweight='bold',fontsize=7,transform=ax[i].transAxes)
    ax[i].text(0,1.05,labels[i],fontweight='bold',fontsize=7,transform=ax[i].transAxes)
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Add a colorbar for the reduced temperature (T_r = T/T_crit).
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
cbar.set_label(r'$T_\mathrm{r}=T/T_\mathrm{crit}$')
cbar.outline.set_linewidth(.5)
cbar.ax.tick_params(width=.5, length=3)

plt.savefig('figure3.4.pdf', dpi=300, bbox_inches='tight')
plt.show()
