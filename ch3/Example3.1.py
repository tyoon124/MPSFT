# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 3 Transport and Dynamic Anomalies
Example 3.1
Demonstrate the temperature and pressure dependence of the shear viscosity eta and the thermal
conductivity lambda of CO2 far from the gas-liquid critical point (Fig. 3.1). The script scans
temperature (220 <= T <= 800 K) along ten isobars, 4 <= p <= 100 MPa, using the CoolProp
Helmholtz-energy (HEOS) backend, which evaluates eta with the Laesecke-Muzny correlation and lambda
with the Huber et al. correlation. Both correlations follow the decomposition of Eqs. 3.1 and 3.3,
i.e., a zero-density term plus a density-dependent excess term. Two limiting regimes can be
identified in the figure. At high temperature and low pressure, the isobars converge toward a common curve, because
the excess terms become small relative to the zero-density limit. There eta and lambda increase with
T, consistent with the Chapman-Enskog behavior of Eqs.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and make figure settings.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
import CoolProp as CP
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

centimeter = 1/2.54
fig, ax = plt.subplots(1, 2, 
                       figsize=(16*centimeter, 10*centimeter),
                       dpi=500, gridspec_kw = {'hspace':0.25,'wspace':0.25})

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Basic properties of CO2
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
fluid = 'CO2'
Tcrit = CP.CoolProp.PropsSI('Tcrit', fluid)    

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Calculate the shear viscosity and thermal conductivity using CoolProp.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TinKs = np.linspace(220,800,1000)
PinMPas = [4,10,12,16,20,30,40,60,80,100]
PinPas = np.array(PinMPas) * 1e6
colors = sns.color_palette('viridis',n_colors=len(PinPas))

for i,PinPa in enumerate(PinPas):
    mus = CP.CoolProp.PropsSI('viscosity','T',TinKs,'P',PinPa,fluid)*1e6
    lambdas = CP.CoolProp.PropsSI('conductivity','T',TinKs,'P',PinPa,fluid)
    ax[0].plot(TinKs,mus,
               color=colors[i],
               label=r'$p=%d$ MPa' % (PinPa/1e6))
    ax[1].plot(TinKs,lambdas,
               color=colors[i],
               label=r'$p=%d$ MPa' % (PinPa/1e6))
        
ax[0].axvline(x=Tcrit, ls='--', color='k', lw=0.8, label=r'$T_\mathrm{crit}$')
ax[0].set_xlim([np.min(TinKs),np.max(TinKs)])
ax[0].set_ylim([0,250])
ax[0].set_xlabel(r'$T$ [K]')
ax[0].set_ylabel(r'$\eta$ [$\mu$Pa s]')

ax[0].legend(frameon=False)

ax[1].axvline(x=Tcrit, ls='--', color='k', lw=0.8, label=r'$T_\mathrm{crit}$')
ax[1].set_xlim([np.min(TinKs),np.max(TinKs)])
ax[1].set_ylim([0,0.25])
ax[1].set_xlabel(r'$T$ [K]')
ax[1].set_ylabel(r'$\lambda$ [W m$^{-1}$ K$^{-1}$]')
ax[1].legend(frameon=False)

ax[0].text(-0.2,1.0,'A',fontweight='bold',fontsize=7,transform=ax[0].transAxes)
ax[1].text(-0.2,1.0,'B',fontweight='bold',fontsize=7,transform=ax[1].transAxes)

plt.savefig('figure3.1.pdf', dpi=300, bbox_inches='tight')
plt.show()