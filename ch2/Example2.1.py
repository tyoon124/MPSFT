# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 2 Phase Behavior and Thermodynamic Anomalies
Example 2.1
Draw the vapor-liquid coexistence curves and p-rho isotherms of CO2 (A) and water (B). Several Python packages are available to calculate the vapor-liquid equilibrium of various substances. Here we use the CoolProp package (https://coolprop.org/) to compute the vapor-liquid coexistence curves and isotherms on the pressure-density (p-rho) diagrams shown in Fig. 2.1, using the Span-Wagner EoS (CO2) and the IAPWS-95 formulation (water) as the backends.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Install CoolProp
Run the cell below once to install CoolProp. You can skip this step if it is already installed in your environment.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'CoolProp', '--quiet'])
print('CoolProp installed successfully.')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Import packages
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import CoolProp.CoolProp as CP

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Compute the coexistence lines and isotherms, and visualize them.
CoolProp uses the Span-Wagner EoS for CO2 and the IAPWS-95 formulation for water.

Span, R., & Wagner, W. (1996). A new equation of state for carbon dioxide covering the fluid region from the triple point temperature to 1100 K at pressures up to 800 MPa. Journal of physical and chemical reference data, 25(6), 1509-1596.
Wagner, W., & Pruss, A. (2002). The IAPWS formulation 1995 for the thermodynamic properties of ordinary water substance for general and scientific use. Journal of physical and chemical reference data, 31(2), 387-535.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 3.0. Basic settings
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

# 3.1. Panel settings
# fluid: (isotherms in deg C, density range [kg/m3], pressure range [MPa], label positions (rho, p) for each isotherm)
FLUIDS = {'CO2':   {'tinCs':  [10, 20, 25, 28, 31, 35, 40, 50],
                    'xlim':   [100, 900],
                    'ylim':   [4, 10],
                    'labels': [(750, 4.5), (650, 5.5), (600, 6.3), (550, 6.8),
                               (450, 7.6), (400, 8.2), (450, 9.5), (200, 9.5)]},
          'Water': {'tinCs':  [300, 330, 350, 365, 374, 380, 400, 420],
                    'xlim':   [0, 800],
                    'ylim':   [8, 30],
                    'labels': [(720, 9.0), (660, 13.0), (620, 17.0), (560, 20.0),
                               (420, 22.6), (320, 24.6), (240, 27.0), (195, 29.0)]},
          }
panels = ['A', 'B']

centimeter = 1 / 2.54
fig, axes = plt.subplots(1,2,
                         figsize=(19*centimeter, 8*centimeter),
                         dpi=500, gridspec_kw = {'hspace':0.25,'wspace':0.25}
                         )

colors = sns.color_palette('viridis',n_colors=8)

for ax, panel, fluid in zip(axes, panels, FLUIDS):
    cfg = FLUIDS[fluid]
    Tcrit = CP.PropsSI('Tcrit',   fluid)
    Pcrit = CP.PropsSI('Pcrit',   fluid) / 1e6
    Dcrit = CP.PropsSI('rhocrit', fluid)
    Ttrpl = CP.PropsSI('Ttriple', fluid)

    # 3.2. Saturation lines (vapor-liquid equilibrium)
    TinKs   = np.linspace(Ttrpl, Tcrit, 2000)
    Dsatls  = CP.PropsSI('D', 'T', TinKs, 'Q', 0, fluid)
    Dsatgs  = CP.PropsSI('D', 'T', TinKs, 'Q', 1, fluid)
    PsatsLG = CP.PropsSI('P', 'T', TinKs, 'Q', 1, fluid) / 1e6
    ax.plot(Dsatls, PsatsLG, color='k')
    ax.plot(Dsatgs, PsatsLG, color='k')

    # 3.3. Isotherms
    Dinkgm3s = np.arange(10, 1000, 0.5)
    TinKs    = np.array(cfg['tinCs']) + 273.15
    for i, TinK in enumerate(TinKs):
        PinMPas = CP.PropsSI('P', 'D', Dinkgm3s, 'T', TinK, fluid) / 1e6
        ax.plot(Dinkgm3s, PinMPas, color=colors[i], zorder=0)
    ax.fill_betweenx(PsatsLG, Dsatgs, Dsatls, color='w', zorder=1)
    ax.plot(Dcrit, Pcrit, 'o', markersize=3, color='k')
    for tinC, (x, y) in zip(cfg['tinCs'], cfg['labels']):
        ax.text(x, y, f'{tinC + 273.15:.0f} K')

    ax.set_xlim(cfg['xlim'])
    ax.set_ylim(cfg['ylim'])
    ax.set_xlabel(r'$\rho$ [kg/m$^3$]')
    ax.set_ylabel(r'$p$ [MPa]')
    ax.text(-0.175, 0.95, panel, fontweight='bold', fontsize=8, transform=ax.transAxes)

plt.savefig('figure2.1.pdf',dpi=500,bbox_inches='tight')
