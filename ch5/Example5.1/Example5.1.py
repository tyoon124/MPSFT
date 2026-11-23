#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 5 Molecular Simulation Analyses
Example 5.1 (Figure 5.1)
Radial distribution function, coordination number, and potential of mean force of the
Lennard-Jones (LJ) fluid on the supercritical isotherm T^*=1.40 from MD simulations (LAMMPS) at
0.10 <= rho^* <= 0.70. Reduced units: r^*=r/sigma, T^*=k_BT/epsilon, rho^*=rho sigma^3.
Panel A: g(r^*). Panel B: coordination number N_c at the first minimum of g(r^*).
Panel C: potential of mean force W(r^*)=-k_BT ln g(r^*) with the pair potential u(r^*).
The g(r) files (rdf_rhoX.XX.dat, columns r g n(r)) are written by the LAMMPS runs in the rhoX.XX
folders. Edit the settings block, then run this script directly.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline

font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 7}
mpl.rc('font', **font)
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = .5
mpl.rcParams['xtick.minor.width'] = .5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = .5
mpl.rcParams['ytick.minor.width'] = .5
mpl.rcParams['lines.linewidth'] = .75

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Settings
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

T_STAR = 1.40                                    # simulation temperature
RHOS_GR = np.arange(0.10, 0.75, 0.05)            # densities for panels A and B
RHOS_PMF = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]   # densities for panel C
G_MIN = 1e-3                                     # ln g diverges inside the core
R_MAX = 4.0
SAVE_FIG = '../figure5.1.pdf'
CMAP_RANGE = (0.0, 0.9)                          # avoid the light-yellow end of viridis

centimeter = 1/2.54
cmap = mpl.colormaps['viridis']

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- read_rdf: returns r and g(r) from rhoX.XX/rdf_rhoX.XX.dat
- color: color of a density on the viridis scale
- u_lj: LJ pair potential in reduced units
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def read_rdf(rho):
    # Two file formats occur:
    # (a) plain columns 'r g n(r)' below '#' header lines;
    # (b) LAMMPS fix ave/time blocks: a 2-number line (timestep, nrows) followed by rows
    #     'index r g coord'; g is averaged over all blocks.
    fname = 'rho%.2f/rdf_rho%.2f.dat' % (rho, rho)
    lines = []
    with open(fname) as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.strip():
                lines.append(line.split())
    if len(lines[0]) == 2:
        blocks = []
        i = 0
        while i < len(lines):
            nrow = int(float(lines[i][1]))
            block = np.array([[float(v) for v in x] for x in lines[i + 1:i + 1 + nrow]])
            blocks.append(block)
            i += 1 + nrow
        r = blocks[0][:, 1]
        g = np.mean([blk[:, 2] for blk in blocks], axis=0)
    else:
        data = np.array([[float(v) for v in x] for x in lines])
        r = data[:, 0]
        g = data[:, 1]
    return r, g

def color(rho, rhos):
    x = (rho - min(rhos)) / (max(rhos) - min(rhos))
    return cmap(CMAP_RANGE[0] + (CMAP_RANGE[1] - CMAP_RANGE[0]) * x)

def u_lj(r):
    return 4.0 * (r**-12 - r**-6)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Panels A and B: g(r) and the coordination number at the first minimum
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig, ax = plt.subplots(1, 3,
                       figsize=(19*centimeter, 6.5*centimeter),
                       dpi=500,
                       gridspec_kw={'wspace': 0.42})

Ncs = []
for rho in RHOS_GR:
    r, g = read_rdf(rho)
    g_sm = savgol_filter(g, 32, 3)
    g_sm[g == 0] = 0
    g_sm[g_sm < 0] = 0
    loc, _ = find_peaks(-g, prominence=0.02)
    n_r = cumulative_trapezoid(4.0 * np.pi * rho * g * r**2, r, initial=0)
    Ncs.append(n_r[loc[0]])
    print('rho* = %.2f  r_min = %.2f  N_c = %.2f' % (rho, r[loc[0]], n_r[loc[0]]))
    if np.any(np.isclose(rho, RHOS_PMF)):
        label = r'$\rho^*=%.2f$' % rho
    else:
        label = None
    ax[0].plot(r, g_sm, color=color(rho, RHOS_GR), label=label)

spline = UnivariateSpline(RHOS_GR, Ncs)
ax[1].plot(RHOS_GR, spline(RHOS_GR), color='k', marker='o', lw=0, markersize=2.5)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Panel C: potential of mean force W(r) = -k_BT ln g(r)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

r_u = np.linspace(0.9, R_MAX, 400)
ax[2].plot(r_u, u_lj(r_u), 'k--', label=r'$u(r^*)$')
for rho in RHOS_PMF:
    r, g = read_rdf(rho)
    mask = (g > G_MIN) & (r <= R_MAX)
    g_sm = savgol_filter(g[mask], 31, 3)
    g_sm = np.clip(g_sm, G_MIN, None)
    pmf = -T_STAR * np.log(g_sm)
    ax[2].plot(r[mask], pmf, color=color(rho, RHOS_PMF), label=r'$\rho^*=%.2f$' % rho)
ax[2].axhline(0, color='gray', lw=0.4)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Axes, labels, and save
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

ax[0].set_xlim([0, 4])
ax[0].set_ylim([-0.01, 2.3])
ax[0].legend(frameon=False, fontsize=5, ncol=1, loc='upper right', handlelength=1.5)
ax[0].set_xlabel(r'$r^*=r/\sigma$')
ax[0].set_ylabel(r'$g(r^*)$')
ax[1].set_xlabel(r'$\rho^*=\rho\sigma^3$')
ax[1].set_ylabel(r'$N_\mathrm{c}$')
ax[2].set_xlim([0.9, R_MAX])
ax[2].set_ylim([-1.2, 0.9])
ax[2].set_xlabel(r'$r^*=r/\sigma$')
ax[2].set_ylabel(r'$W(r^*)/\epsilon$')
ax[2].legend(frameon=False, fontsize=5, ncol=2)
for a, lab in zip(ax, ['A', 'B', 'C']):
    a.text(-0.26, 1, lab, transform=a.transAxes, fontweight='bold', fontsize=8)

plt.savefig(SAVE_FIG, dpi=500, bbox_inches='tight')
print('saved', SAVE_FIG)
