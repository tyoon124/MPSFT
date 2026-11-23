#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 5 Molecular Simulation Analyses
Example 5.2
This example extracts the local density distribution P(rho_loc) of the Lennard-Jones fluid at
rho^*=0.1 and 0.7 from MD trajectories, and fits an inverse-gamma distribution to the gas-like state
and a normal distribution to the liquid-like state. The analysis uses the Python packages numpy,
freud, and scipy.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import freud

import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import cumulative_trapezoid

font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 7}
mpl.rc('font', **font)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

for axis in ['x', 'y']:
    mpl.rcParams[f'{axis}tick.major.size'] = 3
    mpl.rcParams[f'{axis}tick.minor.size'] = 3
    mpl.rcParams[f'{axis}tick.major.width'] = .5
    mpl.rcParams[f'{axis}tick.minor.width'] = .5
    mpl.rcParams['lines.linewidth'] = .75

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define functions
- dump_to_array: Return (traj, box); optionally save them with np.savez.
- voronoi_volumes: Return volume array (N_timestep, N_atoms) of periodic Voronoi cells.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def dump_to_array(dump_file, out_file=None):
    """Return (traj, box); optionally save them with np.savez."""
    frames, boxes = [], []
    with open(dump_file) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            f.readline()                                   # timestep value
            f.readline()                                   # ITEM: NUMBER OF ATOMS
            n = int(f.readline())
            f.readline()                                   # ITEM: BOX BOUNDS
            bb = np.array([[float(v) for v in f.readline().split()[:2]] for _ in range(3)])
            lo, L = bb[:, 0], bb[:, 1] - bb[:, 0]   # bb = [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
            cols = f.readline().split()[2:]                # after "ITEM: ATOMS"
            ix = {c: i for i, c in enumerate(cols)}
            data = np.loadtxt([f.readline() for _ in range(n)], ndmin=2)
 
            for names, scaled in ((("x", "y", "z"), False),
                                  (("xs", "ys", "zs"), True),
                                  (("xu", "yu", "zu"), False)):
                if all(c in ix for c in names):
                    ci = [ix[c] for c in names]
                    break
            else:
                raise ValueError(f"No coordinate columns found in: {cols}")
 
            xyz = data[:, ci]
            if scaled:                                     # fractional -> box coordinates
                xyz = lo + xyz * L
            frame = np.column_stack([data[:, ix["id"]], xyz])
            frames.append(frame[np.argsort(frame[:, 0])])  # sort by atom id
            boxes.append(bb)
 
    traj, box = np.array(frames), np.array(boxes)
    if out_file:
        np.savez(out_file, traj=traj, box=box)
    return traj, box

def voronoi_volumes(traj, box, check=True):
    """Return volume array (N_timestep, N_atoms) of periodic Voronoi cells."""
    nt, n, _ = traj.shape
    vol = np.empty((nt, n))
    voro = freud.locality.Voronoi()
    for t in range(nt):
        lo, hi = box[t, :, 0], box[t, :, 1]
        L = hi - lo
        fbox = freud.Box(Lx=L[0], Ly=L[1], Lz=L[2])           # freud box is centered at the origin
        xyz = traj[t, :, 1:4] - 0.5 * (lo + hi)               # shift box center to the origin
        xyz = fbox.wrap(xyz)                                   # wrap into the box
        voro.compute((fbox, xyz))
        vol[t] = voro.volumes                                  # same order as the input points
        if check and abs(vol[t].sum() / fbox.volume - 1) > 1e-5:
            raise RuntimeError(f"frame {t}: sum of Voronoi volumes {vol[t].sum():.6f} "
                               f"!= box volume {fbox.volume:.6f}")
        if (t + 1) % 100 == 0:
            print(f"  {t + 1}/{nt} frames")
    return vol

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

centimeter = 1/2.54
rhos = np.arange(0.1,0.75,0.05)
colors = sns.color_palette('viridis',n_colors=len(rhos))

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Settings
Edit the parameters below to change the calculation.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# SETTINGS
W = 0.5          # width of the bottom subplot as a fraction of the figure width (0 < W <= 1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. Calculate and plot the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
centimeter = 1/2.54

fig,ax = plt.subplots(1,2,figsize=(18*centimeter, 7*centimeter), dpi=300)

# Voronoi cell volumes of every particle in every frame (volume.npy, provided in the GitHub
# repository). They are computed from the trajectory (traj.npz or the LAMMPS dump file of
# Example 5.1) only if volume.npy is missing.
for rho in (0.10, 0.70):
    addr = 'rho%.2f/' % rho
    if not os.path.isfile(addr+'volume.npy'):
        if os.path.isfile(addr+'traj.npz'):
            data = np.load(addr+'traj.npz')
            traj, box = data['traj'], data['box']
        else:
            fname = addr + 'traj_rho%.2f.lammpstrj' % rho
            traj, box = dump_to_array(fname, addr+'traj.npz')
        np.save(addr+'volume.npy', voronoi_volumes(traj, box))

Ds = np.load('rho0.10/volume.npy')**(-1)
Ds = Ds.flatten()
bins = np.linspace(0,0.5,20)
ax[0].hist(Ds,bins,density=True,edgecolor='w',linewidth=.5,color='r',alpha=.5)


Ds = np.load('rho0.70/volume.npy')**(-1)
Ds = Ds.flatten()
bins = np.linspace(0.3,1.1,20)
ax[1].hist(Ds,bins,density=True,edgecolor='w',linewidth=.5,color='b',alpha=.5)

from scipy import stats

x0 = np.linspace(0, 0.5, 400)
x1 = np.linspace(0.3, 1.1, 400)

# ax[0]: rho* = 0.1 -> inverse gamma (maximum-likelihood fit with loc fixed at 0)
Ds0 = 1.0 / np.load('rho0.10/volume.npy').flatten()
a, loc, scale = stats.invgamma.fit(Ds0, floc=0)
print(a,loc,scale)
ax[0].plot(x0, stats.invgamma.pdf(x0, a, loc, scale), 'k-', lw=1,
           label=r'$\rho^*=0.1$ (gas-like)')
ax[0].legend(frameon=False)

# ax[1]: rho* = 0.7 -> normal
Ds1 = 1.0 / np.load('rho0.70/volume.npy').flatten()
mu, sigma = stats.norm.fit(Ds1)
ax[1].plot(x1, stats.norm.pdf(x1, mu, sigma), 'k-', lw=1,
           label=r'$\rho^*=0.7$ (liquid-like)')
ax[1].legend(frameon=False,loc='upper right')
ax[1].set_ylim([0,5])
print(mu,sigma)

ax[0].set_xlabel(r'$\rho_\mathrm{loc}$')
ax[1].set_xlabel(r'$\rho_\mathrm{loc}$')
ax[0].set_ylabel(r'Probability density')
ax[1].set_ylabel(r'Probability density')
ax[0].text(-0.15,1,'A',fontweight='bold',fontsize=8,
           transform=ax[0].transAxes)
ax[1].text(-0.15,1,'B',fontweight='bold',fontsize=8,
           transform=ax[1].transAxes)

plt.savefig('Example5.2.pdf', dpi=500, bbox_inches='tight')
