#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 5 Molecular Simulation Analyses
Example 5.4
Hydrogen-bond network and tetrahedral order of water: This example characterizes the local structure
of liquid water at 298, 423, 453, 473, 503, and 523 K with two geometric descriptors computed from
MD trajectories: the number of hydrogen bonds per molecule and the tetrahedral order parameter. Both
are the molecular analogs of the descriptors used for the LJ fluid in this section. The
hydrogen-bond criterion plays the role of the distance cutoff of the cluster analysis, and the
tetrahedral order parameter plays the role of the local density, because both ask how the
neighborhood of a molecule is arranged. The analysis uses the Python package numpy.

The traj.npz files in the GitHub repository contain every tenth frame of the production runs (251 of
2501 frames, float32) to limit the file size, and hbstat.npy and tetrahedral.npy hold the results
for these frames (they are recomputed only if missing). The averages therefore differ from the values
quoted in the book, which use all 2501 frames, by less than 0.01.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Define functions
- boxit: Reducing a vector xs into a periodic box
- diffvec: Difference between vectors xs1 and xs2
- count_hbonds
- calc_tetrahedral_order: Calculates the tetrahedral order parameter (q) for each oxygen atom
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def boxit(xs,L):
    ''' Reducing a vector xs into a periodic box 
    of size Ls '''
    xs = np.array(xs)
    xs = np.mod(xs,L)
    xs[xs < 0] += L    
    return xs

def diffvec(xs1, xs2, L):
    ''' Difference between vectors xs1 and xs2 
    in a box size of Ls, considering the PBC '''
    xs1 = np.array(xs1)
    xs2 = np.array(xs2)
    ds = xs2 - xs1    
    ds = boxit(ds,L)
    ds[ds > L/2] = ds[ds > L/2] - L
    
    return ds

def count_hbonds(traj_O, traj_H, traj_water_O, Ls,
                 cutoff_OO=3.3, cutoff_OH=2.5, angle_min=95.7):

    def pbc_diff(a, b, box):
        ds = b[None, :, :] - a[:, None, :]
        ds -= np.round(ds / box) * box  
        return ds

    n_snap = traj_O.shape[0]
    total_count = 0
    total_H = 0
    
    hblists = np.zeros((traj_O.shape[1],n_snap)) # number of molecules x times
    for t,L in zip(range(n_snap),Ls):
        O_donor = traj_O[t]
        O_donor = np.repeat(O_donor, 2, axis=0)
        H_donor = traj_H[t]
        O_accept = traj_water_O[t]
        O_accept = np.repeat(O_accept, 2, axis=0)

        # 벡터 계산
        r_OO = pbc_diff(O_donor, O_accept, L)  
        r_OH = pbc_diff(H_donor, O_accept, L)
       
        d_OO = np.linalg.norm(r_OO, axis=2)
        d_OH = np.linalg.norm(r_OH, axis=2)

        a_vec = - H_donor[:, None, :] + O_donor[:, None, :]  
        b_vec = O_accept[None, :, :] - H_donor[:, None, :]  
        dot = np.sum(a_vec * b_vec, axis=2)
        cos_theta = dot / (np.linalg.norm(a_vec, axis=2) * np.linalg.norm(b_vec, axis=2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_theta))
 
        mask = (d_OO <= cutoff_OO) & (d_OH <= cutoff_OH) & (theta >= angle_min)
        
        # Sum the number of hydrogen bonds for each molecule
        for EachAtom in range(traj_O.shape[1]):            
            hblists[EachAtom,t] = np.sum(mask[EachAtom:EachAtom+2,:])
                    
        hbond_counts = np.sum(mask, axis=1)
        total_count += np.sum(hbond_counts)
        total_H += len(hbond_counts)

    avg_hbond_count = total_count / total_H*2
    
    return avg_hbond_count, hblists

def calc_tetrahedral_order(water_snap, o_type, L):
    """
    Calculates the tetrahedral order parameter (q) for each oxygen atom
    using standard NumPy operations.
    
    Parameters:
    -----------
    water_snap : np.ndarray (n_atoms, 6) -> [id, mol, type, x, y, z]
    L : np.ndarray or float -> Box dimensions [L_x, L_y, L_z]
    
    Returns:
    --------
    q_values : np.ndarray (N_oxy,) containing the q parameter for each oxygen
    """
    oxy_mask = water_snap[:, 2] == o_type
    oxy_coords = water_snap[oxy_mask, 3:6]
    n_oxy = len(oxy_coords)
    
    q_values = np.zeros(n_oxy)
    
    # 2. Loop through each central oxygen atom
    for i in range(n_oxy):
        # Vectorized displacement from central oxygen i to all other oxygens
        vecs = oxy_coords - oxy_coords[i]
        
        # Apply Minimum Image Convention (PBC)
        vecs = np.mod(vecs + L/2, L) - L/2
        
        # Calculate distances
        dists = np.linalg.norm(vecs, axis=1)
        
        # 3. Find the 4 nearest neighbors
        # argsort sorts from smallest to largest distance.
        # Index 0 is the atom itself (distance = 0).
        # Indices 1, 2, 3, 4 are the 4 nearest neighbors.
        nearest_indices = np.argsort(dists)[1:5]
        
        # Extract the directional vectors to these 4 neighbors
        r_ij = vecs[nearest_indices]
        
        # Normalize the vectors
        norms = np.linalg.norm(r_ij, axis=1, keepdims=True)
        r_ij_norm = r_ij / norms
        
        # 4. Calculate the sum term
        # The 6 unique pairs among the 4 neighbors: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        sum_term = 0.0
        
        for j, k in pairs:
            # The dot product of two normalized vectors is the cosine of the angle between them
            cos_psi = np.dot(r_ij_norm[j], r_ij_norm[k])
            
            # Accumulate the Errington and Debenedetti term
            sum_term += (cos_psi + 1.0 / 3.0)**2
            
        # 5. Final q calculation for oxygen i
        q_values[i] = 1.0 - (3.0 / 8.0) * sum_term
        
    return q_values

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Hydrogen bond statistics
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

TinKs = [298,423,453,473,503,523]
for TinK in TinKs:
    addr = '%d/' % TinK
    if os.path.isdir(addr) and not os.path.isfile(addr+'hbstat.npy'):
        filename = addr+'/traj.npz'
        locals().update(np.load(filename))
        Ls = np.copy(L)
        water_snaps = traj[:,traj[0,:,2]>=np.max(traj[0,:,2])-1,:]
        OHtypes = [np.max(water_snaps[0,:,2])-1,np.max(water_snaps[0,:,2])]
        traj_O = water_snaps[:,water_snaps[0,:,2]==OHtypes[0],3:]
        traj_H = water_snaps[:,water_snaps[0,:,2]==OHtypes[1],3:]
        hb_count,hblists = count_hbonds(traj_O, traj_H, traj_O, Ls,
                                        cutoff_OO=3.3, cutoff_OH=2.5, angle_min=95.7)
        np.save(addr+'hbstat.npy',hblists)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Tetrahedral order
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

for TinK in TinKs:
    addr = '%d/' % TinK
    if os.path.isdir(addr) and not os.path.isfile(addr+'tetrahedral.npy'):
        filename = addr+'/traj.npz'
        locals().update(np.load(filename))
        Ls = np.copy(L)
        water_snaps = traj[:,traj[0,:,2]>=np.max(traj[0,:,2])-1,:]
        o_type = np.max(water_snaps[0,:,2])-1
        qs = []
        for water_snap,L in zip(water_snaps,Ls):     
            q = calc_tetrahedral_order(water_snap, o_type, L)
            qs.append(q)
        qs = np.array(qs).T
        np.save(addr+'tetrahedral.npy',qs)
        
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Visualize the result
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib as mpl

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 7}
mpl.rc('font', **font)

for axis in ['x', 'y']:
    mpl.rcParams[f'{axis}tick.major.size'] = 3
    mpl.rcParams[f'{axis}tick.minor.size'] = 3
    mpl.rcParams[f'{axis}tick.major.width'] = .5
    mpl.rcParams[f'{axis}tick.minor.width'] = .5
    mpl.rcParams['lines.linewidth'] = .75

centimeter = 1/2.54
centimeter = 1/2.54
fig = plt.figure(dpi=750, figsize=(19*centimeter, 9*centimeter))
gs = fig.add_gridspec(2, 2, wspace=0.5, hspace=0.12)

axA  = fig.add_subplot(gs[:, 0])               # left column: whole height
axB1 = fig.add_subplot(gs[0, 1])               # right column, top
axB2 = fig.add_subplot(gs[1, 1], sharex=axB1, sharey=axB1)  # right column, bottom
ax = [axA, axB1, axB2]

hbbulks = []
qbulks = []
qdata = {}
for TinK in TinKs:
    addr = '%d' % TinK
    if os.path.isdir(addr):
        qs = np.load(addr+'/tetrahedral.npy')
        qdata[TinK] = qs.flatten()
        qbulks.append(np.mean(qs))
        hbstat = np.load(addr+'/hbstat.npy')
        hbbulks.append(np.mean(hbstat))

hbbulks = np.array(hbbulks)
qbulks = np.array(qbulks)

# ---------------- Panel A: <n_HB> (left axis) and <q> (right axis) ----------------
ax[0].plot(TinKs[1:], hbbulks[1:], 'o', markersize=4, color='C0',
           label=r'$\langle n_\mathrm{HB}\rangle$')
ax[0].set_xlim(ax[0].get_xlim()[0], ax[0].get_xlim()[1])
ax[0].hlines(hbbulks[0], ax[0].get_xlim()[0], ax[0].get_xlim()[1], color='C0',
             label=r'$\langle n_\mathrm{HB}\rangle$ (298 K)')
ax[0].set_ylim([2.2, 3.6])
ax[0].set_xlabel(r'$T$ [K]')
ax[0].set_ylabel(r'$\langle n_\mathrm{HB}\rangle$', color='C0')
ax[0].tick_params(axis='y', colors='C0')

ax1 = ax[0].twinx()
ax1.plot(TinKs[1:], qbulks[1:], 's', markersize=4, color='r',
         label=r'$\langle q\rangle$')
ax1.hlines(qbulks[0], ax[0].get_xlim()[0], ax[0].get_xlim()[1], color='r', ls='--',
           label=r'$\langle q\rangle$ (298 K)')
ax1.set_ylim([0.4, 0.7])
ax1.set_ylabel(r'$\langle q\rangle$', color='r')
ax1.tick_params(axis='y', colors='r')

h0, l0 = ax[0].get_legend_handles_labels()
h1, l1 = ax1.get_legend_handles_labels()
ax1.legend(h0 + h1, l0 + l1, frameon=False, loc='lower left', ncol=2)

# ---------------- Panels B and C: P(q) at the lowest and highest temperature ----------------
bins = np.linspace(-1, 1, 20)          # bin width 0.02; wide range so the density is normalised correctly
width = bins[1] - bins[0]
centers = 0.5*(bins[1:] + bins[:-1])
for axd, TinK in zip(ax[1:], [TinKs[0], TinKs[-1]]):
    P, _ = np.histogram(qdata[TinK], bins=bins, density=True)
    axd.bar(centers, P, width=width, color='b', alpha=0.5,
            edgecolor='w', linewidth=1)
    #axd.axvline(np.mean(qdata[TinK]), color='k', ls='--', lw=0.5)   # <q>
    axd.text(0.04, 0.85, '%d K' % TinK, transform=axd.transAxes)
    axd.set_ylabel(r'$P(q)$')
axB1.set_xlim([-0.2, 1.0])
axB1.set_ylim(bottom=0)
plt.setp(axB1.get_xticklabels(), visible=False)
axB2.set_xlabel(r'$q$')

# ---------------- Panel labels ----------------
ax[0].text(-0.175, 1, 'A', fontweight='bold', fontsize=8, transform=ax[0].transAxes)
ax[1].text(-0.175, 1, 'B', fontweight='bold', fontsize=8, transform=ax[1].transAxes)
ax[2].text(-0.175, 1, 'C', fontweight='bold', fontsize=8, transform=ax[2].transAxes)

fig.savefig('../figure5.4.pdf', bbox_inches='tight')