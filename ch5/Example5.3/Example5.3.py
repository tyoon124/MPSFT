#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 5 Molecular Simulation Analyses
Example 5.3
Characterization of the percolation transition in supercritical fluids: This example locates the
percolation transition densities of the Lennard-Jones fluid at T^*=1.40 from the Voronoi-based
classification of particles into gas-like and liquid-like ones (Sec. 2.7), and tests them by a
direct percolation analysis of each class. The trajectories of Example 5.1 (N=1000) are used at
rho^*=0.10-0.70 in steps of 0.05. The classification uses all 2000 stored frames, and the
percolation analysis uses every tenth frame (200 frames per state point). The analysis uses the
Python packages numpy, scipy, and freud; the cluster counting algorithm of Stoll is implemented in
Section 3.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import freud
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
from PIL import Image
from functools import reduce


font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 7}
mpl.rc('font', **font)

for axis in ['x', 'y']:
    mpl.rcParams[f'{axis}tick.major.size'] = 3
    mpl.rcParams[f'{axis}tick.minor.size'] = 3
    mpl.rcParams[f'{axis}tick.major.width'] = .5
    mpl.rcParams[f'{axis}tick.minor.width'] = .5
    mpl.rcParams['lines.linewidth'] = .75

centimeter = 1/2.54    
Dcrit = 0.316
stride = 1
PERC_STRIDE = 10                    # the percolation analysis uses every PERC_STRIDE-th frame

# Percolation analysis (Stoll's algorithm, Section 3)
CLASSES = {'gas': 1, 'liquid': 0}   # name -> value in the classification array
PERC_CRITERION      = 'any'         # 'any': wraps along >= 1 axis, 'all': wraps along x, y and z
EXCLUDE_PERCOLATING = True          # leave the percolating cluster out of the size distribution

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Classify particles into gas-like and liquid-like ones.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# The labels of each state point are stored in rho*/PiGas.npy (provided in the GitHub repository).
# The classification below runs only for a state point whose PiGas.npy is missing, because it takes long.
rhos = np.arange(0.10,0.75,0.05)
PiGases = []
for rho in rhos:    
    addr = 'rho%.2f/' % rho
    if not os.path.isfile(addr+'/PiGas.npy'):
        data = np.load(addr + 'traj.npz')
        traj, Ls = data['traj'], data['box']      # (2001,1000,4), (2001,3,2)
        traj = traj[:,:,1:]
        traj, Ls = traj[::stride], Ls[::stride]
        labels = []
        for positions, box in zip(traj, Ls):
            lo, hi = box[:, 0], box[:, 1]
            fbox = freud.Box(*(hi - lo))                         # Lx, Ly, Lz
            pos = fbox.wrap((positions - (lo + hi) / 2).astype(np.float32))
            N = len(pos)
        
            voro = freud.locality.Voronoi()
            voro.compute((fbox, pos))
            nlist = voro.nlist
            q, p = nlist.query_point_indices, nlist.point_indices
        
            d = 1.0 / voro.volumes                               # local density per particle
            c1 = np.bincount(q, minlength=N)                     # number of neighbors
            S1 = np.bincount(q, weights=d[p], minlength=N)       # sum of neighbor densities
            S2 = np.bincount(q, weights=S1[p], minlength=N)      # sum over neighbors' neighbors
            C2 = np.bincount(q, weights=c1[p], minlength=N)      # count of neighbors' neighbors
        
            mean_d = (d + S1 + S2) / (1 + c1 + C2)
            labels.append((mean_d <= Dcrit).astype(np.int8))    # 1 = gas-like, 0 = liquid-like
        
        labels = np.array(labels)                                # (n_frames, N)
        np.save(addr+'/PiGas.npy',labels)
        print(labels.shape, "gas fraction:", labels.mean())
    PiG = np.load(addr+'/PiGas.npy')                             # (n_frames, N), 1 = gas-like
    PiGases.append(np.mean(PiG))                                 # gas-like fraction pi^g

PiGases = np.array(PiGases)

def sigmoid(x,a,b):
    return 1/(1+a*np.exp(-b*x))

fig = plt.figure(figsize=(18*centimeter, 14*centimeter),dpi=750)
gs = fig.add_gridspec(2, 4)

ax_top = fig.add_subplot(gs[0, :])                       # top row merged into one subplot
ax_bot = [fig.add_subplot(gs[1, i]) for i in range(4)]   # 4 subplots on the bottom row

from scipy.optimize import brentq
popt, pcov = curve_fit(sigmoid, rhos, PiGases)
a,b = popt
xsol = brentq(lambda x:sigmoid(x,*popt)-0.5,0.2,0.4)

rhoINTERPs = np.linspace(0.1,0.7,1000)
yslope = a*b*np.exp(-b*xsol)/(1+a*np.exp(-b*xsol))**2*(rhoINTERPs - xsol) + 0.5
ax_top.plot(rhoINTERPs,yslope,color='r',ls='--')
ax_top.text((np.log(a)-2)/b,-0.1,r'$\rho_\mathrm{pa}$',color='k',fontsize=9)
ax_top.text((np.log(a)+2)/b,1.075,r'$\rho_\mathrm{pb}$',color='k',fontsize=9)
ax_top.plot(rhos,PiGases,'o',markersize=3,color='k')
ax_top.plot(rhoINTERPs,sigmoid(rhoINTERPs,*popt),color='k')
ax_top.set_xlim([0.1,0.6])
ax_top.set_ylim([-0.02,1.02])
ax_top.set_xlabel(r'$\rho^*$')
ax_top.set_ylabel(r'$\pi^\mathrm{g}$')
ax_top.text(0.975,0.9,r'$\mathbfit{T^*}\mathbf{=1.40}$',horizontalalignment='right',
            transform=ax_top.transAxes)
        
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Define the functions of Stoll's cluster counting algorithm (Comput. Phys. Commun. 109 (1998) 1-5).
Site percolation: only particles of one class are "occupied" sites; two sites are bonded if and only
if both belong to the class and are Voronoi neighbors.
- stoll_label: Cluster labels (0..ncl-1) of n sites connected by bonds (bi[k], bj[k]).
- winding_flags: Flags clusters that connect to their own periodic image.
- wrap: Wrap coordinates into the box.
- get_pairs: Voronoi-neighbor pairs (i<j) of all particles and the image shift of each bond.
- analyze_frame: Clusters and percolation flags of each class in one frame.
- analyze_trajectory: Percolation probability and cluster statistics over all frames.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ---------------------------------------------------------------
# Stoll's algorithm: two index registers
#   N[k] : next member in the (sorted) chain of concatenated labels, 0 = chain top
#   M[k] : head (smallest label) of the chain that contains k
# ---------------------------------------------------------------
def stoll_label(n, bi, bj):
    """Cluster labels (0..ncl-1) of n sites connected by bonds (bi[k], bj[k]).
    Singletons are included as clusters of size 1."""
    lab = [0] * n
    maxl = n // 2 + 2                  # each new label consumes two unlabeled sites
    N = [0] * maxl                     # first register
    M = list(range(maxl))              # second register
    l = 0                              # largest order number so far

    for a, b in zip(bi.tolist(), bj.tolist()):
        la, lb = lab[a], lab[b]
        if la == 0:
            if lb == 0:                # neither enumerated: new order number for both
                l += 1
                lab[a] = lab[b] = l
            else:                      # take the order number of the other one
                lab[a] = lb
        elif lb == 0:
            lab[b] = la
        elif la != lb:                 # two different order numbers
            i1, j1 = M[la], M[lb]
            if i1 == j1:               # already concatenated: nothing to do
                continue
            if i1 > j1:
                i1, j1 = j1, i1
            # merge sorted chain starting at j1 into the one starting at i1,
            # updating both registers (the "changing procedure")
            prev, p, q = i1, N[i1], j1
            while q:
                if p and p < q:
                    N[prev] = p
                    prev, p = p, N[p]
                else:
                    nxt = N[q]
                    N[prev] = q
                    M[q] = i1
                    prev, q = q, nxt
            N[prev] = p

    # renumber: afterwards l = number of clusters with more than one element
    nm = 0
    for j in range(1, l + 1):
        if M[j] == j:
            nm += 1
            M[j] = nm
        else:
            M[j] = M[M[j]]

    lab = np.asarray(lab, dtype=np.int64)
    Mr = np.asarray(M, dtype=np.int64)
    out = np.where(lab > 0, Mr[lab] - 1, -1)
    single = out < 0                   # sites without any bond
    out[single] = nm + np.arange(single.sum())
    return out, nm + int(single.sum())


# ---------------------------------------------------------------
# Percolation test. For the axis under test the periodic border is NOT concatenated
# (Stoll: "borders not considered for percolation tests"), so clusters are open along
# that axis; bonds crossing the border then link these open clusters to their images.
# A cluster percolates if following such links returns to it displaced by a nonzero
# number of box lengths. Image offsets are tracked in a small union-find.
# ---------------------------------------------------------------
def winding_flags(ncl, A, B, s):
    """A[k] -> B[k] are open clusters linked across the border with image shift s[k]
    (n_B = n_A + s). Returns a bool array: True if the open cluster is in a group that wraps."""
    parent = list(range(ncl))
    off = [0] * ncl                    # image offset of node relative to its parent
    flag = [False] * ncl

    def find(x):
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        acc = 0
        for y in reversed(path):       # path compression with offset accumulation
            acc += off[y]
            off[y] = acc
            parent[y] = x
        return x

    for a, b, sh in zip(A.tolist(), B.tolist(), s.tolist()):
        ra, rb = find(a), find(b)
        oa = off[a] if a != ra else 0
        ob = off[b] if b != rb else 0
        if ra == rb:
            if ob - oa != sh:          # closed loop with net winding
                flag[ra] = True
        else:
            parent[rb] = ra
            off[rb] = oa + sh - ob
            flag[ra] = flag[ra] or flag[rb]
    return np.array([flag[find(c)] for c in range(ncl)], dtype=bool)


# ---------------------------------------------------------------
def wrap(pos, lo, L):
    x = np.mod(pos - lo, L)
    return np.where(x >= L, x - L, x)


def get_pairs(pos, L):
    """Voronoi-neighbor pairs (i<j) of all particles and the image shift w of each bond.
    pos must lie in [0, L). Bond i->j connects i with the periodic image of j in box
    n_i + w, i.e. the minimum-image bond vector is  pos[j] - pos[i] + w * L."""
    box = freud.Box(*L)
    pts = box.wrap((pos - L / 2).astype(np.float32))
    voro = freud.locality.Voronoi()
    voro.compute((box, pts))
    nl = voro.nlist
    q, p, vec = nl.query_point_indices, nl.point_indices, nl.vectors
    keep = q < p                           # the list contains both i->j and j->i
    pi, pj, vec = q[keep], p[keep], vec[keep]
    w = np.rint((vec - (pts[pj] - pts[pi])) / L).astype(np.int64)
    return pi, pj, w


def analyze_frame(pos, L, cls_row, class_defs=None, criterion=None):
    class_defs = CLASSES if class_defs is None else class_defs
    criterion = PERC_CRITERION if criterion is None else criterion
    N = len(pos)
    pi, pj, w = get_pairs(pos, L)
    res = {}
    for name, val in class_defs.items():
        mask = cls_row == val
        sites = np.flatnonzero(mask)
        n = len(sites)
        if n == 0:
            res[name] = None
            continue
        local = np.full(N, -1)
        local[sites] = np.arange(n)
        keep = mask[pi] & mask[pj]
        bi, bj, wk = local[pi[keep]], local[pj[keep]], w[keep]

        labels, ncl = stoll_label(n, bi, bj)                  # full PBC
        sizes = np.bincount(labels, minlength=ncl)

        perc_ids = []
        for a in range(3):
            cross = wk[:, a] != 0
            if not cross.any():                               # cannot wrap along a
                perc_ids.append(np.array([], dtype=np.int64))
                continue
            lab_o, ncl_o = stoll_label(n, bi[~cross], bj[~cross])
            flags = winding_flags(ncl_o, lab_o[bi[cross]], lab_o[bj[cross]], wk[cross, a])
            perc_ids.append(np.unique(labels[flags[lab_o]]))
        comb = np.union1d if criterion == 'any' else np.intersect1d
        sel = reduce(comb, perc_ids)
        res[name] = dict(n=n, sizes=sizes, perc_ids=sel,
                         axes=[len(x) > 0 for x in perc_ids],
                         labels=labels, bi=bi, bj=bj)
    return res


def analyze_trajectory(traj, Ls, classes, class_defs=None, criterion=None,
                       exclude_percolating=None, verbose=False):
    """traj (F,N,3), Ls (F,3,2) = [lo, hi] per axis, classes (F,N) class value per particle.
    Returns {class name: dict} with

      P_perc, P_perc_err : probability of an infinite (box-spanning) cluster
      P_axes             : probability of spanning along x, y, z
      ns                 : ns[s] = mean number of clusters of size s per frame
      size_fraction      : probability that a particle sits in a cluster of size s
      mean_size_w        : weight-average cluster size  sum s^2 ns / sum s ns
      mean_size_n        : number-average cluster size  sum s ns / sum ns
      smax, n_particles  : largest cluster / class population in every frame
      perc_flag, perc_axes : per-frame percolation results (all arrays have length F)
      nvalid             : number of frames in which the class is non-empty
    All averages run over all F frames: a frame without particles of a class has no
    cluster and no spanning cluster, so a class that never occurs gives P_perc = 0,
    ns = 0 and mean sizes = nan (check nvalid == 0).
    ns and mean sizes leave out the percolating cluster if exclude_percolating."""
    class_defs = CLASSES if class_defs is None else class_defs
    exclude_percolating = EXCLUDE_PERCOLATING if exclude_percolating is None else exclude_percolating
    F, N, _ = traj.shape
    if not (len(Ls) == len(classes) == F):
        raise ValueError(f'frame mismatch: traj {F}, box {len(Ls)}, classes {len(classes)}')

    hist   = {k: np.zeros(N + 1) for k in class_defs}
    perc   = {k: np.zeros(F, dtype=bool) for k in class_defs}
    paxes  = {k: np.zeros((F, 3), dtype=bool) for k in class_defs}
    smax   = {k: np.zeros(F, dtype=np.int64) for k in class_defs}
    npart  = {k: np.zeros(F, dtype=np.int64) for k in class_defs}
    for f in range(F):
        lo, hi = Ls[f][:, 0], Ls[f][:, 1]
        L = hi - lo
        res = analyze_frame(wrap(traj[f], lo, L), L, classes[f], class_defs, criterion)
        for name, r in res.items():
            if r is None:                  # class empty in this frame: no clusters, no spanning
                continue
            sizes = r['sizes']
            npart[name][f] = r['n']
            perc[name][f] = len(r['perc_ids']) > 0
            paxes[name][f] = r['axes']
            smax[name][f] = sizes.max()
            if exclude_percolating and len(r['perc_ids']):
                sizes = np.delete(sizes, r['perc_ids'])
            hist[name] += np.bincount(sizes, minlength=N + 1)[:N + 1]
        if verbose and (f + 1) % 100 == 0:
            print(f'frame {f + 1}/{F}')

    s = np.arange(N + 1)
    out = {}
    for name in class_defs:                # every class is always returned
        ns = hist[name] / F                # averages run over ALL frames; empty frames count as 0
        sum_s, sum_s2, sum_n = (s * ns).sum(), (s**2 * ns).sum(), ns.sum()
        pp = float(perc[name].mean())
        out[name] = dict(
            P_perc=pp, P_perc_err=float(np.sqrt(pp * (1 - pp) / F)),
            P_axes=paxes[name].mean(axis=0),
            ns=ns, size_fraction=s * ns / sum_s if sum_s > 0 else np.zeros_like(ns),
            mean_size_w=sum_s2 / sum_s if sum_s > 0 else np.nan,
            mean_size_n=sum_s / sum_n if sum_n > 0 else np.nan,
            smax=smax[name], n_particles=npart[name],
            perc_flag=perc[name], perc_axes=paxes[name],
            nvalid=int((npart[name] > 0).sum()))
    return out

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. Use Stoll's algorithm to calculate the percolation probability and locate the percolation transition loci.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PinfGases = []
PinfLiquids = []
rhos = [0.2,0.25,0.3,0.35]
for rho in rhos:
    addr = 'rho%.2f/' % rho
    data = np.load(addr+'traj.npz')
    traj, Ls = data['traj'], data['box']
    classes = np.load(addr+'PiGas.npy')[::PERC_STRIDE]
    # The traj.npz files of the GitHub repository already contain only every PERC_STRIDE-th frame
    # (to limit the file size); a full trajectory is thinned here in the same way.
    if len(traj) != len(classes):
        traj, Ls = traj[::PERC_STRIDE], Ls[::PERC_STRIDE]
    traj = traj[:,:,1:]
    
    res = analyze_trajectory(traj, Ls, classes)   # gas = 1, liquid = 0
    PinfGases.append(res['gas']['P_perc'])
    PinfLiquids.append(res['liquid']['P_perc'])
    print('rho* = %.2f: P_inf(gas) = %.4f, P_inf(liquid) = %.4f' % (rho, PinfGases[-1], PinfLiquids[-1]))

for i,d in enumerate(rhos):
    img = mpimg.imread('lj-%.2f.png' % d)
    ax_bot[i].imshow(img)
    ax_bot[i].axis('off')
    ax_bot[i].text(0.0,1.05,r'$\mathbfit{\rho^*}=\mathbf{%.2f}$' % d,
                   transform=ax_bot[i].transAxes)
    ax_bot[i].text(0.5,-0.1,r'$P_\mathrm{inf}^\mathrm{gas}=%.2f$; $P_\mathrm{inf}^\mathrm{liquid}=%.2f$' 
                   % (PinfGases[i],PinfLiquids[i]),
                   transform=ax_bot[i].transAxes,
                   horizontalalignment='center')
    
plt.savefig('../figure5.3.pdf',dpi=750,bbox_inches='tight')