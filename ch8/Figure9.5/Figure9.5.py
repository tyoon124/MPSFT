# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 9 Supercritical Fluid Extraction and Fractionation
Figure 9.5
Periods of a supercritical fluid extraction and the shrinking-core particle. A Schematic cumulative
extraction curve (black) against time t or the solvent-to-feed ratio S/F, with the tangents drawn at
the start (blue), in the intermediate region (green), and at the end of the extraction (orange).
Their intersections define the end of the constant extraction rate (CER) period, t_CER, and the end
of the falling extraction rate (FER) period, t_FER; the remaining part of the curve is the
diffusion-controlled (DC) period. B Spherical particle of the shrinking-core model: the unextracted
core of radius r_c is surrounded by the extracted layer, through which the solute diffuses to the
particle surface at radius R, and by the external fluid film.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

font = {'family':'Dejavu Sans','weight':'normal','size':7}
mpl.rc('font',**font)
for ax_ in ('xtick','ytick'):
    mpl.rcParams[ax_+'.major.size'] = 3
    mpl.rcParams[ax_+'.minor.size'] = 3
    mpl.rcParams[ax_+'.major.width'] = .5
    mpl.rcParams[ax_+'.minor.width'] = .5
mpl.rcParams['lines.linewidth']=.75

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Calculate the results, plot them, and save the figure
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

t = np.linspace(0,1000,100000)
cm = 1 / 2.54
fig, ax = plt.subplots(1,2,dpi=750,figsize=(19*cm,9*cm),
                       gridspec_kw={'width_ratios':[1,0.8]})
tau1 = 0.003
tau2 = 0.006
func0 = lambda t:1-1/2*(np.exp(-tau1*t)+np.exp(-tau2*t))
ax[0].plot(t,func0(t),color='k')

func = lambda x:1/2*(tau1*np.exp(-tau1*x) + tau2*np.exp(-tau2*x))

tangent = lambda x,x1: func(x1)*(x-x1)+func0(x1)
t1 = 0
ax[0].plot(t[t<200],tangent(t[t<200],t1))

t2 = 900
ax[0].plot(t[t>300],tangent(t[t>300],t2))

t3 = 300
ax[0].plot(t,tangent(t,t3))

tCER = t[np.argmin(np.abs(tangent(t,t3) - tangent(t,t1)))]   # scalar
tFER = t[np.argmin(np.abs(tangent(t,t3) - tangent(t,t2)))]   # scalar

ax[0].fill_betweenx([0,1], 0, tCER, color='blue', alpha=0.2, lw=0)   # CER period
ax[0].fill_betweenx([0,1], tCER, tFER, color='green',  alpha=0.2, lw=0)   # FER period
ax[0].fill_betweenx([0,1], tFER, 1000, color='orange',   alpha=0.2, lw=0)   # DC period

ax[0].set_xlim([0,1000])
ax[0].set_ylim([0,1])
ax[0].set_xlabel(r'$t$ or $S/F$ ratio')
ax[0].set_ylabel(r'yield / (g kg$^{-1}$)')
ax[0].text(-0.05,1.02,'A',fontsize=9,fontweight='bold',
           transform=ax[0].transAxes)
ax[0].text(tCER+25,0.05,r'$t_\mathrm{CER}$')
ax[0].text(tFER+25,0.05,r'$t_\mathrm{FER}$')
ax[0].text(25,1.025,"CER",fontweight='bold')
ax[0].text(275,1.025,"FER",fontweight='bold')
ax[0].text(750,1.025,"DC",fontweight='bold')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].text(-0.05,1.1,'B',fontsize=9,fontweight='bold',
           transform=ax[1].transAxes)
import io
from cairosvg import svg2png
from PIL import Image

SVG_FILE = "shrinking-core.svg"
svg_dpi  = 600
crop_pad = 20                      # px of white kept around the drawing

img = Image.open(io.BytesIO(svg2png(url=SVG_FILE, dpi=svg_dpi,
                                    background_color='white'))).convert('RGB')
arr = np.asarray(img)
mask = (arr < 250).any(axis=2)                     # non-white pixels
rows, cols = np.where(mask)
r0, r1 = max(rows.min()-crop_pad, 0), min(rows.max()+crop_pad, arr.shape[0])
c0, c1 = max(cols.min()-crop_pad, 0), min(cols.max()+crop_pad, arr.shape[1])
img = arr[r0:r1, c0:c1]

ax[1].imshow(img)
ax[1].set_aspect('equal')
ax[1].axis('off')

plt.savefig('figure9.5.pdf',dpi=750,bbox_inches='tight')