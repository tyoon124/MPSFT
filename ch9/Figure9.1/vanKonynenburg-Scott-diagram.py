# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 9 Supercritical Fluid Extraction and Fractionation
Supporting script: vanKonynenburg-Scott-diagram.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages and set the plot style
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif"]
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
font = {'size':7}
mpl.rc('font',**font)
for ax_ in ('xtick','ytick'):
    mpl.rcParams[ax_+'.major.size'] = 3
    mpl.rcParams[ax_+'.minor.size'] = 3
    mpl.rcParams[ax_+'.major.width'] = .5
    mpl.rcParams[ax_+'.minor.width'] = .5
mpl.rcParams['lines.linewidth']=.75

from scipy.interpolate import interp1d

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Calculate and plot the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

cm = 1/2.54

fig,ax = plt.subplots(2,3,figsize=(19*cm,11*cm),dpi=300,
                      gridspec_kw={'wspace':0.2,'hspace':0.2})

# The last point
# [0.0, 0.0], [0.9,0.5], [0.3, 0.04] [0.6, 0.16]
# The first point
# [0.0, 0.1] [0.05 0.08] [0.15 0.15] [0.3,0.35]

x1 = np.linspace(0,0.9,500)

p = np.array([0.0,0.12,0.3,0.6,0.9])
p = np.column_stack((p**4,p**3,p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[0,0.006,0.04,0.16,0.5])
y1 = np.polyval(p,x1)
ax[0,0].plot(x1,y1,label='1',color='k')
x2 = np.linspace(0,0.3,500)
p = np.array([-1.2,0.075,0.15,0.3])
p = np.column_stack((p**3,p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[0.07,0.08,0.15,0.35])
y2 = np.polyval(p,x2)
ax[0,0].plot(x2,y2,label='2',color='k')
ax[0,0].set_xlim([0,1])
ax[0,0].set_ylim([0,1])

last1 = [x1[-1],y1[-1]]
last2 = [x2[-1],y2[-1]]
last3 = [0.6,0.6]
p = np.array([x1[-1],x2[-1],0.6])
p = np.column_stack((p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[y1[-1],y2[-1],0.6])
x3 = np.linspace(x1[-1],x2[-1],100)
y3b = np.polyval(p,x3)
ax[0,0].plot(x3,y3b,'--',label='(b)',color='b')

ax[0,0].plot([x1[-1],x2[-1]],[y1[-1],y2[-1]],':',label='(a)',color='r')

p = np.array([x1[-1],x2[-1],0.6])
p = np.column_stack((p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[y1[-1],y2[-1],0.33])
y3c = np.polyval(p,x3)
ax[0,0].plot(x3,y3c,'--',label='(c)',color='g')
ax[0,0].text(0.15,0.2,'1',fontsize=7)
ax[0,0].text(0.75,0.2,'2',fontsize=7)
ax[0,0].text(0.55,0.275,'a',fontsize=7, fontweight='bold')
ax[0,0].text(0.55,0.45,'b',fontsize=7, fontweight='bold')
ax[0,0].text(0.55,0.65,'c',fontsize=7, fontweight='bold')
#ax[0,0].legend(frameon=False)

ax[0,0].set_xlabel('Temperature')
ax[0,0].set_ylabel('Pressure')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
ax[0,0].text(0.05,0.9,'Type I',fontweight='bold',fontsize=7,horizontalalignment='left',
             transform=ax[0,0].transAxes)

# The last point
# [0.0, 0.0], [0.9,0.5], [0.3, 0.04] [0.6, 0.16]
# The first point
# [0.0, 0.1] [0.05 0.08] [0.15 0.15] [0.3,0.35]

x1 = np.linspace(0.0,0.9,500)

p = np.array([0.0,0.12,0.3,0.6,0.9])
p = np.column_stack((p**4,p**3,p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[0,0.006,0.04,0.16,0.5])
y1 = np.polyval(p,x1)
ax[0,1].plot(x1,y1,label='1',color='k')
x2 = np.linspace(0,0.3,500)
p = np.array([-1.2,0.075,0.15,0.3])
p = np.column_stack((p**3,p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[0.07,0.08,0.15,0.35])
y2 = np.polyval(p,x2)
ax[0,1].plot(x2,y2,label='2',color='k')
ax[0,1].set_xlim([0,1])
ax[0,1].set_ylim([0,1])

x3 = np.linspace(0,0.2,200)
p = interp1d([0,0.05,0.1,0.2],
             [0.03,0.032,0.04,0.08],
             kind='cubic')
y3 = p(x3)
ax[0,1].plot(x3,y3,color='b',label='LLV')

last1 = [x1[-1],y1[-1]]
last2 = [x2[-1],y2[-1]]
last3 = [0.6,0.6]
p = np.array([x1[-1],x2[-1],0.6])
p = np.column_stack((p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[y1[-1],y2[-1],0.6])
x3 = np.linspace(x1[-1],x2[-1],100)
y3b = np.polyval(p,x3)
ax[0,1].plot(x3,y3b,'--',color='k')

ax[0,1].plot([0.2,0.23],[0.08,0.8],'--',color='b',label='UCST')
ax[0,1].text(0.1,0.4,'LL',fontsize=7)
ax[0,1].text(0.25,0.12,'UCEP',fontsize=7)
ax[0,1].text(0.05,0.075,'LLV',fontsize=7)
ax[0,1].text(0.15,0.2,'1',fontsize=7)
ax[0,1].text(0.75,0.2,'2',fontsize=7)
ax[0,1].text(0.25,0.6,'UCST',fontsize=7)

ax[0,1].set_xlabel('Temperature')
ax[0,1].set_ylabel('Pressure')
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,1].text(0.05,0.9,'Type II',fontweight='bold',fontsize=7,horizontalalignment='left',
             transform=ax[0,1].transAxes)

# The last point
# [0.0, 0.0], [0.9,0.5], [0.3, 0.04] [0.6, 0.16]
# The first point
# [0.0, 0.1] [0.05 0.08] [0.15 0.15] [0.3,0.35]

x1 = np.linspace(0.0,0.9,500)

p = np.array([0.0,0.12,0.3,0.6,0.9])
p = np.column_stack((p**4,p**3,p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[0,0.006,0.04,0.16,0.5])
y1 = np.polyval(p,x1)
ax[0,2].plot(x1,y1,label='1',color='k')
x2 = np.linspace(0.0,0.35,500)
y2 = 1.63*x2**2+.2747*x2+.1065
ax[0,2].plot(x2,y2,label='2',color='k')
ax[0,2].set_xlim([0,1])
ax[0,2].set_ylim([0,1])

x3 = np.linspace(0.0,0.4,500)
y3 = 1.73*x3**2 + .0402*x3 + 0.08
ax[0,2].plot(x3,y3,color='k')

lb = [x2[-1],y2[-1]]
ub = [x3[-1],y3[-1]]
mb = [x3[-1]*0.99,y2[-1]*0.95]
p = np.array([x2[-1],x3[-1],x3[-1]*0.99])
p = np.column_stack((p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[y2[-1],y3[-1],y2[-1]*0.95])
x4 = np.linspace(x2[-1],x3[-1],100)
y4 = np.polyval(p,x4)
ax[0,2].plot(x4,y4,'--',label='UCEP',color='k')
ax[0,2].text(0.35,0.45,'UCEP',fontsize=7)
ax[0,2].text(0.35,0.2,'LLV',fontsize=7)
ax[0,2].text(0.1,0.2,'1',fontsize=7)
ax[0,2].text(0.75,0.2,'2',fontsize=7)
ax[0,2].text(0.45,0.6,'a',fontsize=7, fontweight='bold')
ax[0,2].text(0.6,0.9,'b',fontsize=7, fontweight='bold')
ax[0,2].text(0.8,0.9,'c',fontsize=7, fontweight='bold')
ax[0,2].text(0.85,0.6,'d',fontsize=7, fontweight='bold')

a = 3.1786157471572
b = -16.372059920931
c = 28.084457223414
d = 7.4069375936914
e = -54.05792831217
f = 32.909636842296
p = np.array([a,b,c,d,e,f])[::-1]
x = np.linspace(0.05,0.83,100)
y5 = np.polyval(p,x)
x = x + 0.07
y5 = y5 + 0.02
ax[0,2].plot(x,y5,'--',color='r',label='(a)')
ax[0,2].plot(x,np.polyval([5.8486,-11.056,5.7111],x),
         '--',color='b',label='(b)')
y = np.linspace(y5[-1],1,200)
x = 1.2213*y**2-2.1111*y+1.6463
ax[0,2].plot(x,y,'--',color='g',label='(c)')
z = np.linspace(0.9,2,100)
ax[0,2].plot(z,(0.7-0.5)/(1.1-0.89)*(z-0.89)+0.49,
         '--',label='(d)',
         color='m')
ax[0,2].set_xlabel('Temperature')
ax[0,2].set_ylabel('Pressure')
ax[0,2].set_xticks([])
ax[0,2].set_yticks([])

ax[0,2].text(0.05,0.9,'Type III',fontweight='bold',fontsize=7,horizontalalignment='left',
             transform=ax[0,2].transAxes)

# The last point
# [0.0, 0.0], [0.9,0.5], [0.3, 0.04] [0.6, 0.16]
# The first point
# [0.0, 0.1] [0.05 0.08] [0.15 0.15] [0.3,0.35]

x1 = np.linspace(0.0,0.9,500)

p = np.array([0.0,0.12,0.3,0.6,0.9])
p = np.column_stack((p**4,p**3,p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[0,0.006,0.04,0.16,0.5])
y1 = np.polyval(p,x1)
ax[1,0].plot(x1,y1,label='2',color='k')
x2 = np.linspace(0.0,0.375,500)
y2 = 1.63*x2**2+.2747*x2+.1065
ax[1,0].plot(x2[x2<0.15],y2[x2<0.15],label='1',color='k')
ax[1,0].plot(x2[x2>0.225],y2[x2>0.225],label='1',color='k')
ax[1,0].set_xlim([0,1])
ax[1,0].set_ylim([0,1])

x3 = np.copy(x2)
y3 = y2+0.1*np.linspace(0.5,1,len(y2))
x3 = x3[y3<=y2[-1]]
y3 = y3[y3<=y2[-1]]
ax[1,0].plot(x3,y3,color='k')

point1 = (x3[-1],y3[-1])
point2 = (x2[-1],y2[-1])
point3 = [(x3[-1]+x2[-1])/2,y2[-1]*1.05]

x4 = np.linspace(x3[-1],x2[-1],100)
p = np.array([point1[0],point2[0],point3[0]])
p = np.column_stack((p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[point1[1],point2[1],point3[1]])
y4 = np.polyval(p,x4)
ax[1,0].plot(x4,y4,'--',color='b',label='UCEP')

point1 = [x2[x2>.225][0],y2[x2>.225][0]]
point2 = [x1[-1],y1[-1]]
point3 = [0.3,0.5]
x5 = np.linspace(point1[0],point2[0],500)
p = np.array([point1[0],point2[0],point3[0]])
p = np.column_stack((p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[point1[1],point2[1],point3[1]])
y5 = np.polyval(p,x5)
ax[1,0].plot(x5,y5,'--',color='r',label='LCST')

point1 = [x2[x2<0.15][-1],y2[x2<0.15][-1]]
point2 = [0.1,1.05]
p = np.array([point1[0],point2[0]])
p = np.column_stack((p**1,p**0))
p = np.matmul(np.linalg.inv(p),[point1[1],point2[1]])
x6 = np.linspace(point1[0],point2[0],100)
y6 = np.polyval(p,x6)
ax[1,0].plot(x6,y6,'--',color='g',label='UCST')
ax[1,0].text(0.55,0.85,'LCST',fontsize=7)
ax[1,0].text(0.375,0.45,'UCEP',fontsize=7)
#ax[1,0].legend(frameon=False)
ax[1,0].text(0.375,0.35,'LLV',fontsize=7)
ax[1,0].text(0.25,0.2,'LCEP',fontsize=7)
ax[1,0].text(0.05,0.6,'LL',fontsize=7)
ax[1,0].text(0.2,0.8,'UCST',fontsize=7)
ax[1,0].set_xlabel('Temperature')
ax[1,0].set_ylabel('Pressure')
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])

ax[1,0].text(0.05,0.9,'Type IV',fontweight='bold',fontsize=7,horizontalalignment='left',
             transform=ax[1,0].transAxes)

# The last point
# [0.0, 0.0], [0.9,0.5], [0.3, 0.04] [0.6, 0.16]
# The first point
# [0.0, 0.1] [0.05 0.08] [0.15 0.15] [0.3,0.35]

x1 = np.linspace(0.0,0.9,500)

p = np.array([0.0,0.12,0.3,0.6,0.9])
p = np.column_stack((p**4,p**3,p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[0,0.006,0.04,0.16,0.5])
y1 = np.polyval(p,x1)
ax[1,1].plot(x1,y1,label='2',color='k')
x2 = np.linspace(0.0,0.375,500)
y2 = 1.63*x2**2+.2747*x2+.1065
ax[1,1].plot(x2[x2<0.15],y2[x2<0.15],label='1',color='k')
ax[1,1].plot(x2[x2>0.225],y2[x2>0.225],label='1',color='k')
ax[1,1].set_xlim([0,1])
ax[1,1].set_ylim([0,1])

x3 = np.copy(x2)
y3 = y2+0.1*np.linspace(0.5,1,len(y2))
x3 = x3[y3<=y2[-1]]
y3 = y3[y3<=y2[-1]]
ax[1,1].plot(x3,y3,color='k')

point1 = (x3[-1],y3[-1])
point2 = (x2[-1],y2[-1])
point3 = [(x3[-1]+x2[-1])/2,y2[-1]*1.05]

x4 = np.linspace(x3[-1],x2[-1],100)
p = np.array([point1[0],point2[0],point3[0]])
p = np.column_stack((p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[point1[1],point2[1],point3[1]])
y4 = np.polyval(p,x4)
ax[1,1].plot(x4,y4,'--',color='b',label='UCEP')

point1 = [x2[x2>.225][0],y2[x2>.225][0]]
point2 = [x1[-1],y1[-1]]
point3 = [0.3,0.5]
x5 = np.linspace(point1[0],point2[0],500)
p = np.array([point1[0],point2[0],point3[0]])
p = np.column_stack((p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[point1[1],point2[1],point3[1]])
y5 = np.polyval(p,x5)
ax[1,1].plot(x5,y5,'--',color='r',label='LCST')

ax[1,1].text(0.55,0.85,'LCST',fontsize=7)
ax[1,1].text(0.375,0.45,'UCEP',fontsize=7)
#ax[1,1].legend(frameon=False)
ax[1,1].text(0.375,0.35,'LLV',fontsize=7)
ax[1,1].text(0.25,0.2,'LCEP',fontsize=7)
ax[1,1].set_xlabel('Temperature')
ax[1,1].set_ylabel('Pressure')
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])
ax[1,1].text(0.05,0.9,'Type V',fontweight='bold',fontsize=7,horizontalalignment='left',
             transform=ax[1,1].transAxes)

# The last point
# [0.0, 0.0], [0.9,0.5], [0.3, 0.04] [0.6, 0.16]
# The first point
# [0.0, 0.1] [0.05 0.08] [0.15 0.15] [0.3,0.35]

x1 = np.linspace(0.0,0.9,500)

p = np.array([0.0,0.12,0.3,0.6,0.9])
p = np.column_stack((p**4,p**3,p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[0,0.006,0.04,0.16,0.5])
y1 = np.polyval(p,x1)
plt.plot(x1,y1,label='1',color='k')
x2 = np.linspace(0.0,0.3,500)
p = np.array([0,0.05,0.15,0.3])
p = np.column_stack((p**3,p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[0.07,0.08,0.15,0.35])
y2 = np.polyval(p,x2)
plt.plot(x2,y2,label='2',color='k')
plt.xlim([0,1])
plt.ylim([0,1])

last1 = [x1[-1],y1[-1]]
last2 = [x2[-1],y2[-1]]
last3 = [0.6,0.6]
p = np.array([x1[-1],x2[-1],0.6])
p = np.column_stack((p**2,p**1,p**0))
p = np.matmul(np.linalg.inv(p),[y1[-1],y2[-1],0.6])
x3 = np.linspace(x1[-1],x2[-1],100)
y3b = np.polyval(p,x3)
plt.plot(x3,y3b,'--',color='r')

# Closed-loop liquid--liquid immiscibility: the LLV line lies just above the vapor-pressure curve of
# component 1, below T_crit,1, between the LCEP and the UCEP; the LL critical curve joins the two end
# points through a pressure maximum. The gas--liquid critical curve runs separately from C1 to C2.
pv1 = np.array([0,0.05,0.15,0.3])
pv1 = np.matmul(np.linalg.inv(np.column_stack((pv1**3,pv1**2,pv1**1,pv1**0))),[0.07,0.08,0.15,0.35])
lcep = [0.06, np.polyval(pv1,0.08)+0.025]
ucep = [0.24, np.polyval(pv1,0.22)+0.025]
x6 = np.linspace(lcep[0],ucep[0],100)
y6 = np.polyval(pv1,x6)+0.025
plt.plot(x6,y6,'-',color='k')
pk = [0.15,0.50]
q = np.array([lcep[0],pk[0],ucep[0]])
q = np.matmul(np.linalg.inv(np.column_stack((q**2,q**1,q**0))),[lcep[1],pk[1],ucep[1]])
plt.plot(x6,np.polyval(q,x6),'--',color='b')
plt.text(0.005,0.035,'LCEP',fontsize=7)
plt.text(0.255,0.19,'UCEP',fontsize=7)
plt.text(0.12,0.3,'LL',fontsize=7)
plt.text(0.32,0.28,'1',fontsize=7)
plt.text(0.75,0.2,'2',fontsize=7)
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.xticks([])
plt.yticks([])

ax[1,2].text(0.05,0.9,'Type VI',fontweight='bold',fontsize=7,horizontalalignment='left',
             transform=ax[1,2].transAxes)

plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure9.1.pdf"), bbox_inches="tight")
print("saved figure9.1.pdf")
