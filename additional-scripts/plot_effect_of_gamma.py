# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:47:11 2019

@author: egomez
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage


## Define the exponential, n_gamma and distance function for the plots.
def func_exp_pure(x, a, c):    
    return a*np.exp(-c*(x))
def n_gamma_function(gamma_par, a, c):
    gamma_par = np.array(gamma_par, dtype=float)
    a = np.array(a, dtype=float) 
    c = np.array(c, dtype=float)     
    return  np.floor((-1/c)*np.log((gamma_par)/(c*a)))


def distance(alpha_par, n, a, c):
    n = np.array(n, dtype=float)
    alpha_par = np.array(alpha_par, dtype=float) 
    a = np.array(a, dtype=float) 
    c = np.array(c, dtype=float)       
        
    A_alpha = alpha_par*n 
    A = ((1./c)*a)*(1-np.exp(-n*c))
    return A_alpha - A

###############################################################################
# VISUALIZE THE EFFECT OF GAMMA IN FINAL DECISSION: COLOR MAP
###############################################################################

## Make a mesh in the space of parameterisation variables u and v for color map
up_a = 0.5
up_c = 0.07
a = np.linspace(0,up_a, endpoint=True, num=600)
c = np.linspace(0,up_c, endpoint=True, num=800)
a = a[1:]
c = c[1:]
a, c = np.meshgrid(a, c)

# Statistical significance threshold
alpha_par = 0.05

# Create a vector with gamma values separated by powers of 10.
up_gamma= 0.0005
low_gamma=5.0e-08
index = int(np.log10(up_gamma/low_gamma))
for i in range(index+1):
    aux = np.linspace(low_gamma*(10**i),low_gamma*(10**(i+1)),
                      endpoint=False, num=15)
    if i == 0:
        gamma_par = aux
    else:
        gamma_par = np.concatenate((gamma_par,aux))
        
# Compute the distance for each value of a, c and gamma
d = np.empty((a.shape[0],a.shape[1], len(gamma_par)), dtype = 'float')
for g in range(len(gamma_par)):
    n = n_gamma_function(gamma_par[g], a, c)
    n[n<0]=0
    d[:,:,g] =  distance(alpha_par, n, a, c)
    
    d[d[:,:,g]>=0,g] = 1
    d[d[:,:,g]<0,g] = 0
d_gamma = np.sum(d,axis=2)
im = np.copy(np.rot90(np.transpose(d_gamma, [1,0])))

# Obtain the threshold for gamma=5e-06 to plot the line at that point
th = np.where( gamma_par.round(8)==5e-06)
th = th[0][0]
im_th = np.copy(d[:,:,th])
im_th = np.rot90(np.transpose(im_th, [1,0]))
edges = skimage.filters.sobel(im_th)
edges = edges>0
edges = edges.astype(np.int8)
im_th = np.copy(im)
im_th[edges==1] = -1

#------------------------------------------------------------------------------
# INCLUDE MARKERS FOR THE EXPONENTIAL PARAMETERS - PHASE CONTRAST MICROSCOPY
#------------------------------------------------------------------------------

# Coordinates of the exponetial parameters for cell body in phase contrast 
# images.
# We need to obtain the position (a,c) that corresponds with each exponential 
# parameter. 
XC = np.array([0.258,0.263,0.272, 0.258, 0.256, 0.264, 0.259, 0.282, 0.292, 
               0.435, 0.198, 0.195])
YC = np.array([0.0026, 0.0075, 0.0216, 0.0017, 0.0014, 0.0072, 0.0029, 0.04,
               0.0648, -0.0005, 0.0345, 0.0351])

a_row = np.copy(np.rot90(np.transpose(a, [1,0])))
a_row = a_row[0,:]
a_row = a_row.round(3)

c_col = np.copy(np.rot90(np.transpose(c, [1,0])))
c_col = c_col[:,0]
c_col = c_col.round(3)

YC_coor = np.copy(YC)
XC_coor = np.copy(XC)
for j in range(len(XC)):
    YC_coor[j] = np.where(c_col==YC[j].round(3))[0][-1]
    XC_coor[j] = np.where(a_row==XC[j].round(3))[0][-1]
    
# Coordinates of the exponetial parameters for protrusions in phase contrast 
# images. 
# We need to obtain the position (a,c) that corresponds with each exponential 
# parameter. 
XP = np.array([0.250, 0.246, 0.250, 0.248, 0.241, 0.256, 0.251, 0.250, 0.255,
               0.251, 0.250, 0.247])
YP = np.array([0.0031, 0.0221, 0.01, 0.0019, 0.0276, 0.0175, 0.0011, 0.0289,
               0.0211, 0.0023, 0.0248, 0.0134])

a_row = np.copy(np.rot90(np.transpose(a, [1,0])))
a_row = a_row[0,:]
a_row = a_row.round(3) 

c_col = np.copy(np.rot90(np.transpose(c, [1,0])))
c_col = c_col[:,0]
c_col = c_col.round(3)
 
YP_coor = np.copy(YP)
XP_coor = np.copy(XP)
for j in range(len(XP)):
    YP_coor[j] = np.where(c_col==YP[j].round(3))[0][0]
    XP_coor[j] = np.where(a_row==XP[j].round(3))[0][0]

#------------------------------------------------------------------------------

# Obtain the position of a=alpha, to plot a line.
a_row = np.copy(np.rot90(np.transpose(a, [1,0])))
a_row = a_row[0,:]
x_alpha = np.where(a_row.round(3)==alpha_par)
x_alpha = x_alpha[0][0]


###############################################################################

# PLOT
fig=plt.figure(figsize=(10,10))
plt.imshow(im_th,  cmap='CMRmap')
plt.colorbar()
plt.scatter(XC_coor, YC_coor, marker='o',c='black', s=100)
plt.scatter(XP_coor,YP_coor, marker='P',c='magenta', s=100)
plt.plot([x_alpha,x_alpha],[0,im.shape[0]], 'k--')
plt.show()
#fig.savefig('im_decision_gammavar_amax'+ str(up_a) + '_cmax' +  str(up_c) + '.eps', format='eps', dpi = 350)


###############################################################################
# VISUALIZE THE EFFECT OF GAMMA IN FINAL DECISSION: SUBPLOTS
###############################################################################

## Make a mesh in the space of parameterisation variables u and v for subplots
up_a = 1
up_c = 10
alpha_par = 0.05
a = np.linspace(0,up_a, endpoint=True, num=500)
c = np.linspace(0,up_c, endpoint=True, num=500)
a = a[1:]
c = c[1:]
a, c = np.meshgrid(a, c)

up_gamma = 0.1 
low_gamma = 1.0e-12
index = int(np.log10(up_gamma/low_gamma))
for i in range(index+1):
    aux = np.linspace(low_gamma*(10**i),low_gamma*(10**(i+1)), 
                      endpoint=False, num=5)
    if i == 0:
        gamma_par = aux
    else:
        gamma_par = np.concatenate((gamma_par,aux))
#gamma_par = np.concatenate((gamma_par, [1]))
        
# Compute the distance for each value of a, c and gamma
d = np.empty((a.shape[0],a.shape[1], len(gamma_par)), dtype = 'float')
for g in range(len(gamma_par)):
    n = n_gamma_function(gamma_par[g], a, c)
    n[n<0]=0
    d[:,:,g] =  distance(alpha_par, n, a, c)
    
    d[d[:,:,g]>=0,g] = 1
    d[d[:,:,g]<0,g] = 0

###############################################################################
# PLOT
    
Nr = 8
Nc = 7
k = 0
fig = plt.figure(1, figsize=(9,15))
#cmap = plt.cm.get_cmap("rainbow") 
from matplotlib.colors import LinearSegmentedColormap
cmap_name = 'my_list'
colors = [(1,1,1), (0, 0, 0.35)]
cm = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=2)
for i in range(1, Nr*Nc+1):
    ax = plt.subplot(Nr,Nc,i)
    ax.contourf(a,c, d[:,:,k], cmap=cm)
    plt.tick_params(labelsize = 7, which = 'major', direction = 'out', size = 0.05)
    plt.title("gamma = %.2E" % gamma_par[k], fontsize = 7)
    k = k+1
fig.tight_layout()
plt.show()
## fig.savefig('subplots_gamma.pdf, format='pdf', dpi = 250)
