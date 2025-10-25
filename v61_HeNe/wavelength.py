import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds

n, x = np.genfromtxt("data/Interferenz/Gitter.txt",skip_header=1, unpack=True) # number and distance of max

g = 0.001/80 # lattice constant
d = unp.uarray(0.852,0.002) # screen distance
dx = 0.002 # uncertainty

x=unp.uarray(x/100,dx) # (x was in cm)

lamda = g/n*unp.sin(unp.arctan(x/d))

lamda_mean_nom = np.mean(noms(lamda)) # mean value
lamda_mean_std=np.sqrt(np.sum(stds(lamda)**2))/len(lamda) # std of mean

lamda_mean=unp.uarray(lamda_mean_nom,lamda_mean_std)
print("Mean wavelength: ",lamda_mean)