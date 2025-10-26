import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

deg, P, dP = np.genfromtxt("data/Polarisation.txt",skip_header=2, unpack=True)
deg=deg*2*np.pi/360

x=np.linspace(0,2*np.pi,100)

def f(x,I,x_0):
   return I*np.cos(x-x_0)**2

params, covariance_matrix = curve_fit(f, deg, P, sigma=dP)

uncertainties = np.sqrt(np.diag(covariance_matrix))

print("Polarization fit params: ")
for name, value, uncertainty in zip("Ix", params, uncertainties):
    print(f"{name} = {value} ± {uncertainty}")


fig, ax = plt.subplots(1, 1, layout="constrained")
ax.errorbar(deg, P, yerr=dP,fmt="k." ,label="Measurement")
ax.plot(x,f(x,*params),label="Fit")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$P\,[\unit{\milli\watt}]$")

# Radian ticks
custom_ticks = [0, np.pi/2, np.pi, (3*np.pi)/2, 2*np.pi]
custom_tick_labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
ax.set_xticks(custom_ticks)
ax.set_xticklabels(custom_tick_labels)

ax.legend(loc="best")

fig.savefig("build/Polarisation.pdf")