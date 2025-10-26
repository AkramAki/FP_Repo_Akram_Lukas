import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds

x_00, I_00, dI_00 = np.genfromtxt("data/T-Moden/TEM00.txt",skip_header=1,unpack=True) #x in mm, I in mA
x_01, I_01, dI_01 = np.genfromtxt("data/T-Moden/TEM01.txt",skip_header=1,unpack=True)

x=np.linspace(-30,20,200)

# TEM00 Fit
def f00(x,I_0,x_0,sigma):
    return I_0*np.exp(-(x-x_0)**2/sigma**2)

params00, covariance_matrix = curve_fit(f00, x_00, I_00, sigma=dI_00)

uncertainties00 = np.sqrt(np.diag(covariance_matrix))

print("TEM00 fit params: ")
for name, value, uncertainty in zip("Ixo", params00, uncertainties00):
    print(f"{name} = {value} ± {uncertainty}")

# TEM00 plot
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.errorbar(x_00, I_00, yerr=dI_00,fmt="k." ,label="TEM00-Measurement")
ax.plot(x,f00(x,*params00),label="TEM00-Fit")
ax.set_xlabel(r"$x\,[\unit{\mm}]$")
ax.set_ylabel(r"$I\,[\unit{\milli\A}]$")
ax.legend(loc="best")

fig.savefig("build/TEM00.pdf")


#TEM01 Fit
def f01(x,I_0,x_0,sigma):
    return I_0*2*np.sqrt(2)*(x-x_0)/sigma*np.exp(-(x-x_0)**2/sigma**2)

params01, covariance_matrix = curve_fit(f01, x_01, I_01, sigma=dI_01)

uncertainties01 = np.sqrt(np.diag(covariance_matrix))

print("TEM01 fit params: ")
for name, value, uncertainty in zip("Ixo", params01, uncertainties01):
    print(f"{name} = {value} ± {uncertainty}")

# TEM01 plot
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.errorbar(x_01, I_01, yerr=dI_01,fmt="k." ,label="TEM01-Measurement")
ax.plot(x,f01(x,*params01),label="TEM01-Fit")
# ax.set_xlabel(r"$x\,[\unit{\mm}]$")
# ax.set_ylabel(r"$I\,[\unit{\milli\A}]")
ax.legend(loc="best")

fig.savefig("build/TEM01.pdf")