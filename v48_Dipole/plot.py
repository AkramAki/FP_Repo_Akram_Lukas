import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds


kB = 8.617*10**(-5) # boltzmann constant

T_raw,I_raw,F,P=np.genfromtxt("data/run_1_corr.txt",skip_header=1,unpack=True) # run_1_corr corrects the .3 scale

T=T_raw+273.15
I=unp.uarray(-I_raw*F,F/10) # I in pA and change sign for log later

T0=T[0]
Tend=T[-1]


fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(np.arange(T.size),T,".")
ax.set_ylabel(r"$T \mathbin{/} \unit{\celsius}$")
ax.set_xlabel(r"$t \mathbin{/} \unit{\minute}$")


fig.savefig("build/heating_rate_1.pdf")

# heating rate
b_array = np.gradient(T[20:], 1)
b=np.mean(b_array)

# mean uncertainty
std_b = np.std(b_array, ddof=1)
std_mean_b = std_b / np.sqrt(len(b_array))

b=unp.uarray(b,std_mean_b)

print("heating rate: ", b, "+- ",std_mean_b)

# define background
I_background=unp.uarray(np.concatenate([noms(I[0:32]),noms(I[-11:])]),np.concatenate([stds(I[0:32]),stds(I[-11:])]))
T_background=np.concatenate([T[0:32],T[-11:]])
print(T[0],T[32],T[-11],T[-1])

# Fit background
def f(x,a,y_0,b):
    # return (np.exp(a*x))+y_0 
    return b*np.exp(-a/x)+y_0

# unp Fit background
def uf(x,a,y_0,b):
    # return (np.exp(a*x))+y_0 
    return b*unp.exp(-a/x)+y_0

params, covariance_matrix = curve_fit(f, T_background, noms(I_background),p0=[0.0001,1,1],sigma=stds(I_background))

uncertainties = np.sqrt(np.diag(covariance_matrix))

print("Background fit params: ")
for name, value, uncertainty in zip("ayb", params, uncertainties):
    print(f"{name} = {value} ± {uncertainty}")

params=unp.uarray(params,uncertainties)

x=np.linspace(T[0],T[-1],500)

# Plot current with background
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(T,noms(I),".",label="Data")
ax.plot(T_background,noms(I_background),"gx",label="Backgorund")
ax.plot(x,f(x,*noms(params)),"b",label="Background fit")
ax.set_xlabel(r"$T \mathbin{/} \unit{\celsius}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

ax.legend(loc="best")

fig.savefig("build/Current1.pdf")

# Clear relaxation current of background
I_sig=I-uf(T,*params)

# Print background corrected current
print("Background corrected current: ")
for t,i in zip(T,I_sig):
    print(f" {t} \t {i} \n")


# Plot current without background
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(T,noms(I_sig),".",label="Background corrected data")
ax.set_xlabel(r"$T \mathbin{/} \unit{\celsius}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

ax.legend(loc="best")

fig.savefig("build/Current1_no background.pdf")


# T index where I is max
T_max_ind = 43
T_max=-13.9+273.15 # T_max in kelvin


# use linear fit to determine W for T<T_max (Polarization method)
I_pol=I_sig[32:T_max_ind]
T_pol=T[32:T_max_ind]


# Fit background
def lin(x,m,a):
    return m*x+a

# Print values for polarization method
print("Polarization method data points: ")
for t, i in zip(T_pol, I_pol):
    print(rf" {t:.1f} & {1/t:.3e} & {np.log(noms(i)):.1f} \pm {stds((unp.log(i))):.1f} \\")

params_pol, covariance_matrix_pol = curve_fit(lin, 1/T_pol, np.log(noms(I_pol)),sigma=stds(unp.log(I_pol)))

uncertainties_pol = np.sqrt(np.diag(covariance_matrix_pol))

print("Polarization fit params: ")
for name, value, uncertainty in zip("ma", params_pol, uncertainties_pol):
    print(f"{name} = {value} ± {uncertainty}")

W1=-unp.uarray(params_pol[0],uncertainties_pol[0])*kB

print("Activation energy for polarization fit: ",W1) # with k_b in ev/K

x_pol=np.linspace(T_pol[0],T_pol[-1],500)

# only take positive I values to plot the log
# mask = I_sig > 0
# T_pol_plot = T[mask]
# I_pol_plot = I_sig[mask]

# Plot Polarization method
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(1/T_pol,np.log(noms(I_pol)),".",label="Filtered current")
# ax.plot(x_pol,np.exp(params_pol[0]/x_pol+params_pol[1]),"b",label="Activation fit")
ax.plot(1/x_pol,params_pol[0]*(1/x_pol)+params_pol[1],"b",label="Activation fit")
ax.set_xlabel(r"$\frac{1}{T} \mathbin{/} \unit{\kelvin^{-1}}$")
ax.set_ylabel(r"$\ln\left(\frac{I}{\unit{\pico\ampere}}\right)$")

ax.legend(loc="best")

fig.savefig("build/Polarization_method1.pdf")

# Determine W with integral method

Int_complete = integrate.cumulative_trapezoid(noms(I_sig), T, initial=0) #Integration over complete T[:]
Int_Max = Int_complete[T_max_ind] # Integration until T_max
Int_i = Int_Max - Int_complete # Integration from T[i] to T_max

# Values needed for fitting, again start at 32, as this is the beginning of the peak
I_int = I_sig[32:T_max_ind]
T_int = T[32:T_max_ind]
Int = Int_i[32:T_max_ind]

# Define x and y for fitting
Y = unp.log(Int / I_int)
X = 1 / T_int


# print important values for integral method
print("Integration method values: ")
for i, t, F in zip(I_int, T_int, Int):
    print(rf" {t:.1f} & {1/t:.3e} & {np.log(noms(i)):.1f} \pm {stds((unp.log(i))):.1f} & {F:.1f} & {np.log(noms(F/i)):.1f} \pm {stds((unp.log(F/i))):.1f} \\")

params_int, covariance_matrix_int = curve_fit(lin, X, noms(Y),sigma=stds(Y))

uncertainties_int = np.sqrt(np.diag(covariance_matrix_int))

print("Integration fit params: ")
for name, value, uncertainty in zip("ma", params_int, uncertainties_int):
    print(f"{name} = {value} ± {uncertainty}")

W1_int = unp.uarray(params_int[0],uncertainties_int[0]) *kB  
print("Activation energy for integration fit: ",W1_int) # with k_b in ev/K

X_fit = np.linspace(min(X), max(X), 500)
Y_fit = params_int[0] * X_fit + params_int[1]

# Plot Integration method
fig, ax = plt.subplots(1, 1, layout="constrained")
# ax.plot(T,I_sig,".",label="Filtered current")
ax.plot(X, noms(Y), ".", label="Data (ln(Int/I))")
ax.plot(X_fit, Y_fit, "b", label="Activation fit")
# ax.plot(x_int,Int/np.exp(params_int[0]/x_int+params_int[1]),"b",label="Activation fit")
ax.set_xlabel(r"$\frac{1}{T} \mathbin{/} \unit{\kelvin^{-1}}$")
ax.set_ylabel(r"$\ln\left(\frac{F}{I}\right)$")

ax.legend(loc="best")

fig.savefig("build/Integration_method1.pdf")

# Get tau_zero

tau_zero_pol_1 = kB * T_max**2 / (W1 * b) * unp.exp(-W1 / (kB * T_max))
tau_zero_int_1 = kB * T_max**2 / (W1_int * b) * unp.exp(-W1_int / (kB * T_max))

print("tau_zero pol: ", tau_zero_pol_1, "tau_zero int: ", tau_zero_int_1)





######      Run 2           ######
print("RUN2\n")

T_raw,I_raw,F,P=np.genfromtxt("data/run_2_corr.txt",skip_header=1,unpack=True)

T=T_raw+273.15
I=unp.uarray(-I_raw*F,F/10) # I in pA and change sign for log later


fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(np.arange(T.size),T,".",label="Data")
ax.set_ylabel(r"$T \mathbin{/} \unit{\celsius}$")
ax.set_xlabel(r"$t \mathbin{/} \unit{\minute}$")


fig.savefig("build/heating_rate_2.pdf")

# heating rate
b_array = np.gradient(T, 1)
b=np.mean(b_array)

# mean uncertainty
std_b = np.std(b_array, ddof=1)
std_mean_b = std_b / np.sqrt(len(b_array))

b=unp.uarray(b,std_mean_b)


print("heating rate: ", b, "+- ",std_mean_b)

# define background
I_background=unp.uarray(np.concatenate([noms(I[0:15]),noms(I[-16:])]),np.concatenate([stds(I[0:15]),stds(I[-16:])]))
T_background=np.concatenate([T[0:15],T[-16:]])
print(T[0],T[15],T[-16],T[-1])

params, covariance_matrix = curve_fit(f, T_background, noms(I_background),p0=[-2200,0.0001,1.6])

uncertainties = np.sqrt(np.diag(covariance_matrix))

print("Background fit params: ")
for name, value, uncertainty in zip("aby", params, uncertainties):
    print(f"{name} = {value} ± {uncertainty}")

params=unp.uarray(params,uncertainties)


x=np.linspace(T[0],T[-1],500)

# Plot current with background
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(T,noms(I),".",label="Data")
ax.plot(T_background,noms(I_background),"gx",label="Backgorund")
ax.plot(x,f(x,*noms(params)),"b",label="Background fit")
ax.set_xlabel(r"$T \mathbin{/} \unit{\celsius}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

ax.legend(loc="best")

fig.savefig("build/Current2.pdf")

# Clear relaxation current of background
I_sig=I-uf(T,*params)

# Print background corrected current
print("Background corrected current: ")
for t,i in zip(T,I_sig):
    print(rf" {t:.1f} & {noms(i):.1f} $\pm$ {stds(i):.1f} \\")


# Plot current without background
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(T,noms(I_sig),".",label="Background corrected data")
ax.set_xlabel(r"$T \mathbin{/} \unit{\celsius}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

ax.legend(loc="best")

fig.savefig("build/Current2_no background.pdf")


# T index where I is max
T_max_ind = 25
T_max=-14.7+273.15 # T_max in kelvin

# use linear fit to determine W for T<T_max (Polarization method)
I_pol=I_sig[15:T_max_ind]
T_pol=T[15:T_max_ind]


# Fit background
def lin(x,m,a):
    return m*x+a

# Print values for polarization method
print("Polarization method data points: ")
for t, i in zip(T_pol, I_pol):
    print(rf" {t:.1f} & {1/t:.3e} & {np.log(noms(i)):.1f} \pm {stds((unp.log(i))):.1f} \\")

params_pol, covariance_matrix_pol = curve_fit(lin, 1/T_pol, np.log(noms(I_pol)),sigma=stds((unp.log(I_pol))))

uncertainties_pol = np.sqrt(np.diag(covariance_matrix_pol))

print("Polarization fit params: ")
for name, value, uncertainty in zip("ma", params_pol, uncertainties_pol):
    print(f"{name} = {value} ± {uncertainty}")

W2=-unp.uarray(params_pol[0],uncertainties_pol[0])*kB

print("Activation energy for polarization fit: ",W2) # with k_b in ev/K

x_pol=np.linspace(T_pol[0],T_pol[-1],500)

# only take positive I values to plot the log
# mask = I_sig > 0
# T_pol_plot = T[mask]
# I_pol_plot = I_sig[mask]

# Plot Polarization method
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(1/T_pol,noms(unp.log(I_pol)),".",label="Filtered current")
# ax.plot(x_pol,np.exp(params_pol[0]/x_pol+params_pol[1]),"b",label="Activation fit")
ax.plot(1/x_pol,params_pol[0]*(1/x_pol)+params_pol[1],"b",label="Activation fit")
ax.set_xlabel(r"$\frac{1}{T} \mathbin{/} \unit{\kelvin^{-1}}$")
ax.set_ylabel(r"$\ln\left(\frac{I}{\unit{\pico\ampere}}\right)$")

ax.legend(loc="best")

fig.savefig("build/Polarization_method2.pdf")

# Determine W with integral method

Int_complete = integrate.cumulative_trapezoid(noms(I_sig), T, initial=0) #Integration over complete T[:]
Int_Max = Int_complete[T_max_ind] # Integration until T_max
Int_i = Int_Max - Int_complete # Integration from T[i] to T_max

# Values needed for fitting, again start at 15, as this is the beginning of the peak
I_int = I_sig[15:T_max_ind]
T_int = T[15:T_max_ind]
Int = Int_i[15:T_max_ind]

# print important values for integral method
print("Integration method values: ")
for i, t, F in zip(I_int, T_int, Int):
    print(rf" {t:.1f} & {1/t:.3e} & {np.log(noms(i)):.1f} \pm {stds((unp.log(i))):.1f} & {F:.1f} & {np.log(noms(F/i)):.1f} \pm {stds((unp.log(F/i))):.1f} \\")


# Define x and y for fitting
Y = unp.log(Int / I_int)
X = 1 / T_int


params_int, covariance_matrix_int = curve_fit(lin, X, noms(Y),sigma=stds(Y))

uncertainties_int = np.sqrt(np.diag(covariance_matrix_int))

print("Integration fit params: ")
for name, value, uncertainty in zip("ma", params_int, uncertainties_int):
    print(f"{name} = {value} ± {uncertainty}")

W2_int = unp.uarray(params_int[0],uncertainties_int[0]) *kB   
print("Activation energy for integration fit: ",W2_int) # with k_b in ev/K

X_fit = np.linspace(min(X), max(X), 500)
Y_fit = params_int[0] * X_fit + params_int[1]

# Plot Integration method
fig, ax = plt.subplots(1, 1, layout="constrained")
# ax.plot(T,I_sig,".",label="Filtered current")
ax.plot(X, noms(Y), ".", label="Data (ln(Int/I))")
ax.plot(X_fit, Y_fit, "b", label="Activation fit")
# ax.plot(x_int,Int/np.exp(params_int[0]/x_int+params_int[1]),"b",label="Activation fit")
ax.set_xlabel(r"$\frac{1}{T} \mathbin{/} \unit{\kelvin^{-1}}$")
ax.set_ylabel(r"$\ln\left(\frac{F}{I}\right)$")

ax.legend(loc="best")

fig.savefig("build/Integration_method2.pdf")

# Get tau_zero

tau_zero_pol_2 = kB * T_max**2 / (W2 * b) * unp.exp(-W2 / (kB * T_max))
tau_zero_int_2 = kB * T_max**2 / (W2_int * b) * unp.exp(-W2_int / (kB * T_max))

print("tau_zero pol: ", tau_zero_pol_2, "tau_zero int: ", tau_zero_int_2)


# plot tau
x_tau=np.linspace(T0,Tend,300)
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(x_tau,noms(tau_zero_pol_1)*np.exp(noms(W1)/(kB*x_tau)),label=r"Polarization method $b_1$")
ax.plot(x_tau,noms(tau_zero_int_1)*np.exp(noms(W1_int)/(kB*x_tau)),label=r"Integration method $b_1$")
ax.plot(x_tau,noms(tau_zero_pol_2)*np.exp(noms(W2)/(kB*x_tau)),label=r"Polarization method $b_2$")
ax.plot(x_tau,noms(tau_zero_int_2)*np.exp(noms(W2_int)/(kB*x_tau)),label=r"Integration method $b_2$")

ax.set_yscale("log")
ax.legend(loc="best")

fig.savefig("build/tau.pdf")

