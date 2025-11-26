import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate


kB = 8.617*10**(-5) # boltzmann constant

T_raw,I_raw,F,P=np.genfromtxt("data/run_1_corr.txt",skip_header=1,unpack=True) # run_1_corr corrects the .3 scale

T=T_raw+273.15
I=-I_raw*10*F # I in pA and change sign for log later

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

print("heating rate: ", b, "+- ",std_mean_b)

# define background
I_background=np.concatenate([I[0:32],I[-11:]])
T_background=np.concatenate([T[0:32],T[-11:]])

# Fit background
def f(x,a,y_0,b):
    # return (np.exp(a*x))+y_0 
    return b*np.exp(-a/x)+y_0

params, covariance_matrix = curve_fit(f, T_background, I_background,p0=[0.0001,1,1])

uncertainties = np.sqrt(np.diag(covariance_matrix))

print("Background fit params: ")
for name, value, uncertainty in zip("ayb", params, uncertainties):
    print(f"{name} = {value} ± {uncertainty}")

x=np.linspace(T[0],T[-1],50)

# Plot current with background
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(T,I,".",label="Data")
ax.plot(T_background,I_background,"gx",label="Backgorund")
ax.plot(x,f(x,*params),"b",label="Background fit")
ax.set_xlabel(r"$T \mathbin{/} \unit{\celsius}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

ax.legend(loc="best")

fig.savefig("build/Current1.pdf")

# Clear relaxation current of background
I_sig=I-f(T,*params)


# T index where I is max
T_max_ind = 43
T_max=-13.9+273.15 # T_max in kelvin

# use linear fit to determine W for T<T_max (Polarization method)
I_lin=I_sig[32:T_max_ind]
T_lin=T[32:T_max_ind]


# Fit background
def lin(x,m,a):
    return m*x+a

params_lin, covariance_matrix_lin = curve_fit(lin, 1/T_lin, np.log(I_lin))

uncertainties_lin = np.sqrt(np.diag(covariance_matrix_lin))

print("Linear fit params: ")
for name, value, uncertainty in zip("ma", params_lin, uncertainties_lin):
    print(f"{name} = {value} ± {uncertainty}")

W1=-params_lin[0]*kB

print("Activation energy for linear fit: ",W1) # with k_b in ev/K

x_lin=np.linspace(T_lin[0],T_lin[-1],50)

# Plot Polarization method
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(T,I_sig,".",label="Filtered current")
ax.plot(x_lin,np.exp(params_lin[0]/x_lin+params_lin[1]),"b",label="Activation fit")

ax.legend(loc="best")

fig.savefig("build/Polarization_method1.pdf")

# Determine W with integral method

Int_complete = integrate.cumulative_trapezoid(I_sig, T, initial=0) #Integration over complete T[:]
Int_Max = Int_complete[T_max_ind] # Integration until T_max
Int_i = Int_Max - Int_complete # Integration from T[i] to T_max

# Values needed for fitting, again start at 32, as this is the beginning of the peak
I_int = I_sig[32:T_max_ind]
T_int = T[32:T_max_ind]
Int = Int_i[32:T_max_ind]

# Define x and y for fitting
Y = np.log(Int / I_int)
X = 1 / T_int


params_int, covariance_matrix_int = curve_fit(lin, X, Y)

uncertainties_int = np.sqrt(np.diag(covariance_matrix_int))

print("Integration fit params: ")
for name, value, uncertainty in zip("ma", params_int, uncertainties_int):
    print(f"{name} = {value} ± {uncertainty}")

W1_int = params_int[0] *kB  
print("Activation energy for integration fit: ",W1_int) # with k_b in ev/K

x_int=np.linspace(T_int[0],T_int[-1],50)

# Plot Integration method
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(T,I_sig,".",label="Filtered current")
ax.plot(T_int,Int/np.exp(params_int[0]/T_int+params_int[1]),"b",label="Activation fit")

ax.legend(loc="best")

fig.savefig("build/Integration_method1.pdf")

# Get tau_zero

tau_zero_lin_1 = kB * T_max**2 / (W1 * b) * np.exp(-W1 / (kB * T_max))
tau_zero_int_1 = kB * T_max**2 / (W1_int * b) * np.exp(-W1_int / (kB * T_max))

print("tau_zero lin: ", tau_zero_lin_1, "tau_zero int: ", tau_zero_int_1)





######      Run 2           ######
print("RUN2\n")

T_raw,I_raw,F,P=np.genfromtxt("data/run_2_corr.txt",skip_header=1,unpack=True)

T=T_raw+273.15
I=-I_raw*10*F


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

print("heating rate: ", b, "+- ",std_mean_b)

# define background
I_background=np.concatenate([I[0:15],I[-16:]])
T_background=np.concatenate([T[0:15],T[-16:]])

# Fit background
def f(x,a,b,y_0):
    # return (np.exp(a*x))+y_0 
    return b*np.exp(-a/x)+y_0


params, covariance_matrix = curve_fit(f, T_background, I_background,p0=[0.0001,1,-36])

uncertainties = np.sqrt(np.diag(covariance_matrix))

print("Background fit params: ")
for name, value, uncertainty in zip("aby", params, uncertainties):
    print(f"{name} = {value} ± {uncertainty}")

x=np.linspace(T[0],T[-1],50)

# Plot current with background
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(T,I,".",label="Data")
ax.plot(T_background,I_background,"gx",label="Backgorund")
ax.plot(x,f(x,*params),"b",label="Background fit")
ax.set_xlabel(r"$T \mathbin{/} \unit{\celsius}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

ax.legend(loc="best")

fig.savefig("build/Current2.pdf")

# Clear relaxation current of background
I_sig=I-f(T,*params)


# T index where I is max
T_max_ind = 28
T_max=-14.7+273.15 # T_max in kelvin

# use linear fit to determine W for T<T_max (Polarization method)
I_lin=I_sig[15:T_max_ind]
T_lin=T[15:T_max_ind]


# Fit background
def lin(x,m,a):
    return m*x+a

params_lin, covariance_matrix_lin = curve_fit(lin, 1/T_lin, np.log(I_lin))

uncertainties_lin = np.sqrt(np.diag(covariance_matrix_lin))

print("Linear fit params: ")
for name, value, uncertainty in zip("ma", params_lin, uncertainties_lin):
    print(f"{name} = {value} ± {uncertainty}")

W2=-params_lin[0]*kB

print("Activation energy for linear fit: ",W2) # with k_b in ev/K

x_lin=np.linspace(T_lin[0],T_lin[-1],50)

# Plot Polarization method
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(T,I_sig,".",label="Filtered current")
ax.plot(x_lin,np.exp(params_lin[0]/x_lin+params_lin[1]),"b",label="Activation fit")

ax.legend(loc="best")

fig.savefig("build/Polarization_method2.pdf")

# Determine W with integral method

Int_complete = integrate.cumulative_trapezoid(I_sig, T, initial=0) #Integration over complete T[:]
Int_Max = Int_complete[T_max_ind] # Integration until T_max
Int_i = Int_Max - Int_complete # Integration from T[i] to T_max

# Values needed for fitting, again start at 32, as this is the beginning of the peak
I_int = I_sig[15:T_max_ind]
T_int = T[15:T_max_ind]
Int = Int_i[15:T_max_ind]


# Define x and y for fitting
Y = np.log(Int / I_int)
X = 1 / T_int


params_int, covariance_matrix_int = curve_fit(lin, X, Y)

uncertainties_int = np.sqrt(np.diag(covariance_matrix_int))

print("Integration fit params: ")
for name, value, uncertainty in zip("ma", params_int, uncertainties_int):
    print(f"{name} = {value} ± {uncertainty}")

W2_int = params_int[0] *kB  
print("Activation energy for integration fit: ",W2_int) # with k_b in ev/K

x_int=np.linspace(T_int[0],T_int[-1],50)

# Plot Integration method
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(T,I_sig,".",label="Filtered current")
ax.plot(T_int,Int/np.exp(params_int[0]/T_int+params_int[1]),"b",label="Activation fit")

ax.legend(loc="best")

fig.savefig("build/Integration_method2.pdf")

# Get tau_zero

tau_zero_lin_2 = kB * T_max**2 / (W2 * b) * np.exp(-W2 / (kB * T_max))
tau_zero_int_2 = kB * T_max**2 / (W2_int * b) * np.exp(-W2_int / (kB * T_max))

print("tau_zero lin: ", tau_zero_lin_2, "tau_zero int: ", tau_zero_int_2)


# plot tau
x_tau=np.linspace(T0,Tend,100)
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(x_tau,tau_zero_lin_1*np.exp(W1/(kB*x_tau)),label=r"Polarization method $b_1$")
ax.plot(x_tau,tau_zero_int_1*np.exp(W1_int/(kB*x_tau)),label=r"Integration method $b_1$")
ax.plot(x_tau,tau_zero_lin_2*np.exp(W2/(kB*x_tau)),label=r"Polarization method $b_2$")
ax.plot(x_tau,tau_zero_int_2*np.exp(W2_int/(kB*x_tau)),label=r"Integration method $b_2$")

ax.set_yscale("log")
ax.legend(loc="best")

fig.savefig("build/tau.pdf")

