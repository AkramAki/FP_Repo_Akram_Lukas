# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit


# T1_raw,I1_raw,F1,P1=np.genfromtxt("data/run_1.txt",skip_header=1,unpack=True) # run_1_corr corrects the .3 scale

# T1=T1_raw-273.15
# I1=I1_raw/10*F1

# # define background
# I_background1=np.concatenate([I1[0:25],I1[-11:]])
# T_background1=np.concatenate([T1[0:25],T1[-11:]])

# # Fit background
# def f(x,I_0,a,y_0):
#     return -I_0*(np.exp(a*x))+y_0 

# params1, covariance_matrix1 = curve_fit(f, T_background1, I_background1,p0=[1,0.0001,1])

# uncertainties1 = np.sqrt(np.diag(covariance_matrix1))

# print("Background fit params: ")
# for name, value, uncertainty in zip("Iay", params1, uncertainties1):
#     print(f"{name} = {value} ± {uncertainty}")

# x=np.linspace(T1[0],T1[-1],50)

# # Plot current with coloured background
# fig, ax = plt.subplots(1, 1, layout="constrained")
# ax.plot(T1,I1,".",label="Data")
# ax.plot(T_background1,I_background1,"gx",label="Backgorund")
# ax.plot(x,f(x,*params1),"b",label="Background fit")
# ax.set_xlabel(r"$T \mathbin{/} \unit{\celsius}$")
# ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

# ax.legend(loc="best")

# fig.savefig("build/Current1.pdf")

# # Clear relaxation current of background
# I1_sig=I1-f(T1,*params1)



# # T index where I is max
# T1_max = 43

# # use linear fit to determine W for T<T_max
# I1_lin=I1_sig[32:T1_max]
# T1_lin=T1[32:T1_max]

# print(I1_lin)
# print(T1_lin)

# # Fit background
# def lin(x,m,a):
#     return m*x+a

# params1_lin, covariance_matrix1_lin = curve_fit(lin, 1/T1_lin, np.log(-I1_lin))

# uncertainties1_lin = np.sqrt(np.diag(covariance_matrix1_lin))

# print("Linear fit params: ")
# for name, value, uncertainty in zip("Ima", params1_lin, uncertainties1_lin):
#     print(f"{name} = {value} ± {uncertainty}")

# print("Activation energy for linear fit: ",params1_lin[0]*8.617*10**(-5)) # with k_b in ev/K

# x_lin=np.linspace(T1_lin[0],T1_lin[-1],50)

# # Plot new current
# fig, ax = plt.subplots(1, 1, layout="constrained")
# ax.plot(1/T1,np.log(np.abs(-I1_sig)),".",label="Filtered current")
# ax.plot(1/x_lin,lin(x,*params1_lin),"b",label="Activation fit")

# ax.legend(loc="best")

# fig.savefig("build/Current_filtered.pdf")