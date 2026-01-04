import matplotlib.pyplot as plt
import numpy as np
from uncertainties import unumpy as unp

# hopg real space
distances11 = np.array([1.11/5,1.01/5,0.78/4,0.86/4,0.80/4,0.52/3]) # distances from gwyddion readout, divided by the number of atoms it covers
distances12=np.array([1.68/8,1.74/8,1.51/8,0.89/5])
distances21=np.array([1.88/10,1.47/8,1.61/10,1.95/10])
distances22=np.array([1.00/5,0.75/4,1.10/6])
distances31=np.array([1.45/8,1.28/6,0.90/5,1.43/8])
distances32=np.array([1.71/8,1.09/6,1.26/7,1.26/6])
distances51=np.array([1.44/10,1.49/10,1.48/8,1.40/7])
distances52=np.array([1.59/8,1.55/8,1.55/9,0.68/5])

fft_distances11 = np.array([24.8/4,24.2/4])
fft_distances12 = np.array([24.4/4,23.9/4])
fft_distances21 = np.array([15.4/2,24.4/4])
fft_distances22 = np.array([28.4/4,14.8/2])
fft_distances31 = np.array([13.2/2,14.9/2])
fft_distances32 = np.array([12.2/2,15.2/2])
fft_distances51 = np.array([15.2/2,12.2/2])
fft_distances52 = np.array([12.0/2,15.2/2])


# create uarrays with uncertainty 0.15
distances11_u = unp.uarray(distances11, 0.15 * np.ones(len(distances11)))
distances12_u = unp.uarray(distances12, 0.15 * np.ones(len(distances12)))
distances21_u = unp.uarray(distances21, 0.15 * np.ones(len(distances21)))
distances22_u = unp.uarray(distances22, 0.15 * np.ones(len(distances22)))
distances31_u = unp.uarray(distances31, 0.15 * np.ones(len(distances31)))
distances32_u = unp.uarray(distances32, 0.15 * np.ones(len(distances32)))
distances51_u = unp.uarray(distances51, 0.15 * np.ones(len(distances51)))
distances52_u = unp.uarray(distances52, 0.15 * np.ones(len(distances52)))

fft_distances11_u = unp.uarray(fft_distances11, 1.2 * np.ones(len(fft_distances11)))
fft_distances12_u = unp.uarray(fft_distances12, 1.2 * np.ones(len(fft_distances12)))
fft_distances21_u = unp.uarray(fft_distances21, 1.2 * np.ones(len(fft_distances21)))
fft_distances22_u = unp.uarray(fft_distances22, 1.2 * np.ones(len(fft_distances22)))
fft_distances31_u = unp.uarray(fft_distances31, 1.2 * np.ones(len(fft_distances31)))
fft_distances32_u = unp.uarray(fft_distances32, 1.2 * np.ones(len(fft_distances32)))
fft_distances51_u = unp.uarray(fft_distances51, 1.2 * np.ones(len(fft_distances51)))
fft_distances52_u = unp.uarray(fft_distances52, 1.2 * np.ones(len(fft_distances52)))

# concatenate uarrays
distances_u = np.concatenate([distances11_u, distances12_u, distances21_u, distances22_u, distances31_u, distances32_u, distances51_u, distances52_u])
fft_distances_u = np.concatenate([fft_distances11_u, fft_distances12_u, fft_distances21_u, fft_distances22_u, fft_distances31_u, fft_distances32_u, fft_distances51_u, fft_distances52_u])

# calculate mean with uncertainty
dist_mean_u = np.mean(distances_u)

# inverse for fft
inv_fft_distances_u = 1 / fft_distances_u
inv_mean_u = np.mean(inv_fft_distances_u)

# print results
print("All distances:", distances_u)


print(f"Mean distance: {dist_mean_u}")
print(f"Mean inv fft distance: {inv_mean_u}")

print("All inv fft distances:", inv_fft_distances_u)




# gold avg and std
gold_heights = np.array([4.14, 4.86, 8.25, 5.21, 6.75, 11.55])
gold_avg = np.mean(gold_heights)
gold_std = np.std(gold_heights)/np.sqrt(len(gold_heights))

# print results
print(f"Gold Average Height: {gold_avg:.2f} nm")
print(f"Gold Standard Deviation: {gold_std:.2f} nm")
