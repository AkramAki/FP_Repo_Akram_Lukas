import numpy as np
import matplotlib.pyplot as plt

# speed of light in vacuum
c = 2.99792458e8  # m/s

# mapping from filename to nominal resonator length (in cm)
L_nominal_cm = {
    "Moden_1.txt": 53.7,
    "Moden_2.txt": 70.2,
    "Moden_3.txt": 90.0,
    "Moden_4.txt": 110.0,
    "Moden_5.txt": 130.0,
    "Moden_6.txt": 149.9,
    "Moden_7.txt": 170.2,
    "Moden_8.txt": 190.0
}

L_corr_m = []
df_means_Hz = []

for fname, L_cm in L_nominal_cm.items():
    # correct the resonator length (+2.6 cm) and convert to meters
    L_corr = (L_cm + 2.6) * 1e-2   # m
    L_corr_m.append(L_corr)

    # load the data
    path = f"data/L-Moden/{fname}"
    data = np.loadtxt(path, skiprows=1)   # skip header
    freqs_MHz = np.sort(data[:, 1])        # take second column (MHz)

    # compute mean adjacent spacing (fundamental mode spacing)
    spacings_MHz = np.diff(freqs_MHz)
    df_mean_MHz = np.mean(spacings_MHz)

    # convert to Hz
    df_means_Hz.append(df_mean_MHz * 1e6)

L_corr_m = np.array(L_corr_m)
df_means_Hz = np.array(df_means_Hz)

# theoretical longitudinal mode spacing
df_theo_Hz = c / (2*L_corr_m)

################################  Second part  ################################
fig, ax = plt.subplots()

# convert back to cm and MHz for plotting
L_corr_cm = L_corr_m * 1e2
df_means_MHz = df_means_Hz * 1e-6

# create a smooth L axis (cm) for theory
L_smooth_cm = np.linspace(L_corr_cm.min(), L_corr_cm.max(), 500)
L_smooth_m = L_smooth_cm * 1e-2

df_theo_smooth_MHz = (c/(2*L_smooth_m)) * 1e-6
df_theo_at_means_MHz = (c/(2*L_corr_m)) * 1e-6

# measured data: markers only
ax.scatter(L_corr_cm, df_means_MHz, label="measured Δf", marker='o', s=45)

# theoretical: smooth line
ax.plot(L_smooth_cm, df_theo_smooth_MHz,
        label="theoretical Δf = c/(2L)", linestyle='--', linewidth=1.5, alpha=0.6, color="grey")

# theoretical values at measured L: marker only (no line)
ax.scatter(L_corr_cm, df_theo_at_means_MHz, marker='s',
           s=22, alpha=0.5, label="theory at measured L")

ax.set_xlabel(r"Resonator length $L$ / cm")
ax.set_ylabel(r"beat frequency $\Delta f$ / MHz")
ax.set_title(r"Beat frequency vs resonator length")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

fig.savefig("build/L-Moden.pdf")
