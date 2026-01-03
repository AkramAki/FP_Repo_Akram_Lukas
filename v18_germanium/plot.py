import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from plot_functions import read_spe, subtract_background, centroid_in_window, pick_calibration_peaks
from plot_functions import plot_spectrum, activity_at_time, line_content_sideband, full_energy_efficiency

# --- Eu-152 reference energies (keV) from Radiacode https://www.radiacode.com/isotope/eu-152?lang=de ---
# key is energy in keV
eu152_lines = {
    121.7817: {"intensity": 0.2853},
    244.6974: {"intensity": 0.0755},
    344.2785: {"intensity": 0.2659},
    411.1165: {"intensity": 0.02237},
    443.9606: {"intensity": 0.02827},
    778.9045: {"intensity": 0.1293},
    867.3800: {"intensity": 0.0423},
    964.0570: {"intensity": 0.1451},
    1085.8370: {"intensity": 0.1011},
    1112.0760: {"intensity": 0.1367},
    1408.0130: {"intensity": 0.2087},
}

######### Read Data and create plot from which the peaks are picked 
Eu_channels, Eu_time = read_spe("data/Eu_Akram_Lukas.Spe")
background_channels, background_time = read_spe("data/Germanium_Background.Spe")

eu_channels_corrected = subtract_background(Eu_channels, Eu_time, background_channels, background_time)

peaks_idx = pick_calibration_peaks(eu_channels_corrected, prominence=35, distance=15)
print("Energys to pick from sorted:")
print(sorted(eu152_lines.keys()))
print()

# Pick the indexes of the peaks to choose from the created plot stored in "build/pick_peaks.pdf"
chosen = [3, 5, 6, 7, 8, 9, 10] # first pick [3, 5, 6, 7, 8, 9, 10]
if len(chosen) == 0:
    raise ValueError("Look at build/pick_peaks.pdf and choose the peak positions to be used and matched.", 
    "Do choose open plot.py and fill the \"chosen\" array. Match with the energies_keV array.")
peak_channels = peaks_idx[chosen].astype(float)
# Matched by hand by looking at plot and trying values until fit looked good
matching_energies_keV = [121.7817, 244.6974, 344.2785, 411.1165, 443.9606, 778.9045, 964.057] 

if len(peak_channels) != len(matching_energies_keV):
    raise ValueError("channels must have the same length as energies_keV (one channel per reference line).")


######### Do the linear fit
# Fit E = a*ch + b
a, b = np.polyfit(peak_channels, matching_energies_keV, deg=1)

print(f"Calibration: E(keV) = a*channel + b")
print(f"a = {a:.6f} keV/channel")
print(f"b = {b:.3f} keV")
print()


fig, ax = plot_spectrum(
    eu_channels_corrected,
    calibration=(a, b),
    logy=True,
    peaks_idx=peak_channels,
    matching_energies=matching_energies_keV,
    title="Eu-152 — Calibrated.",
    # savepath="build/Eu-152.pdf", do not provide savepath so i can change something in the plot after
)

# add vertical lines at assigned gamma energies
ymin = 0.1 # position where to start the text

for E in matching_energies_keV:
    ax.axvline(
        E,
        linestyle="--",
        linewidth=1,
        color="gray",
        alpha=0.8
    )

    ax.text(
        E, ymin,
        f"{E:.1f} keV",
        rotation=90,
        va="top",
        ha="right",
        fontsize=8,
        color="gray"
    )

fig.savefig("build/Eu-152.pdf")

############ Test the fit 

# --- residuals ---
E_fit = a * peak_channels + b
residuals_keV = matching_energies_keV - E_fit
rms = np.sqrt(np.mean(residuals_keV**2))

print(f"RMS residual = {rms:.3f} keV")
print()

# --- residual plot ---
fig, ax = plt.subplots()

ax.axhline(0.0, linestyle="--", linewidth=1)
ax.plot(matching_energies_keV, residuals_keV, "o")

ax.set_xlabel(r"Literature energy $E_\gamma$ (\unit{\kilo\electronvolt})")
ax.set_ylabel(r"Residual $E_\text{lit} - E_\text{fit}$ (\unit{\kilo\electronvolt})")
ax.set_title("Energy calibration residuals")

ax.grid(True, alpha=0.3, color="white")
ax.set_facecolor("gainsboro")

fig.savefig("build/calibration_residuals.pdf", bbox_inches="tight")


ch_line = np.linspace(peak_channels.min(), peak_channels.max(), 200)

fig, ax = plt.subplots()
ax.plot(peak_channels, matching_energies_keV, "o", label="Assigned peaks")
ax.plot(ch_line, a * ch_line + b, "-", label="Linear fit")

ax.set_xlabel("Channel")
ax.set_ylabel(r"Energy (\unit{\kilo\electronvolt})")
ax.legend()
ax.grid(True, alpha=0.3, color="white")
ax.set_facecolor("gainsboro")

fig.savefig("build/calibration_fit.pdf", bbox_inches="tight")
plt.close(fig)







################# Efficiency part of the analysis
# --- Activity decay correction example ---
A0_bq = 4130.0  # activity given           TODO +-60 need to add uncertainty calculations
t0 = datetime(2000, 10, 1, 0, 0, 0)   # reference date 
t_meas = datetime(2025, 8, 12, 10, 20, 18)  # measurement date from $DATE_MEA in Eu_Akram_Lukas

half_life_s = 13.537 * 365.25 * 24 * 3600  #  half-life (~13.537 years)
A_meas = activity_at_time(A0_bq, t0, t_meas, half_life_s)


# --- settings ---
t_live_s = Eu_time          # live time (s)
peak_half_width = 15
sideband_width = 10
sideband_gap = 4

# save ONE diagnostic figure for this peak (set to None to disable)
save_diag_index = 0
save_diag_path = "build/line_content_example.pdf"

# --- compute for all selected peaks ---
energies_keV = np.asarray(matching_energies_keV, dtype=float)
peak_centers = np.asarray(peak_channels, dtype=float)

N_lines = np.empty_like(energies_keV, dtype=float)
eps_FE  = np.empty_like(energies_keV, dtype=float)

rows = []

for i, (E, c) in enumerate(zip(energies_keV, peak_centers)):
    # emission probability (fraction, not percent)
    I_gamma = eu152_lines[float(E)]["intensity"]

    # enable diagnostics only for the chosen index
    do_diag = (save_diag_index is not None) and (i == save_diag_index)

    N_line, N_raw, bg = line_content_sideband(
        spectrum=eu_channels_corrected,
        peak_center_idx=c,
        peak_half_width=peak_half_width,
        sideband_width=sideband_width,
        sideband_gap=sideband_gap,
        plot_diagnostics=do_diag,
        savepath=(save_diag_path if do_diag else None),
        energy=matching_energies_keV[i]
    )

    eps = full_energy_efficiency(N_line, A_meas, t_live_s, I_gamma)

    N_lines[i] = N_line
    eps_FE[i] = eps

    rows.append((E, I_gamma, c, N_line, eps))

# --- console output ---
print(f"A(meassurment) = {A_meas:.6g} Bq")
print(f"t_live  = {t_live_s:.0f} s")
print()
print("E_keV\t\tI_gamma\t\tpeak_ch\t\tN_line\t\teps_FE")
for E, I, c, N, e in rows:
    print(f"{E:8.3f}\t{I:0.5f}\t\t{c:7.2f}\t\t{N:10.2f}\t{e:0.6e}")

# --- arrays ready for efficiency fit later ---
# energies_keV: x-values
# eps_FE:       y-values
# N_lines:      line contents (if you need them later)







######## Next task

Cs_channels, Cs_time = read_spe("data/Cs_Akram_Lukas.Spe")

Cs_channels_corrected = subtract_background(Cs_channels, Cs_time, background_channels, background_time)
fig, ax = plot_spectrum(
    Cs_channels_corrected,
    calibration=(a, b),
    logy=True,
    title="Cs — Calibrated.",
    #savepath="build/Cs.pdf"
)
# --- find the three features by "largest bins" ---
y = np.asarray(Cs_channels_corrected, dtype=float).copy()

# --- photo peak: global maximum ---
photo_idx = int(np.argmax(y))

# --- backscatter peak: maximum in 1500-1800 channels ---
bs_lo, bs_hi = 1500, 1800
backscatter_idx = int(bs_lo + np.argmax(y[bs_lo:bs_hi + 1]))

# --- Compton edge: maximum in 3800-4050 channels ---
ce_lo, ce_hi = 3800, 4050
compton_edge_idx = int(ce_lo + np.argmax(y[ce_lo:ce_hi + 1]))

# --- convert to x positions in keV with a and b from calibration ---
photo_E = a * photo_idx + b
backscatter_E = a * backscatter_idx + b
compton_edge_E = a * compton_edge_idx + b

# --- add vertical lines + labels to existing ax ---
ymin = 1 # point where text should appear 

for x, txt in [
    (backscatter_E, "backscatter peak"),
    (compton_edge_E, "Compton edge"),
    (photo_E, "photo peak"),
]:
    ax.axvline(x, linestyle="--", linewidth=1, color="grey")
    ax.text(x, ymin, txt, rotation=90, va="top", ha="right", fontsize=9)

fig.savefig("build/Cs.pdf", bbox_inches="tight")




##### Next task

Ba_channels, Ba_time = read_spe("data/Ba_Akram_Lukas.Spe")

Ba_channels_corrected = subtract_background(Ba_channels, Ba_time, background_channels, background_time)
fig, ax = plot_spectrum(
    Ba_channels_corrected,
    calibration=(a, b),
    logy=True,
    title="Ba — Calibrated.",
    savepath="build/Ba.pdf"
)

