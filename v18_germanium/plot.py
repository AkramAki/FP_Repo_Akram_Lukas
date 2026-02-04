import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
from plot_functions import read_spe, subtract_background, centroid_in_window
from plot_functions import plot_spectrum, activity_at_time, pick_calibration_peaks, omega
from plot_functions import line_content_sideband, full_energy_efficiency, peak_widths_with_baseline, klein_nishina_dsigma_dE

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

# Read Data and create plot from which the peaks are picked
Eu_channels, Eu_time = read_spe("data/Eu_Akram_Lukas.Spe")
background_channels, background_time = read_spe(
    "data/Germanium_Background.Spe")

eu_channels_corrected = subtract_background(
    Eu_channels, Eu_time, background_channels, background_time)

peaks_idx = pick_calibration_peaks(
    eu_channels_corrected, prominence=35, distance=15)
print("Energys to pick from sorted:")
print(sorted(eu152_lines.keys()))
print()

# Pick the indexes of the peaks to choose from the created plot stored in "build/pick_peaks.pdf"
chosen = [3, 5, 6, 7, 8, 9, 10]  # first pick [3, 5, 6, 7, 8, 9, 10]
if len(chosen) == 0:
    raise ValueError("Look at build/pick_peaks.pdf and choose the peak positions to be used and matched.",
                     "Do choose open plot.py and fill the \"chosen\" array. Match with the energies_keV array.")
peak_channels = peaks_idx[chosen].astype(float)
# Matched by hand by looking at plot and trying values until fit looked good
matching_energies_keV = [121.7817, 244.6974,
                         344.2785, 411.1165, 443.9606, 778.9045, 964.057]

if len(peak_channels) != len(matching_energies_keV):
    raise ValueError(
        "channels must have the same length as energies_keV (one channel per reference line).")


# Do the linear fit
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
ymin = 0.1  # position where to start the text

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

# Test the fit

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
ax.set_ylabel(
    r"Residual $E_\text{lit} - E_\text{fit}$ (\unit{\kilo\electronvolt})")
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


# Efficiency part of the analysis
# --- Activity decay correction example ---
A0_bq = 4130.0  # activity given           TODO +-60 need to add uncertainty calculations
t0 = datetime(2000, 10, 1, 0, 0, 0)   # reference date
# measurement date from $DATE_MEA in Eu_Akram_Lukas
t_meas = datetime(2025, 8, 12, 10, 20, 18)

half_life_s = 13.537 * 365.25 * 24 * 3600  # half-life (~13.537 years)
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
eps_FE = np.empty_like(energies_keV, dtype=float)

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
print(f"Omega is {omega(8e-2, 2.25e-2)}")
print()
print("E_keV\t\tI_gamma\t\tpeak_ch\t\tN_line\t\teps_FE")
for E, I, c, N, e in rows:
    print(f"{E:8.3f}\t{I:0.5f}\t\t{c:7.2f}\t\t{N:10.2f}\t{e:0.6e}")
print()

# power-law model


def efficiency_powerlaw(E_keV, a, b):
    return a * (E_keV)**b


# --- fit ---
popt, pcov = curve_fit(
    efficiency_powerlaw,
    energies_keV[1:],  # only above 150kev should be considered
    eps_FE[1:],
)

a_fit, b_fit = popt
a_err, b_err = np.sqrt(np.diag(pcov))

print("Power Law Fit of efficiency Q = a * E ** b")
print(f"a = {a_fit:.4f} ± {a_err:.4f}")
print(f"b = {b_fit:.4f} ± {b_err:.4f}")
print()


E_plot = np.linspace(min(energies_keV)*0.9, max(energies_keV)*1.1, 300)

fig, ax = plt.subplots()

ax.plot(energies_keV[1:], eps_FE[1:] * 100, "o",
        label="Measured values above 150keV")
ax.plot(
    E_plot,
    efficiency_powerlaw(E_plot, a_fit, b_fit) * 100,
    "-",
    label=r"Fit: $Q(E) = a\,(E/1\,\mathrm{keV})^b$"
)

# ax.set_xscale("log")
# ax.set_yscale("log")

ax.set_xlabel(r"Energy $E$ (\unit{\kilo\electronvolt})")
ax.set_ylabel(r"Efficiency $Q$ (\%)")

ax.legend()
ax.grid(True, which="both", alpha=0.3, color="white")
ax.set_facecolor("gainsboro")

fig.savefig("build/efficiency_powerlaw_fit.pdf", bbox_inches="tight")


# Next task

Cs_channels, Cs_time = read_spe("data/Cs_Akram_Lukas.Spe")

Cs_channels_corrected = subtract_background(
    Cs_channels, Cs_time, background_channels, background_time)
fig, ax = plot_spectrum(
    Cs_channels_corrected,
    calibration=(a, b),
    logy=True,
    title="Cs — Calibrated.",
    # savepath="build/Cs.pdf"
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
ymin = 1  # point where text should appear

for x, txt in [
    (backscatter_E, "backscatter peak"),
    (compton_edge_E, "Compton edge"),
    (photo_E, "photo peak"),
]:
    ax.axvline(x, linestyle="--", linewidth=1, color="grey")
    ax.text(x, ymin, txt, rotation=90, va="top", ha="right", fontsize=9)

fig.savefig("build/Cs.pdf", bbox_inches="tight")


# Axes in energy (keV)
ch = np.arange(len(Cs_channels_corrected), dtype=float)
E_axis_keV = a * ch + b
y_cs = np.asarray(Cs_channels_corrected, dtype=float)

# 1) Photopeak energy
E_gamma_keV = float(photo_E)
# 2) Photopeak line content + local background
N_photopeak, N_raw_pp, bg_pp = line_content_sideband(
    spectrum=Cs_channels_corrected,
    peak_center_idx=photo_idx,
    peak_half_width=23,
    sideband_width=10,
    sideband_gap=4,
    plot_diagnostics=True,
    savepath="build/Cs_photopeak_linecontent.pdf",
    energy=E_gamma_keV,
)
# 3) Half-width (FWHM) and tenth-width using SAME baseline
widths, peak_height = peak_widths_with_baseline(
    E_axis_keV, y_cs, photo_idx, bg_pp, levels=(0.5, 0.1)
)

xL_50, xR_50, fwhm_keV = widths[0.5]
xL_10, xR_10, tenth_width_keV = widths[0.1]
# 4) Compton edge and backscatter line (from above)
E_compton_edge_keV = float(compton_edge_E)
E_backscatter_keV = float(backscatter_E)
# 5) Compton continuum content
E_low_keV = 350.0
mask_cont = (E_axis_keV >= E_low_keV) & (E_axis_keV <= E_compton_edge_keV)


def fit_func(E, A, B):  # A is scaling factor and B is constant offset
    return klein_nishina_dsigma_dE(E, E_gamma_keV, A) + B


popt, pcov = curve_fit(
    fit_func,
    E_axis_keV[mask_cont], y_cs[mask_cont],
    maxfev=20000
)

A_fit, B_fit = popt
A_err, B_err = np.sqrt(np.diag(pcov))

print("\n--- Compton fit (Klein-Nishina-based) ---")
print(f"Fit window: [{E_low_keV:.1f}, {E_compton_edge_keV:.1f}] keV")
print(f"A  = {A_fit:.3g} ± {A_err:.3g}")
print(f"B  = {B_fit:.3g} ± {B_err:.3g}")

E_grid = np.linspace(50, E_compton_edge_keV, 6000)
y_model = fit_func(E_grid, A_fit, B_fit)

# integrate only the Compton part (subtract fitted constant offset B)
N_compton_continuum_fit = float(np.trapz(y_model - B_fit, E_grid))
N_compton_continuum_fit_with_offset = float(np.trapz(y_model, E_grid))

# ----------------------------
# Diagnostic plot of the fit
# ----------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.step(E_axis_keV, y_cs, where="mid", label="Cs spectrum")
ax.plot(E_axis_keV[mask_cont], y_cs[mask_cont],
        "o", markersize=3, label="Fit region")
ax.plot(E_grid, y_model, "-", label="KN fit")

ax.axvline(E_compton_edge_keV, linestyle="--", linewidth=1,
           label="Measured Compton edge", color="red")
ax.axvline(E_backscatter_keV, linestyle="--", linewidth=1,
           label="Backscatter peak", color="orange")

ax.set_xlim(30, E_compton_edge_keV + 30)
ax.set_ylim(1, 65)
# ax.set_yscale("log")
ax.set_xlabel(r"$E_{\mathrm{dep}}$ (keV)")
ax.set_ylabel("Counts")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)

fig.savefig("build/Cs_compton_KN_fit.pdf", bbox_inches="tight")


# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------
print("\n=== Cs-137 monoenergetic spectrum results ===")
print(f"Photopeak energy:        {E_gamma_keV:.2f} keV")
print(f"FWHM (half-width):       {fwhm_keV:.2f} keV")
print(f"Tenth-width (10%):       {tenth_width_keV:.2f} keV")
print(f"half-width / Tenth-width:{(fwhm_keV / tenth_width_keV):.2f}")
print(f"Tenth-width / half-width:{(tenth_width_keV / fwhm_keV):.2f}")
print(f"Compton edge position:   {E_compton_edge_keV:.2f} keV")
print(f"Backscatter line pos.:   {E_backscatter_keV:.2f} keV")
print(f"Photopeak line content:  {N_photopeak:.2f} counts")
print(f"Integrated Compton content: {N_compton_continuum_fit:.2f} counts")
print(
    f"Integrated Compton content with offset added: {N_compton_continuum_fit_with_offset:.2f} counts")

# Plot for all of these printed values
fig, (ax_comp, ax_pp) = plt.subplots(
    1, 2, figsize=(11, 4)
)

# ============================================================
# Right panel: Photopeak region
# ============================================================
ax_pp.step(E_axis_keV, y_cs - bg_pp, where="mid", color="C0")
ax_pp.set_yscale("log")

# photopeak + widths
ax_pp.axvline(E_gamma_keV, color="C3", linestyle="--", label="Photopeak")
ax_pp.axvspan(xL_50, xR_50, color="C3", alpha=0.25, label="FWHM")
ax_pp.axvspan(xL_10, xR_10, color="C3", alpha=0.12, label="Tenth-width")

# zoom around photopeak
pp_margin = 3 * fwhm_keV
ax_pp.set_xlim(E_gamma_keV - pp_margin, E_gamma_keV + pp_margin)
ax_pp.set_ylim(1, 1000)


ax_pp.set_xlabel(r"$E_{\mathrm{dep}}$ (keV)")
ax_pp.set_ylabel("Counts")
ax_pp.set_title("Photopeak region minus local background")
ax_pp.legend(fontsize=9)
ax_pp.grid(True, which="both", alpha=0.3)

# ============================================================
# Left panel: Compton region
# ============================================================
ax_comp.step(E_axis_keV, y_cs, where="mid", color="C0")
# ax_comp.set_yscale("log")

# Compton features
ax_comp.axvline(E_backscatter_keV, color="C1",
                linestyle="--", label="Backscatter peak")
ax_comp.axvline(E_compton_edge_keV, color="C2",
                linestyle="--", label="Compton edge")
ax_comp.axvspan(
    E_low_keV,
    E_compton_edge_keV,
    color="C2",
    alpha=0.10,
    label="Compton continuum",
)

# zoom on Compton region
ax_comp.set_xlim(E_low_keV - 20, E_compton_edge_keV * 1.05)
ax_comp.set_ylim(0, 60)

ax_comp.set_xlabel(r"$E_{\mathrm{dep}}$ (keV)")
ax_comp.set_title("Compton region")
ax_comp.legend(fontsize=9)
ax_comp.grid(True, which="both", alpha=0.3)

# ============================================================
# Finalize
# ============================================================
fig.suptitle("Cs-137 monoenergetic spectrum analysis", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])

fig.savefig("build/Cs_analysis_twopanel.pdf", bbox_inches="tight")


# Next task
# We now the full energy efficiency with efficiency_powerlaw(E, a_fit, b_fit)

Ba_channels, Ba_time = read_spe("data/Ba_Akram_Lukas.Spe")

Ba_channels_corrected = subtract_background(
    Ba_channels, Ba_time, background_channels, background_time)
E_axis_keV = a * Ba_channels_corrected + b

peaks_idx = pick_calibration_peaks(
    Ba_channels_corrected, prominence=35, distance=15, savepath="build/Ba_pick_peaks.pdf")
picked_peaks = [1, 3, 4, 5, 6]
peak_channels = peaks_idx[picked_peaks].astype(int)

fig, ax = plot_spectrum(
    Ba_channels_corrected,
    calibration=(a, b),
    logy=True,
    title="Ba with marked peaks for activity calculation.",
    peaks_idx=peak_channels,
    matching_energies=a * peak_channels + b,
    savepath="build/Ba.pdf"
)

activitys = []
# https://www.radiacode.com/isotope/ba-133?lang=de
I_gamma = [0.329, 0.0716, 0.1834, 0.6205, 0.0894]


for i, peak_position in enumerate(peak_channels):
    E_i = a * peak_channels[i] + b
    I_gamma_i = I_gamma[i]
    N_line, _, _ = line_content_sideband(
        spectrum=Ba_channels_corrected,
        peak_center_idx=peak_position,
        peak_half_width=15,
        sideband_width=10,
        sideband_gap=4,
        plot_diagnostics=False,  # set to true to see how the sideband background was calculated
        savepath=f"build/sideband_Ba_peaknumber_{picked_peaks[i]}",
        energy=E_i
    )
    eps_i = efficiency_powerlaw(E_i, a_fit, b_fit)
    A_i = N_line / (I_gamma_i * eps_i * Ba_time)
    activitys.append(A_i)
    print(f"Activity for energy line {E_i:.2f} is {A_i:.2f}")
    print(f"Its line content is {N_line:.0f}")
A_mean = np.mean(activitys)
A_std = np.std(activitys)
print(f"Mean: {A_mean:.2f}")
print(f"Std: {A_std:.2f}")

print("Without the first value")
print(f"Mean: {np.mean(activitys[1:]):.2f}")
print(f"Std: {np.std(activitys[1:]):.2f}")


# Next task

unknown_channels, unknown_time = read_spe("data/unknown_Akram_Lukas.Spe")

unknown_channels_corrected = subtract_background(
    unknown_channels, unknown_time, background_channels, background_time)
E_axis_keV = a * Ba_channels_corrected + b

peaks_idx = pick_calibration_peaks(
    unknown_channels_corrected, prominence=100, distance=15, savepath="build/unknown_pick_peaks.pdf")
picked_peaks = [3, 7, 10, 11, 12, 13, 14, 15]
peak_channels = peaks_idx[picked_peaks].astype(int)

fig, ax = plot_spectrum(
    unknown_channels_corrected,
    calibration=(a, b),
    logy=True,
    title="unknown — Calibrated.",
    peaks_idx=peak_channels,
    matching_energies=a * peak_channels + b,
    savepath="build/unknown.pdf"
)

print("Peaks at energy 77, 92 and 960 keV have bad sideband calculation because there are other peaks in there side band region. Keep in mind!!!")
for i, peak_position in enumerate(peak_channels):
    E_i = a * peak_channels[i] + b
    N_line, _, _ = line_content_sideband(
        spectrum=unknown_channels_corrected,
        peak_center_idx=peak_position,
        peak_half_width=15,
        sideband_width=10,
        sideband_gap=4,
        plot_diagnostics=False,  # set to true to see how the sideband background was calculated
        savepath=f"build/sideband_unknown_peaknumber_{picked_peaks[i]}",
        energy=E_i
    )
    # eps_i = efficiency_powerlaw(E_i, a_fit, b_fit)
    # A_i = N_line / (I_gamma_i * eps_i * Ba_time)
    # activitys.append(A_i)
    # print(f"Activity for energy line {E_i:.2f} is {A_i:.2f}")
    print(f"Line content for energy {E_i:.2f} is {N_line:.0f}")
