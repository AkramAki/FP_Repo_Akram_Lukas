import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds

def load_spe_file(filename):
    """
    Load spectrum data from a .Spe file (Maestro format).
    
    Parameters:
    -----------
    filename : str
        Path to the .Spe file
        
    Returns:
    --------
    channels : np.ndarray
        Channel numbers
    counts : np.ndarray
        Counts for each channel
    metadata : dict
        Dictionary containing metadata like date, time, detector info
    """
    metadata = {}
    channels = None
    counts = []
    in_data_section = False
    data_start_line = None
    data_end_line = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Extract metadata
        if line.startswith('$SPEC_ID:'):
            continue
        elif line.startswith('$SPEC_REM:'):
            if i + 1 < len(lines):
                metadata['remarks'] = lines[i + 1].strip()
        elif line.startswith('$DATE_MEA:'):
            if i + 1 < len(lines):
                metadata['date'] = lines[i + 1].strip()
        elif line.startswith('$MEAS_TIM:'):
            if i + 1 < len(lines):
                metadata['time'] = lines[i + 1].strip()
        elif line.startswith('$DATA:'):
            # Next line contains the data range
            if i + 1 < len(lines):
                range_line = lines[i + 1].strip().split()
                metadata['data_range'] = (int(range_line[0]), int(range_line[1]))
                in_data_section = True
                data_start_line = i + 2
                data_end_line = int(range_line[1]) - int(range_line[0]) + 1 + data_start_line
                # Extract counts
                for j in range(data_start_line, min(data_end_line, len(lines))):
                    try:
                        count = int(lines[j].strip())
                        counts.append(count)
                    except ValueError:
                        if lines[j].strip().startswith('$'):
                            break
                        continue
                break
    
    # Create channel array
    if metadata['data_range']:
        start, end = metadata['data_range']
        channels = np.arange(start, end + 1)
    
    counts = np.array(counts)
    
    # Ensure channels and counts have the same length
    if len(channels) > len(counts):
        channels = channels[:len(counts)]
    elif len(counts) > len(channels):
        counts = counts[:len(channels)]
    
    return channels, counts, metadata

# Load data from files
data_dir = os.path.dirname(__file__)
spe_file = os.path.join(data_dir, 'data', 'Aki_Fiene_20260112.Spe')

channels, counts, metadata = load_spe_file(spe_file)

# print data
# print("Channel numbers:", channels)
# print("Counts:", counts)

fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(channels, counts, label="Spectrum Data")
ax.set_xlabel(r"Channel number")
ax.set_ylabel(r"Counts")
ax.legend(loc="best")

fig.savefig("build/Spectrum_raw.pdf")

print("Total count number:", np.sum(counts))
# cutoff channels at the end where every value is nearly zero
cutoff_index = 235
channels_cut = channels[:cutoff_index]
counts_cut = counts[:cutoff_index]

# print again
# print("Cutoff Channel numbers:", channels_cut)
# print("Cutoff Counts:", counts_cut)


# load calibration from txt (ignore header line)
delay, channel_calib = np.genfromtxt("data/calibration.txt", skip_header=1, unpack=True)

# Calibration function
def f(x,a,b):
    return a*x + b

# Fit calibration
params, cov = curve_fit(f, channel_calib, delay)
a= unp.uarray(params[0], np.sqrt(cov[0][0]))
b= unp.uarray(params[1], np.sqrt(cov[1][1]))
print(f"Calibration parameters: time per channel a = {a}, Offset b = {b}")

# linspace for fitting
x=np.linspace(channel_calib[0],channel_calib[-1],200)

# Plot
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(channel_calib, delay, "x",label="Calibration Data")
ax.plot(x, f(x, noms(a), noms(b)), label="Calibration Fit")
ax.set_xlabel(r"Channel number")
ax.set_ylabel(r"Delay $\mathbin{/} \unit{\us}$")
ax.legend(loc="best")

fig.savefig("build/Calibration.pdf")

# Convert channels to time using calibration
t = f(channels, noms(a), noms(b))
t_cut = f(channels_cut, noms(a), noms(b))
# Plot spectrum
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(t_cut, counts_cut, label="Spectrum Data")
ax.set_xlabel(r"Time $\mathbin{/} \unit{\us}$")
ax.set_ylabel(r"Counts")
ax.legend(loc="best")

fig.savefig("build/Spectrum.pdf")

# fit with exponential decay
def exp(x, A, tau, U):
    return A * np.exp(-x / tau) + U

# filter zeros at the beginning for fitting
valid_indices = np.cumsum(counts) > 0
t_fit = t[valid_indices]
counts_fit = counts[valid_indices]

# filter peak region for fitting
peak_start = 18  
peak_end = 30   
peak_indices = (t_fit < peak_start) | (t_fit > peak_end)
t_fit = t_fit[peak_indices]
counts_fit = counts_fit[peak_indices]

params, cov = curve_fit(exp, t_fit, counts_fit, p0=[100, 5,1])
uncertainties = np.sqrt(np.diag(cov))

print("exp decay params: ")
for name, value, uncertainty in zip("AtU", params, uncertainties):
    print(f"{name} = {value} ± {uncertainty}")

# print values that are used for fitting
# print("Fitting data points (time, counts):")
# for time_val, count_val in zip(t_fit, counts_fit):
#     print(f"({time_val}, {count_val})")

# Plot spectrum in log scale, lines at filtered regions
fig, ax = plt.subplots(1, 1, layout="constrained")
ax.semilogy(t_cut, counts_cut, ".", markersize=3, label="Spectrum Data")
# Limit fit plot to t_cut range
t_fit_plot = t_fit[t_fit <= np.max(t_cut)]
ax.semilogy(t_fit_plot, exp(t_fit_plot, *params), label="Exponential Fit", linewidth=2)
ax.axvspan(0, f(valid_indices[-1], noms(a), noms(b)), color='blue', alpha=0.3, label="Excluded Leading Zeros")
ax.axvspan(f(peak_start, noms(a), noms(b)), f(peak_end, noms(a), noms(b)), color='red', alpha=0.3, label="Excluded Peak Region")
ax.set_xlabel(r"Time $\mathbin{/} \unit{\us}$")
ax.set_ylabel(r"Counts")
ax.legend(loc="best")
fig.savefig("build/Spectrum_log.pdf")