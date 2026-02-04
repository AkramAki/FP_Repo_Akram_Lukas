import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Function to read the spe files
# The channel meaning the index of the spectrum array is proportional to Energy
# The number at each position in the spectrum array indicates the counts for that channel.


def read_spe(filename):
    """
    Plot a .Spe spectrum from filepath.

    Parameters
    ----------
    filepath : str
        Path to .Spe file.

    Returns
    -------
    spectrum: float
        spectrum array
    live_meassurment_time: float
        Meassurment time in seconds without dead times: 
    """
    counts = []
    in_data = False
    live_time = None
    n_channels_expected = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("$MEAS_TIM"):
                lt_rt = next(f).strip().split()
                # there are two time values where one is the live time and the other the real time thus with dead times
                live_time = float(lt_rt[0])

            # Enter data block (In Spe file after $DATA there are two numbers indicating channel start and end.
            # The data then comes until another section starts with $)
            if line.startswith("$DATA"):
                in_data = True
                # next line contains channel range
                channel_range = next(f).strip().split()
                ch_start, ch_end = map(int, channel_range)
                n_channels_expected = ch_end - ch_start + 1
                continue

            # Exit data block when next section starts after having entered the data block
            if in_data and line.startswith("$"):
                in_data = False

            # Read counts when we arrived inside the data block and the line is not None
            if in_data and line:
                counts.append(int(line))

    # convert to numpy array
    spectrum = np.array(counts, dtype=float)

    if n_channels_expected is not None and len(spectrum) != n_channels_expected:
        raise ValueError(
            f"Expected {n_channels_expected} channels, got {len(spectrum)}"
        )

    return spectrum, live_time


def subtract_background(spec_sig, t_sig, spec_bg, t_bg):
    """
    Scale background spectrum to the live time of the signal spectrum and subtract.
    Returns background-corrected spectrum (float array).
    """
    if t_sig is None or t_bg is None:
        raise ValueError(
            "Need live times t_sig and t_bg for background subtraction.")

    scale = t_sig / t_bg
    return spec_sig.astype(float) - scale * spec_bg.astype(float)

# This function calculates and x-weighted average to refine the x position of our peak because channels are discrete.
# This should improve the energy fitting if needed. For now kept it out


def centroid_in_window(ch, y, center_idx, half_window=20):
    """
    Compute a simple centroid around a peak near center_idx using a window.
    Returns centroid channel (float).
    """
    lo = max(int(center_idx - half_window), 0)
    hi = min(int(center_idx + half_window), len(y) - 1)

    xw = ch[lo:hi+1]
    yw = y[lo:hi+1].copy()

    # Prevent negative weights after background subtraction:
    yw[yw < 0] = 0.0

    s = yw.sum()
    if s <= 0:
        return float(center_idx)
    return float((xw * yw).sum() / s)


def pick_calibration_peaks(
    spectrum_corr,
    *,
    prominence=200,
    distance=15,
    logy=True,
    xlim=None,
    savepath="build/pick_peaks.pdf"
):
    """
    Find peaks, plot them with indices, and return peak indices + positions.
    You then choose which peaks correspond to which Eu energies.
    """
    y = spectrum_corr.copy()
    y[y < 0] = 0.0
    ch = np.arange(len(y), dtype=float)

    peaks_idx, props = find_peaks(y, prominence=prominence, distance=distance)

    fig, ax = plt.subplots()
    ax.step(ch, y, where="mid")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts (background corrected)")
    if logy:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(1, np.max(y))
    ax.grid(True, which="both", alpha=0.3, color="white")
    ax.set_facecolor('gainsboro')

    # mark peaks and label with running number
    ax.plot(peaks_idx, y[peaks_idx], "x",
            label="Peak candidates for energy calibration")
    ax.legend()
    for k, p in enumerate(peaks_idx):
        ax.text(p, y[p], f"{k}", fontsize=9, va="bottom", ha="center")

    fig.savefig(savepath)

    return peaks_idx


def plot_spectrum(
    spectrum,
    *,                             # force keywords
    calibration=None,              # None or (a, b) with E_keV = a*ch + b
    logy=True,
    xlim=None,
    ylim=None,
    title=None,
    label=None,
    savepath=None,
    show_grid=True,
    step_where="mid",
    peaks_idx=None,                # array-like of int indices in channel-space
    # list/array of indices into peaks_idx to highlight
    matching_energies=None,
    marker_style="x",
):
    """
    Plot a binned spectrum (counts per channel). The spectrum can be background-corrected or raw.

    Parameters
    ----------
    spectrum : array-like
        1D array of counts per channel (can include negatives if background-subtracted).
    calibration : (a, b) or None
        If provided, x-axis is energy in keV via E = a*ch + b. Otherwise x-axis is channel.
    peaks_idx : array-like of int or None
        Peak positions in channel indices to highlight.
    matching_energies : array-like of float or None
        Energys to write above the peaks provided by peaks_idx.
    """

    y = np.asarray(spectrum, dtype=float)
    ch = np.arange(len(y), dtype=float)

    # x-axis mapping
    if calibration is None:
        x = ch
        xlabel = "Channel"
    else:
        a, b = calibration
        x = a * ch + b
        xlabel = r"Deposited energy $E_\mathrm{dep}$ (\unit{\kilo\electronvolt})"

    # For log plots: don't modify data for analysis, only for plotting
    y_plot = y.copy()
    if logy:
        y_plot[y_plot <= 0] = np.nan

    fig, ax = plt.subplots()
    ax.plot(x, y_plot, label=label, alpha=0.75)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")

    if logy:
        ax.set_yscale("log")
    if show_grid:
        ax.grid(True, which="both", alpha=0.3, color="white")
    ax.set_facecolor('gainsboro')

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Peak markers (peaks_idx are the already-filtered indices to mark)
    if peaks_idx is not None:
        peaks_idx = np.asarray(peaks_idx, dtype=int)

        # map x-positions (channel or calibrated energy)
        if calibration is None:
            px = peaks_idx.astype(float)
        else:
            a, b = calibration
            px = a * peaks_idx.astype(float) + b

        # y-values at peaks (avoid nan for marker placement)
        py = np.nan_to_num(y_plot[peaks_idx], nan=0.0)

        # draw markers
        ax.plot(px, py, marker_style, linestyle="None")

        # optionally annotate with matched energies (same length/order as peaks_idx)
        if matching_energies is not None:
            matching_energies = np.asarray(matching_energies, dtype=float)
            if matching_energies.size != peaks_idx.size:
                raise ValueError(
                    "matching_energies must have the same length as peaks_idx.")

            for xi, yi, Ei in zip(px, py, matching_energies):
                ax.text(xi, yi, f"{Ei:.1f} keV", fontsize=9,
                        va="bottom", ha="center")

    if label is not None:
        ax.legend()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    return fig, ax


def activity_at_time(A0_bq, t0, t, half_life_s):
    """
    Decay-correct an activity A0 (in Bq) given at reference time t0 to time t.

    Parameters
    ----------
    A0_bq : float
        Activity at reference time t0 in Bq.
    t0 : datetime-like
        Reference datetime (same timezone handling as t).
    t : datetime-like
        Target datetime.
    half_life_s : float
        Half-life in seconds.

    Returns
    -------
    A_t_bq : float
        Activity at time t in Bq.
    """
    dt_s = (t - t0).total_seconds()
    lam = np.log(2.0) / half_life_s
    return float(A0_bq * np.exp(-lam * dt_s))


def omega(d, R):
    return (2 * np.pi * (1 - (d / np.sqrt(d * d + R * R))))


def full_energy_efficiency(N_photopeak, A_bq, t_live_s, I_gamma):
    """
    Compute full-energy peak efficiency:
        eps_FE = N_photopeak / (A * t_live * I_gamma)

    Parameters
    ----------
    N_photopeak : float
        Background-corrected photopeak line content (counts).
    A_bq : float
        Activity at measurement time in Bq (decays/s).
    t_live_s : float
        Live time in seconds.
    I_gamma : float
        Emission probability (branching ratio) for the gamma line (0..1).

    Returns
    -------
    eps : float
        Full-energy peak efficiency (dimensionless).
    """
    denom = A_bq * t_live_s * I_gamma
    if denom <= 0:
        raise ValueError("A_bq, t_live_s, and I_gamma must be positive.")
    return float(N_photopeak / denom)


def line_content_sideband(
    spectrum,
    peak_center_idx,
    *,
    peak_half_width=20,
    sideband_width=10,
    sideband_gap=5,
    plot_diagnostics=False,
    savepath=None,
    energy=None
):
    """
    Compute background-corrected line content using sidebands:
    - integrate counts in peak window
    - estimate background per channel from left/right sidebands
    - subtract background contribution from peak window

    Parameters
    ----------
    spectrum : array-like
        1D counts per channel (can be background-corrected or raw; if raw, this estimates local baseline).
    peak_center_idx : int
        Approximate peak center (channel index), e.g. from find_peaks or a centroid.
    peak_half_width : int
        Half-width of the integration window around the peak center.
    sideband_width : int
        Number of channels in each sideband region.
    sideband_gap : int
        Gap between peak window edge and sideband start (to avoid including peak tails).
    plot_diagnostics : bool
            If plot_diagnostics is True, a diagnostic plot illustrating the peak window, sidebands, and background level is saved.

    Returns
    -------
    N_line : float
        Background-corrected line content (counts).
    N_peak_raw : float
        Raw integrated counts in peak window.
    bg_per_ch : float
        Estimated background counts per channel (local).
    """
    y = np.asarray(spectrum, dtype=float)
    n = y.size
    c = int(peak_center_idx)
    # Naming conventions hi = high, lo = low, bg = background, ch = channel

    # --- Peak window ---
    p_lo = max(c - peak_half_width, 0)
    p_hi = min(c + peak_half_width, n - 1)
    peak_window = y[p_lo:p_hi + 1]
    N_peak_raw = float(np.sum(peak_window))
    n_peak_ch = (p_hi - p_lo + 1)

    # --- Sidebands ---
    left_hi = max(p_lo - sideband_gap - 1, 0)
    left_lo = max(left_hi - sideband_width + 1, 0)

    right_lo = min(p_hi + sideband_gap + 1, n - 1)
    right_hi = min(right_lo + sideband_width - 1, n - 1)

    left_band = y[left_lo:left_hi +
                  1] if left_hi >= left_lo else np.array([], float)
    right_band = y[right_lo:right_hi +
                   1] if right_hi >= right_lo else np.array([], float)

    side = np.concatenate([left_band, right_band])
    if side.size == 0:
        raise ValueError("Sidebands are empty. Adjust sideband parameters.")

    bg_per_ch = float(np.mean(side))
    N_bg_in_peak = bg_per_ch * n_peak_ch
    N_line = float(N_peak_raw - N_bg_in_peak)

    # --- Optional diagnostic plot ---
    if plot_diagnostics:
        if savepath is None:
            savepath = f"build/linecontent_peak_{c}.pdf"

        # zoom range
        z_lo = max(left_lo - 10, 0)
        z_hi = min(right_hi + 10, n - 1)
        ch = np.arange(z_lo, z_hi + 1)

        fig, ax = plt.subplots()
        ax.step(ch, y[z_lo:z_hi + 1], where="mid")

        # peak window
        ax.axvspan(p_lo, p_hi, alpha=0.2, label="Peak window")

        # sidebands
        ax.axvspan(left_lo, left_hi, alpha=0.2,
                   label="Sidebands", color="green")
        ax.axvspan(right_lo, right_hi, alpha=0.2, color="green")

        # background level
        ax.hlines(
            bg_per_ch,
            xmin=z_lo,
            xmax=z_hi,
            linestyles="--",
            label="Estimated background",
            color="grey"
        )

        ax.set_xlabel("Channel")
        ax.set_ylabel("Counts")
        ax.set_title(
            f"Line content determination (peak at energy {energy:.1f} keV)")
        ax.legend()
        ax.grid(True, alpha=0.3, color="white")
        ax.set_facecolor('gainsboro')

        fig.savefig(savepath, bbox_inches="tight")

    return N_line, N_peak_raw, bg_per_ch


def peak_widths_with_baseline(x, y, peak_idx, bg_per_ch, levels=(0.5, 0.1)):
    """
    Compute peak widths at given relative levels using a provided local baseline.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    p = int(peak_idx)

    y_peak = float(y[p])
    peak_height = y_peak - bg_per_ch
    if peak_height <= 0:
        raise ValueError("Peak height above baseline is not positive.")

    widths = {}
    for lvl in levels:
        thr = bg_per_ch + lvl * peak_height

        # left crossing
        i = p
        while i > 0 and y[i] > thr:
            i -= 1
        x_left = x[i]

        # right crossing
        j = p
        while j < len(y) - 1 and y[j] > thr:
            j += 1
        x_right = x[j]

        widths[lvl] = (float(x_left), float(x_right), float(x_right - x_left))

    return widths, peak_height


def klein_nishina_dsigma_dE(E, E_gamma, A=1.0, m_ec2=510.99895):
    """
    Klein-Nishina differential cross section written as dσ/dE,
    where E is the kinetic energy transferred to the electron.

    Parameters
    ----------
    E : array_like
        Electron energy transfer (same units as E_gamma), e.g. keV.
    E_gamma : float
        Incident photon energy (same units as E), e.g. keV.
    A : float, optional
        Overall scale factor for fitting (absorbs detector effects, bin width, etc.).
        Default: 1.0
    m_ec2 : float, optional
        Electron rest energy in same units as E (default 510.99895 keV).

    Returns
    -------
    y : ndarray
        Model values proportional to dσ/dE (scaled by A).
        Values outside the physical range 0 < E < E_max are set to 0.

    Notes
    -----
    Physical upper limit (Compton edge for electron):
        E_max = E_gamma * (1 - 1/(1 + 2*E_gamma/m_ec2))
              = E_gamma * (2*E_gamma/m_ec2) / (1 + 2*E_gamma/m_ec2)

    For fitting counts, A typically captures:
      - overall normalization
      - bin width
      - efficiency / acceptance in the chosen region
    """
    E = np.asarray(E, dtype=float)

    # Physical maximum electron energy (Compton edge)
    alpha = E_gamma / m_ec2
    E_max = E_gamma * (2 * alpha) / (1 + 2 * alpha)

    # Initialize output
    y = np.zeros_like(E, dtype=float)

    # Mask physical region and avoid singular endpoints
    eps = 1e-12
    mask = (E > eps) & (E < (E_gamma - eps)) & (E <= (E_max - eps))

    if not np.any(mask):
        return y

    Em = E[mask]
    Eg = float(E_gamma)

    # Core bracket term in your LaTeX expression
    ratio = Em / (Eg - Em)                 # E / (Eγ - E)
    bracket = 2.0 + (ratio**2) * (1.0 + (Eg - Em) / Eg - 2.0 * (Eg - Em) / Em)

    # Constant prefactor (3/8 * σ_Th / (m_ec2)) is absorbed into A for fitting
    # If you want absolute units, multiply A by (3/8)*sigma_th/m_ec2.
    y[mask] = A * (bracket / m_ec2)

    return y
