"""
EECS 351 room acoustics: three additional figures.

Usage:
    python generate_plots.py --input task1/claps.wav --recordings-dir recordings

All input audio must be WAV: task1/claps.wav plus bedroom1.wav–bedroom4.wav
and bathroom1.wav–bathroom4.wav in the recordings directory. Convert any
M4A originals to WAV before running; this script does not decode M4A or use
ffmpeg. The sample rates can differ (the input is resampled when needed).

Plot 3 uses the eight Table 4 measurements from the November 2026 progress
report. Plots 1 and 2 are calculated from the WAV waveforms.
"""

import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt

BEDROOM_COLOR = "#2465aa"
BATHROOM_COLOR = "#d16b35"
FIG_DPI = 220
BAND_CENTERS = (500, 1000, 2000, 4000)
PNR_THRESHOLD_DB = 12.0

# Table 4 values from the progress report.
TABLE4 = [
    ("Bedroom", 1, 0.28, 10.7),
    ("Bedroom", 2, 0.31, 9.3),
    ("Bedroom", 3, 0.33, 10.0),
    ("Bedroom", 4, 0.36, 8.8),
    ("Bathroom", 1, 0.42, 7.1),
    ("Bathroom", 2, 0.38, 7.0),
    ("Bathroom", 3, 0.38, 8.1),
    ("Bathroom", 4, 0.42, 8.3),
]


def save_figure(fig, out_dir, stem):
    filename = os.path.join(out_dir, stem + ".png")
    fig.savefig(filename, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("Created:", filename)


def read_mono(filename):
    """Read WAV audio and convert to floating-point mono."""
    if os.path.splitext(filename)[1].lower() != ".wav":
        raise ValueError("Expected a .wav file: " + filename)

    fs, audio = wavfile.read(filename)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        if info.min == 0:
            midpoint = (info.max + 1) / 2
            audio = (audio.astype(np.float64) - midpoint) / midpoint
        else:
            audio = audio.astype(np.float64) / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float64)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.ndim != 1 or not len(audio) or not np.all(np.isfinite(audio)):
        raise ValueError("Invalid or empty WAV file: " + filename)

    return fs, audio


def build_ir(fs, kind):
    """Build the synthetic impulse responses from Table 2."""
    if kind == "Single echo":
        reflections = [(0, 1.0), (0.25, 0.60)]
        tail = False

    elif kind == "Multiple reflections":
        reflections = [
            (0, 1.0),
            (0.15, 0.60),
            (0.30, 0.35),
            (0.50, 0.20),
        ]
        tail = False

    elif kind == "Decay tail":
        reflections = [
            (0, 1.0),
            (0.007, 0.50),
            (0.012, 0.40),
            (0.018, 0.30),
        ]
        tail = True

    else:
        raise ValueError(kind)

    length_s = max(d for d, _ in reflections)
    if tail:
        length_s = max(length_s, 1.5 * 0.40)

    h = np.zeros(round(length_s * fs) + 1)

    for delay, amplitude in reflections:
        h[round(delay * fs)] += amplitude

    if tail:
        n0 = round(0.020 * fs)
        t = np.arange(len(h) - n0) / fs

        decay = 10 ** (-3 * t / 0.40)
        noise = np.random.default_rng(351).standard_normal(len(t)) * decay
        noise /= np.sqrt(np.sum(noise**2))
        h[n0:] += noise

    return h


def simulate_signals(fs, x):
    """Generate the original and three simulated waveforms."""
    result = {"Original clap": x}

    for name in (
        "Single echo",
        "Multiple reflections",
        "Decay tail",
    ):
        y = fftconvolve(x, build_ir(fs, name), mode="full")

        # Prevent clipping.
        y = y / max(1.0, float(np.max(np.abs(y))))
        result[name] = y

    return result


def energy_envelope(x, fs, window_ms=5.0):
    """Calculate short-time mean-square energy."""
    n = max(1, round(fs * window_ms / 1000))

    energy = uniform_filter1d(x * x, size=n, mode="constant")

    return np.maximum(energy, 1e-18)


def peak_index(x, fs, required_tail_ms=400.0):
    """Locate the loudest clap with sufficient following audio."""
    energy = energy_envelope(x, fs)
    latest = len(x) - round(required_tail_ms * fs / 1000)

    if latest > 0:
        return int(np.argmax(energy[:latest]))

    return int(np.argmax(energy))


def relative_envelope(x, fs, pk, start_ms=-20.0, end_ms=400.0):
    """Get short-time energy relative to a selected clap peak."""
    a = round(pk + start_ms * fs / 1000)
    b = round(pk + end_ms * fs / 1000)

    t_ms = np.arange(a, b) / fs * 1000 - pk / fs * 1000

    seg = np.zeros(b - a, dtype=np.float64)
    src_a = max(a, 0)
    src_b = min(b, len(x))

    if src_b > src_a:
        seg[src_a - a : src_b - a] = x[src_a:src_b]

    energy = energy_envelope(seg, fs)

    # Normalize independently within each recording.
    db = 10 * np.log10(energy / max(energy.max(), 1e-18))

    return t_ms, db


def find_recordings(folder):
    """Locate bedroom1.wav–bedroom4.wav and bathroom1.wav–bathroom4.wav."""
    paths = {}
    missing = []

    for room in ("bedroom", "bathroom"):
        for pos in range(1, 5):
            filename = os.path.join(folder, f"{room}{pos}.wav")

            if os.path.isfile(filename):
                paths[(room, pos)] = filename
            else:
                missing.append(filename)

    return paths, missing


# ------------------------------------------------------------
# Plot 1: Short-time energy heatmap
# ------------------------------------------------------------


def plot1_heatmap(out_dir, fs_x, x, room_audio):
    """Compare all 12 original, synthetic, and measured signals."""
    signals = simulate_signals(fs_x, x)
    x_pk = peak_index(x, fs_x)

    rows = []

    # Original + simulated recordings.
    for name, audio in signals.items():
        t, db = relative_envelope(audio, fs_x, x_pk)
        rows.append((name, t, db))

    # Bedroom and bathroom recordings.
    for room, display in (
        ("bedroom", "Bedroom"),
        ("bathroom", "Bathroom"),
    ):
        for pos in range(1, 5):
            fs, audio = room_audio[(room, pos)]
            pk = peak_index(audio, fs)
            t, db = relative_envelope(audio, fs, pk)
            rows.append((f"{display} {pos}", t, db))

    times = np.linspace(-20, 400, 841)

    data = np.array([np.interp(times, t, db, left=-50, right=-50) for _, t, db in rows])

    fig, ax = plt.subplots(figsize=(10.3, 5.8))

    im = ax.imshow(
        np.clip(data, -50, 0),
        origin="upper",
        aspect="auto",
        extent=(-20, 400, len(rows) - 0.5, -0.5),
        cmap="magma",
        vmin=-50,
        vmax=0,
        interpolation="nearest",
    )

    ax.set_yticks(np.arange(len(rows)), [name for name, _, _ in rows])

    ax.axhline(3.5, color="white", lw=1.7, alpha=0.9)
    ax.axhline(7.5, color="white", lw=1.7, alpha=0.9)
    ax.axvline(0, color="white", ls="--", lw=0.8, alpha=0.7)

    for delay in (150, 250, 300):
        ax.axvline(delay, color="white", ls=":", lw=0.6, alpha=0.30)

    ax.set(
        xlabel="Time relative to selected clap peak (ms)",
        title=("Short-time energy across original, " "simulated, and measured audio"),
    )

    colorbar = fig.colorbar(im, ax=ax, shrink=0.93, pad=0.025)
    colorbar.set_label("5-ms energy relative to each row's peak (dB)")

    ax.text(
        0.005,
        -0.14,
        "Each measured recording is aligned to its own "
        "loudest clap; colors are relative, not absolute "
        "sound levels.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout()
    save_figure(fig, out_dir, "plot1_energy_heatmap")


# ------------------------------------------------------------
# Plot 2: Frequency-dependent clarity
# ------------------------------------------------------------


def resample_if_needed(x, old_fs, new_fs):
    """Resample input audio when WAV sampling rates differ."""
    if old_fs == new_fs:
        return x

    g = math.gcd(int(old_fs), int(new_fs))

    return resample_poly(
        x,
        new_fs // g,
        old_fs // g,
    )


def estimate_ir(x, fs_x, y, fs_y, regularization=0.01):
    """Estimate room impulse response by regularized deconvolution."""
    xs = resample_if_needed(x, fs_x, fs_y)

    n = len(xs) + len(y)
    nfft = 1 << (n - 1).bit_length()

    X = np.fft.rfft(xs, nfft)
    Y = np.fft.rfft(y, nfft)

    reg = regularization * np.max(np.abs(X) ** 2)

    H = Y * np.conj(X) / (np.abs(X) ** 2 + reg)

    h = np.fft.irfft(H, nfft)

    # Put lag zero near the center of the response.
    h = np.roll(h, nfft // 2)

    pk = int(np.argmax(np.abs(h)))

    # Keep 2 ms before and 1 s after the strongest peak.
    a = max(0, pk - round(0.002 * fs_y))
    b = min(len(h), pk + round(1.0 * fs_y))

    cropped = h[a:b]

    if len(cropped) < round(0.7 * fs_y):
        raise ValueError(
            "Estimated IR is truncated; " "inspect playback synchronization."
        )

    cropped = cropped / max(np.max(np.abs(cropped)), 1e-20)

    return cropped, pk - a


def octave_c50(h, fs, n_direct, center_hz, pnr_threshold=PNR_THRESHOLD_DB):
    """Estimate noise-compensated clarity in one octave band."""
    lo = center_hz / np.sqrt(2)
    hi = center_hz * np.sqrt(2)

    if hi >= fs / 2:
        return np.nan, np.nan, np.nan

    sos = butter(
        4,
        (lo, hi),
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    hb = sosfiltfilt(sos, h)
    energy = hb**2

    # Estimate the late noise floor.
    late_a = n_direct + round(0.6 * fs)
    late = energy[late_a:]

    if len(late) < round(0.1 * fs):
        return np.nan, np.nan, np.nan

    noise = float(np.mean(late))

    # Compute a 10-ms smoothed energy envelope.
    w = max(1, round(0.010 * fs))

    smooth = uniform_filter1d(energy, size=w, mode="constant")

    direct_peak = smooth[n_direct : n_direct + round(0.005 * fs)].max()

    pnr = 10 * np.log10(max(direct_peak, 1e-30) / max(noise, 1e-30))

    # Determine where energy approaches the noise floor.
    start = n_direct + round(0.020 * fs)
    below = np.flatnonzero(smooth[start:] < 1.5 * noise)

    cutoff = start + int(below[0]) if len(below) else len(energy)

    after_50 = n_direct + round(0.050 * fs)

    # Reject estimates with insufficient signal quality.
    if pnr < pnr_threshold or cutoff < n_direct + round(0.060 * fs):
        return (
            np.nan,
            pnr,
            (cutoff - n_direct) / fs,
        )

    corrected = np.maximum(energy[:cutoff] - noise, 0)

    early = corrected[n_direct:after_50].sum()

    late_energy = corrected[after_50:].sum()

    if early <= 0 or late_energy <= 0:
        return (
            np.nan,
            pnr,
            (cutoff - n_direct) / fs,
        )

    c50 = 10 * np.log10(early / late_energy)

    return (
        c50,
        pnr,
        (cutoff - n_direct) / fs,
    )


def plot2_band_clarity(out_dir, fs_x, x, room_audio):
    """Calculate and visualize octave-band C50 for both rooms."""
    records = []

    for room in ("bedroom", "bathroom"):
        for pos in range(1, 5):
            fs, y = room_audio[(room, pos)]

            h, n_direct = estimate_ir(x, fs_x, y, fs)

            for center in BAND_CENTERS:
                c50, pnr, cutoff_s = octave_c50(h, fs, n_direct, center)

                records.append(
                    {
                        "room": room,
                        "position": pos,
                        "center_hz": center,
                        "c50_db": c50,
                        "pnr_db": pnr,
                        "cutoff_s": cutoff_s,
                        "reliable": np.isfinite(c50),
                    }
                )

    # Save all numerical measurements.
    csv_path = os.path.join(out_dir, "plot2_band_metrics.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()

        for rec in records:
            writer.writerow(
                {
                    k: (
                        f"{v:.5g}"
                        if isinstance(v, (float, np.floating)) and np.isfinite(v)
                        else ("" if isinstance(v, (float, np.floating)) else v)
                    )
                    for k, v in rec.items()
                }
            )

    print("Created:", csv_path)

    # Plot frequency-dependent clarity.
    fig, ax = plt.subplots(figsize=(8.1, 4.9))
    stats = {}

    for room, label, color in (
        ("bedroom", "Bedroom", BEDROOM_COLOR),
        ("bathroom", "Bathroom", BATHROOM_COLOR),
    ):
        means = []
        lows = []
        highs = []
        ns = []

        for fcenter in BAND_CENTERS:
            vals = np.array(
                [
                    r["c50_db"]
                    for r in records
                    if (
                        r["room"] == room
                        and r["center_hz"] == fcenter
                        and r["reliable"]
                    )
                ],
                dtype=float,
            )

            ns.append(len(vals))

            means.append(float(np.mean(vals)) if len(vals) else np.nan)
            lows.append(float(np.min(vals)) if len(vals) else np.nan)
            highs.append(float(np.max(vals)) if len(vals) else np.nan)

            if len(vals):
                ax.scatter(
                    [fcenter] * len(vals),
                    vals,
                    color=color,
                    s=18,
                    alpha=0.38,
                    zorder=3,
                )

        means = np.asarray(means)
        lows = np.asarray(lows)
        highs = np.asarray(highs)

        good = np.isfinite(means)

        if good.any():
            ax.plot(
                np.array(BAND_CENTERS)[good],
                means[good],
                marker="o",
                lw=2.1,
                ms=6,
                color=color,
                label=label + " mean",
            )

            ax.vlines(
                np.array(BAND_CENTERS)[good],
                lows[good],
                highs[good],
                colors=color,
                alpha=0.45,
                linewidth=3,
            )

        stats[room] = {
            f: {"n": n, "mean": m} for f, n, m in zip(BAND_CENTERS, ns, means)
        }

    ax.set(
        xlabel="Octave-band center frequency (Hz)",
        ylabel="Clarity C50 (dB)",
        title=("Frequency-dependent clarity " "in the two measured rooms"),
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(
        BAND_CENTERS,
        ["500", "1000", "2000", "4000"],
    )
    ax.grid(alpha=0.2)

    if any(np.isfinite(r["c50_db"]) for r in records):
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5,
            0.52,
            "No bands passed the reliability check.\n"
            "Inspect plot2_band_metrics.csv "
            "and the recordings.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.text(
        0.0,
        -0.18,
        "Dots: individual positions; heavy lines: means; "
        "bars: observed range. "
        f"Bands require PNR >= {PNR_THRESHOLD_DB:g} dB "
        "and usable tail.",
        fontsize=8,
        color="#555555",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    save_figure(fig, out_dir, "plot2_frequency_clarity")

    return stats, records


# ------------------------------------------------------------
# Plot 3: Early decay time versus clarity
# ------------------------------------------------------------


def plot3_edt_clarity(out_dir):
    """Plot the eight EDT/C50 measurements from Table 4."""
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    summary = {}

    for name, color in (
        ("Bedroom", BEDROOM_COLOR),
        ("Bathroom", BATHROOM_COLOR),
    ):
        rows = [(p, edt, c50) for room, p, edt, c50 in TABLE4 if room == name]

        xs = np.array([r[1] for r in rows])
        ys = np.array([r[2] for r in rows])

        ax.scatter(
            xs,
            ys,
            color=color,
            s=57,
            zorder=3,
            label=name + " positions",
        )

        for p, edt, c50 in rows:
            ax.annotate(
                str(p),
                (edt, c50),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )

        mx = float(np.mean(xs))
        my = float(np.mean(ys))

        ax.scatter(
            [mx],
            [my],
            marker="D",
            s=125,
            color=color,
            edgecolors="white",
            linewidths=1.3,
            zorder=5,
        )

        summary[name] = (mx, my)

    ax.set(
        xlabel="Early decay time, EDT (s)",
        ylabel="Clarity C50 (dB)",
        title=("Relationship between " "early decay time and clarity"),
    )

    ax.set_xlim(0.255, 0.45)
    ax.set_ylim(6.5, 11.3)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right")

    ax.text(
        0.0,
        -0.17,
        "Numbers identify recorder positions (1-4); "
        "diamonds show room means. "
        "Values reproduced from progress report, Table 4.",
        fontsize=8,
        color="#555555",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    save_figure(fig, out_dir, "plot3_edt_vs_clarity")

    csv_path = os.path.join(out_dir, "plot3_table4_metrics.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["room", "position", "edt_s", "c50_db", "source"])

        for row in TABLE4:
            writer.writerow(list(row) + ["Progress report Table 4"])

    return summary


# ------------------------------------------------------------
# Generate Markdown results summary
# ------------------------------------------------------------


def make_generated_notes(out_dir, has_input, has_room_audio, stats, summary):
    """Write results and indicate which plots were computed."""
    be, bc = summary["Bedroom"]
    re, rc = summary["Bathroom"]

    lines = [
        "# EECS 351 — generated results notes",
        "",
        "This file distinguishes plots computed from provided "
        "waveforms from values transcribed from the existing "
        "progress report. See `EXPLANATIONS.md` for full "
        "methods, context, and limitations.",
        "",
        "## Plot 1 — Short-time energy heatmap",
        "",
    ]

    if has_input and has_room_audio:
        lines.extend(
            [
                "Created `plot1_energy_heatmap.png` from the "
                "original clap, three regenerated synthetic "
                "responses, and all eight room recordings.",
                "Each row was normalized to its own peak and "
                "aligned to a selected loud clap (not necessarily "
                "the same clap event across recordings). "
                "White dotted markers at 150, 250, and 300 ms "
                "mark some programmed synthetic reflection delays. "
                "Interpret differences qualitatively; the "
                "heatmap does not compare absolute sound "
                "pressure levels.",
            ]
        )
    else:
        lines.append(
            "Not computed: `claps.wav` and all eight room " "recordings are required."
        )

    lines.extend(
        [
            "",
            "## Plot 2 — Frequency-dependent clarity",
            "",
        ]
    )

    if stats is None:
        lines.append(
            "Not computed: the input and all eight room " "recordings are required."
        )
    else:
        lines.extend(
            [
                "Created `plot2_frequency_clarity.png` using "
                "regularized deconvolution and band-specific, "
                "noise-compensated C50 estimates.",
                "The following estimates passed the stated "
                "exploratory quality filter:",
                "",
                "| Center (Hz) | Bedroom mean C50 (dB) | "
                "Valid positions | Bathroom mean C50 (dB) | "
                "Valid positions |",
                "|---:|---:|---:|---:|---:|",
            ]
        )

        for fc in BAND_CENTERS:
            a = stats["bedroom"][fc]
            b = stats["bathroom"][fc]

            am = f'{a["mean"]:.2f}' if np.isfinite(a["mean"]) else "n/a"
            bm = f'{b["mean"]:.2f}' if np.isfinite(b["mean"]) else "n/a"

            lines.append(f'| {fc} | {am} | {a["n"]}/4 ' f'| {bm} | {b["n"]}/4 |')

        lines.extend(
            [
                "",
                "The source and receiver hardware and automatic "
                "level control may influence these results. "
                "Bands with insufficient signal-to-noise ratio "
                "are not interpreted. Position means with fewer "
                "than two valid estimates are especially uncertain.",
            ]
        )

    lines.extend(
        [
            "",
            "## Plot 3 — EDT versus clarity",
            "",
            "Created `plot3_edt_vs_clarity.png` from the eight "
            "EDT/C50 pairs in Table 4 of the supplied progress "
            "report (not independently recomputed).",
            (
                f"Bedroom mean: EDT = {be:.2f} s, "
                f"C50 = {bc:.2f} dB. "
                f"Bathroom mean: EDT = {re:.2f} s, "
                f"C50 = {rc:.2f} dB."
            ),
            (
                f"Bathroom minus bedroom: EDT = {re-be:+.2f} s "
                f"and C50 = {rc-bc:+.2f} dB."
            ),
            "All four bathroom points have longer EDT and lower "
            "C50 than all four bedroom points. This is an observed "
            "association in two non-controlled setups, not "
            "evidence that any particular surface material "
            "causes the difference.",
            "",
        ]
    )

    notes_path = os.path.join(out_dir, "plot_explanations_generated.md")

    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Created:", notes_path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        default=os.path.join("task1", "claps.wav"),
        help="Original claps.wav path",
    )

    parser.add_argument(
        "--recordings-dir",
        default="recordings",
        help=(
            "Directory with bedroom1.wav ... bedroom4.wav "
            "and bathroom1.wav ... bathroom4.wav (WAV only)"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Figure/CSV/notes output folder",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot 3 uses the progress report's measurements.
    means = plot3_edt_clarity(args.output_dir)

    has_input = os.path.isfile(args.input)

    paths, missing = find_recordings(args.recordings_dir)
    has_room_audio = len(missing) == 0

    if not has_input:
        print("Missing original clap file:", args.input)

    if missing:
        print("Missing room files:", ", ".join(missing))

    stats = None

    if has_input and has_room_audio:
        fs_x, x = read_mono(args.input)

        room_audio = {}

        for key, filename in paths.items():
            room_audio[key] = read_mono(filename)

        plot1_heatmap(args.output_dir, fs_x, x, room_audio)

        stats, _ = plot2_band_clarity(args.output_dir, fs_x, x, room_audio)

    else:
        print(
            "Plots 1 and 2 skipped until claps.wav "
            "and all eight room WAV files are available. "
            "Plot 3 is complete."
        )

    make_generated_notes(
        args.output_dir,
        has_input,
        has_room_audio,
        stats,
        means,
    )


if __name__ == "__main__":
    main()
