from pathlib import Path
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve, spectrogram

# audio folders
A_DIR = Path("task1")
B_DIR = Path("recordings")

# input x[n]
A_INPUT = A_DIR / "claps.wav"

# B played claps.wav
B_TEST_SIGNAL = A_INPUT

# deconvolution settings
DECONV_EPS = 1e-2
METRIC_BAND = (300.0, 8000.0)

# B recordings
B_RECORDINGS = {
    "room1": {
        "label": "Room One (bedroom)",
        "files": {p: B_DIR / f"bedroom{p}.m4a" for p in (1, 2, 3, 4)},
    },
    "room2": {
        "label": "Room Two (bathroom)",
        "files": {p: B_DIR / f"bathroom{p}.m4a" for p in (1, 2, 3, 4)},
    },
}

# A models: (delay, amplitude)
SINGLE = [(0, 1), (0.25, 0.6)]
MULTIPLE = [(0, 1), (0.15, 0.6), (0.30, 0.35), (0.50, 0.2)]

# C model: decay tail
TAIL_EARLY = [(0, 1), (0.007, 0.5), (0.012, 0.4), (0.018, 0.3)]
TAIL_RT60 = 0.40
TAIL_DRR_DB = 0.0

# room size (m)
ROOMS_GEOM = {
    "Room One (bedroom)": (4.10, 3.10, 2.40),
    "Room Two (bathroom)": (1.50, 2.40, 2.40),
}
C_SOUND = 343.0

# output folders
FIG_DIR = Path("figures")
TAB_DIR = Path("tables")
OUT_AUDIO = Path("audio_out")

# plot style
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED = "#0b0b0b", "#8a8984"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "savefig.bbox": "tight",
    }
)


# read audio
def read_mono(path):
    path = Path(path)
    # m4a -> wav
    if path.suffix.lower() != ".wav":
        import shutil
        import subprocess
        import tempfile

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(f"{path}: ffmpeg is needed to read {path.suffix} "
                               "files (macOS: brew install ffmpeg)")
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "decoded.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                            "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
            fs, x = wavfile.read(wav)
    else:
        fs, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(float) / np.iinfo(x.dtype).max
    else:
        x = x.astype(float)
    # stereo -> mono
    if x.ndim == 2:
        x = x.mean(axis=1)
    return fs, x


# build h[n]
def build_h(reflections, fs, tail=None):
    length = max(d for d, _ in reflections)
    if tail is not None:
        length = max(length, 1.5 * tail["rt60"])
    h = np.zeros(round(length * fs) + 1)
    for d, a in reflections:
        h[round(d * fs)] += a
    if tail is not None:
        rng = np.random.default_rng(351)
        n0 = round(tail["start"] * fs)
        t = np.arange(len(h) - n0) / fs
        env = 10 ** (-3 * t / tail["rt60"])
        noise = env * rng.standard_normal(len(t))
        noise *= np.sqrt(10 ** (-tail["drr"] / 10) / np.sum(noise**2))
        h[n0:] += noise
    return h


def mag_db(h, fs, nfft=None):
    nfft = nfft or max(1 << 16, 1 << int(np.ceil(np.log2(len(h)))))
    H = np.fft.rfft(h, nfft)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    return f, 20 * np.log10(np.abs(H) + 1e-12)


# octave smoothing
def smooth_spectrum(f, p_lin, frac=3, fmin=20, fmax=None, npts=200):
    fmax = fmax or f[-1]
    fc = np.geomspace(fmin, fmax / 2 ** (1 / (2 * frac)), npts)
    out = np.empty_like(fc)
    for i, c in enumerate(fc):
        lo, hi = c / 2 ** (1 / (2 * frac)), c * 2 ** (1 / (2 * frac))
        m = (f >= lo) & (f < hi)
        out[i] = p_lin[m].mean() if m.any() else np.nan
    return fc, 10 * np.log10(out + 1e-20)


def power_spectrum(x, fs):
    nfft = 1 << int(np.ceil(np.log2(len(x))))
    X = np.fft.rfft(x, nfft)
    return np.fft.rfftfreq(nfft, 1 / fs), np.abs(X) ** 2


# estimate h[n]
def deconvolve(y, x, eps=1e-2):
    n = len(x) + len(y)
    nfft = 1 << int(np.ceil(np.log2(n)))
    X = np.fft.rfft(x, nfft)
    Y = np.fft.rfft(y, nfft)
    reg = eps * np.max(np.abs(X) ** 2)
    h = np.fft.irfft(Y * np.conj(X) / (np.abs(X) ** 2 + reg), nfft)
    # center lag 0
    return np.roll(h, nfft // 2)


def bandpass(h, fs, band=None):
    from scipy.signal import butter, sosfiltfilt

    lo, hi = band or METRIC_BAND
    sos = butter(4, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, h)


# cut around peak
def crop_ir(h, fs, pre=0.002, post=1.0):
    pk = int(np.argmax(np.abs(h)))
    a = max(0, pk - round(pre * fs))
    b = min(len(h), pk + round(post * fs))
    seg = h[a:b]
    return seg / np.max(np.abs(seg)), pk - a


# Schroeder EDC
def edc_db(h):
    e = np.cumsum((h**2)[::-1])[::-1]
    return 10 * np.log10(e / e[0] + 1e-20)


# fit -> T60
def rt60_from_edc(edc, fs, hi=-5.0, lo=-25.0):
    idx = np.where((edc <= hi) & (edc >= lo))[0]
    if len(idx) < 10:
        return np.nan
    t = idx / fs
    slope, _ = np.polyfit(t, edc[idx], 1)
    return -60.0 / slope if slope < 0 else np.nan


# EDT, T10, DRR, C50
def ir_metrics(h, fs, n_direct):
    hb = bandpass(h, fs)
    e = hb**2
    # noise floor
    tail = e[n_direct + round(0.6 * fs):]
    N = float(np.mean(tail)) if len(tail) > fs // 10 else 0.0
    w = max(1, round(0.01 * fs))
    es = np.convolve(e, np.ones(w) / w, mode="same")
    pnr = 10 * np.log10(es[n_direct : n_direct + round(0.005 * fs)].max() / max(N, 1e-30))
    start = n_direct + round(0.02 * fs)
    below = np.where(es[start:] < 1.5 * N)[0]
    tc = start + (below[0] if len(below) else len(e) - start)
    ec = np.clip(e[:tc] - N, 0, None)
    # noise-compensated EDC
    E = np.cumsum(ec[n_direct:][::-1])[::-1]
    edc = 10 * np.log10(np.maximum(E, 1e-30) / E[0])
    edt = rt60_from_edc(edc, fs, 0, -10)
    t10 = rt60_from_edc(edc, fs, -5, -15)
    t20 = rt60_from_edc(edc, fs, -5, -25) if pnr >= 35 else np.nan
    d = n_direct + round(0.0025 * fs)
    n50 = n_direct + round(0.050 * fs)
    direct, rev = ec[:d].sum(), ec[d:].sum()
    c50 = 10 * np.log10(ec[:n50].sum() / max(ec[n50:].sum(), 1e-30))
    drr = 10 * np.log10(direct / max(rev, 1e-30))
    return dict(edt=edt, t10=t10, t20=t20, pnr=pnr, drr=drr, c50=c50, edc=edc,
                hb=hb / np.max(np.abs(hb)))


def ir_spectrum(r, dur=0.3):
    seg = r["h"][: r["n0"] + round(dur * r["fs"])]
    f, P = power_spectrum(seg, r["fs"])
    fc, Pd = smooth_spectrum(f, P, frac=3, fmin=50, fmax=r["fs"] / 2)
    lo, hi = METRIC_BAND
    m = (fc >= lo) & (fc <= hi)
    return fc, Pd - np.nanmax(Pd[m])


def shade_band(ax):
    lo, hi = METRIC_BAND
    ax.axvspan(20, lo, color="#e4e3df", alpha=0.6, lw=0)
    ax.axvspan(hi, 30000, color="#e4e3df", alpha=0.6, lw=0)
    ax.set_xlim(50, 24000)


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def write_table(name, text):
    TAB_DIR.mkdir(exist_ok=True)
    (TAB_DIR / f"{name}.tex").write_text(text)
    print(f"  wrote tables/{name}.tex")


def fmt(v, d=2):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


# Part 0: models, geometry
def part0(fs=48000):
    print("Part 0: impulse-response models and room geometry")
    tail = dict(start=0.020, rt60=TAIL_RT60, drr=TAIL_DRR_DB)
    models = [
        ("Single echo (A)", build_h(SINGLE, fs)),
        ("Multiple reflections (A)", build_h(MULTIPLE, fs)),
        ("Early reflections + decay tail (C)", build_h(TAIL_EARLY, fs, tail)),
    ]

    h3 = models[2][1]
    t = np.arange(len(h3)) / fs
    # tail model
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 3.6))
    ax[0].plot(t * 1e3, h3, color=COLORS[0], lw=0.6)
    ax[0].set(xlim=(-5, 1.3 * TAIL_RT60 * 1e3), ylabel="Amplitude",
              title="Supplementary h[n]: direct sound, early reflections, decaying tail")
    edc = edc_db(h3)
    ax[1].plot(t * 1e3, edc, color=COLORS[0])
    ax[1].axhline(-60, color=MUTED, ls="--", lw=0.8)
    ax[1].annotate(f"design RT60 = {TAIL_RT60:.2f} s", (TAIL_RT60 * 1e3 * 0.55, -55),
                   color="#52514e", fontsize=8)
    ax[1].set(xlim=(-5, 1.3 * TAIL_RT60 * 1e3), ylim=(-70, 3),
              xlabel="Time (ms)", ylabel="Energy decay (dB)")
    fig.tight_layout()
    save(fig, "fig0_tail_model")

    # |H(f)| of models
    fig, axes = plt.subplots(3, 2, figsize=(6.6, 5.4),
                             gridspec_kw=dict(width_ratios=[1.6, 1]))
    for i, (name, h) in enumerate(models):
        f, H = mag_db(h, fs, nfft=1 << 20)
        axl, axr = axes[i]
        axl.semilogx(f[1:], H[1:], color=COLORS[i], lw=0.5)
        lim = (-30, 15)
        axl.set(xlim=(20, fs / 2), ylim=lim, ylabel="|H(f)| (dB)")
        axl.set_title(name, loc="left")
        m = (f >= 1000) & (f <= 1020)
        axr.plot(f[m], H[m], color=COLORS[i], lw=1.0)
        axr.set(xlim=(1000, 1020), ylim=lim)
        axr.set_title("zoom 1000-1020 Hz", loc="left", color="#52514e")
    axes[-1, 0].set_xlabel("Frequency (Hz, log)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    save(fig, "fig0_h_frequency")

    a, D = SINGLE[1][1], SINGLE[1][0]
    rows = [
        ("Peak gain $1+a$", f"{20*np.log10(1+a):+.1f} dB"),
        ("Notch gain $1-a$", f"{20*np.log10(1-a):+.1f} dB"),
        ("Peak/notch spacing $1/D$", f"{1/D:.1f} Hz"),
    ]
    body = "\n".join(f"{k} & {v} \\\\" for k, v in rows)
    # comb table
    write_table("tab0_comb",
                "\\begin{tabular}{lr}\n\\toprule\nQuantity (single echo, "
                f"$a={a}$, $D={D}$\\,s) & Value \\\\\n\\midrule\n{body}\n"
                "\\bottomrule\n\\end{tabular}\n")

    # geometry table
    lines = []
    for room, (L, W, Hh) in ROOMS_GEOM.items():
        rt = [2 * d / C_SOUND * 1e3 for d in (L, W, Hh)]
        modes = [C_SOUND / (2 * d) for d in (L, W, Hh)]
        V = L * W * Hh
        lines.append(
            f"{room} & ${L:.2f}\\times{W:.2f}\\times{Hh:.2f}$ & {V:.1f} & "
            + " / ".join(f"{v:.1f}" for v in rt) + " & "
            + " / ".join(f"{v:.0f}" for v in modes) + " \\\\"
        )
    write_table(
        "tab0_geometry",
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Room & $L\\times W\\times H$ (m) & $V$ (m$^3$) & "
        "$2d/c$ (ms) & $c/2d$ (Hz) \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n",
    )


# Part A: simulation
def part_a():
    print("Part A: simulation outputs")
    if not A_INPUT.exists():
        print(f"  [skip] {A_INPUT} not found -- skipping Part A")
        return
    fs, x = read_mono(A_INPUT)
    tail = dict(start=0.020, rt60=TAIL_RT60, drr=TAIL_DRR_DB)
    hs = {
        "Single echo": build_h(SINGLE, fs),
        "Multiple reflections": build_h(MULTIPLE, fs),
        "Decay tail (C)": build_h(TAIL_EARLY, fs, tail),
    }
    ys = {}
    OUT_AUDIO.mkdir(exist_ok=True)
    for k, h in hs.items():
        y = fftconvolve(x, h)
        y /= max(1, np.max(np.abs(y)))
        ys[k] = y
    y = ys["Decay tail (C)"]
    wavfile.write(OUT_AUDIO / "decay_tail_C.wav", fs, (y * 32767).astype(np.int16))
    print("  wrote audio_out/decay_tail_C.wav")

    peak = int(np.argmax(np.abs(x)))
    a = max(0, peak - round(0.05 * fs))
    b = min(len(x), peak + round(0.65 * fs))
    t = np.arange(a, b) / fs - peak / fs

    sigs = [("Original x[n]", x, [])] + [
        (k, ys[k], [d for d, _ in r][1:])
        for k, r in (("Single echo", SINGLE), ("Multiple reflections", MULTIPLE))
    ] + [("Decay tail (C)", ys["Decay tail (C)"], [])]
    # time domain
    fig, axes = plt.subplots(len(sigs), 1, figsize=(6.6, 6.2), sharex=True, sharey=True)
    for i, (ax, (name, s, delays)) in enumerate(zip(axes, sigs)):
        ss = np.zeros(b - a)
        seg = s[a:b]
        ss[: len(seg)] = seg
        ax.plot(t * 1e3, ss, color=COLORS[min(i, 3)], lw=0.5)
        for d in delays:
            ax.axvline(d * 1e3, color=MUTED, ls="--", lw=0.8)
        ax.set_title(name, loc="left")
        ax.set_ylabel("Amplitude")
    axes[-1].set_xlabel("Time relative to loudest clap (ms); dashed = designed reflection delay")
    fig.tight_layout()
    save(fig, "figA_time")

    # energy envelope
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    w = round(0.005 * fs)
    for i, (name, s, _) in enumerate(sigs):
        seg = np.zeros(b - a)
        ss = s[a:b]
        seg[: len(ss)] = ss
        env = 10 * np.log10(np.convolve(seg**2, np.ones(w) / w, "same") + 1e-12)
        ax.plot(t * 1e3, env - env.max(), color=COLORS[min(i, 3)], lw=1.0, label=name)
    ax.set(ylim=(-60, 3), xlabel="Time relative to loudest clap (ms)",
           ylabel="Short-time energy (dB)")
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    save(fig, "figA_envelope")

    b2 = min(len(x), peak + round(1.2 * fs))
    # spectra
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ref = None
    for i, (name, s, _) in enumerate(sigs):
        seg = s[a:b2]
        f, P = power_spectrum(seg, fs)
        fc, Pd = smooth_spectrum(f, P, frac=6, fmin=50, fmax=fs / 2)
        if ref is None:
            ref = Pd
        axes[0].semilogx(fc, Pd - np.nanmax(ref), color=COLORS[min(i, 3)], label=name)
        if i:
            axes[1].semilogx(fc, Pd - ref, color=COLORS[min(i, 3)], label=name)
    for ax in axes:
        ax.set_xlim(50, 16000)
    axes[0].set(xlabel="Frequency (Hz)", ylabel="Level (dB, 1/6 oct.)",
                title="Smoothed spectra")
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Difference to original (dB)",
                title="Change vs. original")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, "figA_spectrum")

    # comb check
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    seg_x = x[a:b2]
    seg_y = ys["Single echo"][a:b2]
    nfft = 1 << 18
    X = np.fft.rfft(seg_x, nfft)
    Y = np.fft.rfft(seg_y, nfft)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    m = (f >= 1000) & (f <= 1020)
    ax.plot(f[m], 20 * np.log10(np.abs(Y[m] / X[m])), color=COLORS[1],
            label="measured $|Y(f)/X(f)|$")
    aa, D = SINGLE[1][1], SINGLE[1][0]
    ax.plot(f[m], 20 * np.log10(np.abs(1 + aa * np.exp(-2j * np.pi * f[m] * D))),
            color=INK, ls="--", lw=0.8, label="theory $|1+a e^{-j2\\pi fD}|$")
    ax.set(xlabel="Frequency (Hz)", ylabel="Gain (dB)", xlim=(1000, 1020), ylim=(-10, 9),
           title="Single echo: empirical transfer function, 1000-1020 Hz")
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    save(fig, "figA_comb_zoom")

    # spectrograms
    fig, axes = plt.subplots(1, len(sigs), figsize=(6.8, 2.6), sharey=True)
    for ax, (name, s, _) in zip(axes, sigs):
        seg = s[a:b2]
        f, tt, S = spectrogram(seg, fs, nperseg=512, noverlap=448)
        Sd = 10 * np.log10(S + 1e-14)
        ax.pcolormesh((tt - 0.05) * 1e3, f / 1e3, Sd, vmin=Sd.max() - 70,
                      vmax=Sd.max(), cmap="Blues", shading="auto", rasterized=True)
        ax.set_title(name, fontsize=8, loc="left")
        ax.set_xlabel("ms")
        ax.grid(False)
    axes[0].set_ylabel("Frequency (kHz)")
    axes[0].set_ylim(0, min(12, fs / 2e3))
    fig.tight_layout()
    save(fig, "figA_spectrogram")


# Part B: real rooms
def part_b():
    print("Part B: room recordings")
    x_test, fs_test = None, None
    if B_TEST_SIGNAL is not None and Path(B_TEST_SIGNAL).exists():
        fs_test, x_test = read_mono(B_TEST_SIGNAL)
        print(f"  test signal: {B_TEST_SIGNAL} ({len(x_test)/fs_test:.2f} s)")
    else:
        print("  no test signal -> each recording is treated as an approximate IR (clap)")

    # estimate each IR
    results = {}
    for key, room in B_RECORDINGS.items():
        for pos, path in room["files"].items():
            if not Path(path).exists():
                print(f"  [skip] {path} not found -- skipped")
                continue
            fs, y = read_mono(path)
            if x_test is not None:
                if fs != fs_test:
                    raise ValueError(f"{path}: fs={fs}, test signal fs={fs_test}")
                h_raw = deconvolve(y, x_test, DECONV_EPS)
            else:
                h_raw = y
            h, n0 = crop_ir(h_raw, fs, post=1.0)
            m = ir_metrics(h, fs, n0)
            results.setdefault(key, {})[pos] = dict(fs=fs, y=y, h=h, n0=n0, **m)

    if not results:
        print("  [skip] no recordings found -- skipping Part B")
        return

    for key, res in results.items():
        label = B_RECORDINGS[key]["label"]
        poss = sorted(res)

        r0 = res[poss[0]]
        fs = r0["fs"]
        # input vs recording
        fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.6), sharex=True)
        if x_test is not None:
            xc = fftconvolve(r0["y"], x_test[::-1])
            lag = int(np.argmax(np.abs(xc))) - (len(x_test) - 1)
            ref = np.zeros_like(r0["y"])
            s0 = max(lag, 0)
            seg = x_test[max(-lag, 0):][: len(ref) - s0]
            ref[s0 : s0 + len(seg)] = seg
            pk = int(np.argmax(np.abs(ref)))
        else:
            ref = None
            pk = int(np.argmax(np.abs(r0["y"])))
        a = max(0, pk - round(0.05 * fs))
        b = min(len(r0["y"]), pk + round(1.45 * fs))
        tt = (np.arange(a, b) - pk) / fs * 1e3
        if ref is not None:
            axes[0].plot(tt, ref[a:b] / np.max(np.abs(ref)), color=MUTED, lw=0.5)
            axes[0].set_title("Known input x[n] = claps.wav (time-aligned)", loc="left")
        else:
            axes[0].set_title("No separate input file: recording used as approx. IR",
                              loc="left")
        yy = r0["y"][a:b]
        axes[1].plot(tt, yy / np.max(np.abs(yy)), color=COLORS[0], lw=0.5)
        axes[1].set_title(f"{label}, position {poss[0]}: recording y[n]", loc="left")
        axes[1].set_xlabel("Time relative to loudest clap (ms)")
        for ax in axes:
            ax.set_ylabel("Norm. amplitude")
        fig.tight_layout()
        save(fig, f"figB_{key}_input_vs_rec")

        # IR and EDC
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
        for i, p in enumerate(poss):
            r = res[p]
            t = (np.arange(len(r["h"])) - r["n0"]) / r["fs"] * 1e3
            m = t <= 100
            axes[0].plot(t[m], r["hb"][m] - 2.2 * i, color=COLORS[i % 4], lw=0.5)
            axes[0].text(101, -2.2 * i, f"pos {p}", va="center", fontsize=7,
                         color="#52514e")
            te = np.arange(len(r["edc"])) / r["fs"] * 1e3
            axes[1].plot(te, r["edc"], color=COLORS[i % 4], label=f"pos {p}")
        lo, hi = METRIC_BAND
        axes[0].set(xlabel="Time after direct sound (ms)", yticks=[], xlim=(-3, 112),
                    title=f"Estimated h[n], {lo:.0f} Hz-{hi/1e3:.0f} kHz, first 100 ms")
        axes[1].axhspan(-15, -5, color="#e4e3df", alpha=0.7, lw=0)
        axes[1].set(xlabel="Time after direct sound (ms)", ylabel="Energy decay (dB)",
                    ylim=(-30, 2), title="Noise-compensated EDC (grey: T10 fit range)")
        fig.tight_layout()
        save(fig, f"figB_{key}_ir_edc")

        # frequency domain
        fig, ax = plt.subplots(figsize=(6.6, 2.8))
        for i, p in enumerate(poss):
            r = res[p]
            fc, Pd = ir_spectrum(r)
            ax.semilogx(fc, Pd, color=COLORS[i % 4], label=f"pos {p}")
        shade_band(ax)
        ax.set(xlabel="Frequency (Hz)", ylabel="Level (dB, 1/3 oct.)", ylim=(-40, 3),
               title=f"{label}: magnitude of estimated response (first 300 ms)")
        ax.legend(ncol=4, loc="lower center", fontsize=7)
        fig.tight_layout()
        save(fig, f"figB_{key}_freq")

    if len(results) >= 2:
        # compare rooms
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
        for i, (key, res) in enumerate(results.items()):
            label = B_RECORDINGS[key]["label"]
            fs = next(iter(res.values()))["fs"]
            for j, r in enumerate(res.values()):
                te = np.arange(len(r["edc"])) / fs * 1e3
                axes[0].plot(te, r["edc"], color=COLORS[i], lw=0.8,
                             label=label if j == 0 else None)
            specs = [ir_spectrum(r)[1] for r in res.values()]
            fc = ir_spectrum(next(iter(res.values())))[0]
            axes[1].semilogx(fc, np.nanmean(specs, axis=0), color=COLORS[i], label=label)
        axes[0].axhspan(-15, -5, color="#e4e3df", alpha=0.7, lw=0)
        axes[0].set(xlabel="Time after direct sound (ms)", ylabel="Energy decay (dB)",
                    ylim=(-30, 2), title="EDC, all positions")
        shade_band(axes[1])
        axes[1].set(xlabel="Frequency (Hz)", ylabel="Level (dB)", ylim=(-40, 3),
                    title="Mean 1/3-oct. response per room")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        save(fig, "figB_compare")

    # metrics table
    cols = ["pnr", "edt", "t10", "t20", "drr", "c50"]
    digs = [0, 2, 2, 2, 1, 1]
    rows = []
    for key, res in results.items():
        label = B_RECORDINGS[key]["label"]
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{label}}}}} \\\\")
        for p in sorted(res):
            r = res[p]
            rows.append(f"& {p} & " + " & ".join(fmt(r[c], d) for c, d in zip(cols, digs))
                        + " \\\\")
        means = [np.nanmean([res[p][c] for p in res]) if any(np.isfinite(res[p][c]) for p in res)
                 else np.nan for c in cols]
        sds = [np.nanstd([res[p][c] for p in res]) if np.isfinite(m) else np.nan
               for c, m in zip(cols, means)]
        cells = [fmt(m, d) if not np.isfinite(sd) or c == "pnr"
                 else f"{fmt(m, d)} $\\pm$ {fmt(sd, d)}"
                 for c, m, sd, d in zip(cols, means, sds, digs)]
        rows.append("& \\textit{mean} & " + " & ".join(cells) + " \\\\ \\midrule")
    rows[-1] = rows[-1].replace(" \\midrule", "")
    write_table(
        "tabB_metrics",
        "\\begin{tabular}{llcccccc}\n\\toprule\n"
        "& Pos. & PNR (dB) & EDT (s) & $T_{10}$ (s) & $T_{20}$ (s) & DRR (dB) & "
        "$C_{50}$ (dB) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n",
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        part0()
        part_a()
        part_b()
    print("Done. Recompile main.tex to update the report.")
