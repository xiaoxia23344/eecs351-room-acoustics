import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve

# read wav
fs, x = wavfile.read("claps.wav")
if np.issubdtype(x.dtype, np.integer):
    x = x.astype(float) / np.iinfo(x.dtype).max
else:
    x = x.astype(float)
if x.ndim == 2:
    x = x.mean(axis=1)  # stereo -> mono


def simulate(reflections, filename):
    """
    reflections: list of (delay_seconds, amplitude) pairs
    """
    h = np.zeros(round(max(delay for delay, _ in reflections) * fs) + 1)
    for delay, amplitude in reflections:
        h[round(delay * fs)] += amplitude

    y = fftconvolve(x, h, mode="full")
    y /= max(1, np.max(np.abs(y)))  # avoid clipping
    # 32767: max int16 value
    wavfile.write(filename, fs, (y * 32767).astype(np.int16))
    return y


# reflections: list of (delay_seconds, amplitude) pairs
single = [(0, 1), (0.25, 0.6)]
multiple = [(0, 1), (0.15, 0.6), (0.30, 0.35), (0.50, 0.2)]

y_single = simulate(single, "single_echo.wav")
y_multiple = simulate(multiple, "multiple_reflections.wav")

# plot impulse responses using their nonzero impulses.
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, sharey=True)

for ax, (reflections, title) in zip(
    axes, [(single, "Single echo"), (multiple, "Multiple reflections")]
):
    delays, amplitudes = zip(*reflections)
    ax.stem(delays, amplitudes)
    ax.set_title(title)
    ax.set_ylabel("Amplitude")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Delay (seconds)")
plt.tight_layout()
plt.savefig("impulse_responses.png")
plt.close()

# plot a short window around the loudest clap.
peak = np.argmax(np.abs(x))
start = max(0, peak - round(0.05 * fs))
end = min(len(x), peak + round(0.65 * fs))
t = np.arange(start, end) / fs

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for ax, signal, title in zip(
    axes, [x, y_single, y_multiple], ["Original", "Single echo", "Multiple reflections"]
):
    ax.plot(t, signal[start:end])
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.savefig("waveforms.png")
plt.close()
