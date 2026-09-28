"""Record 5 seconds from the microphone, plot a spectrogram, and list the notes played.

Only fundamental frequencies are reported. Each frame's pitch is found with a
Harmonic Product Spectrum (HPS): the spectrum is multiplied by downsampled copies
of itself, so the overtones (2f, 3f, 4f, ...) all reinforce the fundamental f
while individual overtones get suppressed.

Run from the repo root:
    .venv\\Scripts\\python "Class 5\\spectrogram_notes.py"

Optional: pass a .wav path to analyze a file instead of recording.
"""

import sys
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from scipy.ndimage import median_filter
from scipy.signal import spectrogram

RATE = 44100
DURATION_S = 5
CHUNK = 1024

FRAME = 4096          # samples per analysis frame (~93 ms)
HOP = 1024            # step between frames (~23 ms)
NFFT = 16384          # zero-padded FFT size for finer frequency bins
HPS_HARMONICS = 5     # how many downsampled spectra to multiply together
F_MIN, F_MAX = 60.0, 2000.0  # range of fundamentals we search
SILENCE_DB = -35.0    # frames quieter than (loudest frame + this) are silent
ONSET_DB = 3.0        # an energy jump this big re-triggers the same note
ONSET_LOOKBACK = 4    # no re-trigger within this many frames of a note's start
MIN_NOTE_S = 0.08     # ignore notes shorter than this
NOISE_DB = -65.0      # spectrogram power below this (dB) is treated as background noise

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
OUT_DIR = Path(__file__).parent


def record(seconds=DURATION_S):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
    print(f"Recording for {seconds} seconds... play now!")
    chunks = [stream.read(CHUNK, exception_on_overflow=False)
              for _ in range(int(RATE / CHUNK * seconds))]
    print("Done recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    raw = b"".join(chunks)

    with wave.open(str(OUT_DIR / "recording.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(raw)

    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0, RATE


def load_wav(path):
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float64) / 32768.0, rate


def freq_to_midi(f):
    return 69 + 12 * np.log2(f / 440.0)


def midi_to_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def fundamental(frame, rate):
    """Return the fundamental frequency of one frame using HPS (0 if none)."""
    window = np.hanning(len(frame))
    spec = np.abs(np.fft.rfft(frame * window, NFFT))
    bin_hz = rate / NFFT

    lo, hi = int(F_MIN / bin_hz), int(F_MAX / bin_hz)

    # Flatten every bin below the noise threshold to a constant floor. Power is
    # scaled the same way as the plotted spectrogram (one-sided PSD), so NOISE_DB
    # means the same thing in both. A floor rather than zero keeps HPS working
    # when one overtone happens to be quiet.
    psd_scale = 2 / (rate * np.sum(window ** 2))
    floor = np.sqrt(10 ** (NOISE_DB / 10) / psd_scale)
    if not (spec[lo:] > floor).any():
        return 0.0
    spec = np.maximum(spec, floor)

    hps = spec.copy()
    for h in range(2, HPS_HARMONICS + 1):
        down = spec[::h]
        hps[:len(down)] *= down
    peak = lo + int(np.argmax(hps[lo:hi]))

    # HPS sometimes lands an octave too low. If there's barely any energy at
    # the chosen bin but a strong peak at double it, the double is the real one.
    double = 2 * peak
    if double < len(spec):
        win = slice(max(double - 2, 0), double + 3)
        if spec[peak] < 0.2 * spec[win].max():
            peak = double - 2 + int(np.argmax(spec[win]))

    # Parabolic interpolation around the peak for sub-bin accuracy.
    if 1 <= peak < len(spec) - 1:
        a, b, c = np.log(spec[peak - 1:peak + 2] + 1e-12)
        denom = a - 2 * b + c
        offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        offset = 0.0
    return (peak + offset) * bin_hz


def detect_notes(signal, rate):
    starts = range(0, len(signal) - FRAME, HOP)
    times = np.array([(s + FRAME / 2) / rate for s in starts])
    rms_db = np.array([20 * np.log10(np.sqrt(np.mean(signal[s:s + FRAME] ** 2)) + 1e-12)
                       for s in starts])
    voiced = rms_db > max(rms_db.max() + SILENCE_DB, -60.0)

    freqs = np.array([fundamental(signal[s:s + FRAME], rate) if v else 0.0
                      for s, v in zip(starts, voiced)])
    midi = np.where(freqs > 0, np.round(freq_to_midi(np.maximum(freqs, 1e-6))), -1).astype(int)
    midi = median_filter(midi, size=5)  # smooth out single-frame glitches

    # Group consecutive frames with the same MIDI note into notes.
    notes = []
    current = None
    for i, m in enumerate(midi):
        # A re-attack of the same note: the note decayed to some low point after
        # its peak, then got louder again by more than ONSET_DB.
        level = rms_db[i]
        attack = (current is not None and i - current["start"] > ONSET_LOOKBACK
                  and level - current["floor"] > ONSET_DB)
        if current and (m != current["midi"] or attack):
            notes.append(current)
            current = None
        if m >= 0 and current is None:
            current = {"midi": m, "start": i, "end": i, "peak": level, "floor": level}
        elif current:
            current["end"] = i
            if level >= current["peak"]:
                current["peak"] = current["floor"] = level
            else:
                current["floor"] = min(current["floor"], level)
    if current:
        notes.append(current)

    frame_s = HOP / rate
    results = []
    for n in notes:
        dur = (n["end"] - n["start"] + 1) * frame_s
        if dur < MIN_NOTE_S:
            continue
        seg = freqs[n["start"]:n["end"] + 1]
        results.append({
            "name": midi_to_name(n["midi"]),
            "freq": float(np.median(seg[seg > 0])),
            "start": times[n["start"]] - FRAME / 2 / rate,
            "duration": dur,
        })
    return results, times, freqs


def plot(signal, rate, notes, times, freqs):
    f, t, sxx = spectrogram(signal, fs=rate, window="hann", nperseg=FRAME, noverlap=FRAME - HOP)
    power_db = 10 * np.log10(sxx + 1e-20)
    power_db[power_db < NOISE_DB] = NOISE_DB  # flatten background noise to the floor
    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(t, f, power_db, shading="gouraud", cmap="magma", vmin=NOISE_DB)
    fig.colorbar(mesh, ax=ax, label="Power (dB)")

    voiced = freqs > 0
    ax.plot(times[voiced], freqs[voiced], ".", color="cyan", ms=3, label="Fundamental")
    for n in notes:
        ax.text(max(n["start"], t[0]), n["freq"] * 1.08, n["name"], color="white",
                fontsize=10, fontweight="bold")

    ax.set_xlim(0, len(signal) / rate)
    ax.set_ylim(0, 4000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram with detected notes")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spectrogram.png", dpi=120)
    plt.show()


def main():
    if len(sys.argv) > 1:
        signal, rate = load_wav(sys.argv[1])
    else:
        signal, rate = record()

    notes, times, freqs = detect_notes(signal, rate)

    if not notes:
        print("\nNo notes detected. Try playing louder or closer to the mic.")
    else:
        print(f"\n{'#':>3}  {'Note':<5} {'Freq (Hz)':>9}  {'Start (s)':>9}  {'Length (s)':>10}")
        for i, n in enumerate(notes, 1):
            print(f"{i:>3}  {n['name']:<5} {n['freq']:>9.1f}  {n['start']:>9.2f}  {n['duration']:>10.2f}")
        print("\nNotes played:", " ".join(n["name"] for n in notes))

    plot(signal, rate, notes, times, freqs)


if __name__ == "__main__":
    main()
