"""Team FFT estimators and single-video plotting experiment."""

import numpy as np
import scipy

if __package__:
    from .filters import band_pass_butter_filtering, smoothness_priors_detrend
else:
    from filters import band_pass_butter_filtering, smoothness_priors_detrend


def ffourier_transform(signal, fps=30):
    fourier = scipy.fft.fft(signal)
    freq_array = scipy.fft.fftfreq(len(signal), 1.0 / fps)
    return fourier, freq_array


def get_hr_bpm(signal, fps=30):
    """Return the team's unrestricted FFT peak in BPM for a sampled signal."""
    fourier, freq_array = ffourier_transform(signal, fps)
    idx = np.abs(fourier).argmax()
    return freq_array[idx] * 60


def extract_hr(raw_signal, fps, fc, N=3):
    """Preserve the raw-signal filtering entry point used by evaluation."""
    detrended_signal = smoothness_priors_detrend(raw_signal, lmbda=(fps / (2 * np.pi * 0.1))**2)
    filtered = band_pass_butter_filtering(detrended_signal, fps, fc, N)
    return get_hr_bpm(filtered, fps)


def sliding_window(signal, window_size_seconds=10, step_size_seconds=1, fps=30):
    """Return complete windows and center times in seconds, including empty results.

    Durations are rounded to the nearest frame for fractional frame rates.
    Windows are read-only views; trailing incomplete windows are omitted.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional")
    if not all(np.isfinite(v) and v > 0 for v in (fps, window_size_seconds, step_size_seconds)):
        raise ValueError("FPS and window/step durations must be finite and positive")
    frame_counts = np.array([window_size_seconds, step_size_seconds]) * fps
    if not np.isfinite(frame_counts).all():
        raise ValueError("Window/step durations exceed the finite sample range")
    window_size_frames, step_size_frames = (int(round(v)) for v in frame_counts)
    if min(window_size_frames, step_size_frames) < 1:
        raise ValueError("Window and step must each span at least one frame")
    if len(signal) < window_size_frames:
        return np.empty((0, window_size_frames), dtype=signal.dtype), np.empty(0)
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size_frames)[::step_size_frames]
    time = (np.arange(len(windows)) * step_size_frames + window_size_frames / 2) / fps
    return windows, time


def extract_hr_signal(signal, fps=30):
    """Estimate the team's HR trace from an already filtered signal."""
    windows, time = sliding_window(signal, fps=fps)
    hr = [get_hr_bpm(window, fps) for window in windows]
    return time, hr


def main():
    from pathlib import Path

    import matplotlib.pyplot as plt

    if __package__:
        from .channels_cheeks import signal_extraction_main_cheeks
        from .channels_forehead import signal_extraction_main_forehead
    else:
        from channels_cheeks import signal_extraction_main_cheeks
        from channels_forehead import signal_extraction_main_forehead

    Path("Results").mkdir(parents=True, exist_ok=True)
    target_video = 'target_video.mp4'

    #---------------------------------
    # 1. Signal extraction from video
    #---------------------------------
    green_forehead, fps = signal_extraction_main_forehead(target_video, 'g')
    green_cheeks, fps = signal_extraction_main_cheeks(target_video, 'g')

    time = [i/fps for i in range(len(green_forehead))]
    print(f"fps: {fps}")

    # Cheeks - raw signal
    plt.figure(figsize=(15,5))
    plt.plot(time, green_cheeks, linestyle='-', marker='o', color="green")
    plt.title("Green Cheeks")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel intensity")
    plt.savefig("Results/raw_cheeks.png")

    # Forehead
    plt.figure(figsize=(15,5))
    plt.plot(time, green_forehead, linestyle='-', marker='o', color="green")
    plt.title("Green Forehead")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel intensity")
    plt.savefig("Results/raw_forehead.png")

    #--------------
    # 2. Filtering
    #--------------
    # Detrending filter - raw signal
    fc = 0.2
    lmbda = (fps/(2 * np.pi * fc))**2
    detrended_signal_forehead = smoothness_priors_detrend(green_forehead, lmbda)
    detrended_signal_cheeks = smoothness_priors_detrend(green_cheeks, lmbda)

    # Band-pass filter definition
    fc = [1, 2.0]
    N = 3
    filtered_signal_forehead = band_pass_butter_filtering(detrended_signal_forehead, fps, fc, N)
    filtered_signal_cheeks = band_pass_butter_filtering(detrended_signal_cheeks, fps, fc, N)

    plt.figure(figsize=(10,5))
    plt.plot(filtered_signal_forehead, linestyle='-', marker='o', color="green")
    plt.title("Filtered Green Channel Forehead")
    plt.xlabel("Frame")
    plt.ylabel("Mean green intensity")
    plt.grid(True)
    plt.savefig("Results/filtered_forehead.png")

    plt.figure(figsize=(10,5))
    plt.plot(filtered_signal_cheeks, linestyle='-', marker='o', color="green")
    plt.title("Filtered Green Channel Cheeks")
    plt.xlabel("Frame")
    plt.ylabel("Mean green intensity")
    plt.grid(True)
    plt.savefig("Results/filtered_cheeks.png")

    #------------------------------
    # 3.1 Dominant HR - Simple FFT
    #------------------------------
    fourier_forehead, freq_array_forehead = ffourier_transform(filtered_signal_forehead, fps)
    fourier_cheeks, freq_array_cheeks = ffourier_transform(filtered_signal_cheeks, fps)

    plt.figure(figsize=(10,5))
    plt.plot(freq_array_forehead, np.abs(fourier_forehead), linestyle='-', marker='o', color="green")
    plt.title("Fourier Transform Green Channel Forehead")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.savefig("Results/forehead_fft.png")

    plt.figure(figsize=(10,5))
    plt.plot(freq_array_cheeks, np.abs(fourier_cheeks), linestyle='-', marker='o', color="green")
    plt.title("Fourier Transform Green Channel Cheeks")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.savefig("Results/cheeks_fft.png")

    # heart rate
    hr_forehead = get_hr_bpm(filtered_signal_forehead, fps=fps)
    print(f"Heart Rate (forehead): {hr_forehead:.2f} BPM")
    hr_cheeks = get_hr_bpm(filtered_signal_cheeks, fps=fps)
    print(f"Heart Rate (cheeks): {hr_cheeks:.2f} BPM")

    #--------------------------------------------------
    # 3.2 HR signal extraction - FFT to sliding window
    #--------------------------------------------------

    time, hr_forehead = extract_hr_signal(filtered_signal_forehead, fps=fps)
    time_c, hr_cheeks = extract_hr_signal(filtered_signal_cheeks, fps=fps)

    plt.figure(figsize=(10,5))
    plt.plot(time, hr_forehead, linestyle='-', marker='o', color="green")
    plt.title("HR signal - Forehead")
    plt.xlabel("Time (s)")
    plt.ylabel("Heart rate (BPM)")
    plt.grid(True)
    plt.savefig("Results/hr_forehead.png")


    plt.figure(figsize=(10,5))
    plt.plot(time_c, hr_cheeks, linestyle='-', marker='o', color="green")
    plt.title("HR signal - Cheeks")
    plt.xlabel("Time (s)")
    plt.ylabel("Heart rate (BPM)")
    plt.grid(True)
    plt.savefig("Results/hr_cheeks.png")

    plt.close("all")
    print("For paired ground-truth metrics, use python -m Round1_Baseline.evaluate --help")


if __name__ == "__main__":
    main()
