import scipy
import numpy as np
from filters import *
import matplotlib.pyplot as plt
import csv
from pathlib import Path

from channels_forehead import signal_extraction_main_forehead
from channels_cheeks import signal_extraction_main_cheeks
from metrics import calculate_metrics, write_to_csv, bland_altman_plot

# Fourier Transform to get the most relevant frequencies of the signal
def ffourier_transform(signal, fps=60):
  # Applies Fourier Transform to the signal
  fourier = scipy.fft.fft(signal)
  # Gets indexes of the signal and converts them into timestamps for each value of FFT
  freq_array = scipy.fft.fftfreq(len(signal), 1.0/fps)

  return fourier, freq_array

def detect_peaks(signal, time_array):
   indexes, _ = scipy.signal.find_peaks(signal)
   timestamp_peaks = [time_array[i] for i in indexes]
   return timestamp_peaks

# Returns value of heart rate for a computed Fourier transform
def get_hr_bpm(signal, fps=60):
    fourier, freq_array = ffourier_transform(signal, fps)
    idx = np.abs(fourier).argmax()
    hr = freq_array[idx]*60
    return hr

# def extract_hr(raw_signal, fc, N=3, fps=30):
#     """
#         KEY INPUTS:
#         - Raw signal
#         - Band pass filter: cut-off frequency (fc=[fc_lower, fc_upper]), order (N), fps (get from video)

#         OUTPUT:
#         - Heart Rate value

#         Inputs for testing:
#         - Ground Truth
#     """

#     detrended_signal = smoothness_priors_detrend(raw_signal, lmbda=(fps/(2 * np.pi * 0.1))**2)
#     filtered = band_pass_butter_filtering(detrended_signal, fps, fc, N)
#     heart_rate = get_hr_bpm(filtered, fps)

#     return heart_rate

def sliding_window(signal, fps=60, window_size_seconds=10, step_size_seconds=1):
   window_size_frames = window_size_seconds*fps
   step_size_frames = step_size_seconds*fps

   samples = len(signal)

   if samples < window_size_frames:
      return []

   n_windows = (samples-window_size_frames)//step_size_frames + 1
   shape = (n_windows, window_size_frames)
   strides = (signal.strides[0] * step_size_frames, signal.strides[0])
    
   windows = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
   time = np.array([(i * step_size_frames + window_size_frames / 2) / fps for i in range(n_windows)])
   return windows, time

def extract_hr_signal(signal, fps=60, mode="fft", window_time_seconds=10):
       
   hr = []

   if mode == "fft":
      windows, time = sliding_window(signal, fps, window_size_seconds=window_time_seconds)
      for window in windows:
         hr.append(get_hr_bpm(window, fps))

   if mode == "peak_detection":
      windows, time = sliding_window(signal, fps, window_size_seconds=window_time_seconds)
      t = []
      for window in windows:
            win_time = np.arange(len(window), dtype=float) / float(fps)
            timestamp_peaks = detect_peaks(window, win_time)
            delta_t = np.diff(timestamp_peaks)
            hr.append(60/np.mean(delta_t))

   return time, hr

# -------------------------------------- TEST --------------------------------------
# Test single video - outputs raw signal, filtered signal, FFT result and estimated HR signal
# Computes evaluation metrics for that video

def main():

    target_video = 'target_video.avi'
    results_dir = Path("Results")
    results_dir.mkdir(parents=True, exist_ok=True)

    #---------------------------------
    # 1. Signal extraction from video
    #---------------------------------
    green_forehead, fps = signal_extraction_main_forehead(target_video, 'g')
    green_cheeks, fps = signal_extraction_main_cheeks(target_video, 'g')

    time_video = [i/fps for i in range(len(green_forehead))]
    print(f"fps: {fps}")

    # Cheeks - raw signal
    plt.figure(figsize=(15,5))
    plt.plot(time_video, green_cheeks, linestyle='-', marker='o', color="green")
    plt.title("Green Cheeks")
    plt.xlabel("Time (ms)")
    plt.ylabel("Channel intensity")
    plt.savefig(results_dir / "raw_cheeks.png")
    plt.close()

    # Forehead
    plt.figure(figsize=(15,5))
    plt.plot(time_video, green_forehead, linestyle='-', marker='o', color="green")
    plt.title("Green Forehead")
    plt.xlabel("Time (ms)")
    plt.ylabel("Channel intensity")
    plt.savefig(results_dir / "raw_forehead.png")
    plt.close()

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
    plt.plot(time_video, filtered_signal_forehead, linestyle='-', color="green")
    plt.title("Filtered Green Channel Forehead")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean green intensity")
    plt.grid(True)
    plt.savefig(results_dir / "filtered_forehead.png")
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(time_video, filtered_signal_cheeks, linestyle='-', color="green")
    plt.title("Filtered Green Channel Cheeks")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean green intensity")
    plt.grid(True)
    plt.savefig(results_dir / "filtered_cheeks.png")
    plt.close()

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
    plt.savefig(results_dir / "forehead_fft.png")
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(freq_array_cheeks, np.abs(fourier_cheeks), linestyle='-', marker='o', color="green")
    plt.title("Fourier Transform Green Channel Cheeks")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.savefig(results_dir / "cheeks_fft.png")
    plt.close()

    # heart rate
    hr_forehead = get_hr_bpm(filtered_signal_forehead, fps=fps)
    print(f"Heart Rate (forehead): {hr_forehead:.2f} BPM")
    hr_cheeks = get_hr_bpm(filtered_signal_cheeks, fps=fps)
    print(f"Heart Rate (cheeks): {hr_cheeks:.2f} BPM")

    #--------------------------------------------------
    # 3.2 HR signal extraction - FFT to sliding window
    #--------------------------------------------------

    time, hr_forehead = extract_hr_signal(filtered_signal_forehead, fps=fps, mode="peak_detection", window_time_seconds=10)
    time_c, hr_cheeks = extract_hr_signal(filtered_signal_cheeks, fps=fps, mode="peak_detection", window_time_seconds=10)

   #  hr_forehead = median_filter(hr_forehead, 7)
   #  hr_cheeks = median_filter(hr_cheeks, 7)

    plt.figure(figsize=(10,5))
    plt.plot(time, hr_forehead, linestyle='-', color="green")
    plt.title("HR signal - Forehead")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (BPM)")
    plt.grid(True)
    plt.savefig(results_dir / "hr_forehead.png")
    plt.close()


    plt.figure(figsize=(10,5))
    plt.plot(time_c, hr_cheeks, linestyle='-', color="green")
    plt.title("HR signal - Cheeks")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (BPM)")
    plt.grid(True)
    plt.savefig(results_dir / "hr_cheeks.png")
    plt.close()

    #---------------
    # 4. Evaluation
    #---------------
    # Import and sample Ground Truth
    with open("gtdump.xmp", "r") as f:
       reader = csv.reader(f)
       time_gt_ms, hr_gt, _, _ = [list(c) for c in zip(*reader)]

    plt.figure()
    plt.plot(time_gt_ms, hr_gt)
    plt.savefig(results_dir / "gt.png")
    plt.close()

    # Interpolate high-res values onto the low-res timestamps
    time = [i * 1000 for i in time]
    time = np.asarray(time, dtype=float)
    time_gt_ms = np.asarray(time_gt_ms, dtype=float)
    hr_gt = np.asarray(hr_gt, dtype=float)
    gt = np.interp(time, time_gt_ms, hr_gt)

    plt.figure()
    plt.plot(time_gt_ms, hr_gt)
    plt.plot(time, gt)
    plt.plot(time, hr_forehead)
    plt.plot(time, hr_cheeks)
    plt.legend(["ground truth", "gt aligned", "forehead", "cheeks"])
    plt.title("GT and Estimation signal comparison")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (BPM)")
    plt.savefig("Results/gt_aligned.png")
    plt.close()

    # Compute evaluation metrics
    eval_report_forehead = calculate_metrics(hr_forehead, gt)
    print("Forehead")
    for k in eval_report_forehead.keys():
       print(f"{k}: {eval_report_forehead[k]}")
    save_path = "Results/eval_report_forehead.csv"
    write_to_csv(save_path, eval_report_forehead)

    eval_report_cheeks = calculate_metrics(hr_cheeks, gt)

    print("\nCheeks")
    for k in eval_report_cheeks.keys():
      print(f"{k}: {eval_report_cheeks[k]}")
    save_path = "Results/eval_report_cheeks.csv"
    write_to_csv(save_path, eval_report_cheeks)

    plt.figure()
    bland_altman_plot(hr_forehead, gt)
    plt.ylabel("Difference")
    plt.xlabel("Average")
    plt.title("Bland Altman Plot - Forehead")
    plt.savefig(results_dir / "bland_altman_forehead")
    plt.close()

    plt.figure()
    bland_altman_plot(hr_cheeks, gt)
    plt.ylabel("Difference")
    plt.xlabel("Average")
    plt.title("Bland Altman Plot - Cheeks")
    plt.savefig(results_dir / "bland_altman_cheeks")
    plt.close()



if __name__ == "__main__":
   main()