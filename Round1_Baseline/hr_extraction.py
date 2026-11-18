import scipy
import numpy as np
from filters import *

from channels_forehead import signal_extraction_main_forehead
from channels_cheeks import signal_extraction_main_cheeks

# Fourier Transform to get the most relevant frequencies of the signal
def ffourier_transform(signal, fps):
  # Applies Fourier Transform to the signal
  fourier = scipy.fft.fft(signal)
  # Gets indexes of the signal and converts them into timestamps for each value of FFT
  freq_array = scipy.fft.fftfreq(len(signal), 1.0/fps)

  return fourier, freq_array

# Returns value of heart rate for a computed Fourier transform
def get_hr_bpm(fourier, freq_array):
    idx = np.abs(fourier).argmax()
    hr = freq_array[idx]*60
    return hr

def extract_hr(raw_signal, fps, fc, N=3):

    """
        KEY INPUTS:
        - Raw signal
        - Pass band filter: cut-off frequency (fc=[fc_lower, fc_upper]), order (N), fps (get from video)

        OUTPUT:
        - Heart Rate signal

        Inputs for testing:
        - Ground Truth signal
    """

    detrended_signal = smoothness_priors_detrend(raw_signal, lmbda=(fps/(2 * np.pi * 0.1))**2)
    filtered = band_pass_butter_filtering(detrended_signal, fps, fc, N)
    fourier, freq_array = ffourier_transform(filtered, fps)
    heart_rate = get_hr_bpm(fourier, freq_array)

    return heart_rate

def sliding_window():
   ...
   return 0

def extract_hr_signal():
   ...
   return 0

# -------------------------------------- TEST --------------------------------------
import matplotlib.pyplot as plt
target_video = 'target_video.mp4'

green_forehead, fps = signal_extraction_main_forehead(target_video, 'g')
green_cheeks, fps = signal_extraction_main_cheeks(target_video, 'g')

time = [i/fps for i in range(len(green_forehead))]

# Cheeks - raw signal
plt.figure(figsize=(15,5))
plt.plot(time, green_cheeks, linestyle='-', marker='o', color="green")
plt.title("Green Cheeks")
plt.xlabel("Time (ms)")
plt.ylabel("Channel intensity")
plt.savefig("Results/cheeks_raw.png")

# Forehead
plt.figure(figsize=(15,5))
plt.plot(time, green_forehead, linestyle='-', marker='o', color="green")
plt.title("Green Forehead")
plt.xlabel("Time (ms)")
plt.ylabel("Channel intensity")
plt.savefig("Results/forehead_raw.png")

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
plt.savefig("Results/forehead_filtered.png")

plt.figure(figsize=(10,5))
plt.plot(filtered_signal_cheeks, linestyle='-', marker='o', color="green")
plt.title("Filtered Green Channel Cheeks")
plt.xlabel("Frame")
plt.ylabel("Mean green intensity")
plt.grid(True)
plt.savefig("Results/cheeks_filtered.png")

# FFT
fourier_forehead, freq_array_forehead = ffourier_transform(filtered_signal_forehead, fps)
fourier_cheeks, freq_array_cheeks = ffourier_transform(filtered_signal_cheeks, fps)

plt.figure(figsize=(10,5))
plt.plot(freq_array_forehead, np.abs(fourier_forehead), linestyle='-', marker='o', color="green")
plt.title("Fourier Transform Green Channel Forehead")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.savefig("Results/fft_forehead.png")

plt.figure(figsize=(10,5))
plt.plot(freq_array_cheeks, np.abs(fourier_cheeks), linestyle='-', marker='o', color="green")
plt.title("Fourier Transform Green Channel Cheeks")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.savefig("Results/fft_cheeks.png")

# heart rate
hr_forehead = extract_hr(green_forehead, fps, fc, N=3)
print(f"Heart Rate (forehead): {hr_forehead:.2f} BPM")
hr_cheeks = extract_hr(green_cheeks, fps, fc, N=3)
print(f"Heart Rate (cheeks): {hr_cheeks:.2f} BPM")