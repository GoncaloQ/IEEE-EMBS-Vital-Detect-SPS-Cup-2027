import os
import time as time_module
import urllib.request
import cv2
import scipy
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from filters import *

def download_model_if_missing(model_path="face_landmarker.task"):
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading face_landmarker.task model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete!")

#sample skin colour
def get_calibrated_skin_mask(hsv_frame, sample_point, radius=5):
    x, y = sample_point
    h, w = hsv_frame.shape[:2]
    x1, x2 = max(0, x - radius), min(w, x + radius)
    y1, y2 = max(0, y - radius), min(h, y + radius)

    sample_roi = hsv_frame[y1:y2, x1:x2]
    if sample_roi.size == 0:
        return cv2.inRange(hsv_frame, np.array([0, 20, 70]), np.array([25, 255, 255]))

    mean_hsv = np.mean(sample_roi, axis=(0, 1))
    lower_skin = np.array([max(0, mean_hsv[0] - 15), max(20, mean_hsv[1] - 40), max(40, mean_hsv[2] - 60)], dtype=np.uint8)
    upper_skin = np.array([min(180, mean_hsv[0] + 15), min(255, mean_hsv[1] + 40), min(255, mean_hsv[2] + 60)], dtype=np.uint8)

    return cv2.inRange(hsv_frame, lower_skin, upper_skin)

#find where the forehead ends
def find_hairline_above_mesh(hsv_frame, skin_mask, start_x, start_y, max_search_up):
    h_frame, w_frame = hsv_frame.shape[:2]
    strip_w = 6
    x1 = max(0, int(start_x - strip_w))
    x2 = min(w_frame, int(start_x + strip_w))

    y_start = int(start_y)
    y_end = max(0, int(start_y - max_search_up))

    if y_start <= y_end or x2 <= x1:
        return y_start

    skin_strip = skin_mask[y_end:y_start, x1:x2]
    if skin_strip.size == 0:
        return y_start

    skin_profile = np.mean(skin_strip, axis=1)

    for idx in range(len(skin_profile) - 1, -1, -1):
        if skin_profile[idx] < 100:
            return y_end + idx

    return y_end


def extract_forehead_rgb_signals(video_path: str, model_path: str = "face_landmarker.task"):
    """
    Tracks the forehead mask and calculates average R, G, and B channel values for each frame.
    Returns: r_means, g_means, b_means: 1D numpy arrays containing mean values per frame.
    """
    #run the model on video of 1 face
    download_model_if_missing(model_path)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    #face landmarks
    BROW_LINE = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
    UPPER_MESH_POINTS = [103, 67, 109, 10, 338, 297, 332] #horizontal line in the middle of the forehead

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None

    #get information about the video file
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_timestamp_ms = 0
    frame_duration_ms = int(1000 / fps)
    time = []

    prev_upper_pts = None
    smooth_factor = 0.6

    # Output vectors
    r_means, g_means, b_means = [], [], []


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += frame_duration_ms
        time.append(frame_timestamp_ms)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            # Skin color calibration
            #sample_x and sample_y correspnd to the middle of the frehead
            sample_x = int(landmarks[151].x * width)
            sample_y = int(landmarks[151].y * height)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            skin_mask = get_calibrated_skin_mask(hsv_frame, (sample_x, sample_y))

            #if a person is bald we need to establish when to stop searching for a hairline
            #by adjusting the coefficient of max_search_up we can adjust that
            brow_center_y = landmarks[9].y * height
            nose_y = landmarks[1].y * height
            face_height_scale = abs(nose_y - brow_center_y)
            max_search_up = face_height_scale * 1.0

            #determine eyebrow boundary
            poly_points = []
            for idx in BROW_LINE:
                lm = landmarks[idx]
                poly_points.append([int(lm.x * width), int(lm.y * height)])

            #determine hairline boundary
            raw_upper_pts = []
            for idx in UPPER_MESH_POINTS:
                lm = landmarks[idx]
                lm_x = int(lm.x * width)
                lm_y = int(lm.y * height)

                auto_y = find_hairline_above_mesh(hsv_frame, skin_mask, lm_x, lm_y, max_search_up)
                raw_upper_pts.append([lm_x, auto_y])

            # Temporal smoothing (to prevent glitches when the face moves)
            raw_upper_pts = np.array(raw_upper_pts, dtype=np.float32)
            if prev_upper_pts is None:
                smooth_upper_pts = raw_upper_pts
            else:
                #exponntial moving average: position of the point is based on the position of the current frame and the previous frame
                #smooth factor can be adjusted: currently it's 0.6
                smooth_upper_pts = (smooth_factor * raw_upper_pts) + ((1 - smooth_factor) * prev_upper_pts)

            prev_upper_pts = smooth_upper_pts
            upper_pts_int = smooth_upper_pts.astype(np.int32).tolist()

            #create a polygon of the forehead
            poly_points.extend(reversed(upper_pts_int))
            pts = np.array([poly_points], dtype=np.int32)

            # Create binary mask (255 inside forehead polygon, 0 outside)
            forehead_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(forehead_mask, pts, 255)

            # Calculate mean RGB values strictly within the masked region
            # OpenCV frames are BGR by default, we need to convert to RGB
            b_mean, g_mean, r_mean, _ = cv2.mean(frame, mask=forehead_mask)

            r_means.append(r_mean)
            g_means.append(g_mean)
            b_means.append(b_mean)
        else:
            # Append 0 if no face is detected in the frame
            r_means.append(0)
            g_means.append(0)
            b_means.append(0)

    cap.release()
    detector.close()

    print(f"Extraction complete! Extracted {len(r_means)} frame measurements.")
    return np.array(r_means), np.array(g_means), np.array(b_means), fps

def signal_extraction_main_forehead(input_video, input_channel):
  '''
  input_video is a video file .mp4
  input_chanel is either:
  'r' for red channel,'g' for green channel or 'b' for blue channel
  '''

  # Run signal extraction
  r_channel, g_channel, b_channel, fps = extract_forehead_rgb_signals(input_video)
  if input_channel == 'r':
      return r_channel, fps
  elif input_channel == 'g':
      return g_channel, fps
  elif input_channel == 'b':
      return b_channel, fps
  else:
      print('invalid input channel')

def extract_upper_cheeks_rgb(video_path: str, model_path: str = "face_landmarker.task"):
    """
    Tracks upper cheek masks (left + right) and calculates average R, G, and B channel
    values for each frame across the combined cheek regions.

    Returns: r_means, g_means, b_means: 1D numpy arrays containing mean values per frame.
    """

    #run model on video for 1 face
    download_model_if_missing(model_path)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,running_mode=vision.RunningMode.VIDEO, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    #upper cheek landmarks
    LEFT_UPPER_CHEEK = [117, 118, 101, 36, 205, 207, 187, 147, 123]
    RIGHT_UPPER_CHEEK = [346, 347, 330, 266, 425, 427, 411, 376, 352]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None

    #get information about the video file
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_timestamp_ms = 0
    frame_duration_ms = int(1000 / fps)

    # Buffers for temporal smoothing
    prev_left_pts = None
    prev_right_pts = None
    smooth_factor = 0.6

    # Output vectors
    r_means, g_means, b_means = [], [], []


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += frame_duration_ms

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            def get_raw_pts(landmark_indices):
                pts = []
                for idx in landmark_indices:
                    lm = landmarks[idx]
                    pts.append([int(lm.x * width), int(lm.y * height)])
                return np.array(pts, dtype=np.float32)

            # Extract raw points of the cheeks
            raw_left_pts = get_raw_pts(LEFT_UPPER_CHEEK)
            raw_right_pts = get_raw_pts(RIGHT_UPPER_CHEEK)

            # Temporal Smoothing (to prevent glitches when the face moves)
            if prev_left_pts is None:
                smooth_left_pts = raw_left_pts
            else:
                #exponntial moving average: position of the point is based on the position of the current frame and the previous frame
                #smooth factor can be adjusted: currently it's 0.6
                smooth_left_pts = (smooth_factor * raw_left_pts) + ((1 - smooth_factor) * prev_left_pts)
            prev_left_pts = smooth_left_pts

            if prev_right_pts is None:
                smooth_right_pts = raw_right_pts
            else:
                smooth_right_pts = (smooth_factor * raw_right_pts) + ((1 - smooth_factor) * prev_right_pts)
            prev_right_pts = smooth_right_pts

            #convert to integer pixel coordinates
            left_int = smooth_left_pts.astype(np.int32)
            right_int = smooth_right_pts.astype(np.int32)

            #convex hulls to guarantee non-intersecting shapes
            left_hull = cv2.convexHull(left_int)
            right_hull = cv2.convexHull(right_int)

            # Create binary mask for both cheeks (255 inside cheeks, 0 outside)
            cheeks_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(cheeks_mask, [left_hull], 255)
            cv2.fillPoly(cheeks_mask, [right_hull], 255)

            # Calculate mean BGR values strictly within the cheek regions
            # OpenCV frames are BGR by default, we need to convert to RGB
            b_mean, g_mean, r_mean, _ = cv2.mean(frame, mask=cheeks_mask)

            r_means.append(r_mean)
            g_means.append(g_mean)
            b_means.append(b_mean)
        else:
            # If no face is detected, append 0
            r_means.append(0)
            g_means.append(0)
            b_means.append(0)

    cap.release()
    detector.close()

    print(f"Extraction complete! Extracted {len(r_means)} frame measurements.")
    return np.array(r_means), np.array(g_means), np.array(b_means), fps


def signal_extraction_main_cheeks(input_video, input_channel):
  '''
  input_video is a video file .mp4
  input_chanel is either:
  'r' for red channel,'g' for green channel or 'b' for blue channel
  '''

  # Run signal extraction
  r_vector, g_vector, b_vector, fps = extract_upper_cheeks_rgb(input_video)
  if input_channel == 'r':
      return r_vector, fps
  elif input_channel == 'g':
      return g_vector, fps
  elif input_channel == 'b':
      return b_vector, fps
  else:
      print('invalid input channel')

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

"""
KEY INPUTS:
   - Raw signal
   - The filter: cut-off frequency (fc=[fc_lower, fc_upper]), order (N), fps (get from video)

Inputs for testing:
   - Ground Truth

OUTPUT:
   - Heart Rate
"""

def extract_hr(raw_signal, fps, fc, N=3):
  detrended_signal = smoothness_priors_detrend(raw_signal, lmbda=(fps/(2 * np.pi * 0.1))**2)
  filtered = band_pass_butter_filtering(detrended_signal, fps, fc, N)
  fourier, freq_array = ffourier_transform(filtered, fps)
  heart_rate = get_hr_bpm(fourier, freq_array)

  return heart_rate

# -------------------------------------- TEST --------------------------------------
import matplotlib.pyplot as plt
green_forehead, fps = signal_extraction_main_forehead('input_person.mp4', 'g')
green_cheeks, fps = signal_extraction_main_cheeks('input_person.mp4', 'g')

time = [i/fps for i in range(len(green_forehead))]

# Cheeks
plt.figure(figsize=(20,5))
plt.plot(time, green_cheeks, linestyle='-', marker='o', color="green")
plt.title("Green Cheeks")
plt.xlabel("Time (ms)")
plt.ylabel("Channel intensity")
plt.show()

# Forehead
plt.figure(figsize=(20,5))
plt.plot(time, green_forehead, linestyle='-', marker='o', color="green")
plt.title("Green Forehead")
plt.xlabel("Time (ms)")
plt.ylabel("Channel intensity")
plt.show()

# Detrending filter
fc = 0.2
lmbda = (fps/(2 * np.pi * fc))**2
detrended_signal_forehead = smoothness_priors_detrend(green_forehead, lmbda)
detrended_signal_cheeks = smoothness_priors_detrend(green_cheeks, lmbda)

# band-pass filter
fc = [1, 2.0]
N = 3
filtered_signal_forehead = band_pass_butter_filtering(detrended_signal_forehead, fps, fc, N)
filtered_signal_cheeks = band_pass_butter_filtering(detrended_signal_cheeks, fps, fc, N)

plt.plot(filtered_signal_forehead, linestyle='-', marker='o', color="green")
plt.title("Filtered Green Channel Forehead")
plt.xlabel("Frame")
plt.ylabel("Mean green intensity")
plt.grid(True)
plt.show()

plt.plot(filtered_signal_cheeks, linestyle='-', marker='o', color="green")
plt.title("Filtered Green Channel Cheeks")
plt.xlabel("Frame")
plt.ylabel("Mean green intensity")
plt.grid(True)
plt.show()

# FFT
fourier_forehead, freq_array_forehead = ffourier_transform(filtered_signal_forehead, fps)
fourier_cheeks, freq_array_cheeks = ffourier_transform(filtered_signal_cheeks, fps)

plt.plot(freq_array_forehead, np.abs(fourier_forehead), linestyle='-', marker='o', color="green")
plt.title("Fourier Transform Green Channel Forehead")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

plt.plot(freq_array_cheeks, np.abs(fourier_cheeks), linestyle='-', marker='o', color="green")
plt.title("Fourier Transform Green Channel Cheeks")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

# heart rate
hr_forehead = extract_hr(green_forehead, fps, fc, N=3)
print(f"Heart Rate (forehead): {hr_forehead:.2f} BPM")
hr_cheeks = extract_hr(green_cheeks, fps, fc, N=3)
print(f"Heart Rate (cheeks): {hr_cheeks:.2f} BPM")