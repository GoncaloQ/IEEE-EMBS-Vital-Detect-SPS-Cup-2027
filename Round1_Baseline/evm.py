import cv2
import numpy as np
from scipy.signal import butter, lfilter


# -------------------------------
# Butterworth Bandpass Filter
# -------------------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data, axis=0)

# -------------------------------
# Gaussian Pyramid Functions
# -------------------------------
def build_gaussian_pyramid(frame, levels):
    pyramid = [frame]
    for _ in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstruct_from_pyramid(pyramid):
    frame = pyramid[-1]
    for level in reversed(pyramid[:-1]):
        frame = cv2.pyrUp(frame)
        frame = cv2.resize(frame, (level.shape[1], level.shape[0]))
    return frame

# -------------------------------
# Eulerian Video Magnification
# -------------------------------
def eulerian_video_magnification(video_path, output_path,
                                 lowcut=0.4, highcut=3.0, amplification=50,
                                 pyramid_levels=3, fps_override=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file.")

    fps = fps_override if fps_override else cap.get(cv2.CAP_PROP_FPS)
    frames = []

    # Read frames and build pyramid
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pyramid = build_gaussian_pyramid(frame.astype(np.float32), pyramid_levels)
        frames.append(pyramid[-1])  # Use top level for temporal filtering

    cap.release()
    frames = np.array(frames)

    # Apply temporal bandpass filter
    filtered = apply_bandpass(frames, lowcut, highcut, fps, order=3)

    # Amplify and reconstruct frames
    output_frames = []
    for i in range(len(frames)):
        amplified = frames[i] + amplification * filtered[i]
        pyramid = build_gaussian_pyramid(amplified, pyramid_levels)
        reconstructed = reconstruct_from_pyramid(pyramid)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        output_frames.append(cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))

    # Save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = output_frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in output_frames:
        out.write(frame)
    out.release()
    print(f"Output saved to {output_path}")

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    eulerian_video_magnification(
        video_path="target_video.mp4",
        output_path="output_evm.mp4",
        lowcut=0.4,       # Lower bound frequency (Hz)
        highcut=2.0,      # Upper bound frequency (Hz)
        amplification=40, # Amplification factor
        pyramid_levels=3  # Gaussian pyramid levels
    )
