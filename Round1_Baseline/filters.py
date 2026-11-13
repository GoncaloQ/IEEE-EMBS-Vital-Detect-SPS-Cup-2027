import numpy as np
import scipy
from scipy import sparse


def smoothness_priors_detrend(signal, lmbda):
    """
    Separate a slowly varying trend from a 1D signal.

    ``lmbda`` controls the trend smoothness: larger values favor a smoother
    trend. Returns the detrended signal and the estimated trend.
    """
    sample_count = len(signal)
    identity = sparse.eye(sample_count)

    # Each row computes a second difference: x[i] - 2*x[i+1] + x[i+2].
    offsets = [0, 1, 2]
    data = [np.ones(sample_count), -2 * np.ones(sample_count), np.ones(sample_count)]
    second_difference = sparse.dia_matrix((data, offsets), shape=(sample_count - 2, sample_count)).tocsr()

    # Estimate the trend by balancing closeness to the signal with smoothness:
    # (I + lambda^2 * D2.T @ D2) * trend = signal.
    trend = sparse.linalg.spsolve(identity + (lmbda**2) * (second_difference.T @ second_difference),signal)
    # Removing the estimated trend leaves the detrended signal.
    detrended_signal = signal - trend
    return detrended_signal

def exponential_moving_average(signal, alpha):
    """
    Compute the exponential moving average of a 1D signal.

    ``alpha`` controls the smoothing factor: larger values favor a smoother
    signal.

    este filtro é para ser aplicado ao sinal em tempo real
    """
    smoothed_signal = np.zeros_like(signal)
    smoothed_signal[0] = signal[0]
    for i in range(1, len(signal)):
        smoothed_signal[i] = alpha * signal[i] + (1 - alpha) * smoothed_signal[i - 1]
    return smoothed_signal


def band_pass_butter_filtering(signal, fps, fc, N):
    fc_lower, fc_upper = fc
    coefficients = scipy.signal.butter(
        N, [fc_lower, fc_upper], btype="bandpass", fs=fps, output="sos"
    )
    return scipy.signal.sosfiltfilt(coefficients, signal)