import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fftpack import fft

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.1
        self.error_covariance = 1.1

    def filter(self, measurement):
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.error_covariance = (1 - kalman_gain) * self.error_covariance + self.process_variance
        return self.estimate


def apply_kalman_filter(audio_data, process_variance=1e-5, measurement_variance=1e-2):
    kf = KalmanFilter(process_variance, measurement_variance)
    return np.array([kf.filter(x) for x in audio_data])


def read_audio_file(file_path):
    file_path="/Users/ayush/Documents/python/sample1.wav"
    audio_data, sample_rate = sf.read(file_path)
    if audio_data.ndim > 1:  # If stereo, take one channel
        audio_data = audio_data[:, 0]
    return audio_data, sample_rate


def frequency_analysis(audio_data, sample_rate):
    N = len(audio_data)  # Total number of samples
    fft_data = fft(audio_data)
    magnitude = np.abs(fft_data[:N // 2])  # Use only positive frequencies
    freqs = np.fft.fftfreq(N, d=1 / sample_rate)[:N // 2]
    return freqs, magnitude


def plot_time_domain(raw_audio, filtered_audio, sample_rate):
    time = np.arange(len(raw_audio)) / sample_rate
    plt.figure(figsize=(12, 6))
    plt.plot(time, raw_audio, label="Raw Audio", alpha=0.7)
    plt.plot(time, filtered_audio, label="Filtered Audio", alpha=0.7)
    plt.title("Time Domain Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


def plot_frequency_spectrum(freqs, magnitude, title="Frequency Spectrum"):
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, magnitude)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()


def main():
    file_path = "/Users/ayush/Documents/python/audio_sample.wav"  # Replace with your audio file path
    audio_data, sample_rate = read_audio_file(file_path)
    print(f"Sample Rate: {sample_rate} Hz, Total Samples: {len(audio_data)}")
    filtered_audio = apply_kalman_filter(audio_data)
    raw_freqs, raw_magnitude = frequency_analysis(audio_data, sample_rate)
    filtered_freqs, filtered_magnitude = frequency_analysis(filtered_audio, sample_rate)
    plot_time_domain(audio_data, filtered_audio, sample_rate)
    plot_frequency_spectrum(raw_freqs, raw_magnitude, title="Raw Frequency Spectrum")
    plot_frequency_spectrum(filtered_freqs, filtered_magnitude, title="Filtered Frequency Spectrum")


if __name__ == "__main__":
    main()
