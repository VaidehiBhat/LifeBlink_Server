from flask import Flask, request, jsonify
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

app = Flask(__name__)

# Function to process EOG (eye blink detection)
def process_eog(data):
    values = np.array(data)
    peaks, _ = find_peaks(values, height=0.5, distance=50)  
    blink_count = len(peaks)
    return {"blink_count": blink_count}

# Function to process PPG (with advanced signal processing)
def process_ppg_signal(data, fs=100, low_cutoff=8, high_cutoff=0.4, notch_freq=50, notch_quality=30, wavelet_type='sym8', sub_coeff_of_decomp=2):
    """
    Processes PPG signals using Butterworth filters, notch filter, and wavelet denoising.
    Returns the estimated heartbeat count.
    """
    data = np.array(data, dtype=np.float64)

    # Handle NaN and Inf values
    if not np.isfinite(data).all():
        valid_data = data[np.isfinite(data)]
        replacement_value = np.mean(valid_data) if valid_data.size > 0 else 0
        data[~np.isfinite(data)] = replacement_value

    # Butterworth Low-pass filter (5th order)
    b_low, a_low = butter(N=5, Wn=low_cutoff / (fs / 2), btype='low')
    filtered_signal = filtfilt(b_low, a_low, data)

    # Butterworth High-pass filter (2nd order)
    b_high, a_high = butter(N=2, Wn=high_cutoff / (fs / 2), btype='high')
    filtered_signal = filtfilt(b_high, a_high, filtered_signal)

    # Notch filter for 50 Hz noise removal
    b_notch, a_notch = iirnotch(notch_freq / (fs / 2), notch_quality)
    filtered_signal = filtfilt(b_notch, a_notch, filtered_signal)

    # Wavelet-based Denoising
    def bayes_shrink(coeff):
        """BayesShrink thresholding for wavelet coefficients."""
        var = np.median(np.abs(coeff)) / 0.6745
        return coeff * (var**2 / (var**2 + (coeff**2).mean()))

    wavelet = pywt.Wavelet(wavelet_type)
    maxlev = pywt.dwt_max_level(len(filtered_signal), wavelet.dec_len) - sub_coeff_of_decomp
    coeffs = pywt.wavedec(filtered_signal, wavelet_type, level=maxlev)

    # Apply thresholding
    for i in range(1, len(coeffs)):
        coeffs[i] = bayes_shrink(coeffs[i])

    # Reconstruct the signal
    cleaned_signal = pywt.waverec(coeffs, wavelet_type)[:len(filtered_signal)]

    # Peak detection for heart rate estimation
    peaks, _ = find_peaks(cleaned_signal, height=0.5, distance=50)
    heartbeat = len(peaks) * 3  # Estimated BPM

    return {"heart_rate": heartbeat}

# Route for EOG sensor data
@app.route('/process_eog', methods=['POST'])
def eog_endpoint():
    data = request.json.get("values", [])
    if not data:
        return jsonify({"error": "No data received"}), 400
    result = process_eog(data)
    return jsonify(result)

# Route for PPG sensor data
@app.route('/process_ppg', methods=['POST'])
def ppg_endpoint():
    data = request.json.get("values", [])
    if not data:
        return jsonify({"error": "No data received"}), 400
    result = process_ppg_signal(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Run on Render
