from scipy.fft import rfft
from scipy.fftpack import fftfreq
from scipy.signal import welch, find_peaks, find_peaks_cwt
import numpy as np


class METHODS:
    RMS = 0
    FFT = 1
    PSD = 2
    NUM_PSD_PEAKS = 3
    NUM_FFT_PEAKS = 3


def extract_features(sequences, methods=[METHODS.RMS, METHODS.FFT, METHODS.RMS], peak_distance=None):
    inpts = sequences.shape[2]
    N = sequences.shape[1]
    peak_distance = None  # peak_distance or (N/2)/10
    n_features = (((METHODS.NUM_FFT_PEAKS + METHODS.NUM_PSD_PEAKS) * 2) + 1) * inpts
    fft_amps = np.abs(rfft(sequences, axis=1))
    fft_frqs = fftfreq(N)
    psd_frqs, psd_amps = welch(sequences, axis=1, nperseg=N)  # psd_frqs shoud be equal to fft_frqs' first half
    features = np.zeros((sequences.shape[0],
                         n_features))  # 3 peaks psd (x,y) + 3 peaks fft (x,y) + rms = (6 + 6 + 1) * 9 = 117 features
    for i in range(sequences.shape[0]):
        fft_peaks = np.zeros(METHODS.NUM_FFT_PEAKS * 2 * inpts)
        for j in range(fft_amps[i].T.shape[0]):
            col = fft_amps[i].T[j]
            peaks, properties = find_peaks(col, height=0, distance=peak_distance)
            highest = np.argsort(properties['peak_heights'])
            for k in range(METHODS.NUM_FFT_PEAKS):  # it's only a few peaks, so it's okay that I'm double nesting :)
                if k < highest.size:
                    ind = peaks[highest[-(k + 1)]]
                    x = fft_frqs[ind]  # the frequency
                    y = col[ind]  # the amplitude
                else:
                    x = 0.0
                    y = 0.0
                fft_peaks[(j * METHODS.NUM_FFT_PEAKS * 2) + (k * 2)] = x
                fft_peaks[(j * METHODS.NUM_FFT_PEAKS * 2) + (k * 2) + 1] = y

        psd_peaks = np.zeros(METHODS.NUM_PSD_PEAKS * 2 * inpts)
        for j in range(psd_amps[i].T.shape[0]):
            col = psd_amps[i].T[j]
            peaks, properties = find_peaks(col, height=0, distance=peak_distance)
            highest = np.argsort(properties['peak_heights'])
            for k in range(METHODS.NUM_PSD_PEAKS):
                if k < highest.size:
                    ind = peaks[highest[-(k + 1)]]
                    x = psd_frqs[ind]
                    y = col[ind]
                else:
                    x = 0.0
                    y = 0.0
                psd_peaks[(j * METHODS.NUM_PSD_PEAKS * 2) + (k * 2)] = x
                psd_peaks[(j * METHODS.NUM_PSD_PEAKS * 2) + (k * 2) + 1] = y

        rms = np.zeros(inpts)
        for j in range(sequences[i].T.shape[0]):
            rms[j] = np.square(sequences[i].T[j]).mean()

        features[i] = np.concatenate((fft_peaks, psd_peaks, rms))

    return features
