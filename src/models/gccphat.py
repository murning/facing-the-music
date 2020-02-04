import numpy as np
import constants


def gcc_phat(signal, reference_signal, fs=8000, max_tau=constants.max_tau, interpolation=16):
    '''
    Returns the offset between two signals using GCC-PHAT
    '''

    fft_size = signal.shape[0] + reference_signal.shape[0]

    # GCC-PHAT
    SIGNAL = np.fft.rfft(signal, n=fft_size)
    REFERENCE_SIGNAL = np.fft.rfft(reference_signal, n=fft_size)
    R = SIGNAL * np.conj(REFERENCE_SIGNAL)
#     cross_correlation = np.fft.irfft(R / np.abs(R), n=(interpolation * fft_size))
    cross_correlation = np.fft.irfft(np.divide(R,np.abs(R), where=np.abs(R)!=0) , n=(interpolation * fft_size))

    # FFT shift
    max_shift = int(interpolation * fs * max_tau)
    cross_correlation = np.concatenate((cross_correlation[-max_shift:], cross_correlation[:max_shift + 1]))

    # Offset Calculation
    shift = np.argmax(np.abs(cross_correlation)) - max_shift
    tau = shift / float(interpolation * fs)

    return tau, cross_correlation


def tdoa(signal, reference_signal, fs):
    '''
    Compute time difference of arrival using GCC-PHAT
    '''
    tau, _ = gcc_phat(signal, reference_signal, fs=fs)
    theta = -np.arcsin(tau / constants.max_tau) * 180 / np.pi + 90
    return theta



