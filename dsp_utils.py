"""
DSP Utilities 
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def design_rrc_filter(
    sample_rate: float,
    symbol_rate: float,
    alpha: float = 0.35,
    filter_span_symbols: int = 10
) -> Tuple[np.ndarray, int]:
    """
    Design high-precision root raised cosine filter

    Args:
        sample_rate: Sample rate in Hz
        symbol_rate: Symbol rate in Hz
        alpha: Roll-off factor (0-1.0)
        filter_span_symbols: Filter length in symbols

    Returns:
        Tuple of (filter coefficients, samples per symbol)
    """
    samples_per_symbol = int(sample_rate / symbol_rate)
    num_taps = filter_span_symbols * samples_per_symbol + 1

    # Time vector normalized to symbol period
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / samples_per_symbol

    # RRC filter impulse response
    h = np.zeros(num_taps)

    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
        elif alpha != 0 and abs(abs(ti) - 1.0 / (4.0 * alpha)) < 1e-10:
            h[i] = (alpha / np.sqrt(2.0)) * (
                ((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))) +
                ((1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha)))
            )
        else:
            numerator = np.sin(np.pi * ti * (1.0 - alpha)) + \
                        4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            denominator = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = numerator / denominator

    # Normalize for unity gain at DC
    h = h / np.sqrt(np.sum(h ** 2))

    return h, samples_per_symbol


def apply_pulse_shaping(
    symbols: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    filter_type: str = 'rrc',
    alpha: float = 0.35
) -> np.ndarray:
    """
    Apply precise pulse shaping to symbols

    Args:
        symbols: Input symbols (complex or real)
        sample_rate: Sample rate in Hz
        symbol_rate: Symbol rate in Hz
        filter_type: 'rrc', 'rc', 'gaussian'
        alpha: Roll-off factor for RRC/RC, BT product for Gaussian

    Returns:
        Pulse-shaped signal
    """
    samples_per_symbol = int(sample_rate / symbol_rate)

    # Zero-stuff (insert zeros between symbols)
    upsampled = np.zeros(len(symbols) * samples_per_symbol, dtype=symbols.dtype)
    upsampled[::samples_per_symbol] = symbols

    if filter_type == 'rrc':
        h, _ = design_rrc_filter(sample_rate, symbol_rate, alpha)
    elif filter_type == 'rc':
        h, _ = design_rc_filter(sample_rate, symbol_rate, alpha)
    elif filter_type == 'gaussian':
        h = design_gaussian_filter(sample_rate, symbol_rate, alpha)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Apply filter
    shaped = signal.lfilter(h, 1.0, upsampled)

    # Compensate for filter gain
    shaped = shaped * samples_per_symbol

    return shaped


def design_rc_filter(
    sample_rate: float,
    symbol_rate: float,
    alpha: float = 0.35,
    filter_span_symbols: int = 10
) -> Tuple[np.ndarray, int]:
    """
    Design raised cosine filter

    Args:
        sample_rate: Sample rate in Hz
        symbol_rate: Symbol rate in Hz
        alpha: Roll-off factor (0-1.0)
        filter_span_symbols: Filter length in symbols

    Returns:
        Tuple of (filter coefficients, samples per symbol)
    """
    samples_per_symbol = int(sample_rate / symbol_rate)
    num_taps = filter_span_symbols * samples_per_symbol + 1

    # Time vector normalized to symbol period
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / samples_per_symbol

    # RC filter impulse response
    h = np.zeros(num_taps)

    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0
        elif alpha != 0 and abs(abs(ti) - 1.0 / (2.0 * alpha)) < 1e-10:
            h[i] = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * alpha))
        else:
            h[i] = np.sinc(ti) * np.cos(np.pi * alpha * ti) / (1.0 - (2.0 * alpha * ti) ** 2)

    # Normalize
    h = h / np.sum(h)

    return h, samples_per_symbol


def design_gaussian_filter(
    sample_rate: float,
    symbol_rate: float,
    bt: float = 0.5,
    filter_span_symbols: int = 4
) -> np.ndarray:
    """
    Design Gaussian filter for GMSK

    Args:
        sample_rate: Sample rate in Hz
        symbol_rate: Symbol rate in Hz
        bt: Bandwidth-time product
        filter_span_symbols: Filter length in symbols

    Returns:
        Filter coefficients
    """
    samples_per_symbol = int(sample_rate / symbol_rate)
    num_taps = filter_span_symbols * samples_per_symbol + 1

    # Time vector normalized to symbol period
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / samples_per_symbol

    # Gaussian filter impulse response
    h = np.exp(-2.0 * np.pi ** 2 * bt ** 2 * t ** 2)

    # Normalize
    h = h / np.sum(h)

    return h


def generate_fm_modulation(
    baseband: np.ndarray,
    sample_rate: float,
    deviation: float,
    pre_emphasis: bool = False
) -> np.ndarray:
    """
    Generate high-fidelity FM modulation

    Args:
        baseband: Baseband modulating signal
        sample_rate: Sample rate in Hz
        deviation: Frequency deviation in Hz
        pre_emphasis: Apply pre-emphasis (for broadcast FM)

    Returns:
        Complex FM signal
    """
    # Apply pre-emphasis if requested (75 μs time constant)
    if pre_emphasis:
        tau = 75e-6  # 75 microseconds
        b, a = signal.bilinear([tau, 0], [tau, 1], sample_rate)
        baseband = signal.lfilter(b, a, baseband)

    # Integrate baseband to get phase
    phase = 2.0 * np.pi * deviation * np.cumsum(baseband) / sample_rate

    # Generate FM signal
    fm_signal = np.exp(1j * phase)

    return fm_signal


def generate_am_modulation(
    baseband: np.ndarray,
    modulation_index: float = 0.8,
    carrier_suppressed: bool = False
) -> np.ndarray:
    """
    Generate high-fidelity AM modulation

    Args:
        baseband: Baseband modulating signal
        modulation_index: Modulation index (0-1.0)
        carrier_suppressed: Suppress carrier (DSB-SC)

    Returns:
        Complex AM signal (baseband representation)
    """
    if carrier_suppressed:
        # DSB-SC: just the baseband
        am_signal = modulation_index * baseband
    else:
        # Standard AM: carrier + modulation
        am_signal = 1.0 + modulation_index * baseband

    # Convert to complex
    return am_signal.astype(complex)


def generate_ssb_modulation(
    baseband: np.ndarray,
    sideband: str = 'usb'
) -> np.ndarray:
    """
    Generate high-fidelity SSB modulation using Hilbert transform

    Args:
        baseband: Baseband modulating signal
        sideband: 'usb' or 'lsb'

    Returns:
        Complex SSB signal
    """
    # Generate analytic signal using Hilbert transform
    analytic = signal.hilbert(baseband)

    if sideband.lower() == 'usb':
        return analytic
    elif sideband.lower() == 'lsb':
        return np.conj(analytic)
    else:
        raise ValueError(f"Unknown sideband: {sideband}")


def design_bandpass_filter(
    sample_rate: float,
    center_freq: float,
    bandwidth: float,
    filter_order: int = 8,
    filter_type: str = 'butter'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design high-quality bandpass filter

    Args:
        sample_rate: Sample rate in Hz
        center_freq: Center frequency in Hz
        bandwidth: Bandwidth in Hz
        filter_order: Filter order
        filter_type: 'butter', 'cheby1', 'cheby2', 'ellip'

    Returns:
        Tuple of (b, a) filter coefficients
    """
    # Calculate cutoff frequencies
    low_freq = center_freq - bandwidth / 2.0
    high_freq = center_freq + bandwidth / 2.0

    # Normalize to Nyquist frequency
    nyquist = sample_rate / 2.0
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist

    # Design filter
    if filter_type == 'butter':
        b, a = signal.butter(filter_order, [low_norm, high_norm], btype='band')
    elif filter_type == 'cheby1':
        b, a = signal.cheby1(filter_order, 0.5, [low_norm, high_norm], btype='band')
    elif filter_type == 'cheby2':
        b, a = signal.cheby2(filter_order, 40, [low_norm, high_norm], btype='band')
    elif filter_type == 'ellip':
        b, a = signal.ellip(filter_order, 0.5, 40, [low_norm, high_norm], btype='band')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return b, a


def upsample_and_filter(
    signal_in: np.ndarray,
    upsample_factor: int,
    filter_cutoff: float = 0.4
) -> np.ndarray:
    """
    Upsample signal with anti-aliasing filter

    Args:
        signal_in: Input signal
        upsample_factor: Upsampling factor
        filter_cutoff: Filter cutoff relative to new Nyquist (0-1.0)

    Returns:
        Upsampled signal
    """
    if upsample_factor == 1:
        return signal_in

    # Zero-stuff
    upsampled = np.zeros(len(signal_in) * upsample_factor, dtype=signal_in.dtype)
    upsampled[::upsample_factor] = signal_in

    # Design anti-aliasing filter
    num_taps = 64 * upsample_factor + 1
    h = signal.firwin(num_taps, filter_cutoff / upsample_factor, window='blackmanharris')

    # Apply filter and compensate for gain
    filtered = signal.lfilter(h, 1.0, upsampled) * upsample_factor

    return filtered


def measure_signal_quality(signal_in: np.ndarray, sample_rate: float) -> dict:
    """
    Measure signal quality metrics

    Args:
        signal_in: Input signal
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary of quality metrics
    """
    # Power measurements
    power_rms = np.sqrt(np.mean(np.abs(signal_in) ** 2))
    power_peak = np.max(np.abs(signal_in))
    papr_db = 20 * np.log10(power_peak / power_rms) if power_rms > 0 else 0

    # Spectral measurements
    freqs, psd = signal.welch(signal_in, sample_rate, nperseg=min(1024, len(signal_in)))
    occupied_bw = estimate_occupied_bandwidth(freqs, psd)

    metrics = {
        'rms_power': power_rms,
        'peak_power': power_peak,
        'papr_db': papr_db,
        'occupied_bandwidth_hz': occupied_bw,
        'length_samples': len(signal_in),
        'duration_ms': len(signal_in) / sample_rate * 1000
    }

    return metrics


def estimate_occupied_bandwidth(freqs: np.ndarray, psd: np.ndarray, threshold_db: float = -20) -> float:
    """
    Estimate occupied bandwidth from PSD

    Args:
        freqs: Frequency array
        psd: Power spectral density
        threshold_db: Threshold below peak in dB

    Returns:
        Occupied bandwidth in Hz
    """
    psd_db = 10 * np.log10(psd + 1e-12)
    peak_db = np.max(psd_db)
    threshold = peak_db + threshold_db

    # Find indices above threshold
    above_threshold = psd_db >= threshold
    if not np.any(above_threshold):
        return 0.0

    indices = np.where(above_threshold)[0]
    bw = freqs[indices[-1]] - freqs[indices[0]]

    return abs(bw)
