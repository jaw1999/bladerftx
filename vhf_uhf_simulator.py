"""
VHF/UHF Communications Simulator
Generates various VHF/UHF modulated signals for educational purposes
Supports AM, FM, SSB, and digital modes
"""

import numpy as np
from typing import Optional
import logging
from bladerf_base import BladeRFTransmitter, normalize_signal

logger = logging.getLogger(__name__)

# Common VHF/UHF frequency bands
VHF_LOW = (30e6, 88e6)          # 30-88 MHz
VHF_HIGH = (108e6, 174e6)       # 108-174 MHz (includes FM broadcast, air band)
UHF_LOW = (300e6, 512e6)        # 300-512 MHz
UHF_HIGH = (512e6, 806e6)       # 512-806 MHz

# Aviation band
AIR_BAND = (108e6, 137e6)       # 108-137 MHz

# Common modulation parameters (accurate specs)
AM_MODULATION_INDEX = 0.8       # 80% modulation
FM_DEVIATION_NARROW = 5e3       # ±5 kHz (narrowband FM - amateur/LMR)
FM_DEVIATION_NARROW_2_5 = 2.5e3 # ±2.5 kHz (narrow - 902 MHz)
FM_DEVIATION_WIDE = 75e3        # ±75 kHz (wideband FM broadcast)


class VHFUHFSimulator:
    """VHF/UHF Signal Simulator"""

    def __init__(self, sample_rate: float = 2e6):
        """
        Initialize VHF/UHF simulator

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        logger.info(f"VHF/UHF Simulator initialized with sample rate: {sample_rate/1e6:.3f} MHz")

    def generate_audio_tone(
        self,
        frequency: float,
        duration_ms: float,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate audio tone

        Args:
            frequency: Audio frequency in Hz
            duration_ms: Duration in milliseconds
            amplitude: Amplitude (0-1.0)

        Returns:
            Audio samples
        """
        num_samples = int(self.sample_rate * duration_ms / 1000)
        t = np.arange(num_samples) / self.sample_rate
        audio = amplitude * np.sin(2 * np.pi * frequency * t)
        return audio

    def generate_am_signal(
        self,
        audio_freq: float,
        duration_ms: float,
        modulation_index: float = AM_MODULATION_INDEX
    ) -> np.ndarray:
        """
        Generate AM (Amplitude Modulation) signal

        Args:
            audio_freq: Audio frequency in Hz
            duration_ms: Duration in milliseconds
            modulation_index: Modulation index (0-1.0)

        Returns:
            Complex baseband AM signal
        """
        logger.info(f"Generating AM signal with {audio_freq} Hz audio tone")

        # Generate audio signal
        audio = self.generate_audio_tone(audio_freq, duration_ms)

        # AM modulation: s(t) = (1 + m*audio(t)) * carrier(t)
        # In baseband (carrier removed), just the modulation envelope
        am_signal = 1.0 + modulation_index * audio

        # Convert to complex (AM is real, but represented as complex for consistency)
        signal = am_signal.astype(complex)

        return normalize_signal(signal, target_power=0.7)

    def generate_fm_signal(
        self,
        audio_freq: float,
        duration_ms: float,
        deviation: float = FM_DEVIATION_NARROW
    ) -> np.ndarray:
        """
        Generate FM (Frequency Modulation) signal

        Args:
            audio_freq: Audio frequency in Hz
            duration_ms: Duration in milliseconds
            deviation: Frequency deviation in Hz

        Returns:
            Complex baseband FM signal
        """
        logger.info(f"Generating FM signal with {audio_freq} Hz audio, deviation {deviation/1e3:.1f} kHz")

        # Generate audio signal
        audio = self.generate_audio_tone(audio_freq, duration_ms)

        # FM modulation: phase(t) = integral of (2*pi*deviation*audio(t))
        num_samples = len(audio)
        t = np.arange(num_samples) / self.sample_rate

        # Integrate audio to get phase
        phase = 2 * np.pi * deviation * np.cumsum(audio) / self.sample_rate

        # Generate FM signal
        fm_signal = np.exp(1j * phase)

        return normalize_signal(fm_signal, target_power=0.7)

    def generate_ssb_signal(
        self,
        audio_freq: float,
        duration_ms: float,
        sideband: str = 'usb'
    ) -> np.ndarray:
        """
        Generate SSB (Single Sideband) signal

        Args:
            audio_freq: Audio frequency in Hz
            duration_ms: Duration in milliseconds
            sideband: 'usb' for upper sideband, 'lsb' for lower sideband

        Returns:
            Complex baseband SSB signal
        """
        logger.info(f"Generating {sideband.upper()} signal with {audio_freq} Hz audio")

        # Generate audio signal
        audio = self.generate_audio_tone(audio_freq, duration_ms)

        # Hilbert transform for SSB generation
        # SSB = audio(t) +/- j*hilbert(audio(t))
        from scipy import signal as sp_signal
        analytic_signal = sp_signal.hilbert(audio)

        if sideband.lower() == 'usb':
            ssb_signal = analytic_signal
        elif sideband.lower() == 'lsb':
            ssb_signal = np.conj(analytic_signal)
        else:
            raise ValueError(f"Unknown sideband: {sideband}")

        return normalize_signal(ssb_signal, target_power=0.7)

    def generate_nbfm_voice(
        self,
        duration_ms: float = 1000,
        speech_simulation: bool = True
    ) -> np.ndarray:
        """
        Generate Narrowband FM voice signal

        Args:
            duration_ms: Duration in milliseconds
            speech_simulation: Simulate speech-like characteristics

        Returns:
            Complex NBFM signal
        """
        logger.info("Generating Narrowband FM voice signal")

        num_samples = int(self.sample_rate * duration_ms / 1000)
        t = np.arange(num_samples) / self.sample_rate

        if speech_simulation:
            # Simulate speech with multiple formants
            audio = (
                0.5 * np.sin(2 * np.pi * 500 * t) +
                0.3 * np.sin(2 * np.pi * 1500 * t) +
                0.2 * np.sin(2 * np.pi * 2500 * t)
            )
            # Add some amplitude modulation to simulate speech envelope
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
            audio = audio * envelope
        else:
            # Simple tone
            audio = np.sin(2 * np.pi * 1000 * t)

        # Apply FM modulation
        phase = 2 * np.pi * FM_DEVIATION_NARROW * np.cumsum(audio) / self.sample_rate
        fm_signal = np.exp(1j * phase)

        return normalize_signal(fm_signal, target_power=0.7)

    def generate_ctcss_tone(
        self,
        ctcss_freq: float,
        duration_ms: float,
        carrier_audio_freq: float = 1000
    ) -> np.ndarray:
        """
        Generate FM signal with CTCSS (Continuous Tone-Coded Squelch System)

        Args:
            ctcss_freq: CTCSS tone frequency (67-254.1 Hz)
            duration_ms: Duration in milliseconds
            carrier_audio_freq: Main audio frequency in Hz

        Returns:
            Complex FM signal with CTCSS
        """
        logger.info(f"Generating FM signal with CTCSS {ctcss_freq} Hz")

        num_samples = int(self.sample_rate * duration_ms / 1000)
        t = np.arange(num_samples) / self.sample_rate

        # Generate main audio
        audio = np.sin(2 * np.pi * carrier_audio_freq * t)

        # Add CTCSS tone (typically 10-15% of main audio level)
        ctcss_tone = 0.15 * np.sin(2 * np.pi * ctcss_freq * t)
        audio_with_ctcss = audio + ctcss_tone

        # Apply FM modulation
        phase = 2 * np.pi * FM_DEVIATION_NARROW * np.cumsum(audio_with_ctcss) / self.sample_rate
        fm_signal = np.exp(1j * phase)

        return normalize_signal(fm_signal, target_power=0.7)

    def generate_p25_signal(self, duration_ms: float = 1000) -> np.ndarray:
        """
        Generate P25 Phase 1 (APCO Project 25) C4FM signal

        Accurate specs:
        - 4800 baud symbol rate
        - C4FM: 4-level FSK at ±1800, ±600 Hz
        - 9600 bps (2 bits/symbol)
        - 12.5 kHz channel spacing

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            Complex P25 C4FM signal
        """
        logger.info("Generating P25 Phase 1 C4FM signal (4800 baud, ±1800/±600 Hz)")

        # P25 Phase 1 specs
        symbol_rate = 4800  # symbols/sec (4800 baud)
        num_symbols = int(symbol_rate * duration_ms / 1000)

        # Generate random 2-bit symbols (0, 1, 2, 3)
        symbols_2bit = np.random.randint(0, 4, num_symbols)

        # Map to C4FM deviation levels
        # Symbol 00 -> -1800 Hz, 01 -> -600 Hz, 10 -> +600 Hz, 11 -> +1800 Hz
        deviation_map = {0: -1800, 1: -600, 2: 600, 3: 1800}
        deviations = np.array([deviation_map[s] for s in symbols_2bit])

        # Upsample to sample rate with smooth transitions
        samples_per_symbol = int(self.sample_rate / symbol_rate)

        # Create smooth deviation profile (not rectangular)
        from scipy import signal as sp_signal
        upsampled = np.repeat(deviations, samples_per_symbol)

        # Apply Gaussian smoothing for spectral shaping
        gaussian_window = sp_signal.windows.gaussian(11, std=2)
        gaussian_window /= gaussian_window.sum()
        smoothed = sp_signal.convolve(upsampled, gaussian_window, mode='same')

        # Generate C4FM signal (frequency modulation)
        phase = 2 * np.pi * np.cumsum(smoothed) / self.sample_rate
        p25_signal = np.exp(1j * phase)

        # Truncate to exact duration
        target_samples = int(self.sample_rate * duration_ms / 1000)
        if len(p25_signal) > target_samples:
            p25_signal = p25_signal[:target_samples]

        return normalize_signal(p25_signal, target_power=0.7)

    def generate_dstar_signal(self, duration_ms: float = 1000) -> np.ndarray:
        """
        Generate simplified D-STAR digital signal

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            Complex D-STAR signal (GMSK modulation)
        """
        logger.info("Generating D-STAR digital signal")

        # D-STAR uses GMSK at 4800 bps
        bit_rate = 4800
        num_bits = int(bit_rate * duration_ms / 1000)

        # Generate random data bits
        data_bits = np.random.randint(0, 2, num_bits)

        # Convert to NRZ (+1/-1)
        nrz = 2 * data_bits - 1

        # Apply Gaussian filter for GMSK
        from scipy import signal as sp_signal

        # Gaussian filter parameters
        bt = 0.5  # Bandwidth-time product
        samples_per_bit = int(self.sample_rate / bit_rate)

        # Create Gaussian filter
        t = np.arange(-4, 4, 1.0/samples_per_bit)
        h = np.exp(-2 * np.pi**2 * bt**2 * t**2)
        h = h / np.sum(h)

        # Upsample and filter
        upsampled = np.repeat(nrz, samples_per_bit)
        filtered = sp_signal.convolve(upsampled, h, mode='same')

        # Integrate to get phase
        phase = 2 * np.pi * FM_DEVIATION_NARROW * np.cumsum(filtered) / self.sample_rate

        # Generate GMSK signal
        gmsk_signal = np.exp(1j * phase)

        return normalize_signal(gmsk_signal, target_power=0.7)

    def generate_signal(
        self,
        modulation: str = 'fm',
        duration_ms: float = 1000,
        **kwargs
    ) -> np.ndarray:
        """
        Generate VHF/UHF signal

        Args:
            modulation: Modulation type ('am', 'fm', 'ssb', 'nbfm', 'p25', 'dstar')
            duration_ms: Duration in milliseconds
            **kwargs: Additional parameters for specific modulation types

        Returns:
            Complex baseband signal
        """
        mod = modulation.lower()

        if mod == 'am':
            audio_freq = kwargs.get('audio_freq', 1000)
            return self.generate_am_signal(audio_freq, duration_ms)

        elif mod == 'fm':
            audio_freq = kwargs.get('audio_freq', 1000)
            deviation = kwargs.get('deviation', FM_DEVIATION_NARROW)
            return self.generate_fm_signal(audio_freq, duration_ms, deviation)

        elif mod == 'ssb':
            audio_freq = kwargs.get('audio_freq', 1000)
            sideband = kwargs.get('sideband', 'usb')
            return self.generate_ssb_signal(audio_freq, duration_ms, sideband)

        elif mod == 'nbfm':
            return self.generate_nbfm_voice(duration_ms)

        elif mod == 'p25':
            return self.generate_p25_signal(duration_ms)

        elif mod == 'dstar':
            return self.generate_dstar_signal(duration_ms)

        elif mod == 'ctcss':
            ctcss_freq = kwargs.get('ctcss_freq', 100.0)
            audio_freq = kwargs.get('audio_freq', 1000)
            return self.generate_ctcss_tone(ctcss_freq, duration_ms, audio_freq)

        else:
            raise ValueError(f"Unknown modulation type: {modulation}")


def main():
    """Example usage"""
    # Create VHF/UHF simulator
    vhf_sim = VHFUHFSimulator(sample_rate=2e6)

    # Generate narrowband FM signal
    signal = vhf_sim.generate_signal(modulation='nbfm', duration_ms=1000)

    # Transmit using BladeRF on 2m band
    with BladeRFTransmitter(
        frequency=146.52e6,  # 2m calling frequency
        sample_rate=2e6,
        bandwidth=2e6,
        gain=30
    ) as tx:
        tx.transmit(signal, repeat=1)

    logger.info("VHF/UHF signal transmission complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
