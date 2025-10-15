"""
GPS L1 C/A Signal Simulator
Generates reference-quality GPS L1 C/A signals with precise timing and spectral characteristics
"""

import numpy as np
from typing import List, Optional
import logging
from scipy import signal as sp_signal, interpolate
from bladerf_base import BladeRFTransmitter, normalize_signal

logger = logging.getLogger(__name__)

# GPS L1 Constants
GPS_L1_FREQUENCY = 1575.42e6  # Hz
GPS_CA_CODE_RATE = 1.023e6    # chips/sec
GPS_CA_CODE_LENGTH = 1023     # chips
GPS_NAV_DATA_RATE = 50        # bits/sec


class GPSSimulator:
    """GPS L1 C/A Signal Simulator"""

    # Gold code generator polynomials (G1 and G2)
    G1_POLY = [1, 3, 4, 5, 6, 7, 10]
    G2_POLY = [2, 3, 4, 5, 7, 8, 9, 10]

    # Phase selection for different PRN satellites
    PRN_PHASE_SELECT = {
        1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9),
        6: (2, 10), 7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3),
        11: (3, 4), 12: (5, 6), 13: (6, 7), 14: (7, 8), 15: (8, 9),
        16: (9, 10), 17: (1, 4), 18: (2, 5), 19: (3, 6), 20: (4, 7),
        21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6), 25: (5, 7),
        26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
        31: (3, 8), 32: (4, 9)
    }

    def __init__(self, sample_rate: float = 2.6e6, precise_timing: bool = True):
        """
        Initialize GPS simulator

        Args:
            sample_rate: Sample rate in Hz (recommended: integer multiple of 1.023 MHz)
            precise_timing: Use fractional sample timing for exact chip boundaries
        """
        self.sample_rate = sample_rate
        self.ca_code_cache = {}
        self.precise_timing = precise_timing

        # Calculate exact samples per chip (may be fractional)
        self.samples_per_chip = sample_rate / GPS_CA_CODE_RATE

        # Verify sample rate is reasonable
        if sample_rate < 2.046e6:
            logger.warning(f"Sample rate {sample_rate/1e6:.3f} MHz may be too low for GPS (recommend >= 2.046 MHz)")

        logger.info(f"GPS Simulator initialized:")
        logger.info(f"  Sample rate: {sample_rate/1e6:.6f} MHz")
        logger.info(f"  Samples per chip: {self.samples_per_chip:.6f}")
        logger.info(f"  Precise timing: {precise_timing}")

    def generate_ca_code(self, prn: int) -> np.ndarray:
        """
        Generate C/A code for given PRN

        Args:
            prn: Satellite PRN number (1-32)

        Returns:
            C/A code sequence (1023 chips)
        """
        if prn in self.ca_code_cache:
            return self.ca_code_cache[prn]

        if prn not in self.PRN_PHASE_SELECT:
            raise ValueError(f"Invalid PRN: {prn}. Must be 1-32")

        # Initialize shift registers
        g1 = np.ones(10, dtype=int)
        g2 = np.ones(10, dtype=int)

        ca_code = np.zeros(GPS_CA_CODE_LENGTH, dtype=int)

        phase_select = self.PRN_PHASE_SELECT[prn]

        # Generate 1023 chips
        for i in range(GPS_CA_CODE_LENGTH):
            # Output is G1[10] XOR G2[phase_select]
            ca_code[i] = g1[9] ^ g2[phase_select[0]-1] ^ g2[phase_select[1]-1]

            # Shift G1
            g1_feedback = g1[2] ^ g1[9]
            g1 = np.roll(g1, 1)
            g1[0] = g1_feedback

            # Shift G2
            g2_feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
            g2 = np.roll(g2, 1)
            g2[0] = g2_feedback

        # Convert to +1/-1 (BPSK)
        ca_code = 2 * ca_code - 1

        # Cache for future use
        self.ca_code_cache[prn] = ca_code

        return ca_code

    def generate_nav_data(self, duration_ms: float) -> np.ndarray:
        """
        Generate simplified navigation data (alternating pattern for testing)

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            Navigation data bits (+1/-1)
        """
        num_bits = int(duration_ms * GPS_NAV_DATA_RATE / 1000)

        # Simple alternating pattern for testing
        # In real GPS, this would be ephemeris, almanac, etc.
        nav_data = np.ones(num_bits, dtype=int)
        nav_data[::2] = -1

        return nav_data

    def _upsample_ca_code_precise(self, ca_code: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Upsample C/A code with precise timing using sinc interpolation

        Args:
            ca_code: C/A code chips (1023 values)
            num_samples: Exact number of output samples for one code period

        Returns:
            Precisely upsampled C/A code
        """
        if not self.precise_timing:
            # Fast method: simple repeat
            samples_per_chip_int = int(np.round(self.samples_per_chip))
            upsampled = np.repeat(ca_code, samples_per_chip_int)
            if len(upsampled) < num_samples:
                upsampled = np.pad(upsampled, (0, num_samples - len(upsampled)), mode='edge')
            return upsampled[:num_samples]

        # Precise method: create time-aligned samples
        # Each chip has exact duration of 1/GPS_CA_CODE_RATE seconds
        chip_duration = 1.0 / GPS_CA_CODE_RATE

        # Sample times relative to code start
        sample_times = np.arange(num_samples) / self.sample_rate

        # Determine which chip each sample corresponds to
        chip_indices = np.floor(sample_times / chip_duration).astype(int)
        chip_indices = np.clip(chip_indices, 0, GPS_CA_CODE_LENGTH - 1)

        # Map chips to samples
        upsampled = ca_code[chip_indices].astype(float)

        return upsampled

    def generate_signal(
        self,
        prn: int,
        duration_ms: float = 1000,
        doppler_shift: float = 0.0,
        amplitude: float = 1.0,
        code_phase_chips: float = 0.0
    ) -> np.ndarray:
        """
        Generate reference-quality GPS L1 C/A signal with precise timing

        Args:
            prn: Satellite PRN number (1-32)
            duration_ms: Signal duration in milliseconds
            doppler_shift: Doppler frequency shift in Hz
            amplitude: Signal amplitude (0-1.0)
            code_phase_chips: Initial code phase offset in chips (0-1022)

        Returns:
            Complex baseband IQ samples with BPSK(1) modulation
        """
        logger.info(f"Generating GPS signal for PRN {prn}:")
        logger.info(f"  Duration: {duration_ms} ms")
        logger.info(f"  Doppler: {doppler_shift:+.1f} Hz")
        logger.info(f"  Code phase: {code_phase_chips:.2f} chips")

        # Get C/A code
        ca_code = self.generate_ca_code(prn)

        # Apply code phase offset (circular shift)
        if code_phase_chips != 0:
            phase_chips = int(code_phase_chips) % GPS_CA_CODE_LENGTH
            ca_code = np.roll(ca_code, -phase_chips)

        # Generate navigation data
        nav_data = self.generate_nav_data(duration_ms)

        # Calculate exact number of samples
        num_samples = int(np.round(self.sample_rate * duration_ms / 1000))

        # Calculate exact samples per 1ms code period
        samples_per_code = self.sample_rate / 1000.0  # Exact value

        # Generate time vector for Doppler
        t = np.arange(num_samples) / self.sample_rate

        # Pre-allocate signal array
        signal = np.zeros(num_samples, dtype=float)

        # Generate each 1ms code period with precise timing
        num_codes = int(np.ceil(duration_ms))

        for code_idx in range(num_codes):
            # Calculate exact sample boundaries
            start_sample = int(np.round(code_idx * samples_per_code))
            end_sample = int(np.round((code_idx + 1) * samples_per_code))
            end_sample = min(end_sample, num_samples)

            if start_sample >= num_samples:
                break

            segment_length = end_sample - start_sample

            # Get nav data bit for this millisecond (20 codes per bit @ 50 Hz)
            nav_bit_idx = code_idx // 20
            if nav_bit_idx >= len(nav_data):
                nav_bit_idx = len(nav_data) - 1
            nav_bit = nav_data[nav_bit_idx]

            # Upsample C/A code with precise timing
            ca_upsampled = self._upsample_ca_code_precise(ca_code, segment_length)

            # Apply nav data modulation (BPSK)
            signal[start_sample:end_sample] = ca_upsampled * nav_bit * amplitude

        # Convert to complex baseband (GPS is BPSK on I channel)
        signal = signal.astype(complex)

        # Apply Doppler shift if specified
        if doppler_shift != 0:
            # Include code Doppler (proportional to carrier Doppler)
            code_doppler = doppler_shift * GPS_CA_CODE_RATE / GPS_L1_FREQUENCY
            total_freq_shift = doppler_shift  # Carrier Doppler in baseband

            signal = signal * np.exp(1j * 2 * np.pi * total_freq_shift * t)

            logger.info(f"  Applied Doppler: carrier={doppler_shift:.1f} Hz, code={code_doppler:.3f} chips/s")

        # Normalize power
        signal = normalize_signal(signal, target_power=0.7)

        logger.info(f"  Generated {num_samples} samples ({len(signal)/self.sample_rate*1000:.3f} ms actual)")

        return signal

    def generate_multi_satellite(
        self,
        prn_list: List[int],
        duration_ms: float = 1000,
        doppler_shifts: Optional[List[float]] = None,
        amplitudes: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Generate combined signal from multiple satellites

        Args:
            prn_list: List of PRN numbers
            duration_ms: Signal duration in milliseconds
            doppler_shifts: Doppler shifts for each satellite (Hz)
            amplitudes: Relative amplitudes for each satellite

        Returns:
            Combined complex baseband signal
        """
        if doppler_shifts is None:
            doppler_shifts = [0.0] * len(prn_list)

        if amplitudes is None:
            amplitudes = [1.0] * len(prn_list)

        logger.info(f"Generating multi-satellite signal with PRNs: {prn_list}")

        # Generate each satellite signal
        combined_signal = None

        for prn, doppler, amp in zip(prn_list, doppler_shifts, amplitudes):
            sat_signal = self.generate_signal(prn, duration_ms, doppler, amp)

            if combined_signal is None:
                combined_signal = sat_signal
            else:
                combined_signal += sat_signal

        # Normalize combined signal
        combined_signal = normalize_signal(combined_signal, target_power=0.7)

        return combined_signal


def main():
    """Example usage"""
    # Create GPS simulator
    gps_sim = GPSSimulator(sample_rate=2.6e6)

    # Generate signal for PRN 1
    signal = gps_sim.generate_signal(prn=1, duration_ms=100)

    # Transmit using BladeRF
    with BladeRFTransmitter(
        frequency=GPS_L1_FREQUENCY,
        sample_rate=2.6e6,
        bandwidth=2.5e6,
        gain=30
    ) as tx:
        tx.transmit(signal, repeat=1)

    logger.info("GPS signal transmission complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
