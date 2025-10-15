"""
Inmarsat Signal Simulator
Generates reference-quality Inmarsat L-band signals with precise modulation
"""

import numpy as np
from typing import Optional
import logging
from bladerf_base import BladeRFTransmitter, normalize_signal
from dsp_utils import design_rrc_filter, apply_pulse_shaping

logger = logging.getLogger(__name__)

# Inmarsat frequency bands (L-band)
INMARSAT_UPLINK_START = 1626.5e6    # Hz
INMARSAT_UPLINK_END = 1660.5e6      # Hz
INMARSAT_DOWNLINK_START = 1525.0e6  # Hz
INMARSAT_DOWNLINK_END = 1559.0e6    # Hz


class InmarsatSimulator:
    """Inmarsat Signal Simulator"""

    def __init__(self, sample_rate: float = 5e6, alpha: float = 0.35):
        """
        Initialize Inmarsat simulator with reference-quality modulation

        Args:
            sample_rate: Sample rate in Hz
            alpha: RRC roll-off factor (0.20-0.50 typical for satellite)
        """
        self.sample_rate = sample_rate
        self.alpha = alpha

        logger.info(f"Inmarsat Simulator initialized:")
        logger.info(f"  Sample rate: {sample_rate/1e6:.3f} MHz")
        logger.info(f"  RRC alpha: {alpha}")

    def generate_bpsk_signal(
        self,
        data_rate: float,
        duration_ms: float
    ) -> np.ndarray:
        """
        Generate BPSK modulated signal with RRC pulse shaping

        Args:
            data_rate: Symbol rate in bps
            duration_ms: Duration in milliseconds

        Returns:
            Complex baseband signal with precise pulse shaping
        """
        logger.info(f"Generating BPSK signal: {data_rate} bps, {duration_ms} ms")

        # Calculate exact number of symbols
        symbol_rate = data_rate
        num_symbols = int(symbol_rate * duration_ms / 1000)

        # Generate random BPSK symbols (+1/-1)
        symbols = 2 * np.random.randint(0, 2, num_symbols) - 1

        # Apply precise RRC pulse shaping
        signal = apply_pulse_shaping(
            symbols.astype(complex),
            self.sample_rate,
            symbol_rate,
            filter_type='rrc',
            alpha=self.alpha
        )

        # Truncate to exact duration
        num_samples = int(self.sample_rate * duration_ms / 1000)
        if len(signal) > num_samples:
            signal = signal[:num_samples]
        elif len(signal) < num_samples:
            signal = np.pad(signal, (0, num_samples - len(signal)), mode='constant')

        logger.info(f"  Generated {len(signal)} samples, {num_symbols} symbols")

        return signal

    def generate_qpsk_signal(
        self,
        data_rate: float,
        duration_ms: float
    ) -> np.ndarray:
        """
        Generate QPSK modulated signal with RRC pulse shaping

        Args:
            data_rate: Data rate in bps
            duration_ms: Duration in milliseconds

        Returns:
            Complex baseband signal with Gray-coded QPSK
        """
        logger.info(f"Generating QPSK signal: {data_rate} bps, {duration_ms} ms")

        # QPSK: 2 bits per symbol
        symbol_rate = data_rate / 2.0
        num_symbols = int(symbol_rate * duration_ms / 1000)

        # Generate random symbols (0-3)
        data_symbols = np.random.randint(0, 4, num_symbols)

        # Gray-coded QPSK constellation (pi/4 offset for better linearity)
        # Gray mapping: 00->0, 01->1, 11->2, 10->3
        qpsk_constellation = np.array([
            (1 + 1j),   # 00
            (-1 + 1j),  # 01
            (-1 - 1j),  # 11
            (1 - 1j)    # 10
        ]) / np.sqrt(2)

        symbols = qpsk_constellation[data_symbols]

        # Apply precise RRC pulse shaping
        signal = apply_pulse_shaping(
            symbols,
            self.sample_rate,
            symbol_rate,
            filter_type='rrc',
            alpha=self.alpha
        )

        # Truncate to exact duration
        num_samples = int(self.sample_rate * duration_ms / 1000)
        if len(signal) > num_samples:
            signal = signal[:num_samples]
        elif len(signal) < num_samples:
            signal = np.pad(signal, (0, num_samples - len(signal)), mode='constant')

        logger.info(f"  Generated {len(signal)} samples, {num_symbols} symbols")

        return signal

    def generate_inmarsat_c(self, duration_ms: float = 1000) -> np.ndarray:
        """
        Generate Inmarsat-C signal (600 bps BPSK with RRC)

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            Reference-quality complex baseband signal
        """
        return self.generate_bpsk_signal(data_rate=600, duration_ms=duration_ms)

    def generate_inmarsat_m(self, duration_ms: float = 1000) -> np.ndarray:
        """
        Generate Inmarsat-M signal (4.8 kbps QPSK with RRC)

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            Reference-quality complex baseband signal
        """
        return self.generate_qpsk_signal(data_rate=4800, duration_ms=duration_ms)

    def generate_bgan(self, duration_ms: float = 1000) -> np.ndarray:
        """
        Generate BGAN signal (64 kbps QPSK with RRC)

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            Reference-quality complex baseband signal
        """
        return self.generate_qpsk_signal(data_rate=64000, duration_ms=duration_ms)

    def generate_signal(
        self,
        service_type: str = 'c',
        duration_ms: float = 1000,
        frequency_offset: float = 0.0,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate Inmarsat signal

        Args:
            service_type: Service type ('c', 'm', 'bgan')
            duration_ms: Duration in milliseconds
            frequency_offset: Frequency offset in Hz
            amplitude: Signal amplitude

        Returns:
            Complex baseband signal
        """
        # Generate baseband signal
        if service_type.lower() == 'c':
            signal = self.generate_inmarsat_c(duration_ms)
        elif service_type.lower() == 'm':
            signal = self.generate_inmarsat_m(duration_ms)
        elif service_type.lower() == 'bgan':
            signal = self.generate_bgan(duration_ms)
        else:
            raise ValueError(f"Unknown service type: {service_type}")

        # Apply frequency offset if specified
        if frequency_offset != 0:
            t = np.arange(len(signal)) / self.sample_rate
            signal = signal * np.exp(1j * 2 * np.pi * frequency_offset * t)

        # Apply amplitude
        signal = signal * amplitude

        # Normalize
        signal = normalize_signal(signal, target_power=0.7)

        logger.info(f"Generated {len(signal)} samples ({duration_ms} ms)")

        return signal


def main():
    """Example usage"""
    # Create Inmarsat simulator
    inmarsat_sim = InmarsatSimulator(sample_rate=5e6)

    # Generate Inmarsat-C signal
    signal = inmarsat_sim.generate_signal(service_type='c', duration_ms=1000)

    # Transmit using BladeRF (downlink frequency)
    with BladeRFTransmitter(
        frequency=1545.0e6,  # Example Inmarsat downlink frequency
        sample_rate=5e6,
        bandwidth=5e6,
        gain=30
    ) as tx:
        tx.transmit(signal, repeat=1)

    logger.info("Inmarsat signal transmission complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
