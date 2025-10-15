"""
BladeRF Base Interface Module
Core interface for BladeRF SDR hardware
"""

import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import bladerf
    from bladerf._bladerf import ChannelLayout, Format
    BLADERF_AVAILABLE = True
except ImportError:
    logger.error("bladeRF library not found. Please install bladeRF.")
    raise ImportError("bladeRF is required - see INSTALL_BLADERF.md")


class BladeRFTransmitter:
    """Base class for BladeRF transmission - this handles all the radio hardware interaction"""

    def __init__(
        self,
        frequency: float,
        sample_rate: float,
        bandwidth: float,
        gain: int = 30,
        channel: int = 0
    ):
        """
        Set up the BladeRF transmitter with your parameters

        Args:
            frequency: Center frequency in Hz (where you're transmitting)
            sample_rate: Sample rate in Hz (how fast we're sampling)
            bandwidth: Bandwidth in Hz (how wide your signal is)
            gain: TX gain in dB (0-60, higher = louder but be careful!)
            channel: Channel number (0 or 1 for bladeRF 2.0)
        """
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.gain = min(max(gain, 0), 60)  # Keep gain in safe range so we don't blow anything up
        self.channel = channel
        self.device = None
        self.is_running = False

        logger.info(f"Initializing BladeRF Transmitter:")
        logger.info(f"  Frequency: {frequency/1e6:.3f} MHz")
        logger.info(f"  Sample Rate: {sample_rate/1e6:.3f} MHz")
        logger.info(f"  Bandwidth: {bandwidth/1e6:.3f} MHz")
        logger.info(f"  Gain: {self.gain} dB")

    def open(self) -> bool:
        """Open a connection to the BladeRF device and configure it"""
        try:
            self.device = bladerf.BladeRF()

            # Grab the TX channel we want to use
            tx_ch = bladerf.CHANNEL_TX(self.channel)

            # Set up all the transmit parameters
            self.device.set_frequency(tx_ch, int(self.frequency))
            self.device.set_sample_rate(tx_ch, int(self.sample_rate))
            self.device.set_bandwidth(tx_ch, int(self.bandwidth))
            self.device.set_gain(tx_ch, self.gain)

            # Set up the sync interface for streaming samples to the device
            # We're using SC16_Q11 format (that's signed complex 16-bit with Q11 fixed point)
            # The buffer config here was tuned to work well - don't mess with it unless you know what you're doing
            self.device.sync_config(
                layout=ChannelLayout.TX_X1,
                fmt=Format.SC16_Q11,
                num_buffers=16,
                buffer_size=8192,
                num_transfers=8,
                stream_timeout=3500
            )

            logger.info("BladeRF device opened and sync TX configured successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to open BladeRF device: {e}")
            return False

    def transmit(self, iq_samples: np.ndarray, repeat: int = 1, continuous: bool = False) -> bool:
        """
        Actually transmit your IQ samples over the air

        Args:
            iq_samples: Complex IQ samples (numpy array with your signal)
            repeat: How many times to repeat (ignored if you set continuous=True)
            continuous: If True, keep transmitting until you hit Ctrl+C

        Returns:
            True if it worked, False if something went wrong
        """
        if self.device is None:
            logger.error("Device not opened")
            return False

        try:
            # Convert our samples to the format the BladeRF wants (interleaved I/Q as int16)
            # Q11 format means we can use values from -2048 to 2047
            scale = 2000  # Stay a bit below max to avoid clipping artifacts

            # Make sure samples are in range
            iq_clipped = np.clip(iq_samples, -1.0, 1.0)

            # Create the interleaved array (I, Q, I, Q, I, Q, ...)
            iq_int16 = np.empty(len(iq_clipped) * 2, dtype=np.int16)
            iq_int16[0::2] = np.real(iq_clipped) * scale  # Real part (I)
            iq_int16[1::2] = np.imag(iq_clipped) * scale  # Imaginary part (Q)

            # Turn on the TX module if it isn't already running
            tx_ch = bladerf.CHANNEL_TX(self.channel)
            if not self.is_running:
                self.device.enable_module(tx_ch, True)
                self.is_running = True
                logger.info("TX module enabled")

            # Now actually transmit the samples
            if continuous:
                logger.info("Starting continuous transmission (press Ctrl+C to stop)...")
                iteration = 0
                try:
                    while True:
                        self.device.sync_tx(iq_int16, len(iq_samples))
                        iteration += 1
                        if iteration % 100 == 0:  # Print a status update every 100 loops
                            duration_sec = iteration * len(iq_samples) / self.sample_rate
                            logger.info(f"Transmitted {iteration} iterations ({duration_sec:.1f}s total)")
                except KeyboardInterrupt:
                    logger.info(f"\nTransmission stopped by user after {iteration} iterations")
                    logger.info(f"Total duration: {iteration * len(iq_samples) / self.sample_rate:.1f}s")
            else:
                # Just transmit it a fixed number of times
                for i in range(repeat):
                    self.device.sync_tx(iq_int16, len(iq_samples))
                    logger.info(f"Transmitted iteration {i+1}/{repeat} ({len(iq_samples)} samples)")
                logger.info(f"Successfully transmitted {len(iq_samples)} samples total")

            return True

        except KeyboardInterrupt:
            logger.info("\nTransmission interrupted by user")
            return True
        except Exception as e:
            logger.error(f"Transmission failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def close(self):
        """Shut down the BladeRF and clean everything up"""
        if self.device is not None:
            try:
                if self.is_running:
                    self.device.enable_module(bladerf.CHANNEL_TX(self.channel), False)
                    self.is_running = False
                self.device.close()
                logger.info("BladeRF device closed")
            except Exception as e:
                logger.error(f"Error closing device: {e}")

        self.device = None

    def __enter__(self):
        """Context manager entry - lets you use 'with BladeRFTransmitter() as tx:'"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically closes when you're done"""
        self.close()


def normalize_signal(signal: np.ndarray, target_power: float = 0.7) -> np.ndarray:
    """
    Scale the signal to hit a target power level (so it's not too loud or too quiet)

    Args:
        signal: Complex signal array
        target_power: Target RMS power (0-1.0, usually 0.7 is good)

    Returns:
        Normalized signal
    """
    current_power = np.sqrt(np.mean(np.abs(signal)**2))
    if current_power > 0:
        signal = signal * (target_power / current_power)
    return signal


def upsample_signal(signal: np.ndarray, upsample_factor: int) -> np.ndarray:
    """
    Upsample the signal to a higher sample rate (by stuffing zeros and filtering)

    Args:
        signal: Input signal
        upsample_factor: How much to upsample (2x, 4x, etc.)

    Returns:
        Upsampled signal
    """
    if upsample_factor == 1:
        return signal

    # Stuff zeros between samples
    upsampled = np.zeros(len(signal) * upsample_factor, dtype=signal.dtype)
    upsampled[::upsample_factor] = signal

    # Filter it to smooth things out (this is pretty basic, could be fancier)
    from scipy import signal as sp_signal
    b = sp_signal.firwin(64, 1.0/upsample_factor)
    upsampled = sp_signal.lfilter(b, 1, upsampled) * upsample_factor

    return upsampled
