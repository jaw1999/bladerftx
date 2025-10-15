"""
Comprehensive Examples for BladeRF Signal Simulators
Demonstrates usage of all signal types with reference-quality waveforms
"""

import logging
import numpy as np
from gps_simulator import GPSSimulator, GPS_L1_FREQUENCY
from inmarsat_simulator import InmarsatSimulator
from link16_simulator import Link16Simulator, LINK16_BASE_FREQUENCY
from vhf_uhf_simulator import VHFUHFSimulator, FM_DEVIATION_NARROW, FM_DEVIATION_WIDE
from bladerf_base import BladeRFTransmitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_gps_single_satellite():
    """
    Example 1: GPS Single Satellite
    Generates a single GPS satellite signal with precise timing
    Transmits continuously until Ctrl+C
    """
    logger.info("=" * 60)
    logger.info("Example 1: GPS L1 C/A - Single Satellite (PRN 1)")
    logger.info("=" * 60)

    # Create GPS simulator with precise timing
    gps_sim = GPSSimulator(sample_rate=2.6e6, precise_timing=True)

    # Generate signal for PRN 1 with slight Doppler
    signal = gps_sim.generate_signal(
        prn=1,
        duration_ms=100,
        doppler_shift=100.0,  # 100 Hz Doppler
        code_phase_chips=0.0
    )

    # Transmit continuously
    with BladeRFTransmitter(
        frequency=GPS_L1_FREQUENCY,
        sample_rate=2.6e6,
        bandwidth=2.5e6,
        gain=30
    ) as tx:
        tx.transmit(signal, continuous=True)

    logger.info("GPS single satellite transmission complete\n")


def example_gps_multi_satellite():
    """
    Example 2: GPS Multi-Satellite Constellation
    Simulates multiple GPS satellites with different Doppler shifts
    Transmits continuously until Ctrl+C
    """
    logger.info("=" * 60)
    logger.info("Example 2: GPS L1 C/A - Multi-Satellite Constellation")
    logger.info("=" * 60)

    gps_sim = GPSSimulator(sample_rate=4.0e6, precise_timing=True)

    # Simulate 4 visible satellites with realistic parameters
    prn_list = [1, 7, 14, 23]
    doppler_shifts = [450.0, -1200.0, 800.0, -300.0]  # Hz
    amplitudes = [1.0, 0.8, 0.9, 0.7]  # Relative signal strengths

    signal = gps_sim.generate_multi_satellite(
        prn_list=prn_list,
        duration_ms=100,
        doppler_shifts=doppler_shifts,
        amplitudes=amplitudes
    )

    with BladeRFTransmitter(
        frequency=GPS_L1_FREQUENCY,
        sample_rate=4.0e6,
        bandwidth=4.0e6,
        gain=30
    ) as tx:
        tx.transmit(signal, continuous=True)

    logger.info("GPS multi-satellite transmission complete\n")


def example_inmarsat_c():
    """
    Example 3: Inmarsat-C Signal
    600 bps BPSK, 4 kHz bandwidth, convolutional coded
    Transmits continuously until Ctrl+C
    """
    logger.info("=" * 60)
    logger.info("Example 3: Inmarsat-C (600 bps BPSK, 4 kHz BW)")
    logger.info("=" * 60)

    # BladeRF minimum sample rate: 521 kHz
    # Use 600 kHz for 600 bps (1000 samples/symbol)
    sample_rate = 600e3
    inmarsat_sim = InmarsatSimulator(sample_rate=sample_rate, alpha=0.35)

    signal = inmarsat_sim.generate_signal(
        service_type='c',
        duration_ms=1000,
        frequency_offset=0,
        amplitude=1.0
    )

    with BladeRFTransmitter(
        frequency=1537.5e6,  # Inmarsat downlink (within 1530-1545 MHz)
        sample_rate=sample_rate,
        bandwidth=10e3,  # 10 kHz (wider than 4 kHz signal for filter rolloff)
        gain=30
    ) as tx:
        tx.transmit(signal, continuous=True)

    logger.info("Inmarsat-C transmission complete\n")


def example_inmarsat_bgan():
    """
    Example 4: Inmarsat BGAN Signal
    151.2 ksym/s 16-QAM, 189 kHz bandwidth, 0.25 RRC
    NOTE: Using QPSK approximation (16-QAM requires more complex implementation)
    Transmits continuously until Ctrl+C
    """
    logger.info("=" * 60)
    logger.info("Example 4: Inmarsat BGAN (151.2 ksym/s QPSK approx, 189 kHz BW)")
    logger.info("=" * 60)

    # BGAN specs: 151.2 ksym/s, 189 kHz BW, 0.25 RRC
    # Using 1 MHz sample rate (6.6 samples/symbol)
    sample_rate = 1e6
    symbol_rate = 151.2e3

    # Create simulator with alpha=0.25 for BGAN
    inmarsat_sim = InmarsatSimulator(sample_rate=sample_rate, alpha=0.25)

    # Generate QPSK at 151.2 ksym/s (302.4 kbps data rate)
    signal = inmarsat_sim.generate_qpsk_signal(
        data_rate=302400,  # 151.2 ksym/s * 2 bits/symbol
        duration_ms=500
    )

    with BladeRFTransmitter(
        frequency=1545.0e6,  # Inmarsat downlink band
        sample_rate=sample_rate,
        bandwidth=250e3,  # 250 kHz (wider than 189 kHz for rolloff)
        gain=30
    ) as tx:
        tx.transmit(signal, continuous=True)

    logger.info("Inmarsat BGAN transmission complete\n")


def example_link16_basic():
    """
    Example 5: Link-16 Tactical Data Link
    CCSK+MSK, 5 Mcps, 77k hops/sec, 57.6 kbps
    Transmits continuously until Ctrl+C
    """
    logger.info("=" * 60)
    logger.info("Example 5: Link-16 (CCSK+MSK, 5 Mcps, 57.6 kbps)")
    logger.info("=" * 60)

    link16_sim = Link16Simulator(sample_rate=20e6)

    # Generate pseudo-random hop pattern (51 channels)
    np.random.seed(42)
    hop_pattern = np.random.randint(0, 51, size=200).tolist()

    signal = link16_sim.generate_signal(
        duration_ms=100,
        data_rate_kbps=57.6,
        hop_pattern=hop_pattern
    )

    with BladeRFTransmitter(
        frequency=LINK16_BASE_FREQUENCY,
        sample_rate=20e6,
        bandwidth=20e6,
        gain=30
    ) as tx:
        tx.transmit(signal, continuous=True)

    logger.info("Link-16 transmission complete\n")


def example_vhf_nbfm():
    """
    Example 6: VHF Narrowband FM (Amateur Radio / Public Safety)
    Voice communication with 5 kHz deviation
    Transmits continuously until Ctrl+C
    """
    logger.info("=" * 60)
    logger.info("Example 6: VHF Narrowband FM (146.52 MHz)")
    logger.info("=" * 60)

    vhf_sim = VHFUHFSimulator(sample_rate=2e6)

    signal = vhf_sim.generate_signal(
        modulation='nbfm',
        duration_ms=1000
    )

    with BladeRFTransmitter(
        frequency=146.52e6,  # 2m calling frequency
        sample_rate=2e6,
        bandwidth=2e6,
        gain=30
    ) as tx:
        tx.transmit(signal, continuous=True)

    logger.info("VHF NBFM transmission complete\n")


def example_vhf_am_airband():
    """
    Example 7: VHF AM (Aviation Band)
    AM voice communication typical of aircraft
    Transmits continuously until Ctrl+C
    """
    logger.info("=" * 60)
    logger.info("Example 7: VHF AM Aviation Band (121.5 MHz)")
    logger.info("=" * 60)

    vhf_sim = VHFUHFSimulator(sample_rate=2e6)

    signal = vhf_sim.generate_signal(
        modulation='am',
        duration_ms=1000,
        audio_freq=1000  # 1 kHz tone
    )

    with BladeRFTransmitter(
        frequency=121.5e6,  # Emergency frequency
        sample_rate=2e6,
        bandwidth=2e6,
        gain=30
    ) as tx:
        tx.transmit(signal, continuous=True)

    logger.info("VHF AM transmission complete\n")


def example_uhf_p25():
    """
    Example 8: UHF P25 Digital Voice (Public Safety)
    C4FM modulation for P25 Phase 1
    Transmits continuously until Ctrl+C
    """
    logger.info("=" * 60)
    logger.info("Example 8: UHF P25 Digital (4800 baud C4FM)")
    logger.info("=" * 60)

    uhf_sim = VHFUHFSimulator(sample_rate=2e6)

    signal = uhf_sim.generate_signal(
        modulation='p25',
        duration_ms=1000
    )

    with BladeRFTransmitter(
        frequency=460.0e6,  # UHF public safety band
        sample_rate=2e6,
        bandwidth=2e6,
        gain=30
    ) as tx:
        tx.transmit(signal, continuous=True)

    logger.info("UHF P25 transmission complete\n")


def example_vhf_ctcss():
    """
    Example 9: VHF FM with CTCSS Tone
    Narrowband FM with 100 Hz CTCSS (PL tone)
    Transmits continuously until Ctrl+C
    """
    logger.info("=" * 60)
    logger.info("Example 9: VHF FM with CTCSS (100 Hz PL)")
    logger.info("=" * 60)

    vhf_sim = VHFUHFSimulator(sample_rate=2e6)

    signal = vhf_sim.generate_signal(
        modulation='ctcss',
        duration_ms=1000,
        ctcss_freq=100.0,  # 100 Hz PL tone
        audio_freq=1000
    )

    with BladeRFTransmitter(
        frequency=146.94e6,
        sample_rate=2e6,
        bandwidth=2e6,
        gain=30
    ) as tx:
        tx.transmit(signal, continuous=True)

    logger.info("VHF FM with CTCSS transmission complete\n")


def example_uhf_dstar():
    """
    Example 10: UHF D-STAR Digital Voice
    GMSK modulation for D-STAR
    Transmits continuously until Ctrl+C
    """
    logger.info("=" * 60)
    logger.info("Example 10: UHF D-STAR Digital (GMSK)")
    logger.info("=" * 60)

    uhf_sim = VHFUHFSimulator(sample_rate=2e6)

    signal = uhf_sim.generate_signal(
        modulation='dstar',
        duration_ms=1000
    )

    with BladeRFTransmitter(
        frequency=445.0e6,  # 70cm band
        sample_rate=2e6,
        bandwidth=2e6,
        gain=30
    ) as tx:
        tx.transmit(signal, continuous=True)

    logger.info("UHF D-STAR transmission complete\n")


def run_all_examples():
    """Run all example simulations"""
    logger.info("\n" + "=" * 60)
    logger.info("BladeRF Signal Simulator - All Examples")
    logger.info("=" * 60 + "\n")

    examples = [
        ("GPS Single Satellite", example_gps_single_satellite),
        ("GPS Multi-Satellite", example_gps_multi_satellite),
        ("Inmarsat-C", example_inmarsat_c),
        ("Inmarsat BGAN", example_inmarsat_bgan),
        ("Link-16", example_link16_basic),
        ("VHF NBFM", example_vhf_nbfm),
        ("VHF AM Airband", example_vhf_am_airband),
        ("UHF P25", example_uhf_p25),
        ("VHF CTCSS", example_vhf_ctcss),
        ("UHF D-STAR", example_uhf_dstar),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            logger.error(f"Error in {name}: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("All examples completed!")
    logger.info("=" * 60)


def main():
    """
    Main entry point
    Run individual examples or all examples
    """
    import sys

    if len(sys.argv) > 1:
        example_map = {
            'gps1': example_gps_single_satellite,
            'gps_multi': example_gps_multi_satellite,
            'inmarsat_c': example_inmarsat_c,
            'inmarsat_bgan': example_inmarsat_bgan,
            'link16': example_link16_basic,
            'vhf_nbfm': example_vhf_nbfm,
            'vhf_am': example_vhf_am_airband,
            'uhf_p25': example_uhf_p25,
            'vhf_ctcss': example_vhf_ctcss,
            'uhf_dstar': example_uhf_dstar,
            'all': run_all_examples,
        }

        example_name = sys.argv[1]
        if example_name in example_map:
            example_map[example_name]()
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available examples: {', '.join(example_map.keys())}")
    else:
        print("Usage: python examples.py <example_name>")
        print("\nAvailable examples:")
        print("  gps1          - GPS single satellite")
        print("  gps_multi     - GPS multiple satellites")
        print("  inmarsat_c    - Inmarsat-C 600 bps")
        print("  inmarsat_bgan - Inmarsat BGAN 64 kbps")
        print("  link16        - Link-16 tactical datalink")
        print("  vhf_nbfm      - VHF narrowband FM")
        print("  vhf_am        - VHF AM airband")
        print("  uhf_p25       - UHF P25 digital")
        print("  vhf_ctcss     - VHF FM with CTCSS")
        print("  uhf_dstar     - UHF D-STAR digital")
        print("  all           - Run all examples")


if __name__ == "__main__":
    main()
