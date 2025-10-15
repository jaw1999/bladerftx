# BladeRF Signal Simulator

RF signal generators for BladeRF SDR. 

## Features

- **GPS L1 C/A**: Satellite simulation with accurate timing and Gold codes, multi-satellite support
- **Inmarsat**: Inmarsat-C (600 bps BPSK) and BGAN (151.2 ksym/s QPSK) with RRC pulse shaping
- **Link-16**: Tactical datalink with CCSK+MSK modulation and frequency hopping
- **VHF/UHF**: AM, FM, SSB, P25, D-STAR, and CTCSS generation

## Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install libbladerf-dev bladerf

# Or from source: https://github.com/Nuand/bladeRF
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Example Scripts

```bash
python examples.py gps1              # Single GPS satellite
python examples.py gps_multi         # GPS constellation
python examples.py inmarsat_c        # Inmarsat-C
python examples.py inmarsat_bgan     # Inmarsat BGAN
python examples.py link16            # Link-16
python examples.py vhf_nbfm          # Narrowband FM
python examples.py vhf_am            # AM (aviation)
python examples.py uhf_p25           # P25 digital
python examples.py vhf_ctcss         # FM with CTCSS
python examples.py uhf_dstar         # D-STAR
python examples.py all               # Run all
```

### GPS Example

```python
from gps_simulator import GPSSimulator, GPS_L1_FREQUENCY
from bladerf_base import BladeRFTransmitter

gps_sim = GPSSimulator(sample_rate=2.6e6, precise_timing=True)

signal = gps_sim.generate_signal(
    prn=1,
    duration_ms=100,
    doppler_shift=450.0,
    code_phase_chips=0.0
)

with BladeRFTransmitter(
    frequency=GPS_L1_FREQUENCY,
    sample_rate=2.6e6,
    bandwidth=2.5e6,
    gain=30
) as tx:
    tx.transmit(signal, continuous=True)
```

### Inmarsat Example

```python
from inmarsat_simulator import InmarsatSimulator

inmarsat_sim = InmarsatSimulator(sample_rate=600e3, alpha=0.35)

signal = inmarsat_sim.generate_signal(
    service_type='c',
    duration_ms=1000,
    frequency_offset=0,
    amplitude=1.0
)
```

### VHF/UHF Example

```python
from vhf_uhf_simulator import VHFUHFSimulator

vhf_sim = VHFUHFSimulator(sample_rate=2e6)

signal = vhf_sim.generate_signal(
    modulation='ctcss',
    duration_ms=1000,
    ctcss_freq=100.0,
    audio_freq=1000
)
```

## Signal Specifications

### GPS L1 C/A
- 1.023 MHz chip rate (exact)
- 1ms code period alignment
- IS-GPS-200 compliant Gold codes (all 32 PRNs)
- Fractional sample timing
- Code Doppler scaling

### Inmarsat
- RRC pulse shaping (alpha=0.35 for C, 0.25 for BGAN)
- Gray-coded QPSK constellation
- Accurate symbol timing
- Correct bandwidth occupation

### Link-16
- CCSK (32-ary cyclic code shift keying)
- MSK modulation
- 13 microsecond hop dwell
- 51 channels, 3 MHz spacing
- Simplified RS structure

### VHF/UHF
- FM deviation: ±5 kHz (narrowband), ±75 kHz (wideband)
- P25 C4FM: ±1800/±600 Hz deviation levels
- D-STAR GMSK: BT=0.5
- CTCSS: accurate frequency and level
- SSB: Hilbert transform

## Sample Rates

- **GPS**: 2.6 MHz minimum (4.0 MHz for multi-satellite)
- **Inmarsat-C**: 600 kHz
- **Inmarsat BGAN**: 1 MHz
- **Link-16**: 20 MHz
- **VHF/UHF**: 2 MHz

All rates within BladeRF 2.0 range (521 kHz - 61.44 MHz).

## BladeRF Configuration

Handled automatically:
- TX frequency
- Sample rate
- Bandwidth filter
- Gain (adjustable)

Signals generated as complex baseband IQ, scaled to SC16_Q11 format.

## File Structure

```
bladerftx/
├── bladerf_base.py          # BladeRF interface
├── dsp_utils.py             # DSP functions
├── gps_simulator.py         # GPS generator
├── inmarsat_simulator.py    # Inmarsat generator
├── link16_simulator.py      # Link-16 generator
├── vhf_uhf_simulator.py     # VHF/UHF generator
├── examples.py              # Examples
├── requirements.txt         # Dependencies
└── README.md                # This file
```

