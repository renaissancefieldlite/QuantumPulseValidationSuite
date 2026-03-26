# QuantumPulseValidationSuite

Pulse-detection experiment for the `0.67 Hz` hypothesis.

This repo now separates three different layers that were previously blended:

1. `simulation_baseline`
   A controlled inject-and-detect run used to validate the detection pipeline.
2. `hardware_derived_model`
  A spectrum study anchored to calibration-style device parameters such as
  `T1`, `T2`, readout error, gate error, drift, leakage, and crosstalk.
3. `real_hardware_validation`
   The still-pending step where raw traces or counts would be analyzed without
   imposing the target carrier in the simulator.

## What It Tests

The detection stack uses FFT-style spectral analysis to answer two separate
questions:

- can the pipeline recover a known 0.67 Hz transition cadence when it is present by design?
- what low-frequency structure appears when the trace is generated from a
  calibration-anchored decoherence model rather than an injected cadence?

## Current Status

- baseline simulation: implemented
- hardware-derived model: implemented
- real hardware path: pending

## Quick Start

```bash
python3 QuantumPulseValidationSuite.py --mode simulation --json
python3 QuantumPulseValidationSuite.py --mode hardware-derived --json
```

To use a custom calibration snapshot:

```bash
python3 QuantumPulseValidationSuite.py \
  --mode hardware-derived \
  --calibration examples/sample_hardware_calibration.json \
  --json
```

See [docs/METHOD.md](docs/METHOD.md) and
[docs/EVIDENCE_BOUNDARY.md](docs/EVIDENCE_BOUNDARY.md).
