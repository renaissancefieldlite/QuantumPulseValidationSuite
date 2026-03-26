# Method

## Simulation baseline

The simulation run is intentionally an `inject -> detect` sanity check. It is
used to confirm that the spectral pipeline reports a target transition cadence
when that cadence is added to a noisy trace by design.

That is pipeline validation, not independent proof of an intrinsic device
frequency.

## Hardware-derived model

The hardware-derived mode loads calibration-style parameters and builds a local
decoherence trajectory. The resulting coherence proxy is then inspected in the
frequency domain without explicitly inserting a 0.67 Hz cadence.

This is stronger than an unconstrained toy signal because the trace is anchored
to device characteristics, but it is still a model layer.

## Real hardware path

To move this repo into a direct evidence category, the same analysis must be
run against traces or counts captured from a real backend without imposing the
target frequency in the simulator.
