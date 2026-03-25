"""Pulse detection suite with bounded evidence labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hardware_profile import extract_noise_parameters, load_calibration, simulate_noise_trajectory


def band_power(freqs: np.ndarray, spectrum: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(spectrum[mask], freqs[mask]))


def spectral_summary(series: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    centered = series - np.mean(series)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    dominant_idx = int(np.argmax(spectrum[1:]) + 1) if len(spectrum) > 1 else 0
    return {
        "dominant_frequency_hz": float(freqs[dominant_idx]),
        "band_power_0p60_0p74_hz": band_power(freqs, spectrum, 0.60, 0.74),
        "band_power_0p10_2p00_hz": band_power(freqs, spectrum, 0.10, 2.00),
        "peak_power": float(spectrum[dominant_idx]),
    }


def run_simulation(duration_seconds: float, sample_rate_hz: float, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    time_axis = np.arange(int(duration_seconds * sample_rate_hz)) / sample_rate_hz
    noise = rng.normal(0.0, 1.0, len(time_axis))
    control = noise.copy()
    injected = noise + 0.2 * np.sin(2 * np.pi * 0.67 * time_axis)
    control_summary = spectral_summary(control, sample_rate_hz)
    injected_summary = spectral_summary(injected, sample_rate_hz)
    return {
        "mode": "simulation",
        "evidence_status": "simulation_baseline",
        "claim_under_test": "Whether the spectral pipeline recovers a known carrier when that carrier is present by design.",
        "control_trace": control_summary,
        "injected_trace": injected_summary,
        "delta_band_power_0p60_0p74_hz": injected_summary["band_power_0p60_0p74_hz"] - control_summary["band_power_0p60_0p74_hz"],
    }


def run_hardware_derived(calibration_path: str | None, duration_seconds: float, sample_rate_hz: float) -> dict[str, object]:
    calibration = load_calibration(calibration_path)
    params = extract_noise_parameters(calibration)
    report = simulate_noise_trajectory(params, duration_seconds=duration_seconds, sample_rate_hz=sample_rate_hz)
    coherence_proxy = np.array(report["time_series"]["coherence_proxy"], dtype=float)
    spectrum = spectral_summary(coherence_proxy, sample_rate_hz)
    return {
        "mode": "hardware-derived",
        "evidence_status": "hardware_derived_model",
        "claim_under_test": "What low-frequency structure appears in a calibration-anchored decoherence trace without imposing a target carrier.",
        "noise_summary": report["summary"],
        "spectral_summary": spectrum,
    }


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Run bounded pulse-detection experiments.")
    parser.add_argument("--mode", choices=["simulation", "hardware-derived"], default="simulation")
    parser.add_argument("--calibration", help="Optional calibration JSON path.")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--sample-rate", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", help="Optional output path.")
    args = parser.parse_args()

    if args.mode == "simulation":
        result = run_simulation(args.duration, args.sample_rate, args.seed)
    else:
        result = run_hardware_derived(args.calibration, args.duration, args.sample_rate)

    result["schema_version"] = "rfl.quantum_pulse_validation.v2"
    result["next_step"] = "Run the same spectral analysis on non-simulated traces before treating the output as direct empirical evidence."

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"mode={result['mode']}")
        print(f"evidence_status={result['evidence_status']}")

    return result


if __name__ == "__main__":
    main()
