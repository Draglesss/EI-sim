#!/usr/bin/env python3
"""
Generate time-series.

CSV format:
timestamp,gas,label
0,0.123,normal
100,0.121,normal
...

Signal synthesis overview:

- Baseline: constant level in [0, 1]
- Noise: Gaussian noise added per sample
- Drift: slow linear change over time (e.g., aging/environment)
- Leak patterns:
  - step leak: exponential rise after a start time (first-order response)
  - intermittent: periodic pulses (on/off)
  - spikes: short impulses at random times
  - noisy environment: low-frequency sinusoid + higher noise
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Series:
    timestamps_ms: List[int]
    values: List[float]
    label: str


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _gaussian(rng: random.Random, mu: float, sigma: float) -> float:
    # Box–Muller
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mu + sigma * z0


def _timestamps(duration_s: float, freq_hz: float) -> List[int]:
    if freq_hz <= 0:
        raise ValueError("freq_hz must be > 0")
    dt_ms = int(round(1000.0 / freq_hz))
    n = int(round(duration_s * freq_hz))
    return [i * dt_ms for i in range(n)]


def generate_normal(
    *,
    duration_s: float,
    freq_hz: float,
    baseline: float,
    noise_sigma: float,
    rng: random.Random,
) -> Series:
    # baseline + Gaussian noise
    ts = _timestamps(duration_s, freq_hz)
    vals = [_clamp(baseline + _gaussian(rng, 0.0, noise_sigma), 0.0, 1.0) for _ in ts]
    return Series(ts, vals, "normal")


def generate_drift(
    *,
    duration_s: float,
    freq_hz: float,
    baseline: float,
    drift_per_s: float,
    noise_sigma: float,
    rng: random.Random,
) -> Series:
    # x(t) = baseline + drift_per_s * t + noise
    ts = _timestamps(duration_s, freq_hz)
    vals: List[float] = []
    for t_ms in ts:
        t_s = t_ms / 1000.0
        x = baseline + drift_per_s * t_s + _gaussian(rng, 0.0, noise_sigma)
        vals.append(_clamp(x, 0.0, 1.0))
    return Series(ts, vals, "drift")


def generate_step_leak(
    *,
    duration_s: float,
    freq_hz: float,
    baseline: float,
    step_at_s: float,
    step_size: float,
    settle_tau_s: float,
    noise_sigma: float,
    rng: random.Random,
) -> Series:
    """
    Step increase at step_at_s with exponential settling (sensor response).

    leak(t) = step_size * (1 - exp(-(t - t0) / tau))   for t >= t0
              0                                       for t <  t0

    x(t) = baseline + leak(t) + noise
    """
    ts = _timestamps(duration_s, freq_hz)
    vals: List[float] = []
    for t_ms in ts:
        t_s = t_ms / 1000.0
        if t_s < step_at_s:
            leak = 0.0
        else:
            # approaches step_size as t increases
            leak = step_size * (1.0 - math.exp(-(t_s - step_at_s) / max(settle_tau_s, 1e-6)))
        x = baseline + leak + _gaussian(rng, 0.0, noise_sigma)
        vals.append(_clamp(x, 0.0, 1.0))
    return Series(ts, vals, "step_leak")


def generate_intermittent_leak(
    *,
    duration_s: float,
    freq_hz: float,
    baseline: float,
    pulse_period_s: float,
    pulse_duty: float,
    pulse_amp: float,
    noise_sigma: float,
    rng: random.Random,
) -> Series:
    """
    Leak pulses on/off (e.g., wind / valve cycling).
    """
    ts = _timestamps(duration_s, freq_hz)
    vals: List[float] = []
    for t_ms in ts:
        t_s = t_ms / 1000.0
        phase = (t_s % max(pulse_period_s, 1e-6)) / max(pulse_period_s, 1e-6)
        leak = pulse_amp if phase < _clamp(pulse_duty, 0.0, 1.0) else 0.0
        x = baseline + leak + _gaussian(rng, 0.0, noise_sigma)
        vals.append(_clamp(x, 0.0, 1.0))
    return Series(ts, vals, "intermittent_leak")


def generate_spikes(
    *,
    duration_s: float,
    freq_hz: float,
    baseline: float,
    spikes_per_min: float,
    spike_amp: float,
    spike_width_s: float,
    noise_sigma: float,
    rng: random.Random,
) -> Series:
    """
    Occasional short spikes (electrical noise). Good to demonstrate robustness.
    """
    ts = _timestamps(duration_s, freq_hz)
    dt_s = 1.0 / freq_hz
    spike_width_steps = max(1, int(round(spike_width_s / dt_s)))
    p_spike = (spikes_per_min / 60.0) * dt_s  # Bernoulli per timestep

    vals: List[float] = []
    remaining = 0
    for _ in ts:
        if remaining <= 0 and rng.random() < p_spike:
            remaining = spike_width_steps
        leak = spike_amp if remaining > 0 else 0.0
        remaining -= 1
        x = baseline + leak + _gaussian(rng, 0.0, noise_sigma)
        vals.append(_clamp(x, 0.0, 1.0))
    return Series(ts, vals, "spikes")


def generate_noisy_environment(
    *,
    duration_s: float,
    freq_hz: float,
    baseline: float,
    noise_sigma: float,
    hum_hz: float,
    hum_amp: float,
    rng: random.Random,
) -> Series:
    """
    Higher noise + a low-frequency oscillation (e.g., vibration / interference).
    """
    ts = _timestamps(duration_s, freq_hz)
    vals: List[float] = []
    for t_ms in ts:
        t_s = t_ms / 1000.0
        hum = hum_amp * math.sin(2.0 * math.pi * hum_hz * t_s)
        x = baseline + hum + _gaussian(rng, 0.0, noise_sigma)
        vals.append(_clamp(x, 0.0, 1.0))
    return Series(ts, vals, "noisy_env")


def _axis_names(axes: int) -> List[str]:
    if axes <= 0:
        raise ValueError("--axes must be >= 1")
    if axes == 1:
        return ["gas"]
    # Multi-axis naming to match typical Edge Impulse expectations
    # (e.g., accX/accY/accZ style).
    suffixes = ["X", "Y", "Z", "W"]
    if axes > len(suffixes):
        raise ValueError("--axes too large (max 4)")
    return [f"gas{suffixes[i]}" for i in range(axes)]


def _expand_axes(base: float, axes: int, rng: random.Random) -> List[float]:
    """
    Turn a single simulated scalar into multi-axis readings.
    The extra axes are highly correlated (realistic for multi-sensor setups),
    with slight independent noise so the model has multiple channels.
    """
    if axes == 1:
        return [base]
    # Add tiny axis-specific perturbations.
    out: List[float] = []
    for i in range(axes):
        jitter = _gaussian(rng, 0.0, 0.0025) * (1.0 + 0.15 * i)
        out.append(_clamp(base + jitter, 0.0, 1.0))
    return out


def write_csv(series: Series, out_path: str, *, axes: int, rng: random.Random) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        axis_names = _axis_names(axes)
        f.write("timestamp," + ",".join(axis_names) + ",label\n")
        for t_ms, v in zip(series.timestamps_ms, series.values):
            axis_vals = _expand_axes(v, axes, rng)
            f.write(
                f"{t_ms},"
                + ",".join(f"{av:.6f}" for av in axis_vals)
                + f",{series.label}\n"
            )


def write_csv_values_only(series: Series, out_path: str, *, axes: int, rng: random.Random) -> None:
    """
    Write a strict time-series CSV for Edge Impulse CLI uploader:
    timestamp + numeric sensor columns only (no label column).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        axis_names = _axis_names(axes)
        f.write("timestamp," + ",".join(axis_names) + "\n")
        for t_ms, v in zip(series.timestamps_ms, series.values):
            axis_vals = _expand_axes(v, axes, rng)
            f.write(f"{t_ms}," + ",".join(f"{av:.6f}" for av in axis_vals) + "\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate gas-sensor time series for Edge Impulse.")
    p.add_argument(
        "--scenario",
        choices=["normal", "drift", "step", "intermittent", "spikes", "noisy"],
        required=True,
        help="Scenario to generate.",
    )
    p.add_argument("--out", required=True, help="Output CSV path (e.g., data/normal_train.csv).")
    p.add_argument("--duration", type=float, default=180.0, help="Duration in seconds.")
    p.add_argument("--freq", type=float, default=10.0, help="Sampling frequency (Hz).")
    p.add_argument("--seed", type=int, default=1, help="Random seed.")
    p.add_argument("--baseline", type=float, default=0.12, help="Baseline sensor value in [0,1].")
    p.add_argument("--noise", type=float, default=0.01, help="Noise sigma.")
    p.add_argument(
        "--axes",
        type=int,
        default=1,
        help="Number of sensor axes to output (1..3 recommended). Use 3 if CLI upload fails on single-axis.",
    )
    p.add_argument(
        "--no-label-column",
        action="store_true",
        help="Write only timestamp+gas (for edge-impulse-uploader --label ...).",
    )

    # Drift params
    p.add_argument("--drift-per-s", type=float, default=0.00015, help="Drift per second.")

    # Step leak params
    p.add_argument("--step-at", type=float, default=30.0, help="Seconds when leak starts.")
    p.add_argument("--step-size", type=float, default=0.35, help="Leak amplitude added to baseline.")
    p.add_argument("--settle-tau", type=float, default=6.0, help="Settling time constant (seconds).")

    # Intermittent params
    p.add_argument("--pulse-period", type=float, default=8.0, help="Pulse period in seconds.")
    p.add_argument("--pulse-duty", type=float, default=0.4, help="Duty cycle in [0,1].")
    p.add_argument("--pulse-amp", type=float, default=0.25, help="Pulse amplitude.")

    # Spikes params
    p.add_argument("--spikes-per-min", type=float, default=3.0, help="Expected spikes per minute.")
    p.add_argument("--spike-amp", type=float, default=0.45, help="Spike amplitude.")
    p.add_argument("--spike-width", type=float, default=0.3, help="Spike width (seconds).")

    # Noisy env params
    p.add_argument("--hum-hz", type=float, default=0.4, help="Hum frequency (Hz).")
    p.add_argument("--hum-amp", type=float, default=0.03, help="Hum amplitude.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    common = dict(
        duration_s=float(args.duration),
        freq_hz=float(args.freq),
        baseline=float(args.baseline),
        noise_sigma=float(args.noise),
        rng=rng,
    )

    if args.scenario == "normal":
        series = generate_normal(**common)
    elif args.scenario == "drift":
        series = generate_drift(**common, drift_per_s=float(args.drift_per_s))
    elif args.scenario == "step":
        series = generate_step_leak(
            **common,
            step_at_s=float(args.step_at),
            step_size=float(args.step_size),
            settle_tau_s=float(args.settle_tau),
        )
    elif args.scenario == "intermittent":
        series = generate_intermittent_leak(
            **common,
            pulse_period_s=float(args.pulse_period),
            pulse_duty=float(args.pulse_duty),
            pulse_amp=float(args.pulse_amp),
        )
    elif args.scenario == "spikes":
        series = generate_spikes(
            **common,
            spikes_per_min=float(args.spikes_per_min),
            spike_amp=float(args.spike_amp),
            spike_width_s=float(args.spike_width),
        )
    elif args.scenario == "noisy":
        series = generate_noisy_environment(
            **common,
            hum_hz=float(args.hum_hz),
            hum_amp=float(args.hum_amp),
        )
    else:
        raise RuntimeError("unreachable")

    axes = int(args.axes)
    if args.no_label_column:
        write_csv_values_only(series, args.out, axes=axes, rng=rng)
    else:
        write_csv(series, args.out, axes=axes, rng=rng)
    print(f"Wrote {args.out} ({len(series.timestamps_ms)} rows, label={series.label})")


if __name__ == "__main__":
    main()

