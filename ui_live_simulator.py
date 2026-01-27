from __future__ import annotations

import os
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import streamlit as st

from ei_local_runner import EIRunner


def _dt_s(interval_ms: int) -> float:
    return float(interval_ms) / 1000.0


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _step_leak(t_s: float, start_s: float, amp: float, tau_s: float) -> float:
    """
    Leak simulation (first-order response):

    We model a leak as a step input that ramps up with an exponential "settling" curve:

        leak(t) = A * (1 - exp(-(t - t0) / tau))   for t >= t0
                  0                                for t <  t0

    - A   : leak amplitude (signal offset)
    - t0  : leak start time
    - tau : time constant; smaller => faster rise, larger => slower rise
    """
    if t_s < start_s:
        return 0.0
    tau_s = max(tau_s, 1e-6)
    return amp * (1.0 - np.exp(-(t_s - start_s) / tau_s))


def _expand_axes(base: float, axes: int, rng: np.random.Generator) -> List[float]:
    """
    Multi-channel simulation:

    The `.eim` expects `axis_count` channels (typically 3). In a real system this can represent
    multiple colocated channels (e.g., CH4 + temperature + humidity). For this demo, channels are
    generated as highly correlated copies of a scalar signal, with small per-channel jitter.
    """
    if axes <= 1:
        return [_clamp01(base)]
    # small axis-specific noise; correlated channels
    out: List[float] = []
    for i in range(axes):
        jitter = rng.normal(0.0, 0.0025) * (1.0 + 0.15 * i)
        out.append(_clamp01(base + jitter))
    return out


def _flatten_window(rows: List[List[float]]) -> List[float]:
    """
    Convert a window of shape (T, axes) into a flat feature vector for the `.eim`.

    Edge Impulse time-series order is time-major:
      [x0,y0,z0,  x1,y1,z1,  x2,y2,z2,  ...]
    """
    flat: List[float] = []
    for r in rows:
        flat.extend(r)
    return flat


def main() -> None:
    st.set_page_config(page_title="Edge Impulse Leak Simulator", layout="wide")
    st.title("Model Simulator")

    st.markdown(
        """

"""
    )

    with st.sidebar:
        st.header("Model")
        model_path = st.text_input(
            "Path to .eim model",
            value=os.path.join(os.getcwd(), "gas-anomaly-demo-linux-x86_64-v1.eim"),
        )
        debug = st.checkbox("Model debug output", value=False)
        threshold = st.number_input("Alert threshold (anomaly score)", min_value=-10.0, max_value=50.0, value=1.0, step=0.1)

        st.header("Signal")
        baseline = st.slider("Baseline", min_value=0.0, max_value=1.0, value=0.12, step=0.01)
        noise_sigma = st.slider("Noise sigma", min_value=0.0, max_value=0.1, value=0.01, step=0.005)

        st.header("Leak scenario")
        leak_on = st.toggle("Leak ON", value=False, help="When turned on, the leak starts immediately.")
        leak_amp = st.slider("Leak amplitude", min_value=0.0, max_value=0.9, value=0.55, step=0.05)
        leak_tau = st.slider("Leak settle tau (s)", min_value=0.5, max_value=20.0, value=6.0, step=0.5)

        st.header("Drift (optional)")
        drift_on = st.toggle("Drift ON", value=False)
        drift_per_s = st.slider(
            "Drift per second",
            min_value=-0.005,
            max_value=0.005,
            value=0.0,
            step=0.0005,
            format="%.4f",
        )

        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=1, step=1)

    # Session state
    if "running" not in st.session_state:
        st.session_state.running = False
    if "t_s" not in st.session_state:
        st.session_state.t_s = 0.0
    if "window" not in st.session_state:
        st.session_state.window = deque()  # type: ignore
    if "scores" not in st.session_state:
        st.session_state.scores = deque(maxlen=300)  # type: ignore
    if "runner" not in st.session_state:
        st.session_state.runner = None
    if "model_info" not in st.session_state:
        st.session_state.model_info = None
    if "leak_start_s" not in st.session_state:
        st.session_state.leak_start_s = 0.0
    if "leak_on_prev" not in st.session_state:
        st.session_state.leak_on_prev = False

    cols = st.columns([1, 1, 2])
    with cols[0]:
        start_pause_label = "Pause" if st.session_state.running else "Start"
        if st.button(start_pause_label, type="primary"):
            st.session_state.running = not st.session_state.running
    with cols[1]:
        if st.button("Reset"):
            st.session_state.running = False
            st.session_state.t_s = 0.0
            st.session_state.window = deque()
            st.session_state.scores = deque(maxlen=300)
    with cols[2]:
        st.write(
            f"**Status**: {'Running' if st.session_state.running else 'Paused'} | "
            f"t={st.session_state.t_s:.1f}s | "
            f"window={len(st.session_state.window)} rows"
        )

    # Always run with the real model
    if st.session_state.runner is None or (st.session_state.runner and st.session_state.runner.model_path != model_path):
        try:
            if st.session_state.runner is not None:
                try:
                    st.session_state.runner.stop()
                except Exception:
                    pass
            runner = EIRunner(model_path, debug=debug, timeout_s=30)
            model_info = runner.start()
            st.session_state.runner = runner
            st.session_state.model_info = model_info
        except Exception as e:
            st.error(f"Failed to start model runner: {e}")
            st.stop()

    mp = (st.session_state.model_info or {}).get("model_parameters", {})
    axis_count = int(mp.get("axis_count", 0) or 0)
    interval_ms = int(mp.get("interval_ms", 0) or 0)
    input_features_count = int(mp.get("input_features_count", 0) or 0)
    if axis_count <= 0 or interval_ms <= 0 or input_features_count <= 0:
        st.error("Model metadata missing axis_count/interval_ms/input_features_count. Try re-downloading the .eim.")
        st.stop()

    if input_features_count % axis_count != 0:
        st.error(
            f"Model expects {input_features_count} features which is not divisible by axis_count={axis_count}."
        )
        st.stop()

    rows_per_window = input_features_count // axis_count
    window_s = rows_per_window * (interval_ms / 1000.0)
    freq_hz = 1000.0 / interval_ms
 
    if st.session_state.model_info:
        with st.expander("Model info", expanded=False):
            st.json(st.session_state.model_info)
        st.caption(
            f"model parameters: axis_count={axis_count}, interval_ms={interval_ms} (≈{freq_hz:.2f} Hz), "
            f"window={rows_per_window} rows (≈{window_s:.1f}s), input_features_count={input_features_count}"
        )

    # If the user just turned Leak ON, start the leak "now"
    if bool(leak_on) and not bool(st.session_state.leak_on_prev):
        st.session_state.leak_start_s = float(st.session_state.t_s)
    st.session_state.leak_on_prev = bool(leak_on)

    # Main loop tick (one sample per rerun while running)
    if st.session_state.running:
        rng = np.random.default_rng(int(seed) + int(st.session_state.t_s * 1000))
        t_s = float(st.session_state.t_s)

        # -------- Signal simulation (model input source) --------
        #
        # A scalar signal is simulated, then expanded to `axis_count` channels.
        #
        # 1) Baseline: constant level in [0, 1]
        # 2) Drift: slow linear change over time (e.g., aging/temperature effects)
        #       drift(t) = drift_per_s * t
        # 3) Leak: first-order exponential rise after leak_start_s (see _step_leak docstring)
        # 4) Noise: Gaussian noise added each sample
        #
        # Final scalar:
        #   x(t) = clamp01( baseline + drift(t) + leak(t) + noise(t) )
        #
        # Then multi-channel:
        #   channel_i(t) = x(t) + small_channel_jitter
        #
        drift = float(drift_per_s) * t_s if drift_on else 0.0
        leak = (
            _step_leak(t_s, float(st.session_state.leak_start_s), float(leak_amp), float(leak_tau))
            if leak_on
            else 0.0
        )
        noise = rng.normal(0.0, float(noise_sigma))
        base = _clamp01(float(baseline) + drift + leak + float(noise))
        axes_vals = _expand_axes(base, axis_count, rng)

        # -------- Rolling window (model input buffer) --------
        #
        # The `.eim` expects a fixed-length window of raw samples:
        #   rows_per_window = input_features_count / axis_count
        #
        # Example:
        #   axis_count=3, interval_ms=100  => 10 Hz
        #   input_features_count=150       => rows_per_window=50 => 5 seconds
        #
        # Windowing can cause the anomaly score to decay over a short period after
        # Leak/Drift is disabled, because prior samples remain inside the window.
        #
        win: Deque[List[float]] = st.session_state.window
        win.append(axes_vals)
        max_rows = int(rows_per_window)
        while len(win) > max_rows:
            win.popleft()

        # once full, run inference
        score: Optional[float] = None
        if len(win) == max_rows:
            rows = list(win)
            features = _flatten_window(rows)
            try:
                out = st.session_state.runner.classify_window(features)
                score = out.anomaly
            except Exception as e:
                st.error(f"Inference failed: {e}")
                score = None

        if score is not None:
            st.session_state.scores.append((t_s, float(score)))

        st.session_state.t_s = t_s + _dt_s(interval_ms)

    # Display charts
    left, right = st.columns(2)
    with left:
        st.subheader("Latest window (axis 1)")
        if len(st.session_state.window) > 0:
            arr = np.array(list(st.session_state.window), dtype=np.float32)
            st.line_chart(arr[:, 0])
        else:
            st.info("Start to generate samples.")

    with right:
        st.subheader("Anomaly score over time")
        if len(st.session_state.scores) > 0:
            scores_arr = np.array(list(st.session_state.scores), dtype=np.float32)
            st.line_chart(scores_arr[:, 1])
            latest = float(scores_arr[-1, 1])
            st.write(f"Latest anomaly score: **{latest:.3f}**")
            if latest >= float(threshold):
                st.error(f"ALERT: anomaly score ≥ {float(threshold):.2f}")
            else:
                st.success(f"OK: anomaly score < {float(threshold):.2f}")
        else:
            st.info("Anomaly scores appear once the rolling window is full.")

    if st.session_state.running:
        time.sleep(0.10)  # throttle reruns (keeps UI responsive)
        st.rerun()


if __name__ == "__main__":
    main()

