from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EIResult:
    raw: Dict[str, Any]
    anomaly: Optional[float]

 
class EIRunner:
    """
    Thin wrapper around Edge Impulse Linux Python SDK.

    Uses a local `.eim` file (Linux EIM executable) and runs inference via IPC.
    Docs: https://docs.edgeimpulse.com/tools/libraries/sdks/inference/linux/python
    """

    def __init__(self, model_path: str, *, debug: bool = False, timeout_s: int = 30):
        self.model_path = model_path
        self.debug = debug
        self.timeout_s = timeout_s
        self._runner = None
        self.model_info: Optional[Dict[str, Any]] = None

    def start(self) -> Dict[str, Any]:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.access(self.model_path, os.X_OK):
            raise PermissionError(
                f'Model file is not executable: {self.model_path}\n'
                f'Run: chmod +x "{self.model_path}"'
            )

        from edge_impulse_linux.runner import ImpulseRunner  # type: ignore

        self._runner = ImpulseRunner(self.model_path, timeout=self.timeout_s)
        self.model_info = self._runner.init(debug=self.debug)
        return self.model_info

    def stop(self) -> None:
        if self._runner is not None:
            self._runner.stop()
            self._runner = None

    def classify_window(self, features: list[float]) -> EIResult:
        if self._runner is None:
            raise RuntimeError("Runner not started. Call start() first.")

        res: Dict[str, Any] = self._runner.classify(features)

        # Most anomaly impulses return `result.anomaly` (float).
        anomaly = None
        try:
            anomaly_val = res.get("result", {}).get("anomaly", None)
            if isinstance(anomaly_val, (int, float)):
                anomaly = float(anomaly_val)
        except Exception:
            anomaly = None

        return EIResult(raw=res, anomaly=anomaly)