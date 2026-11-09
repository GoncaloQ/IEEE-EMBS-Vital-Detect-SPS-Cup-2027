import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    method: str = "green"
    roi: str = "cheeks"
    model_path: str = "models/face_landmarker.task"
    low_hz: float = 0.7
    high_hz: float = 3.0
    filter_order: int = 3
    min_duration_s: float = 10.0
    window_s: float = 15.0
    stride_s: float = 5.0
    window_enabled: bool = True
    max_gap_s: float = 0.25
    min_valid_fraction: float = 0.9
    min_band_power_fraction: float = 0.1
    min_peak_power_fraction: float = 0.35
    pos_window_s: float = 1.6
    smoothing_alpha: float = 0.6
    reference_max_gap_s: float = 1.0
    sync_policy: str = "shared-clock"
    reference_offset_s: float = 0.0
    offset_provenance: str = ""

    def __post_init__(self):
        if not isinstance(self.window_enabled, bool):
            raise ValueError("window_enabled must be a boolean")
        if self.method not in ("green", "pos"):
            raise ValueError("method must be green or pos")
        if self.roi not in ("forehead", "left_cheek", "right_cheek", "cheeks"):
            raise ValueError("Unsupported facial ROI")
        positive = (
            self.low_hz,
            self.high_hz,
            self.min_duration_s,
            self.window_s,
            self.stride_s,
            self.max_gap_s,
            self.pos_window_s,
            self.reference_max_gap_s,
        )
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Durations, gaps and frequencies must be finite and positive")
        if self.low_hz >= self.high_hz:
            raise ValueError("low_hz must be below high_hz")
        if not isinstance(self.filter_order, int) or not 1 <= self.filter_order <= 10:
            raise ValueError("filter_order must be an integer in 1..10")
        if self.window_enabled and self.window_s < self.min_duration_s:
            raise ValueError("window_s must be at least min_duration_s")
        for value in (
            self.min_valid_fraction,
            self.min_band_power_fraction,
            self.min_peak_power_fraction,
            self.smoothing_alpha,
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Fractions and smoothing_alpha must be in (0, 1]")
        if self.sync_policy not in ("shared-clock", "recording-start"):
            raise ValueError("Unknown synchronization policy")
        if not math.isfinite(self.reference_offset_s):
            raise ValueError("reference_offset_s must be finite")
        if self.reference_offset_s and not self.offset_provenance.strip():
            raise ValueError("A nonzero reference offset requires independent offset_provenance")

    def to_dict(self):
        return asdict(self)
