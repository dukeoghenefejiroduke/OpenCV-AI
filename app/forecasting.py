"""Predictive Corrosion Forecasting Module (app/forecasting.py).

Provides deterministic degradation rate modeling and time-to-failure projections
based on historical inspection timestamps and corrosion percentage velocity.
Enforces rigorous data sufficiency checks (requires at least 2 valid historical observations).
"""

from typing import List, Tuple, Optional, Any
import math


def calculate_degradation_velocity(history: List[Tuple[float, float]]) -> float:
    """Calculates corrosion increase rate (% per day) from historical data points (timestamp_days, corrosion_pct).

    Enforces strict data sufficiency and validity checks:
    - Requires at least 2 valid observations.
    - Filters out non-finite, negative, or duplicate timestamps.
    - Requires positive time span (dt > 0).

    Args:
        history: List of tuples (time_in_days, corrosion_percentage).

    Returns:
        Corrosion growth rate percentage per day. Returns 0.0 if insufficient or invalid data.
    """
    if not isinstance(history, list) or len(history) < 2:
        return 0.0

    valid_points = []
    seen_times = set()

    for item in history:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        t, c = item
        try:
            ft = float(t)
            fc = float(c)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(ft) or not math.isfinite(fc):
            continue
        if ft in seen_times:
            continue # Skip duplicate timestamps
        
        seen_times.add(ft)
        valid_points.append((ft, fc))

    if len(valid_points) < 2:
        return 0.0

    # Sort by time
    sorted_history = sorted(valid_points, key=lambda x: x[0])
    t0, c0 = sorted_history[0]
    t1, c1 = sorted_history[-1]

    dt = t1 - t0
    if dt <= 0:
        return 0.0

    dc = c1 - c0
    rate = dc / dt
    return rate if math.isfinite(rate) else 0.0


def project_time_to_critical(current_corrosion: float, daily_rate: float, critical_threshold: float = 10.0) -> Optional[float]:
    """Projects estimated days until corrosion reaches the CRITICAL threshold.

    Args:
        current_corrosion: Current corrosion percentage.
        daily_rate: Corrosion growth rate (% per day).
        critical_threshold: Threshold percentage triggering critical state (default 10.0%).

    Returns:
        Estimated days to critical, or None / float('inf') if rate <= 0 or already critical.
    """
    try:
        fc = float(current_corrosion)
        fr = float(daily_rate)
        fthresh = float(critical_threshold)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(fc) or not math.isfinite(fr) or not math.isfinite(fthresh):
        return None

    if fc >= fthresh:
        return 0.0
    if fr <= 0:
        return None

    remaining = fthresh - fc
    days = remaining / fr
    return round(days, 1) if math.isfinite(days) else None
