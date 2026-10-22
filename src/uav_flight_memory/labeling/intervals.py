from __future__ import annotations

from collections.abc import Iterable

Interval = tuple[float, float]


def normalize_interval(start: float, end: float) -> Interval | None:
    start_f = float(start)
    end_f = float(end)
    if end_f <= start_f:
        return None
    return start_f, end_f


def union_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    cleaned = sorted(
        interval
        for interval in (normalize_interval(start, end) for start, end in intervals)
        if interval is not None
    )
    if not cleaned:
        return []

    merged: list[Interval] = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def overlap_duration(intervals: Iterable[Interval], window_start: float, window_end: float) -> float:
    duration = 0.0
    for start, end in intervals:
        duration += max(0.0, min(float(end), window_end) - max(float(start), window_start))
    return duration


def total_duration(intervals: Iterable[Interval]) -> float:
    return sum(float(end) - float(start) for start, end in intervals)

