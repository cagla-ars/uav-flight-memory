from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uav_flight_memory.config import EXCLUDED_CLASSES, RARE_CLASSES, TARGET_CLASSES
from uav_flight_memory.labeling.intervals import Interval, union_intervals


@dataclass(frozen=True)
class Annotation:
    anomaly_class: str
    note: str
    annotation_timestamp: str
    ranges_by_topic: dict[str, list[Interval]]


@dataclass(frozen=True)
class FlightRecord:
    flight_id: str
    annotations: tuple[Annotation, ...]

    @property
    def class_set(self) -> tuple[str, ...]:
        return tuple(sorted({annotation.anomaly_class for annotation in self.annotations}))

    @property
    def is_multi_annotation(self) -> bool:
        return len(self.class_set) > 1


def load_mapping(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_annotation(raw: dict[str, Any]) -> Annotation:
    ranges_by_topic: dict[str, list[Interval]] = {}
    for topic, intervals in raw.get("ranges") or []:
        parsed = []
        for interval in intervals:
            if len(interval) != 2:
                continue
            start, end = interval
            if float(end) > float(start):
                parsed.append((float(start), float(end)))
        ranges_by_topic[str(topic)] = union_intervals(parsed)

    return Annotation(
        anomaly_class=str(raw.get("class", "")),
        note=str(raw.get("note", "")),
        annotation_timestamp=str(raw.get("timestamp", "")),
        ranges_by_topic=ranges_by_topic,
    )


def parse_mapping(raw_mapping: dict[str, Any]) -> dict[str, FlightRecord]:
    records: dict[str, FlightRecord] = {}
    for flight_id, payload in raw_mapping.items():
        annotations = tuple(parse_annotation(annotation) for annotation in payload.get("annotations", []))
        records[str(flight_id)] = FlightRecord(flight_id=str(flight_id), annotations=annotations)
    return records


def class_distribution(records: dict[str, FlightRecord]) -> dict[str, Any]:
    annotation_counts: Counter[str] = Counter()
    range_counts: Counter[str] = Counter()
    flight_class_sets: Counter[tuple[str, ...]] = Counter()
    for record in records.values():
        flight_class_sets[record.class_set] += 1
        for annotation in record.annotations:
            annotation_counts[annotation.anomaly_class] += 1
            range_counts[annotation.anomaly_class] += sum(
                len(intervals) for intervals in annotation.ranges_by_topic.values()
            )

    return {
        "flight_count": len(records),
        "annotation_counts": dict(sorted(annotation_counts.items())),
        "range_counts": dict(sorted(range_counts.items())),
        "flight_class_sets": {
            " + ".join(key) if key else "<none>": value
            for key, value in flight_class_sets.most_common()
        },
        "multi_annotation_flights": [
            record.flight_id for record in records.values() if record.is_multi_annotation
        ],
    }


def intervals_by_class(record: FlightRecord) -> dict[str, list[Interval]]:
    by_class: dict[str, list[Interval]] = defaultdict(list)
    for annotation in record.annotations:
        for intervals in annotation.ranges_by_topic.values():
            by_class[annotation.anomaly_class].extend(intervals)
    return {klass: union_intervals(intervals) for klass, intervals in by_class.items()}


def flight_selection_rows(
    records: dict[str, FlightRecord],
    seed: int,
    normal_limit: int,
    external_position_limit: int,
    include_multi_annotation: bool = True,
) -> list[dict[str, Any]]:
    import random

    rng = random.Random(seed)
    single_by_class: dict[str, list[FlightRecord]] = defaultdict(list)
    multi_records: list[FlightRecord] = []

    for record in records.values():
        class_set = record.class_set
        if not class_set or any(klass in EXCLUDED_CLASSES for klass in class_set):
            continue
        if any(klass not in TARGET_CLASSES for klass in class_set):
            continue
        if record.is_multi_annotation:
            if include_multi_annotation:
                multi_records.append(record)
            continue
        single_by_class[class_set[0]].append(record)

    selected: list[FlightRecord] = []
    for klass, flights in single_by_class.items():
        flights = sorted(flights, key=lambda item: item.flight_id)
        rng.shuffle(flights)
        if klass == "Normal":
            selected.extend(flights[:normal_limit])
        elif klass == "External Position":
            selected.extend(flights[:external_position_limit])
        elif klass in RARE_CLASSES:
            selected.extend(flights)
        else:
            selected.extend(flights)

    selected.extend(sorted(multi_records, key=lambda item: item.flight_id))
    selected = sorted({record.flight_id: record for record in selected}.values(), key=lambda item: item.flight_id)

    rows = []
    for record in selected:
        rows.append(
            {
                "flight_id": record.flight_id,
                "class_set": "|".join(record.class_set),
                "selection_group": "multi" if record.is_multi_annotation else record.class_set[0],
                "multi_annotation": record.is_multi_annotation,
                "annotation_count": len(record.annotations),
            }
        )
    return rows

