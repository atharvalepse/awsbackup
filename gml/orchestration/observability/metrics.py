"""In-process metrics registry with Prometheus text-format exposition."""
import math
import threading
from collections import defaultdict
from typing import Iterable


_LABEL_VALUE_ESCAPES = (
    ("\\", "\\\\"),
    ("\n", "\\n"),
    ('"', '\\"'),
)


def _escape_label_value(v: str) -> str:
    out = str(v)
    for ch, repl in _LABEL_VALUE_ESCAPES:
        out = out.replace(ch, repl)
    return out


def _format_labels(label_names: list[str], label_values: tuple) -> str:
    if not label_names:
        return ""
    pairs = [
        f'{k}="{_escape_label_value(v)}"'
        for k, v in zip(label_names, label_values)
    ]
    return "{" + ",".join(pairs) + "}"


def _format_labels_with_extra(
    label_names: list[str], label_values: tuple, extra_key: str, extra_value: str
) -> str:
    pairs = [
        f'{k}="{_escape_label_value(v)}"'
        for k, v in zip(label_names, label_values)
    ]
    pairs.append(f'{extra_key}="{_escape_label_value(extra_value)}"')
    return "{" + ",".join(pairs) + "}"


class Counter:
    def __init__(
        self,
        name: str,
        help_text: str,
        label_names: list[str] | None = None,
    ) -> None:
        self.name = name
        self.help_text = help_text
        self.label_names: list[str] = list(label_names) if label_names else []
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def _validate(self, labels: dict[str, str]) -> tuple:
        if set(labels.keys()) != set(self.label_names):
            raise ValueError(
                f"Counter {self.name!r} expects labels {self.label_names}, "
                f"got {sorted(labels.keys())}"
            )
        return tuple(labels[k] for k in self.label_names)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = self._validate(labels)
        with self._lock:
            self._values[key] += amount

    def get(self, **labels: str) -> float:
        key = self._validate(labels)
        return self._values.get(key, 0.0)

    def values(self) -> dict[tuple, float]:
        return dict(self._values)


class Histogram:
    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: list[float],
        label_names: list[str] | None = None,
    ) -> None:
        if list(buckets) != sorted(buckets):
            raise ValueError(f"Histogram {name!r} buckets must be sorted ascending")
        self.name = name
        self.help_text = help_text
        self.upper_bounds: list[float] = list(buckets) + [math.inf]
        self.label_names: list[str] = list(label_names) if label_names else []
        self._bucket_counts: dict[tuple, list[int]] = defaultdict(
            lambda: [0] * len(self.upper_bounds)
        )
        self._sums: dict[tuple, float] = defaultdict(float)
        self._counts: dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def _validate(self, labels: dict[str, str]) -> tuple:
        if set(labels.keys()) != set(self.label_names):
            raise ValueError(
                f"Histogram {self.name!r} expects labels {self.label_names}, "
                f"got {sorted(labels.keys())}"
            )
        return tuple(labels[k] for k in self.label_names)

    def observe(self, value: float, **labels: str) -> None:
        key = self._validate(labels)
        with self._lock:
            counts = self._bucket_counts[key]
            for i, ub in enumerate(self.upper_bounds):
                if value <= ub:
                    counts[i] += 1
            self._sums[key] += value
            self._counts[key] += 1

    def get_buckets(self, **labels: str) -> list[tuple[float, int]]:
        key = self._validate(labels)
        counts = self._bucket_counts.get(key, [0] * len(self.upper_bounds))
        return list(zip(self.upper_bounds, counts))

    def get_sum(self, **labels: str) -> float:
        key = self._validate(labels)
        return self._sums.get(key, 0.0)

    def get_count(self, **labels: str) -> int:
        key = self._validate(labels)
        return self._counts.get(key, 0)

    def label_keys(self) -> Iterable[tuple]:
        return self._counts.keys()


class CounterRegistry:
    _instance: "CounterRegistry | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}

    @classmethod
    def get(cls) -> "CounterRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_counter(self, counter: Counter) -> None:
        self._counters[counter.name] = counter

    def register_histogram(self, histogram: Histogram) -> None:
        self._histograms[histogram.name] = histogram

    def get_counter(self, name: str) -> Counter:
        return self._counters[name]

    def get_histogram(self, name: str) -> Histogram:
        return self._histograms[name]

    def render_prometheus(self) -> str:
        out: list[str] = []
        for c in self._counters.values():
            out.append(f"# HELP {c.name} {c.help_text}")
            out.append(f"# TYPE {c.name} counter")
            vals = c.values()
            if not vals:
                if not c.label_names:
                    out.append(f"{c.name} 0")
            else:
                for label_values, value in vals.items():
                    label_str = _format_labels(c.label_names, label_values)
                    out.append(f"{c.name}{label_str} {value}")
            out.append("")
        for h in self._histograms.values():
            out.append(f"# HELP {h.name} {h.help_text}")
            out.append(f"# TYPE {h.name} histogram")
            label_keys = list(h.label_keys())
            if not label_keys and not h.label_names:
                label_keys = [tuple()]
            for label_values in label_keys:
                buckets = h.get_buckets(**dict(zip(h.label_names, label_values)))
                for ub, count in buckets:
                    le = "+Inf" if math.isinf(ub) else _format_number(ub)
                    label_str = _format_labels_with_extra(h.label_names, label_values, "le", le)
                    out.append(f"{h.name}_bucket{label_str} {count}")
                base_label_str = _format_labels(h.label_names, label_values)
                sum_value = h.get_sum(**dict(zip(h.label_names, label_values)))
                count_value = h.get_count(**dict(zip(h.label_names, label_values)))
                out.append(f"{h.name}_sum{base_label_str} {sum_value}")
                out.append(f"{h.name}_count{base_label_str} {count_value}")
            out.append("")
        return "\n".join(out) + "\n"


def _format_number(n: float) -> str:
    if n == int(n):
        return str(int(n))
    return f"{n:g}"


# ---------------------------------------------------------------------------
# Module-level metric instances (registered at import time)
# ---------------------------------------------------------------------------

ORCHESTRATION_REQUESTS_TOTAL = Counter(
    name="orchestration_requests_total",
    help_text="Total orchestration requests, labeled by target family",
    label_names=["target_family"],
)

ORCHESTRATION_LATENCY_SECONDS = Histogram(
    name="orchestration_latency_seconds",
    help_text="End-to-end orchestration latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    label_names=["target_family"],
)

ORCHESTRATION_RETRIEVAL_LATENCY_SECONDS = Histogram(
    name="orchestration_retrieval_latency_seconds",
    help_text="Retriever latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

ORCHESTRATION_BUDGET_UTILIZATION_RATIO = Histogram(
    name="orchestration_budget_utilization_ratio",
    help_text="Ratio of memory budget actually used to memory budget available",
    buckets=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
)

ORCHESTRATION_CONFLICTS_DETECTED_TOTAL = Counter(
    name="orchestration_conflicts_detected_total",
    help_text="Total memory conflicts detected by SAM",
)

ORCHESTRATION_FAILURES_TOTAL = Counter(
    name="orchestration_failures_total",
    help_text="Failures per pipeline stage",
    label_names=["stage"],
)


def _register_all() -> None:
    reg = CounterRegistry.get()
    reg.register_counter(ORCHESTRATION_REQUESTS_TOTAL)
    reg.register_histogram(ORCHESTRATION_LATENCY_SECONDS)
    reg.register_histogram(ORCHESTRATION_RETRIEVAL_LATENCY_SECONDS)
    reg.register_histogram(ORCHESTRATION_BUDGET_UTILIZATION_RATIO)
    reg.register_counter(ORCHESTRATION_CONFLICTS_DETECTED_TOTAL)
    reg.register_counter(ORCHESTRATION_FAILURES_TOTAL)


_register_all()
