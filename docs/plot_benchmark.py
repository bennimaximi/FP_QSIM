from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INPUT_JSON = ROOT / "_static" / "benchmark_results_q5_16.json"
OUTPUT_RUNTIME_PNG = ROOT / "_static" / "benchmark_q5_16.png"
OUTPUT_CX_HEAVY_PNG = ROOT / "_static" / "benchmark_cx_heavy_q5_16.png"


def _load_medians_by_name(path: Path, lhs_token: str, rhs_token: str) -> tuple[dict[int, float], dict[int, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lhs: dict[int, float] = {}
    rhs: dict[int, float] = {}

    for entry in data.get("benchmarks", []):
        params = entry.get("params", {})
        n_qubits = int(params.get("n_qubits"))
        median_seconds = float(entry["stats"]["median"])
        median_ms = 1000.0 * median_seconds

        name = str(entry.get("name", ""))
        if lhs_token in name:
            lhs[n_qubits] = median_ms
        elif rhs_token in name:
            rhs[n_qubits] = median_ms

    return lhs, rhs


def _validate_qubit_range(qubits: Iterable[int]) -> None:
    expected = set(range(5, 17))
    observed = set(qubits)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Unexpected qubit range. Missing={missing}, extra={extra}")


def _plot_pair(
    qubits: list[int],
    lhs_ms: list[float],
    rhs_ms: list[float],
    ratio: list[float],
    lhs_label: str,
    rhs_label: str,
    ratio_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, (ax_time, ax_ratio) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax_time.plot(qubits, lhs_ms, marker="o", label=lhs_label)
    ax_time.plot(qubits, rhs_ms, marker="o", label=rhs_label)
    ax_time.set_ylabel("Median runtime [ms]")
    ax_time.set_title(title)
    ax_time.grid(alpha=0.3)
    ax_time.legend()

    ax_ratio.plot(qubits, ratio, marker="o", color="tab:red")
    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax_ratio.set_xlabel("Number of qubits")
    ax_ratio.set_ylabel(ratio_label)
    ax_ratio.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote plot to {output_path}")


def main() -> None:
    custom, aer = _load_medians_by_name(
        INPUT_JSON,
        lhs_token="test_benchmark_custom_simulator",
        rhs_token="test_benchmark_aer_simulator",
    )
    qubits = sorted(set(custom) & set(aer))

    if not qubits:
        raise ValueError("No matching simulator-runtime benchmark entries found in JSON input.")

    _validate_qubit_range(qubits)

    custom_ms = [custom[q] for q in qubits]
    aer_ms = [aer[q] for q in qubits]
    ratio = [c / a for c, a in zip(custom_ms, aer_ms)]

    _plot_pair(
        qubits=qubits,
        lhs_ms=custom_ms,
        rhs_ms=aer_ms,
        ratio=ratio,
        lhs_label="Custom simulator median",
        rhs_label="Qiskit Aer median",
        ratio_label="Ratio (custom / aer)",
        title="Benchmark: Qiskit Aer vs Custom Simulator (5 to 16 qubits)",
        output_path=OUTPUT_RUNTIME_PNG,
    )

    baseline, optimized = _load_medians_by_name(
        INPUT_JSON,
        lhs_token="test_benchmark_cx_heavy_baseline_manual",
        rhs_token="test_benchmark_cx_heavy_optimized",
    )
    qubits_cx = sorted(set(baseline) & set(optimized))

    if not qubits_cx:
        raise ValueError("No matching cx-heavy benchmark entries found in JSON input.")

    _validate_qubit_range(qubits_cx)

    baseline_ms = [baseline[q] for q in qubits_cx]
    optimized_ms = [optimized[q] for q in qubits_cx]
    speedup = [b / o for b, o in zip(baseline_ms, optimized_ms)]

    _plot_pair(
        qubits=qubits_cx,
        lhs_ms=baseline_ms,
        rhs_ms=optimized_ms,
        ratio=speedup,
        lhs_label="Manual baseline median",
        rhs_label="Manual optimized median",
        ratio_label="Speedup (baseline / optimized)",
        title="Benchmark: CX-heavy Baseline vs Optimized (5 to 16 qubits)",
        output_path=OUTPUT_CX_HEAVY_PNG,
    )


if __name__ == "__main__":
    main()
