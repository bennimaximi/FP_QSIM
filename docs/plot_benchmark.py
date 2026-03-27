from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUT_JSON = ROOT / "_static" / "benchmark_results_q6_24.json"
OUTPUT_RUNTIME_PNG = ROOT / "_static" / "benchmark_runtime_q6_24.png"
OUTPUT_SPEEDUP_PNG = ROOT / "_static" / "benchmark_speedup_vs_aer_q6_24.png"

SERIES_BY_TEST_NAME = {
    "test_benchmark_custom_simulator": "custom",
    "test_benchmark_optimized_python_runtime": "optimized-python",
    "test_benchmark_optimized_numba_runtime": "optimized-numba",
    "test_benchmark_aer_simulator": "aer",
    "test_benchmark_cuda_runtime": "cuda",
}


def _load_runtime_series(path: Path) -> dict[str, dict[int, float]]:
    """Load benchmark median runtimes grouped by simulator label.

    Args:
        path: Path to the pytest-benchmark JSON results file.

    Returns:
        Mapping of simulator label to {n_qubits: median_ms}.

    """
    data = json.loads(path.read_text(encoding="utf-8"))
    series: dict[str, dict[int, float]] = {label: {} for label in SERIES_BY_TEST_NAME.values()}

    for entry in data.get("benchmarks", []):
        name = str(entry.get("name", ""))
        test_name = name.split("[")[0]
        label = SERIES_BY_TEST_NAME.get(test_name)
        if label is None:
            continue

        params = entry.get("params", {})
        n_qubits_raw = params.get("n_qubits")
        if n_qubits_raw is None:
            continue

        n_qubits = int(n_qubits_raw)
        median_seconds = float(entry["stats"]["median"])
        median_ms = 1000.0 * median_seconds
        series[label][n_qubits] = median_ms

    return series


def _plot_runtime(series: dict[str, dict[int, float]], output_path: Path) -> None:
    """Plot all available runtime curves and save the figure.

    Args:
        series: Mapping of simulator labels to runtime data.
        output_path: Destination path for the rendered PNG.

    Returns:
        None.

    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for label, data in series.items():
        if not data:
            continue
        qubits = sorted(data)
        values = [data[q] for q in qubits]
        ax.plot(qubits, values, marker="o", label=label)

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Median runtime [ms]")
    ax.set_title("Simulator Benchmark Runtime (q = 6, 8, 12, 16, 20, 24 where available)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote plot to {output_path}")


def _plot_speedup_vs_aer(series: dict[str, dict[int, float]], output_path: Path) -> None:
    """Plot speedup curves relative to Aer and save the figure.

    Args:
        series: Mapping of simulator labels to runtime data.
        output_path: Destination path for the rendered PNG.

    Returns:
        None.

    """
    aer = series.get("aer", {})
    if not aer:
        raise ValueError("No Aer benchmark entries found in JSON input.")

    candidates = ["custom", "optimized-python", "optimized-numba", "cuda"]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    has_any_curve = False
    for candidate in candidates:
        runtime = series.get(candidate, {})
        common_qubits = sorted(set(aer) & set(runtime))
        if not common_qubits:
            continue

        speedup = [aer[q] / runtime[q] for q in common_qubits]
        ax.plot(common_qubits, speedup, marker="o", label=f"aer/{candidate}")
        has_any_curve = True

    if not has_any_curve:
        raise ValueError("No overlapping qubit points found for speedup plot.")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Speedup factor")
    ax.set_title("Speedup Relative to Aer (>1 means candidate is faster)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote plot to {output_path}")


def main() -> None:
    """Generate and save benchmark runtime plots from pytest JSON output.

    Returns:
        None.

    """
    series = _load_runtime_series(INPUT_JSON)
    if not any(series.values()):
        raise ValueError("No recognized benchmark entries found in JSON input.")

    _plot_runtime(series, OUTPUT_RUNTIME_PNG)
    _plot_speedup_vs_aer(series, OUTPUT_SPEEDUP_PNG)


if __name__ == "__main__":
    main()
