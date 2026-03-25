from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INPUT_JSON = ROOT / '_static' / 'benchmark_results_q4_15.json'
OUTPUT_PNG = ROOT / '_static' / 'benchmark_q4_15.png'


def _load_medians(path: Path) -> tuple[dict[int, float], dict[int, float]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    custom: dict[int, float] = {}
    aer: dict[int, float] = {}

    for entry in data.get('benchmarks', []):
        params = entry.get('params', {})
        n_qubits = int(params.get('n_qubits'))
        median_seconds = float(entry['stats']['median'])
        median_ms = 1000.0 * median_seconds

        name = str(entry.get('name', ''))
        if 'custom' in name:
            custom[n_qubits] = median_ms
        elif 'aer' in name:
            aer[n_qubits] = median_ms

    return custom, aer


def main() -> None:
    custom, aer = _load_medians(INPUT_JSON)
    qubits = sorted(set(custom) & set(aer))

    if not qubits:
        raise ValueError('No matching custom/aer benchmark entries found in JSON input.')

    custom_ms = [custom[q] for q in qubits]
    aer_ms = [aer[q] for q in qubits]
    ratio = [c / a for c, a in zip(custom_ms, aer_ms)]

    fig, (ax_time, ax_ratio) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax_time.plot(qubits, custom_ms, marker='o', label='Custom simulator median')
    ax_time.plot(qubits, aer_ms, marker='o', label='Qiskit Aer median')
    ax_time.set_ylabel('Median runtime [ms]')
    ax_time.set_title('Benchmark: Qiskit Aer vs Custom Simulator (4 to 15 qubits)')
    ax_time.grid(alpha=0.3)
    ax_time.legend()

    ax_ratio.plot(qubits, ratio, marker='o', color='tab:red')
    ax_ratio.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax_ratio.set_xlabel('Number of qubits')
    ax_ratio.set_ylabel('Ratio (custom / aer)')
    ax_ratio.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f'Wrote plot to {OUTPUT_PNG}')


if __name__ == '__main__':
    main()
