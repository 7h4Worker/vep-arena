"""Quick validation: Spectral-TDCA with intermodulation harmonics at 1.0s window."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.embc_jbhi import (
    JBHI35_HISTORICAL5_SUBJECT_IDS,
    jbhi_receiver_filter_band,
    load_jbhi35_historical5_subject,
    load_target_frequency_pairs,
    resolve_jbhi35_historical5_package,
)
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.tdca import TDCA
from vep_arena.methods.traditional import filterbank_weights


def _spectral_references_baseline(
    frequency_pairs: tuple[tuple[float, float], ...],
    samples: int,
    sampling_rate: int,
    harmonics: int,
) -> list[np.ndarray]:
    """Original: independent harmonics only."""
    times = np.arange(samples, dtype=np.float64) / sampling_rate
    references = []
    for pair in frequency_pairs:
        active = tuple(dict.fromkeys(f for f in pair if f > 0))
        rows = []
        for h in range(1, harmonics + 1):
            for f in active:
                rows.append(np.sin(2.0 * np.pi * h * f * times))
                rows.append(np.cos(2.0 * np.pi * h * f * times))
        references.append(np.stack(rows))
    return references


def _spectral_references_intermod(
    frequency_pairs: tuple[tuple[float, float], ...],
    samples: int,
    sampling_rate: int,
    harmonics: int,
    intermod_order: int = 2,
) -> list[np.ndarray]:
    """Extended: independent harmonics + intermodulation components."""
    times = np.arange(samples, dtype=np.float64) / sampling_rate
    nyquist = sampling_rate / 2.0
    references = []
    for pair in frequency_pairs:
        active = tuple(dict.fromkeys(f for f in pair if f > 0))
        rows = []
        # Independent harmonics (same as baseline)
        for h in range(1, harmonics + 1):
            for f in active:
                rows.append(np.sin(2.0 * np.pi * h * f * times))
                rows.append(np.cos(2.0 * np.pi * h * f * times))
        # Intermodulation terms (only if dual-frequency target)
        if len(active) == 2:
            f1, f2 = active
            im_freqs = set()
            for m in range(1, intermod_order + 1):
                for n in range(1, intermod_order + 1):
                    if m + n > intermod_order + 1:
                        continue
                    candidates = [
                        m * f1 + n * f2,
                        m * f1 - n * f2,
                        m * f2 - n * f1,
                    ]
                    for freq in candidates:
                        if 1.0 < freq < nyquist:
                            im_freqs.add(round(freq, 4))
            # Remove frequencies already covered by independent harmonics
            indep = set()
            for h in range(1, harmonics + 1):
                for f in active:
                    indep.add(round(h * f, 4))
            im_freqs -= indep
            for freq in sorted(im_freqs):
                rows.append(np.sin(2.0 * np.pi * freq * times))
                rows.append(np.cos(2.0 * np.pi * freq * times))
        references.append(np.stack(rows))
    return references


def run_comparison(window_seconds: float = 1.0) -> None:
    config_path = TASK_ROOT / "five_subject_reproduction_config_20260728.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frequency_pairs = load_target_frequency_pairs("jbhi35_historical5")
    package = resolve_jbhi35_historical5_package()

    bands = int(config["tdca_filter_bands"])
    delay = int(config["tdca_n_delay"])
    components = int(config["tdca_n_components"])
    harmonics = int(config["tdca_harmonics"])
    sampling_rate = 1000
    samples = int(round(window_seconds * sampling_rate))

    refs_baseline = _spectral_references_baseline(frequency_pairs, samples, sampling_rate, harmonics)
    refs_intermod2 = _spectral_references_intermod(frequency_pairs, samples, sampling_rate, harmonics, intermod_order=2)
    refs_intermod3 = _spectral_references_intermod(frequency_pairs, samples, sampling_rate, harmonics, intermod_order=3)

    print(f"Window: {window_seconds}s ({samples} samples)")
    print(f"Baseline ref rows (target 0): {refs_baseline[0].shape[0]}")
    print(f"Intermod-2 ref rows (target 0): {refs_intermod2[0].shape[0]}")
    print(f"Intermod-3 ref rows (target 0): {refs_intermod3[0].shape[0]}")

    # Count dual-freq targets
    dual = sum(1 for p in frequency_pairs if p[0] > 0 and p[1] > 0)
    print(f"Dual-frequency targets: {dual}/35, single-frequency: {35 - dual}/35")
    print()

    variants = {
        "SPECTRAL_TDCA_baseline": refs_baseline,
        "SPECTRAL_TDCA_intermod2": refs_intermod2,
        "SPECTRAL_TDCA_intermod3": refs_intermod3,
    }

    results = {v: [] for v in variants}
    labels = np.arange(35, dtype=np.int64)
    weights = filterbank_weights(bands)

    for subject_record in package.subjects:
        subject_id = subject_record.subject_id
        subject = load_jbhi35_historical5_subject(package, subject_id)
        raw = subject.x
        filtered = np.stack([jbhi_receiver_filter_band(raw, sampling_rate, band) for band in range(bands)], axis=2)
        needed = samples + delay
        epochs = filtered[..., :min(needed, filtered.shape[-1])]
        if epochs.shape[-1] < needed:
            epochs = np.pad(epochs, [(0, 0), (0, 0), (0, 0), (0, 0), (0, needed - epochs.shape[-1])])

        for variant_name, refs in variants.items():
            t0 = time.perf_counter()
            correct_total = 0
            pred_total = 0
            for test_block in range(6):
                train_blocks = [b for b in range(6) if b != test_block]
                train_x = epochs[:, train_blocks].reshape(35 * 5, bands, 9, needed)
                train_y = np.repeat(labels, 5)
                test_x = epochs[:, test_block]

                model = TDCA(n_components=components, n_delay=delay, fb_weights=weights)
                model.fit(train_x, train_y, refs)
                predicted, _ = model.predict(test_x)
                correct_total += int(np.sum(predicted == labels))
                pred_total += 35

            accuracy = correct_total / pred_total
            itr = itr_bits_per_minute(accuracy, 35, window_seconds + 0.5)
            elapsed = time.perf_counter() - t0
            results[variant_name].append({"subject": subject_id, "accuracy": accuracy, "itr": itr})
            print(f"  {subject_id} {variant_name}: acc={accuracy:.1%}, ITR={itr:.1f} bpm ({elapsed:.1f}s)")

    print("\n" + "=" * 70)
    print(f"Summary at {window_seconds}s window:")
    print(f"{'Variant':<30s} {'Acc (mean±sem)':<20s} {'ITR (mean±sem)':<20s}")
    print("-" * 70)
    for variant_name in variants:
        accs = np.array([r["accuracy"] for r in results[variant_name]])
        itrs = np.array([r["itr"] for r in results[variant_name]])
        acc_mean, acc_sem = accs.mean(), accs.std(ddof=1) / np.sqrt(len(accs))
        itr_mean, itr_sem = itrs.mean(), itrs.std(ddof=1) / np.sqrt(len(itrs))
        print(f"  {variant_name:<28s} {acc_mean:.1%} ± {acc_sem:.1%}       {itr_mean:.1f} ± {itr_sem:.1f} bpm")

    # Per-subject delta
    print("\nPer-subject accuracy delta (intermod2 - baseline):")
    for i, subj in enumerate(JBHI35_HISTORICAL5_SUBJECT_IDS):
        base = results["SPECTRAL_TDCA_baseline"][i]["accuracy"]
        im2 = results["SPECTRAL_TDCA_intermod2"][i]["accuracy"]
        im3 = results["SPECTRAL_TDCA_intermod3"][i]["accuracy"]
        print(f"  {subj}: baseline={base:.1%}  intermod2={im2:.1%} ({im2-base:+.1%})  intermod3={im3:.1%} ({im3-base:+.1%})")


if __name__ == "__main__":
    run_comparison(window_seconds=1.0)
