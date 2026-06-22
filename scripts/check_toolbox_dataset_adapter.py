from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.data.toolbox_adapter import (
    toolbox_beta,
    toolbox_dataset_info,
    toolbox_wearable_dry,
    toolbox_wearable_wet,
)


def parse_ints(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        elif part:
            values.append(int(part))
    return values


def materialize_dataset(name: str, root: Path):
    if name == "beta":
        return toolbox_beta(root=root)
    if name == "wearable_wet":
        return toolbox_wearable_wet(root=root)
    if name == "wearable_dry":
        return toolbox_wearable_dry(root=root)
    raise ValueError(f"Unknown dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["beta", "wearable_wet", "wearable_dry"], required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--materialize", action="store_true", help="Instantiate toolbox dataset; this can download the full dataset.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--blocks", default="1")
    parser.add_argument("--targets", default="0")
    parser.add_argument("--channels", default="occipital_9ch")
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--filter-window", type=float)
    parser.add_argument("--extra-samples", type=int, default=0)
    parser.add_argument("--preprocess", default="toolbox_fb", choices=["raw", "toolbox_preprocess", "preprocess", "toolbox_fb", "filterbank"])
    parser.add_argument("--n-bands", type=int, default=5)
    args = parser.parse_args()

    info = toolbox_dataset_info(args.dataset, root=args.root)
    print(
        f"{info.id}: subjects={len(info.subjects)} blocks={len(info.blocks)} "
        f"targets={len(info.targets)} channels={len(info.channels)} fs={info.sampling_rate} root={info.root}",
        flush=True,
    )
    print(f"freqs={info.frequencies[:5]}... phases_pi={info.phases[:5]}...", flush=True)

    if not args.materialize:
        print("metadata_only=true", flush=True)
        return

    dataset = materialize_dataset(args.dataset, root=info.root)
    batch = dataset.get_trials(
        subject=args.subject,
        blocks=parse_ints(args.blocks),
        targets=parse_ints(args.targets),
        channels=args.channels,
        window=args.window,
        preprocess=args.preprocess,
        n_bands=args.n_bands,
        filter_window=args.filter_window,
        extra_samples=args.extra_samples,
    )
    print(
        f"batch x={batch.x.shape} y={batch.y.tolist()} channels={batch.info.channels} "
        f"preprocess={batch.preprocess}",
        flush=True,
    )


if __name__ == "__main__":
    main()
