"""Create a LOCAL source hash inventory. Coverage is explicitly caller-declared."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vep_arena.run_contract import source_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("files", nargs="+", help="Paths relative to root; include every consumed metadata file")
    parser.add_argument("--output", type=Path, required=True, help="New local JSON file; not a public upload")
    args = parser.parse_args()
    inventory = source_inventory(args.root, args.files)
    text = json.dumps(inventory, indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(args.output)


if __name__ == "__main__":
    main()
