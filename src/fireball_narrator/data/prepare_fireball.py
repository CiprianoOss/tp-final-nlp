from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from fireball_narrator.data.formatting import record_to_chat


def discover_remote_files(repo_id: str, revision: str) -> list[str]:
    files = HfApi().list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
    )
    return sorted(
        path
        for path in files
        if path.startswith("filtered/") and path.endswith(".jsonl")
    )


def discover_local_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        files = sorted(data_dir.glob("filtered/*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {data_dir}")
    return files


def split_sessions(
    files: list[str] | list[Path],
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, list[str] | list[Path]]:
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    validation_count = round(total * validation_fraction)
    test_count = round(total * test_fraction)
    train_count = total - validation_count - test_count
    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + validation_count],
        "test": shuffled[train_count + validation_count :],
    }


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def materialize_file(
    source: str | Path,
    repo_id: str,
    revision: str,
) -> Path:
    if isinstance(source, Path):
        return source
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=source,
            repo_type="dataset",
            revision=revision,
        )
    )


def prepare_split(
    name: str,
    sources: list[str] | list[Path],
    output_path: Path,
    repo_id: str,
    revision: str,
    max_samples: int | None,
) -> Counter:
    stats: Counter = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output:
        for source in tqdm(sources, desc=f"Preparing {name}", unit="session"):
            path = materialize_file(source, repo_id, revision)
            session_id = Path(source).stem
            stats["sessions"] += 1

            for record in iter_jsonl(path):
                stats["records_seen"] += 1
                example = record_to_chat(record)
                if example is None:
                    stats["records_skipped"] += 1
                    continue

                example["metadata"]["session_id"] = session_id
                output.write(json.dumps(example, ensure_ascii=False) + "\n")
                stats["examples_written"] += 1

                if max_samples and stats["examples_written"] >= max_samples:
                    return stats
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FIREBALL state-to-narration triples into chat SFT JSONL. "
            "Files are split by session to prevent train/eval leakage."
        )
    )
    parser.add_argument("--repo-id", default="lara-martin/FIREBALL")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--local-data-dir",
        type=Path,
        help="Optional local directory containing FIREBALL filtered JSONL files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_data_dir:
        files = discover_local_files(args.local_data_dir)
    else:
        files = discover_remote_files(args.repo_id, args.revision)
    if len(files) < 3:
        raise RuntimeError("At least three session files are required for splitting")

    splits = split_sessions(
        files,
        args.seed,
        args.validation_fraction,
        args.test_fraction,
    )
    limits = {
        "train": args.max_train_samples or None,
        "validation": args.max_validation_samples or None,
        "test": args.max_test_samples or None,
    }

    all_stats = {}
    for name, sources in splits.items():
        all_stats[name] = dict(
            prepare_split(
                name=name,
                sources=sources,
                output_path=args.output_dir / f"{name}.jsonl",
                repo_id=args.repo_id,
                revision=args.revision,
                max_samples=limits[name],
            )
        )

    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "seed": args.seed,
        "split_by": "session_file",
        "session_counts": {name: len(value) for name, value in splits.items()},
        "stats": all_stats,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

