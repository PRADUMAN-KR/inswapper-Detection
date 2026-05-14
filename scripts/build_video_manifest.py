import argparse
import csv
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def iter_videos(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def normalize_fake_stem(stem: str) -> str:
    for prefix in ("inswapper_", "uniface_"):
        if stem.startswith(prefix):
            stem = stem.removeprefix(prefix)
    parts = stem.split("_")
    if parts and parts[-1] in {"man", "woman"}:
        parts = parts[:-1]
    if parts and "-" in parts[-1]:
        parts = parts[:-1]
    return "_".join(parts)


def video_id_for(path: Path, source: str) -> str:
    if source in {"inswapper", "uniface"}:
        return normalize_fake_stem(path.stem)
    return path.stem


def add_rows(
    rows: list[dict[str, object]],
    directory: Path,
    label: int,
    source: str,
    fake_type: str,
    is_inswapper: int,
) -> None:
    for path in iter_videos(directory):
        video_id = video_id_for(path, source)
        rows.append(
            {
                "path": str(path),
                "label": label,
                "source": source,
                "fake_type": fake_type,
                "is_inswapper": is_inswapper,
                "boundary_label": label,
                "quality_label": 0,
                "video_id": video_id,
                "identity_id": video_id,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build video manifest for the INSwapper-original-only dataset layout.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-csv", default="data/video_manifest.csv")
    parser.add_argument("--exclude-uniface", action="store_true", help="Exclude UniFace fake videos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    rows: list[dict[str, object]] = []

    add_rows(
        rows,
        data_root / "inswapper" / "original_videos",
        label=0,
        source="inswapper_original",
        fake_type="real",
        is_inswapper=0,
    )

    add_rows(
        rows,
        data_root / "inswapper" / "inswapper",
        label=1,
        source="inswapper",
        fake_type="inswapper",
        is_inswapper=1,
    )
    if not args.exclude_uniface:
        add_rows(
            rows,
            data_root / "inswapper" / "uniface",
            label=1,
            source="uniface",
            fake_type="uniface",
            is_inswapper=1,
        )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "path",
            "label",
            "source",
            "fake_type",
            "is_inswapper",
            "boundary_label",
            "quality_label",
            "video_id",
            "identity_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[tuple[int, str], int] = {}
    for row in rows:
        key = (int(row["label"]), str(row["source"]))
        counts[key] = counts.get(key, 0) + 1
    for (label, source), count in sorted(counts.items()):
        print(f"label={label} source={source} count={count}")
    real_count = sum(1 for row in rows if int(row["label"]) == 0)
    fake_count = sum(1 for row in rows if int(row["label"]) == 1)
    print(f"real={real_count} fake={fake_count} fake_to_real_ratio={fake_count / max(real_count, 1):.2f}")
    print(f"wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
