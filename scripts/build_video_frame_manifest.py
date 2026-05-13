import argparse
import csv
from pathlib import Path

import pandas as pd

from core.video import sample_scene_aware_frames_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample scene-aware video frames into an image manifest.")
    parser.add_argument("--videos", required=True, help="CSV with path,label,source columns.")
    parser.add_argument("--output-csv", default="data/raw_manifest.csv")
    parser.add_argument("--output-dir", default="data/raw/video_frames")
    parser.add_argument("--frames-per-scene", type=int, default=6)
    parser.add_argument("--scene-threshold", type=float, default=0.55)
    parser.add_argument("--max-scenes", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = pd.read_csv(args.videos)
    required = {"path", "label", "source"}
    missing = required - set(videos.columns)
    if missing:
        raise ValueError(f"Video manifest missing columns: {sorted(missing)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for video_index, row in videos.iterrows():
        video_path = Path(str(row["path"]))
        frames, scene_count = sample_scene_aware_frames_from_path(
            video_path,
            frames_per_scene=args.frames_per_scene,
            scene_threshold=args.scene_threshold,
            max_scenes=args.max_scenes,
        )
        stem = video_path.stem
        for sample_index, frame in enumerate(frames):
            frame_path = output_dir / f"{video_index:06d}_{stem}_s{frame.scene_index:03d}_f{frame.frame_index:06d}.jpg"
            frame.image.save(frame_path, quality=95)
            rows.append(
                {
                    "path": str(frame_path),
                    "label": int(row["label"]),
                    "source": row["source"],
                    "fake_type": row.get("fake_type", row["source"]),
                    "identity_id": row.get("identity_id", ""),
                    "video_id": row.get("video_id", video_path.stem),
                    "video_path": str(video_path),
                    "video_index": video_index,
                    "scene_count": scene_count,
                    "scene_index": frame.scene_index,
                    "frame_index": frame.frame_index,
                    "timestamp_sec": frame.timestamp_sec,
                    "sample_index": sample_index,
                }
            )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
