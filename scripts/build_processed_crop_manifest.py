import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create face crop metadata from a unified raw manifest.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="data/processed_metadata.csv")
    parser.add_argument("--output-dir", default="data/raw/processed_crops")
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def detect_face_box(image: Image.Image) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return int(x), int(y), int(x + w), int(y + h)


def expand_box(box: tuple[int, int, int, int], scale: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1) * scale
    nx1 = max(0, int(round(cx - size / 2)))
    ny1 = max(0, int(round(cy - size / 2)))
    nx2 = min(width, int(round(cx + size / 2)))
    ny2 = min(height, int(round(cy + size / 2)))
    return nx1, ny1, nx2, ny2


def infer_metadata(row) -> dict[str, object]:
    source = str(row.get("source", "unknown")).lower()
    fake_type = str(row.get("fake_type", "real" if int(row["label"]) == 0 else source)).lower()
    return {
        "fake_type": fake_type,
        "is_inswapper": int("inswapper" in fake_type or "inswapper" in source),
        "is_gan": int("gan" in fake_type or "gan" in source),
        "boundary_label": int(row.get("boundary_label", row["label"])),
        "quality_label": int(row.get("quality_label", 0)),
        "source": source,
        "video_id": row.get("video_id", row.get("video_path", "")),
        "identity_id": row.get("identity_id", ""),
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    required = {"path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input manifest missing columns: {sorted(missing)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for index, row in frame.iterrows():
        image_path = Path(str(row["path"]))
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            box = detect_face_box(image)
            if box is None:
                failures.append({"path": str(image_path), "reason": "face_not_found"})
                continue
            metadata = infer_metadata(row)
            for crop_name, scale in [("tight", 1.1), ("expanded", 1.5), ("scene", 2.0)]:
                crop_box = expand_box(box, scale, image.width, image.height)
                crop = image.crop(crop_box).resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
                crop_path = output_dir / f"{index:08d}_{crop_name}.jpg"
                crop.save(crop_path, quality=95)
                rows.append(
                    {
                        "path": str(crop_path),
                        "label": int(row["label"]),
                        "crop_type": crop_name,
                        "box_x1": crop_box[0],
                        "box_y1": crop_box[1],
                        "box_x2": crop_box[2],
                        "box_y2": crop_box[3],
                        **metadata,
                    }
                )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["path", "label", "source"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    failure_path = Path(args.output_csv).with_suffix(".failures.csv")
    with open(failure_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "reason"])
        writer.writeheader()
        writer.writerows(failures)


if __name__ == "__main__":
    main()
