import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.inference import DetectorService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scene-aware video detection.")
    parser.add_argument("video")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames-per-scene", type=int, default=6)
    parser.add_argument("--scene-threshold", type=float, default=0.55)
    parser.add_argument("--max-scenes", type=int, default=12)
    args = parser.parse_args()

    service = DetectorService.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
    )
    result = service.predict_video_path(
        args.video,
        frames_per_scene=args.frames_per_scene,
        scene_threshold=args.scene_threshold,
        max_scenes=args.max_scenes,
    )
    payload = {
        "video": args.video,
        "result": result.result.__dict__,
        "scene_count": result.scene_count,
        "sampled_frame_count": result.sampled_frame_count,
        "frames": [
            {
                "scene_index": frame.scene_index,
                "frame_index": frame.frame_index,
                "timestamp_sec": frame.timestamp_sec,
                "result": frame.result.__dict__,
            }
            for frame in result.frames
        ],
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
