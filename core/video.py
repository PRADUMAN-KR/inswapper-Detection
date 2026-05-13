from dataclasses import dataclass
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SceneSegment:
    scene_index: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class SampledVideoFrame:
    scene_index: int
    frame_index: int
    timestamp_sec: float
    image: Image.Image


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def _detect_scenes(
    capture: cv2.VideoCapture,
    frame_count: int,
    fps: float,
    threshold: float,
    scan_every_seconds: float,
) -> list[SceneSegment]:
    stride = max(1, int(round(fps * scan_every_seconds)))
    scenes: list[SceneSegment] = []
    scene_start = 0
    previous_hist: np.ndarray | None = None
    previous_index = 0

    for frame_index in range(0, frame_count, stride):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        hist = _frame_histogram(frame)
        if previous_hist is not None:
            distance = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if distance >= threshold:
                end_frame = max(scene_start, previous_index)
                scenes.append(SceneSegment(len(scenes), scene_start, end_frame))
                scene_start = frame_index
        previous_hist = hist
        previous_index = frame_index

    if frame_count > 0:
        scenes.append(SceneSegment(len(scenes), scene_start, frame_count - 1))
    return scenes or [SceneSegment(0, 0, max(0, frame_count - 1))]


def _choose_scenes(scenes: list[SceneSegment], max_scenes: int) -> list[SceneSegment]:
    if len(scenes) <= max_scenes:
        return scenes
    longest = sorted(
        scenes,
        key=lambda scene: scene.end_frame - scene.start_frame,
        reverse=True,
    )[:max_scenes]
    return sorted(longest, key=lambda scene: scene.start_frame)


def _sample_indices(scene: SceneSegment, frames_per_scene: int) -> list[int]:
    length = scene.end_frame - scene.start_frame + 1
    if length <= 0:
        return []
    count = min(frames_per_scene, length)
    return sorted({int(round(value)) for value in np.linspace(scene.start_frame, scene.end_frame, count)})


def sample_scene_aware_frames_from_path(
    video_path: str | Path,
    frames_per_scene: int = 6,
    scene_threshold: float = 0.55,
    scan_every_seconds: float = 0.5,
    max_scenes: int = 12,
) -> tuple[list[SampledVideoFrame], int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            raise ValueError("Video has no readable frames.")

        scenes = _detect_scenes(
            capture=capture,
            frame_count=frame_count,
            fps=fps,
            threshold=scene_threshold,
            scan_every_seconds=scan_every_seconds,
        )
        selected_scenes = _choose_scenes(scenes, max_scenes=max_scenes)
        sampled: list[SampledVideoFrame] = []
        for scene in selected_scenes:
            for frame_index in _sample_indices(scene, frames_per_scene):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                sampled.append(
                    SampledVideoFrame(
                        scene_index=scene.scene_index,
                        frame_index=frame_index,
                        timestamp_sec=round(frame_index / fps, 3),
                        image=image,
                    )
                )
        if not sampled:
            raise ValueError("Could not sample frames from video.")
        return sampled, len(scenes)
    finally:
        capture.release()


def sample_scene_aware_frames_from_bytes(
    video_bytes: bytes,
    suffix: str = ".mp4",
    frames_per_scene: int = 6,
    scene_threshold: float = 0.55,
    scan_every_seconds: float = 0.5,
    max_scenes: int = 12,
) -> tuple[list[SampledVideoFrame], int]:
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(video_bytes)
        handle.flush()
        return sample_scene_aware_frames_from_path(
            handle.name,
            frames_per_scene=frames_per_scene,
            scene_threshold=scene_threshold,
            scan_every_seconds=scan_every_seconds,
            max_scenes=max_scenes,
        )
