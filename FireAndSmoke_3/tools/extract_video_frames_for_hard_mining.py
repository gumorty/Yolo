from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def iter_videos(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def extract(video_path: Path, out_dir: Path, every: int, max_frames: int | None) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skip unreadable video: {video_path}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_idx = 0
    stem = video_path.stem
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every == 0:
            out = out_dir / f"{stem}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames for focused hard-case labeling/review.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--every", default=30, type=int, help="Save one frame every N frames.")
    parser.add_argument("--max-frames-per-video", type=int)
    args = parser.parse_args()

    total = 0
    for video in iter_videos(args.source):
        saved = extract(video, args.out / video.stem, args.every, args.max_frames_per_video)
        print(f"{video}: saved {saved}")
        total += saved
    print(f"Total saved frames: {total}")


if __name__ == "__main__":
    main()
