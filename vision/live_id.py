"""
Live face identification: open the webcam, collect several frames, average
their embeddings, and print cosine-similarity scores against every enrolled
person in the database.

This module is intentionally camera-aware — vision/face_id.py stays
camera-free so it can be imported and tested headlessly.

Usage:
    python vision/live_id.py
    python vision/live_id.py --camera 1 --frames 10 --warmup 20
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from enrollment.database import load_database
from vision.face_id import _cosine_similarity, get_face_embedding, FACE_MATCH_THRESHOLD

WARMUP_FRAMES = 10
SAMPLE_FRAMES = 5


class LiveIDError(Exception):
    """Raised when live identification cannot proceed."""


def open_camera(device_index: int = 0) -> cv2.VideoCapture:
    """
    Open the webcam at ``device_index``.

    Args:
        device_index: OpenCV camera device index (default 0).

    Returns:
        An open ``cv2.VideoCapture`` instance.

    Raises:
        LiveIDError: If the camera cannot be opened.
    """
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise LiveIDError(
            f"Cannot open camera {device_index}. "
            "Check it is connected and not in use by another app."
        )
    return cap


def collect_embeddings(
    cap: cv2.VideoCapture,
    n_frames: int = SAMPLE_FRAMES,
    warmup: int = WARMUP_FRAMES,
) -> list[np.ndarray]:
    """
    Discard ``warmup`` frames to let the camera auto-expose, then collect
    face embeddings from up to ``n_frames`` subsequent frames.

    Args:
        cap:      An open ``cv2.VideoCapture`` instance.
        n_frames: Number of frames to sample for embeddings.
        warmup:   Number of frames to discard before sampling.

    Returns:
        List of 512-d face embeddings (may be shorter than ``n_frames`` if
        some frames contain no detectable face).
    """
    print(f"Warming up camera ({warmup} frames) ...")
    for _ in range(warmup):
        cap.read()

    embeddings: list[np.ndarray] = []
    print(f"Sampling {n_frames} frames ...")
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            raise LiveIDError("Camera stopped returning frames during sampling.")
        emb = get_face_embedding(frame)
        if emb is not None:
            embeddings.append(emb)

    return embeddings


def score_against_database(
    embedding: np.ndarray,
    threshold: float = FACE_MATCH_THRESHOLD ,
) -> list[tuple[str, float]]:
    """
    Compare ``embedding`` to every enrolled face and return sorted scores.

    Args:
        embedding: 512-d query face embedding.
        threshold: Match threshold; entries are still returned even if below it.

    Returns:
        List of ``(name, score)`` tuples sorted by score descending.
    """
    db = load_database()
    if not db:
        raise LiveIDError(
            "Database is empty. Enroll at least one person with "
            "`python enrollment/enroll.py` first."
        )

    results: list[tuple[str, float]] = []
    for name, record in db.items():
        stored = np.asarray(record["face_embedding"], dtype=np.float64)
        score = _cosine_similarity(embedding, stored)
        results.append((name, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def run(
    camera_index: int = 0,
    n_frames: int = SAMPLE_FRAMES,
    warmup: int = WARMUP_FRAMES,
    threshold: float = FACE_MATCH_THRESHOLD,
) -> int:
    """
    Full live identification pipeline.

    Opens the camera, collects embeddings, averages them, scores against the
    database, and prints results to stdout.

    Args:
        camera_index: OpenCV camera device index.
        n_frames:     Number of frames to average.
        warmup:       Frames to discard for camera auto-exposure.
        threshold:    Cosine similarity floor for a positive match.

    Returns:
        0 on success, 1 on failure.
    """
    try:
        cap = open_camera(camera_index)
    except LiveIDError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        embeddings = collect_embeddings(cap, n_frames=n_frames, warmup=warmup)
    except LiveIDError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        cap.release()

    if not embeddings:
        print("No face detected in any sampled frame. Try better lighting or move closer.")
        return 1

    print(f"\nFace detected in {len(embeddings)}/{n_frames} frames.")
    avg_embedding = np.mean(np.stack(embeddings), axis=0)

    try:
        scores = score_against_database(avg_embedding, threshold=threshold)
    except LiveIDError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\n{'Name':<20} {'Similarity':>10}  {'Match?':>6}")
    print("-" * 42)
    for name, score in scores:
        match_flag = "YES" if score >= threshold else "no"
        print(f"{name:<20} {score:>10.4f}  {match_flag:>6}")

    best_name, best_score = scores[0]
    print()
    if best_score >= threshold:
        print(f"Identified: {best_name}  (confidence {best_score:.1%})")
    else:
        print(f"Unknown person  (closest: {best_name} at {best_score:.1%}, threshold {threshold:.0%})")

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Live face identification against enrolled database."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default 0).")
    parser.add_argument("--frames", type=int, default=SAMPLE_FRAMES, help=f"Frames to average (default {SAMPLE_FRAMES}).")
    parser.add_argument("--warmup", type=int, default=WARMUP_FRAMES, help=f"Warmup frames to discard (default {WARMUP_FRAMES}).")
    parser.add_argument("--threshold", type=float, default=FACE_MATCH_THRESHOLD, help=f"Match threshold 0–1 (default {FACE_MATCH_THRESHOLD}).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(
        run(
            camera_index=args.camera,
            n_frames=args.frames,
            warmup=args.warmup,
            threshold=args.threshold,
        )
    )
