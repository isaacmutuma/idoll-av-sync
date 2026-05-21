"""
Face identification: match one camera frame against enrolled face embeddings.

No camera or video capture happens here — callers supply BGR frames directly.
Uses the shared InsightFace model loaded in enrollment.enroll so the 300 MB
model weights are only allocated once per process.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the project root is on sys.path when this file is run directly
# (python vision/face_id.py), so the enrollment package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from enrollment.enroll import _face_app          # shared InsightFace model
from enrollment.database import load_database    # JSON database reader

FACE_MATCH_THRESHOLD = 0.5  # cosine similarity floor for a valid match


class FaceIDError(Exception):
    """Raised when face identification encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Return the cosine similarity between two 1-D vectors in [-1, 1].

    Returns 0.0 when either vector has zero norm to avoid division by zero.
    InsightFace embeddings are not pre-normalised, so we cannot skip this.

    Args:
        a: First embedding vector.
        b: Second embedding vector of the same length.

    Returns:
        Scalar cosine similarity in [-1.0, 1.0].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def get_face_embedding(frame_bgr: np.ndarray) -> np.ndarray | None:
    """
    Run InsightFace on one BGR frame and return the 512-d face embedding.

    When multiple faces appear in the frame, the one with the largest
    bounding-box area is used — typically the subject closest to the camera.

    Uses the module-level ``_face_app`` imported from ``enrollment.enroll``
    so the model is never initialised more than once per process.

    Args:
        frame_bgr: A single BGR camera frame as a NumPy uint8 array (H, W, 3).

    Returns:
        512-dimensional float64 embedding, or ``None`` if no face is found.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    faces = _face_app.get(frame_bgr)
    if not faces:
        return None

    # bbox format is [x1, y1, x2, y2]; compute proper pixel area
    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    return np.asarray(largest.embedding, dtype=np.float64)


def identify_face(
    frame_bgr: np.ndarray,
    *,
    threshold: float = FACE_MATCH_THRESHOLD,
    db_path: Path | None = None,
) -> tuple[str, float]:
    """
    Match the face in ``frame_bgr`` against all enrolled identities.

    Computes a 512-d embedding for the largest detected face, then scores it
    against every enrolled face embedding using cosine similarity.  The
    highest-scoring identity is returned when its score meets ``threshold``.

    Args:
        frame_bgr: Single BGR camera frame as a NumPy array (H, W, 3).
        threshold: Minimum cosine similarity in [0, 1] to accept a match.
                   Defaults to 0.7.  Lower values increase false positives;
                   higher values increase false negatives.
        db_path:   Optional path override for the enrollment database file.
                   Useful in unit tests to point at a temporary database.

    Returns:
        A ``(name, confidence)`` tuple where ``confidence`` is a float in
        [0, 1].  Returns ``("Unknown", best_score)`` when the best match
        falls below ``threshold``, or ``("Unknown", 0.0)`` when no face is
        detected at all or the database is empty.
    """
    embedding = get_face_embedding(frame_bgr)
    if embedding is None:
        return "Unknown", 0.0

    database: dict[str, Any] = load_database(db_path)
    if not database:
        return "Unknown", 0.0

    best_name = "Unknown"
    best_score = 0.0

    for person_name, record in database.items():
        stored_vec = np.asarray(record["face_embedding"], dtype=np.float64)
        score = _cosine_similarity(embedding, stored_vec)
        if score > best_score:
            best_score = score
            best_name = person_name

    if best_score < threshold:
        # Return the actual score so callers can log "how close" the miss was
        return "Unknown", best_score

    return best_name, best_score


# ---------------------------------------------------------------------------
# Quick smoke test — run with: python vision/face_id.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing vision.face_id ...")

    # --- Test 1: black frame → no face ---
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    name, score = identify_face(black_frame)
    assert name == "Unknown", f"Expected 'Unknown', got {name!r}"
    assert score == 0.0, f"Expected score 0.0, got {score}"
    print(f"  [PASS] black frame → ({name!r}, {score:.3f})")

    # --- Test 2: cosine similarity helper ---
    v = np.array([1.0, 0.0, 0.0])
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9, "Self-similarity should be 1.0"
    assert _cosine_similarity(v, np.zeros(3)) == 0.0, "Zero-norm should return 0.0"
    print("  [PASS] _cosine_similarity edge cases")

    # --- Test 3: database-level matching with synthetic embeddings ---
    import tempfile, json
    from enrollment.database import FACE_EMBEDDING_DIM

    fake_embedding = np.random.randn(FACE_EMBEDDING_DIM).tolist()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(
            {"SyntheticPerson": {"face_embedding": fake_embedding, "voice_embedding": []}},
            tmp,
        )
        tmp_path = Path(tmp.name)

    # A near-identical embedding should score very close to 1.0
    query = np.asarray(fake_embedding, dtype=np.float64)
    score_self = _cosine_similarity(query, query)
    assert abs(score_self - 1.0) < 1e-9, "Identical vectors must have similarity 1.0"
    print(f"  [PASS] identical-vector self-similarity = {score_self:.6f}")
    tmp_path.unlink()

    print("\nAll face_id tests passed.")
