"""
Speaker identification: match one audio waveform against enrolled voice embeddings.

No microphone or recording happens here — callers supply raw waveform arrays directly.
Uses the shared VoiceEncoder loaded at module level so the model weights are only
allocated once per process.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from resemblyzer import VoiceEncoder

# Ensure the project root is on sys.path when this file is run directly
# (python audio/speaker_id.py), so the enrollment package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from enrollment.database import load_database    # JSON database reader

VOICE_MATCH_THRESHOLD = 0.5  # cosine similarity floor for a valid match

# Load the Resemblyzer GE2E model once at import time — avoids reloading the
# ~17 MB weights on every identify_speaker call.
_voice_encoder = VoiceEncoder()


class SpeakerIDError(Exception):
    """Raised when speaker identification encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Return the cosine similarity between two 1-D vectors in [-1, 1].

    Returns 0.0 when either vector has zero norm to avoid division by zero.
    Resemblyzer embeddings are L2-normalised internally, but we guard anyway
    to keep this function safe for arbitrary inputs.

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

def get_voice_embedding(waveform: np.ndarray) -> np.ndarray | None:
    """
    Run Resemblyzer on a raw waveform and return the 256-d speaker embedding.

    Resemblyzer's ``embed_utterance`` expects a 1-D float32 mono waveform
    sampled at 16 kHz.  Near-silent clips return ``None`` so callers are not
    matched against enrolled speakers based on silence alone.

    Uses the module-level ``_voice_encoder`` so the model is never initialised
    more than once per process.

    Args:
        waveform: 1-D float32 NumPy array of raw audio samples at 16 kHz.

    Returns:
        256-dimensional float64 embedding, or ``None`` if the waveform is
        empty or too quiet to produce a meaningful speaker representation.
    """
    if waveform is None or waveform.size == 0:
        return None

    # Reject near-silent clips — same threshold used during enrollment.
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    if rms < 1e-4:
        return None

    embedding = _voice_encoder.embed_utterance(waveform)
    return np.asarray(embedding, dtype=np.float64).reshape(-1)


def identify_speaker(
    waveform: np.ndarray,
    *,
    threshold: float = VOICE_MATCH_THRESHOLD,
    db_path: Path | None = None,
) -> tuple[str, float]:
    """
    Match the speaker in ``waveform`` against all enrolled identities.

    Computes a 256-d Resemblyzer embedding for the supplied audio, then scores
    it against every enrolled voice embedding using cosine similarity.  The
    highest-scoring identity is returned when its score meets ``threshold``.

    Args:
        waveform:  1-D float32 NumPy array of 16 kHz mono audio.
        threshold: Minimum cosine similarity in [0, 1] to accept a match.
                   Defaults to 0.5.  Lower values increase false positives;
                   higher values increase false negatives.
        db_path:   Optional path override for the enrollment database file.
                   Useful in unit tests to point at a temporary database.

    Returns:
        A ``(name, confidence)`` tuple where ``confidence`` is a float in
        [0, 1].  Returns ``("Unknown", best_score)`` when the best match
        falls below ``threshold``, or ``("Unknown", 0.0)`` when the waveform
        is silent, no voice can be embedded, or the database is empty.
    """
    embedding = get_voice_embedding(waveform)
    if embedding is None:
        return "Unknown", 0.0

    database: dict[str, Any] = load_database(db_path)
    if not database:
        return "Unknown", 0.0

    best_name = "Unknown"
    best_score = 0.0

    for person_name, record in database.items():
        stored_vec = np.asarray(record["voice_embedding"], dtype=np.float64)
        # Skip people enrolled without a voice embedding (e.g. face-only records).
        if stored_vec.size == 0:
            continue
        score = _cosine_similarity(embedding, stored_vec)
        if score > best_score:
            best_score = score
            best_name = person_name

    if best_score < threshold:
        # Return the actual score so callers can log "how close" the miss was.
        return "Unknown", best_score

    return best_name, best_score


# ---------------------------------------------------------------------------
# Quick smoke test — run with: python audio/speaker_id.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing audio.speaker_id ...")

    # --- Test 1: empty waveform → no embedding ---
    empty_wave = np.array([], dtype=np.float32)
    name, score = identify_speaker(empty_wave)
    assert name == "Unknown", f"Expected 'Unknown', got {name!r}"
    assert score == 0.0, f"Expected score 0.0, got {score}"
    print(f"  [PASS] empty waveform → ({name!r}, {score:.3f})")

    # --- Test 2: silent waveform → rejected before embedding ---
    silent_wave = np.zeros(16000, dtype=np.float32)
    name, score = identify_speaker(silent_wave)
    assert name == "Unknown"
    assert score == 0.0
    print(f"  [PASS] silent waveform → ({name!r}, {score:.3f})")

    # --- Test 3: cosine similarity helper edge cases ---
    v = np.array([1.0, 0.0, 0.0])
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9, "Self-similarity should be 1.0"
    assert _cosine_similarity(v, np.zeros(3)) == 0.0, "Zero-norm vector should return 0.0"
    print("  [PASS] _cosine_similarity edge cases")

    # --- Test 4: database-level matching with a synthetic embedding ---
    import tempfile, json
    from enrollment.database import VOICE_EMBEDDING_DIM

    fake_embedding = np.random.randn(VOICE_EMBEDDING_DIM).tolist()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(
            {
                "SyntheticSpeaker": {
                    "face_embedding": [],
                    "voice_embedding": fake_embedding,
                }
            },
            tmp,
        )
        tmp_path = Path(tmp.name)

    # Identical vectors must have cosine similarity 1.0.
    query = np.asarray(fake_embedding, dtype=np.float64)
    score_self = _cosine_similarity(query, query)
    assert abs(score_self - 1.0) < 1e-9, "Identical vectors must have similarity 1.0"
    print(f"  [PASS] identical-vector self-similarity = {score_self:.6f}")

    # With threshold=0.0 the synthetic speaker should be found.
    synth_wave = np.random.randn(VOICE_EMBEDDING_DIM).astype(np.float32)
    # Patch the module-level encoder temporarily to return our fake embedding.
    _original_embed = _voice_encoder.embed_utterance
    _voice_encoder.embed_utterance = lambda *_a, **_kw: np.asarray(fake_embedding)
    matched_name, matched_score = identify_speaker(
        synth_wave, threshold=0.0, db_path=tmp_path
    )
    _voice_encoder.embed_utterance = _original_embed  # restore
    assert matched_name == "SyntheticSpeaker", (
        f"Expected 'SyntheticSpeaker', got {matched_name!r}"
    )
    assert abs(matched_score - 1.0) < 1e-6, f"Expected score ~1.0, got {matched_score}"
    print(f"  [PASS] synthetic match → ({matched_name!r}, {matched_score:.6f})")

    tmp_path.unlink()
    print("\nAll speaker_id tests passed.")
