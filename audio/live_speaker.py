"""
Live speaker identification via microphone.

Opens the system microphone, records several seconds of audio, slices the
recording into short non-overlapping chunks, computes a Resemblyzer embedding
for each chunk, averages those embeddings into a single stable speaker
representation, and then prints cosine-similarity scores against every
enrolled person in the enrollment database.

Design note — why this file exists separately from speaker_id.py
----------------------------------------------------------------
``speaker_id.py`` is intentionally mic-free so it can be imported and unit-
tested in headless environments (CI, servers) where no audio hardware exists.
This module is the mic-aware layer; import it only when a microphone is present.

Usage (quick smoke test):
    python audio/live_speaker.py

Usage (from another module):
    from audio.live_speaker import identify_live
    name, confidence = identify_live(duration=4.0, verbose=True)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Make the project root importable when this file is run directly
# (python audio/live_speaker.py from any working directory).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Re-use the headless helpers from speaker_id — they already handle RMS
# gating, model loading, and cosine similarity.
from audio.speaker_id import (  # noqa: E402  (import after sys.path patch)
    _cosine_similarity,
    get_voice_embedding,
    VOICE_MATCH_THRESHOLD,
)
from enrollment.database import load_database  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16_000
"""Resemblyzer requires exactly 16 000 Hz mono input."""

DEFAULT_DURATION: float = 4.0
"""Seconds of audio captured per identification call.
4 s gives Resemblyzer enough speech on most voices while keeping latency low."""

DEFAULT_CHUNK_DURATION: float = 1.5
"""Length of each embedding chunk in seconds.
Shorter chunks produce more embeddings to average over; longer chunks give
Resemblyzer more context.  1.5 s is a practical balance."""

MIN_CHUNK_DURATION: float = 0.5
"""Chunks shorter than this (tail of the recording) are discarded because
Resemblyzer produces unreliable embeddings on very short clips."""


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class LiveSpeakerError(Exception):
    """Raised for unrecoverable microphone or pipeline errors."""


# ---------------------------------------------------------------------------
# Step 1 — record audio from the microphone
# ---------------------------------------------------------------------------

def record_audio(
    duration: float = DEFAULT_DURATION,
    sample_rate: int = SAMPLE_RATE,
    device: int | str | None = None,
) -> np.ndarray:
    """
    Capture mono audio from the default (or specified) microphone.

    Uses sounddevice's blocking ``sd.rec()`` call, which fills a buffer of
    ``int(duration * sample_rate)`` frames and returns when recording is done.
    The function converts the result to a flat 1-D float32 array so it can be
    passed directly to Resemblyzer utilities.

    Why float32?  Resemblyzer's ``embed_utterance`` internally normalises to
    [-1, 1] and works with float32 or float64.  Using float32 here keeps
    memory usage lower for longer recordings.

    Args:
        duration:    Recording length in seconds.  Must be positive.
        sample_rate: Sampling frequency in Hz.  Resemblyzer requires 16 000.
        device:      sounddevice device index or partial name string.
                     ``None`` uses the OS default input device.

    Returns:
        1-D float32 NumPy array with ``int(duration * sample_rate)`` samples.

    Raises:
        LiveSpeakerError: If sounddevice cannot open or read from the mic
                          (device missing, permission denied, etc.).
    """
    num_frames = int(duration * sample_rate)

    try:
        # sd.rec allocates a (num_frames, channels) array, fills it from the
        # mic, and returns immediately.  sd.wait() blocks until it finishes.
        raw = sd.rec(
            num_frames,
            samplerate=sample_rate,
            channels=1,       # mono — Resemblyzer does not use stereo
            dtype="float32",
            device=device,
        )
        sd.wait()             # block on the calling thread until done
    except (sd.PortAudioError, ValueError) as exc:
        # PortAudioError: hardware-level failure (device unavailable, etc.)
        # ValueError:     sounddevice raises this for unrecognised device names.
        raise LiveSpeakerError(f"Microphone error: {exc}") from exc

    # sd.rec returns shape (num_frames, 1) — squeeze to 1-D.
    return raw.reshape(-1)


# ---------------------------------------------------------------------------
# Step 2 — slice the waveform and compute per-chunk embeddings
# ---------------------------------------------------------------------------

def collect_embeddings(
    waveform: np.ndarray,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    sample_rate: int = SAMPLE_RATE,
) -> list[np.ndarray]:
    """
    Slice a waveform into non-overlapping chunks and embed each one.

    Why chunk instead of embedding the whole recording?
    Resemblyzer's GE2E model was trained on short utterances.  A single 4 s
    embedding captures the whole clip but may be skewed by silence or noise at
    the start/end.  Chunking lets us compute N independent embeddings and then
    average them, which reduces per-chunk noise and gives a more robust speaker
    centroid.

    Chunks shorter than ``MIN_CHUNK_DURATION`` (tail of the recording) are
    dropped.  Near-silent chunks are rejected by ``get_voice_embedding``'s RMS
    gate so they don't pollute the average.

    Args:
        waveform:       1-D float32 NumPy array of raw audio at ``sample_rate``.
        chunk_duration: Length of each non-overlapping chunk in seconds.
        sample_rate:    Sample rate of the input waveform.

    Returns:
        List of 256-d float64 Resemblyzer embeddings, one per voiced chunk.
        Returns an empty list when no chunk passes the silence gate.
    """
    chunk_samples = int(chunk_duration * sample_rate)
    min_samples = int(MIN_CHUNK_DURATION * sample_rate)

    embeddings: list[np.ndarray] = []

    # Stride through the waveform in non-overlapping windows.
    for start in range(0, len(waveform), chunk_samples):
        chunk = waveform[start : start + chunk_samples]

        # Discard the short tail fragment to avoid unreliable embeddings.
        if len(chunk) < min_samples:
            continue

        # get_voice_embedding returns None for near-silent chunks.
        embedding = get_voice_embedding(chunk)
        if embedding is not None:
            embeddings.append(embedding)

    return embeddings


# ---------------------------------------------------------------------------
# Step 3 — average the per-chunk embeddings
# ---------------------------------------------------------------------------

def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray | None:
    """
    Compute the L2-normalised mean of a list of 256-d speaker embeddings.

    Why average?  Each chunk embedding is a noisy estimate of the speaker's
    true position in embedding space.  The element-wise mean moves toward the
    speaker's "centre of mass", reducing the influence of any single noisy
    chunk.

    Why L2-normalise the result?  Resemblyzer embeddings are individually
    normalised to unit length, but their element-wise mean is not.  Re-
    normalising places the average back on the unit hypersphere, which is the
    space in which cosine similarity is well-behaved.

    Args:
        embeddings: Non-empty list of equal-length float64 arrays (each 256-d).

    Returns:
        L2-normalised 256-d float64 array representing the averaged speaker,
        or ``None`` when the list is empty or the mean has zero norm.
    """
    if not embeddings:
        return None

    # Stack to (N, 256) → column-wise mean → (256,)
    mean_vec = np.mean(np.stack(embeddings, axis=0), axis=0)

    norm = np.linalg.norm(mean_vec)
    if norm == 0.0:
        # Pathological case: all embedding vectors exactly cancel out.
        return None

    return (mean_vec / norm).astype(np.float64)


# ---------------------------------------------------------------------------
# Step 4 — score the averaged embedding against the enrollment database
# ---------------------------------------------------------------------------

def score_against_database(
    embedding: np.ndarray,
    db_path: Path | None = None,
) -> list[tuple[str, float]]:
    """
    Compute cosine similarity between one embedding and every enrolled person.

    Loads the full enrollment database, then calls ``_cosine_similarity``
    (from speaker_id) for each stored voice embedding.  People enrolled without
    a voice embedding (face-only enrollments) are silently skipped.

    Returns results sorted highest-first so callers can inspect or print them
    in rank order without an extra sort step.

    Args:
        embedding: 256-d float64 query embedding (typically the averaged result
                   from ``average_embeddings``).
        db_path:   Optional path to the enrollment JSON file.  Defaults to the
                   standard ``enrollment/database.json``.

    Returns:
        List of ``(name, cosine_similarity)`` tuples sorted by similarity
        descending.  Empty list when the database contains no voice records.
    """
    database: dict[str, Any] = load_database(db_path)
    scores: list[tuple[str, float]] = []

    for person_name, record in database.items():
        stored_vec = np.asarray(
            record.get("voice_embedding", []), dtype=np.float64
        )

        # Silently skip face-only enrollments that have no voice data.
        if stored_vec.size == 0:
            continue

        similarity = _cosine_similarity(embedding, stored_vec)
        scores.append((person_name, similarity))

    # Highest similarity first — index 0 is always the best candidate.
    scores.sort(key=lambda pair: pair[1], reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Full pipeline — single public entry point
# ---------------------------------------------------------------------------

def identify_live(
    duration: float = DEFAULT_DURATION,
    *,
    threshold: float = VOICE_MATCH_THRESHOLD,
    db_path: Path | None = None,
    sample_rate: int = SAMPLE_RATE,
    device: int | str | None = None,
    verbose: bool = False,
) -> tuple[str, float]:
    """
    Record from the microphone and return the best-matching enrolled speaker.

    This function chains the four pipeline steps into one call:
      1. ``record_audio``         — capture from mic
      2. ``collect_embeddings``   — chunk + embed
      3. ``average_embeddings``   — compute stable mean representation
      4. ``score_against_database`` — rank against enrolled speakers

    Pass ``verbose=True`` to print a cosine-similarity score table to stdout,
    which is useful for debugging thresholds or demonstrating the system live.

    Args:
        duration:    Seconds of audio to capture.  4 s is a good default.
        threshold:   Minimum cosine similarity to accept a positive match.
                     Scores below this return ``("Unknown", score)``.
        db_path:     Optional database path override (used in testing).
        sample_rate: Sampling rate in Hz; keep at 16 000 for Resemblyzer.
        device:      Optional sounddevice device index or name.
        verbose:     If True, print the full ranked score table to stdout.

    Returns:
        ``(name, confidence)`` tuple where ``confidence`` is in [0, 1].
        Returns ``("Unknown", best_score)`` when the best match is below
        ``threshold``, or ``("Unknown", 0.0)`` when recording fails, no speech
        is detected, or the database is empty.
    """
    # ── Step 1: record ──────────────────────────────────────────────────────
    print(f"[live_speaker] Recording {duration:.1f} s … speak now.")
    try:
        waveform = record_audio(duration, sample_rate=sample_rate, device=device)
    except LiveSpeakerError as exc:
        print(f"[live_speaker] ERROR — {exc}")
        return "Unknown", 0.0

    # ── Step 2: chunk and embed ──────────────────────────────────────────────
    embeddings = collect_embeddings(waveform, sample_rate=sample_rate)
    if not embeddings:
        print("[live_speaker] No speech detected — recording was silent.")
        return "Unknown", 0.0

    print(f"[live_speaker] {len(embeddings)} voiced chunk(s) embedded.")

    # ── Step 3: average ──────────────────────────────────────────────────────
    mean_embedding = average_embeddings(embeddings)
    if mean_embedding is None:
        # Should not happen in practice (would require all embeddings to sum to
        # the zero vector), but guard defensively.
        return "Unknown", 0.0

    # ── Step 4: score ────────────────────────────────────────────────────────
    scores = score_against_database(mean_embedding, db_path)
    if not scores:
        print("[live_speaker] Database is empty — enroll someone first.")
        return "Unknown", 0.0

    # Print the full ranked table when requested.
    if verbose:
        print("\n  Cosine-similarity scores (highest = best match):")
        print(f"  {'Rank':<5} {'Name':<22} {'Score':>6}  Bar")
        print("  " + "─" * 50)
        for rank, (name, sim) in enumerate(scores, start=1):
            # 20-cell ASCII bar where each cell represents 0.05 similarity.
            bar = "█" * int(sim * 20)
            print(f"  {rank:<5} {name:<22} {sim:>6.4f}  {bar}")
        print()

    best_name, best_score = scores[0]

    if best_score < threshold:
        print(
            f"[live_speaker] Best match {best_name!r} scored {best_score:.4f}"
            f" — below threshold {threshold:.2f} → Unknown"
        )
        return "Unknown", best_score

    return best_name, best_score


# ---------------------------------------------------------------------------
# Smoke test — run with: python audio/live_speaker.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import tempfile

    print("=" * 60)
    print("audio.live_speaker — headless unit tests")
    print("=" * 60)

    # ── Test 1: record_audio raises LiveSpeakerError on invalid device ──────
    # We mock a bad device name to trigger the PortAudio error path without
    # needing real hardware.  We catch LiveSpeakerError to confirm it propagates.
    print("\n[1] LiveSpeakerError propagation from record_audio …")
    try:
        record_audio(0.1, device="__nonexistent_device_xyz__")
        print("  [FAIL] Expected LiveSpeakerError was not raised")
    except LiveSpeakerError:
        print("  [PASS] LiveSpeakerError raised for invalid device")

    # ── Test 2: collect_embeddings on a silent waveform returns empty list ──
    print("\n[2] collect_embeddings on silent audio …")
    silent = np.zeros(SAMPLE_RATE * 4, dtype=np.float32)  # 4 s of silence
    result = collect_embeddings(silent)
    assert result == [], f"Expected [], got {result}"
    print("  [PASS] silent waveform → 0 embeddings")

    # ── Test 3: collect_embeddings on random (non-silent) audio ────────────
    print("\n[3] collect_embeddings on random (voiced) audio …")
    # Random noise is above the RMS silence gate, so all chunks should embed.
    rng = np.random.default_rng(seed=42)
    noisy = rng.standard_normal(int(SAMPLE_RATE * 4)).astype(np.float32) * 0.1
    chunks = collect_embeddings(noisy, chunk_duration=1.5)
    # 4 s ÷ 1.5 s/chunk = 2 full chunks (the 1 s tail is below MIN_CHUNK_DURATION)
    # 4 s at 1.5 s/chunk: two full chunks (0–1.5 s, 1.5–3.0 s) plus a 1 s tail
    # (3.0–4.0 s).  The tail is above MIN_CHUNK_DURATION (0.5 s), so it embeds
    # too — giving 3 chunks total.
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    assert chunks[0].shape == (256,), f"Expected (256,) shape, got {chunks[0].shape}"
    print(f"  [PASS] 4 s noisy audio → {len(chunks)} embeddings of shape {chunks[0].shape}")

    # ── Test 4: average_embeddings on empty list returns None ───────────────
    print("\n[4] average_embeddings on empty list …")
    avg = average_embeddings([])
    assert avg is None, f"Expected None, got {avg}"
    print("  [PASS] empty list → None")

    # ── Test 5: average_embeddings result is unit-length ───────────────────
    print("\n[5] average_embeddings produces unit vector …")
    vecs = [rng.standard_normal(256) for _ in range(5)]
    avg = average_embeddings(vecs)
    assert avg is not None
    norm = float(np.linalg.norm(avg))
    assert abs(norm - 1.0) < 1e-9, f"Expected unit norm, got {norm}"
    print(f"  [PASS] averaged embedding has L2 norm = {norm:.10f}")

    # ── Test 6: score_against_database against a synthetic database ─────────
    print("\n[6] score_against_database with synthetic enrollment …")
    from enrollment.database import VOICE_EMBEDDING_DIM

    known_vec = rng.standard_normal(VOICE_EMBEDDING_DIM).tolist()
    other_vec = rng.standard_normal(VOICE_EMBEDDING_DIM).tolist()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(
            {
                "Alice": {
                    "face_embedding": [],
                    "voice_embedding": known_vec,
                },
                "Bob": {
                    "face_embedding": [],
                    "voice_embedding": other_vec,
                },
                "FaceOnly": {
                    # Deliberately no voice_embedding key — should be skipped.
                    "face_embedding": list(rng.standard_normal(512)),
                },
            },
            tmp,
        )
        tmp_path = Path(tmp.name)

    query = np.asarray(known_vec, dtype=np.float64)
    # L2-normalise the query (matches what average_embeddings returns).
    query /= np.linalg.norm(query)

    results = score_against_database(query, db_path=tmp_path)

    assert len(results) == 2, f"Expected 2 scored entries, got {len(results)}"
    assert results[0][0] == "Alice", f"Alice should rank first, got {results[0][0]!r}"
    assert abs(results[0][1] - 1.0) < 1e-6, (
        f"Query identical to Alice's vector → similarity should be ~1.0, got {results[0][1]}"
    )
    print(f"  [PASS] Alice scored {results[0][1]:.6f}, Bob scored {results[1][1]:.6f}")
    print("  [PASS] FaceOnly entry correctly skipped (no voice_embedding)")

    tmp_path.unlink()

    # ── Test 7: identify_live with no enrolled speakers ─────────────────────
    print("\n[7] identify_live when database is empty (no real mic needed) …")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump({}, tmp)
        empty_db_path = Path(tmp.name)

    # Monkey-patch record_audio to return a voiced waveform so we don't need
    # the mic but still exercise the full pipeline past the recording step.
    _original_record = record_audio
    noisy_patch = rng.standard_normal(int(SAMPLE_RATE * 4)).astype(np.float32) * 0.1

    import audio.live_speaker as _self_module
    _self_module.record_audio = lambda *_a, **_kw: noisy_patch  # type: ignore[assignment]

    name, score = identify_live(db_path=empty_db_path)
    assert name == "Unknown", f"Expected 'Unknown' for empty DB, got {name!r}"
    assert score == 0.0, f"Expected 0.0 for empty DB, got {score}"
    print(f"  [PASS] empty database → ({name!r}, {score:.3f})")

    _self_module.record_audio = _original_record  # restore
    empty_db_path.unlink()

    # ── All tests passed ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All headless tests passed.")
    print("=" * 60)

    # ── Optional live mic test ───────────────────────────────────────────────
    # Only runs if there is at least one enrolled person in the database.
    # Skipped automatically in CI / headless environments.
    from enrollment.database import list_enrolled_names

    enrolled = list_enrolled_names()
    if not enrolled:
        print(
            "\n[SKIP] Live mic test skipped — database is empty.\n"
            "       Run enrollment/enroll.py first to add speakers."
        )
    else:
        print(f"\nEnrolled speakers: {', '.join(enrolled)}")
        print("Starting live identification (speak for 4 s) …\n")
        matched_name, confidence = identify_live(
            duration=DEFAULT_DURATION,
            threshold=VOICE_MATCH_THRESHOLD,
            verbose=True,
        )
        print(f"\nResult: {matched_name!r}  (confidence {confidence:.4f})")
