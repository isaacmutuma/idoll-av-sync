"""
Multimodal score fusion for person identification.

Accepts pre-computed face and voice identification results (name + confidence
score from each modality) and fuses them into a single (name, confidence)
decision using a weighted combination rule.

This module is deliberately hardware-free:
- It never opens a camera or microphone.
- It never calls face_id.py or speaker_id.py directly.
- All inputs arrive as plain Python arguments (strings and floats).

That separation makes the fusion logic independently testable and means it can
be swapped out or tuned without touching the per-modality modules.

Fusion rules (in priority order):
1. Both names agree  → weighted average: face × 0.6 + voice × 0.4
2. Only one modality returned a name (the other is "Unknown")
   → use that modality's score at full weight
3. Names disagree    → flag as uncertain, take the lower of the two scores
                       as a conservative confidence estimate

A combined score ≥ COMBINED_THRESHOLD (0.6) is required for a positive ID.

Usage:
    from sync.synchronizer import fuse_scores

    name, confidence = fuse_scores(
        face_name="Isaac", face_score=0.82,
        voice_name="Isaac", voice_score=0.71,
    )
    # → ("Isaac", 0.777)
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

FACE_WEIGHT: float = 0.6
"""Weight given to the face modality in a two-modality weighted fusion.

Face recognition is generally the higher-precision modality for this system
(InsightFace 512-d embeddings vs Resemblyzer 256-d GE2E), so it receives the
larger share of the combined score."""

VOICE_WEIGHT: float = 0.4
"""Weight given to the voice modality in a two-modality weighted fusion.
FACE_WEIGHT + VOICE_WEIGHT must equal 1.0."""

COMBINED_THRESHOLD: float = 0.6
"""Minimum fused confidence score required to accept a positive identification.

Below this value the system returns ("Unknown", combined_score) regardless of
which names were matched.  0.6 was chosen as a starting point — tune upward
to reduce false positives, downward to reduce false negatives."""

UNKNOWN: str = "Unknown"
"""Sentinel name returned by both face_id and speaker_id when no match clears
their respective per-modality thresholds."""

assert abs(FACE_WEIGHT + VOICE_WEIGHT - 1.0) < 1e-9, (
    "FACE_WEIGHT + VOICE_WEIGHT must sum to 1.0"
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FusionResult:
    """
    Immutable container for a fused identification decision.

    Attributes:
        name:       The identified person's name, or ``"Unknown"``.
        confidence: Fused confidence score in [0.0, 1.0].
        method:     Human-readable label describing which fusion branch was
                    used.  Useful for logging and debugging.
        uncertain:  True when the two modalities disagreed on the person's
                    identity.  A name is still returned in that case (the
                    lower-confidence candidate) but callers can treat this flag
                    as a reason to request re-identification.
    """

    name: str
    confidence: float
    method: str
    uncertain: bool = False

    def __str__(self) -> str:
        """Return a compact human-readable summary of the fusion decision."""
        flag = " [UNCERTAIN]" if self.uncertain else ""
        return (
            f"{self.name!r}  confidence={self.confidence:.4f}"
            f"  [{self.method}]{flag}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_known(name: str) -> bool:
    """
    Return True when ``name`` represents a positively identified person.

    Any name that is not the ``UNKNOWN`` sentinel is treated as a successful
    per-modality identification.  The check is case-insensitive and strips
    surrounding whitespace to guard against minor formatting differences.

    Args:
        name: Name string returned by face_id or speaker_id.

    Returns:
        True if the name is not "Unknown" (any case / padding).
    """
    return name.strip().lower() != UNKNOWN.lower()


def _weighted_fuse(face_score: float, voice_score: float) -> float:
    """
    Compute the face-weighted average of two per-modality confidence scores.

    Both scores are assumed to be in [0.0, 1.0].  The result is clamped to
    [0.0, 1.0] to guard against floating-point drift from future weight edits.

    Args:
        face_score:  Confidence score from the face-recognition modality.
        voice_score: Confidence score from the voice-recognition modality.

    Returns:
        Weighted combination: ``FACE_WEIGHT × face_score + VOICE_WEIGHT × voice_score``.
    """
    raw = FACE_WEIGHT * face_score + VOICE_WEIGHT * voice_score
    return float(max(0.0, min(1.0, raw)))


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def fuse_scores(
    face_name: str,
    face_score: float,
    voice_name: str,
    voice_score: float,
) -> FusionResult:
    """
    Fuse face and voice identification results into a single decision.

    Applies three fusion branches in priority order:

    **Branch A — both modalities agree**
        Both names are known and identical.  Compute a weighted average score
        (face 60 %, voice 40 %).  This is the highest-confidence path.

    **Branch B — only one modality succeeded**
        One name is ``"Unknown"`` (its per-modality threshold was not met).
        Use the successful modality's score at full weight without blending in
        a zero-confidence result, which would unfairly penalise single-modality
        detections (e.g. when the speaker is off-camera).

    **Branch C — names disagree**
        Both modalities returned a name but they differ.  This is an
        inconsistency that may indicate occlusion, background noise, or a
        genuine impostor.  The system returns the lower of the two scores as a
        conservative estimate and sets ``uncertain=True`` so callers can
        escalate or request re-identification.

    In all branches a combined score below ``COMBINED_THRESHOLD`` maps to
    ``"Unknown"`` in the returned name, while the raw score is preserved for
    caller inspection.

    Args:
        face_name:   Name returned by the face-recognition modality, or
                     ``"Unknown"`` when no face cleared its threshold.
        face_score:  Confidence score in [0.0, 1.0] from face recognition.
        voice_name:  Name returned by the speaker-identification modality, or
                     ``"Unknown"`` when no voice cleared its threshold.
        voice_score: Confidence score in [0.0, 1.0] from speaker identification.

    Returns:
        A :class:`FusionResult` with the fused name, combined confidence,
        fusion method label, and an uncertainty flag.
    """
    face_known = _is_known(face_name)
    voice_known = _is_known(voice_name)

    # ── Branch A: both modalities agree ─────────────────────────────────────
    if face_known and voice_known and face_name.strip() == voice_name.strip():
        combined = _weighted_fuse(face_score, voice_score)
        final_name = face_name.strip() if combined >= COMBINED_THRESHOLD else UNKNOWN
        return FusionResult(
            name=final_name,
            confidence=combined,
            method="weighted_fusion (face×0.6 + voice×0.4)",
        )

    # ── Branch B: only face succeeded ───────────────────────────────────────
    if face_known and not voice_known:
        final_name = face_name.strip() if face_score >= COMBINED_THRESHOLD else UNKNOWN
        return FusionResult(
            name=final_name,
            confidence=face_score,
            method="face_only (voice=Unknown)",
        )

    # ── Branch B: only voice succeeded ──────────────────────────────────────
    if voice_known and not face_known:
        final_name = voice_name.strip() if voice_score >= COMBINED_THRESHOLD else UNKNOWN
        return FusionResult(
            name=final_name,
            confidence=voice_score,
            method="voice_only (face=Unknown)",
        )

    # ── Branch C: both failed ───────────────────────────────────────────────
    if not face_known and not voice_known:
        # Neither modality identified anyone — take whichever score is higher
        # (both are below per-modality thresholds, so combined stays Unknown).
        best_score = max(face_score, voice_score)
        return FusionResult(
            name=UNKNOWN,
            confidence=best_score,
            method="both_unknown",
        )

    # ── Branch C: names disagree ─────────────────────────────────────────────
    # Both modalities returned a name but they differ.  Use the lower score as
    # a conservative confidence estimate and flag the result as uncertain.
    lower_score = min(face_score, voice_score)
    # Prefer the higher-confidence modality's name as the uncertain candidate.
    candidate_name = face_name.strip() if face_score >= voice_score else voice_name.strip()
    final_name = candidate_name if lower_score >= COMBINED_THRESHOLD else UNKNOWN
    return FusionResult(
        name=final_name,
        confidence=lower_score,
        method=f"disagreement (face={face_name!r}, voice={voice_name!r})",
        uncertain=True,
    )


def identify(
    face_name: str,
    face_score: float,
    voice_name: str,
    voice_score: float,
) -> tuple[str, float]:
    """
    Convenience wrapper around :func:`fuse_scores` for callers that only need
    the ``(name, confidence)`` pair.

    Matches the return-type convention used by ``face_id.identify_face`` and
    ``speaker_id.identify_speaker`` so the calling code in ``main.py`` can
    treat all three functions uniformly.

    Args:
        face_name:   Name from face recognition, or ``"Unknown"``.
        face_score:  Face-recognition confidence in [0.0, 1.0].
        voice_name:  Name from speaker identification, or ``"Unknown"``.
        voice_score: Voice-recognition confidence in [0.0, 1.0].

    Returns:
        ``(name, combined_confidence)`` tuple.
    """
    result = fuse_scores(face_name, face_score, voice_name, voice_score)
    return result.name, result.confidence


# ---------------------------------------------------------------------------
# Smoke tests — run with: python sync/synchronizer.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("sync.synchronizer — fusion logic tests")
    print("=" * 60)

    # Helper that prints the scenario header and result, then asserts.
    def _check(
        scenario: str,
        result: FusionResult,
        *,
        expected_name: str,
        expected_method_fragment: str,
        confident_min: float | None = None,
        confident_max: float | None = None,
        uncertain: bool = False,
    ) -> None:
        """Print result and assert expectations for one test scenario."""
        print(f"\n  {scenario}")
        print(f"    → {result}")
        assert result.name == expected_name, (
            f"Expected name {expected_name!r}, got {result.name!r}"
        )
        assert expected_method_fragment in result.method, (
            f"Expected method containing {expected_method_fragment!r}, "
            f"got {result.method!r}"
        )
        if confident_min is not None:
            assert result.confidence >= confident_min, (
                f"Expected confidence ≥ {confident_min}, got {result.confidence}"
            )
        if confident_max is not None:
            assert result.confidence <= confident_max, (
                f"Expected confidence ≤ {confident_max}, got {result.confidence}"
            )
        assert result.uncertain == uncertain, (
            f"Expected uncertain={uncertain}, got {result.uncertain}"
        )
        print("    [PASS]")

    # ── Scenario 1: both modalities agree, combined score above threshold ───
    r = fuse_scores("Isaac", 0.91, "Isaac", 0.75)
    _check(
        "1. Both agree, high scores → positive ID",
        r,
        expected_name="Isaac",
        expected_method_fragment="weighted_fusion",
        confident_min=COMBINED_THRESHOLD,
    )
    expected_s1 = round(FACE_WEIGHT * 0.91 + VOICE_WEIGHT * 0.75, 10)
    assert abs(r.confidence - expected_s1) < 1e-9, (
        f"Weighted fusion score mismatch: expected {expected_s1}, got {r.confidence}"
    )

    # ── Scenario 2: both agree but combined score below threshold ──────────
    r = fuse_scores("Isaac", 0.42, "Isaac", 0.38)
    _check(
        "2. Both agree, low scores → Unknown despite name match",
        r,
        expected_name=UNKNOWN,
        expected_method_fragment="weighted_fusion",
        confident_max=COMBINED_THRESHOLD,
    )

    # ── Scenario 3: face identified, voice unknown ─────────────────────────
    r = fuse_scores("Nasrat", 0.78, UNKNOWN, 0.30)
    _check(
        "3. Face only (voice=Unknown), face score above threshold",
        r,
        expected_name="Nasrat",
        expected_method_fragment="face_only",
    )
    assert r.confidence == 0.78

    # ── Scenario 4: voice identified, face unknown ─────────────────────────
    r = fuse_scores(UNKNOWN, 0.20, "Nasrat", 0.65)
    _check(
        "4. Voice only (face=Unknown), voice score above threshold",
        r,
        expected_name="Nasrat",
        expected_method_fragment="voice_only",
    )
    assert r.confidence == 0.65

    # ── Scenario 5: face identified, voice unknown, face score too low ─────
    r = fuse_scores("Isaac", 0.45, UNKNOWN, 0.10)
    _check(
        "5. Face only, face score below threshold → Unknown",
        r,
        expected_name=UNKNOWN,
        expected_method_fragment="face_only",
    )

    # ── Scenario 6: both Unknown ───────────────────────────────────────────
    r = fuse_scores(UNKNOWN, 0.15, UNKNOWN, 0.22)
    _check(
        "6. Both Unknown → Unknown, take max score",
        r,
        expected_name=UNKNOWN,
        expected_method_fragment="both_unknown",
    )
    assert r.confidence == 0.22, f"Expected max(0.15, 0.22)=0.22, got {r.confidence}"

    # ── Scenario 7: names disagree, lower score above threshold ───────────
    # min(0.80, 0.70) = 0.70 ≥ COMBINED_THRESHOLD → return the higher-score
    # candidate's name ("Isaac", face scored 0.80) with uncertain=True.
    r = fuse_scores("Isaac", 0.80, "Nasrat", 0.70)
    _check(
        "7. Disagreement, both high scores → uncertain, higher-score name kept",
        r,
        expected_name="Isaac",
        expected_method_fragment="disagreement",
        uncertain=True,
    )
    assert r.confidence == 0.70

    # ── Scenario 8: names disagree, lower score below threshold ───────────
    r = fuse_scores("Isaac", 0.75, "Nasrat", 0.50)
    _check(
        "8. Disagreement, lower score below threshold → Unknown + uncertain",
        r,
        expected_name=UNKNOWN,
        expected_method_fragment="disagreement",
        uncertain=True,
    )
    assert r.confidence == 0.50

    # ── Scenario 9: identify() convenience wrapper ─────────────────────────
    print("\n  9. identify() wrapper returns (name, confidence) tuple")
    name, conf = identify("Isaac", 0.88, "Isaac", 0.72)
    assert name == "Isaac"
    assert abs(conf - (FACE_WEIGHT * 0.88 + VOICE_WEIGHT * 0.72)) < 1e-9
    print(f"    → ('{name}', {conf:.4f})  [PASS]")

    # ── Scenario 10: _is_known handles whitespace and case ─────────────────
    print("\n  10. _is_known edge cases")
    assert _is_known("Isaac") is True
    assert _is_known("  Isaac  ") is True
    assert _is_known(UNKNOWN) is False
    assert _is_known("unknown") is False   # case-insensitive
    assert _is_known("UNKNOWN") is False
    assert _is_known("  Unknown  ") is False
    print("    [PASS]")

    print("\n" + "=" * 60)
    print("All synchronizer tests passed.")
    print("=" * 60)
