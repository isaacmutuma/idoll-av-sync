"""
Terminal responder for IDOLL person identification results.

Accepts a FusionResult from sync.synchronizer and formats it into a
human-readable string that is both returned to the caller and printed to
the terminal.

This module is deliberately hardware-free:
- It never opens a camera or microphone.
- It never calls face_id.py or speaker_id.py directly.
- The only input is a FusionResult dataclass; the only output is text.

Two output formats:
    Identified:             "Identified: {name} (confidence: {pct}%)"
    Unknown person:         "Unknown person detected."

An optional uncertainty flag appends " [low confidence]" to positive IDs
when the fused result was flagged as uncertain by the synchronizer (i.e.
face and voice modalities disagreed).

Usage:
    from sync.synchronizer import fuse_scores
    from response.responder import respond

    result = fuse_scores("Isaac", 0.91, "Isaac", 0.75)
    message = respond(result)          # prints and returns the string
    # → "Identified: Isaac (confidence: 85%)"
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is importable when this file is run directly.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sync.synchronizer import UNKNOWN, FusionResult  # noqa: E402


# ---------------------------------------------------------------------------
# Core formatting logic
# ---------------------------------------------------------------------------

def format_result(result: FusionResult) -> str:
    """
    Convert a FusionResult into a human-readable identification string.

    Two possible outputs:
    - **Positive ID**: ``"Identified: {name} (confidence: {pct}%)"``
      where ``pct`` is the fused confidence rounded to the nearest integer.
      If the result is marked uncertain (modalities disagreed), the suffix
      ``" [low confidence]"`` is appended so downstream code or a human
      operator can decide whether to act on the result.
    - **Unknown**: ``"Unknown person detected."``
      Returned whenever ``result.name`` equals the ``UNKNOWN`` sentinel,
      regardless of the confidence score.

    This function does not print anything — it only builds and returns the
    string.  Use :func:`respond` when you want the message printed too.

    Args:
        result: A :class:`~sync.synchronizer.FusionResult` produced by
                :func:`~sync.synchronizer.fuse_scores`.

    Returns:
        Formatted identification string.
    """
    if result.name == UNKNOWN:
        return "Unknown person detected."

    # Convert the [0.0, 1.0] confidence score to a whole-number percentage.
    pct = round(result.confidence * 100)
    message = f"Identified: {result.name} (confidence: {pct}%)"

    # Append a low-confidence warning when the synchronizer flagged a
    # modality disagreement so the caller knows the result is tentative.
    if result.uncertain:
        message += " [low confidence]"

    return message


def respond(result: FusionResult) -> str:
    """
    Format a FusionResult and print it to the terminal.

    Calls :func:`format_result` to build the message string, prints it with
    ``print()``, and then returns the same string so callers can log, assert,
    or forward it without capturing stdout.

    This is the primary entry point for the response layer.  The robot's
    control loop in ``main.py`` calls this once per identification cycle.

    Args:
        result: A :class:`~sync.synchronizer.FusionResult` from the
                synchronizer fusion step.

    Returns:
        The formatted identification string (same value that was printed).
    """
    message = format_result(result)
    print(message)
    return message


# ---------------------------------------------------------------------------
# Smoke tests — run with: python response/responder.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sync.synchronizer import fuse_scores

    print("=" * 60)
    print("response.responder — output formatting tests")
    print("=" * 60)

    # ── Scenario 1: positive ID, both modalities agree ─────────────────────
    # Face and voice both identify "Isaac" with high confidence.
    # Expected: "Identified: Isaac (confidence: 85%)"
    print("\n[1] Positive ID — both modalities agree")
    r = fuse_scores("Isaac", 0.91, "Isaac", 0.75)
    # Weighted fusion: 0.6×0.91 + 0.4×0.75 = 0.546 + 0.300 = 0.846 → 85%
    msg = respond(r)
    assert msg == "Identified: Isaac (confidence: 85%)", (
        f"Unexpected message: {msg!r}"
    )
    print(f"  [PASS] {msg!r}")

    # ── Scenario 2: unknown person, both modalities return Unknown ──────────
    # Neither modality cleared its per-modality threshold.
    # Expected: "Unknown person detected."
    print("\n[2] Unknown — both modalities returned Unknown")
    r = fuse_scores(UNKNOWN, 0.18, UNKNOWN, 0.22)
    msg = respond(r)
    assert msg == "Unknown person detected.", (
        f"Unexpected message: {msg!r}"
    )
    print(f"  [PASS] {msg!r}")

    # ── Scenario 3: unknown person, scores below combined threshold ─────────
    # Names agree but the fused score is too low to pass COMBINED_THRESHOLD.
    # The synchronizer already returns name="Unknown" in this case.
    # Expected: "Unknown person detected."
    print("\n[3] Unknown — names match but combined score too low")
    r = fuse_scores("Nasrat", 0.42, "Nasrat", 0.38)
    msg = respond(r)
    assert msg == "Unknown person detected.", (
        f"Unexpected message: {msg!r}"
    )
    print(f"  [PASS] {msg!r}")

    # ── Scenario 4: face-only positive ID ──────────────────────────────────
    # Voice modality returned Unknown; face alone cleared the threshold.
    # Expected: "Identified: Nasrat (confidence: 78%)"
    print("\n[4] Positive ID — face only (voice=Unknown)")
    r = fuse_scores("Nasrat", 0.78, UNKNOWN, 0.30)
    msg = respond(r)
    assert msg == "Identified: Nasrat (confidence: 78%)", (
        f"Unexpected message: {msg!r}"
    )
    print(f"  [PASS] {msg!r}")

    # ── Scenario 5: voice-only positive ID ─────────────────────────────────
    # Face modality returned Unknown; voice alone cleared the threshold.
    # Expected: "Identified: Isaac (confidence: 65%)"
    print("\n[5] Positive ID — voice only (face=Unknown)")
    r = fuse_scores(UNKNOWN, 0.20, "Isaac", 0.65)
    msg = respond(r)
    assert msg == "Identified: Isaac (confidence: 65%)", (
        f"Unexpected message: {msg!r}"
    )
    print(f"  [PASS] {msg!r}")

    # ── Scenario 6: uncertain result — modalities disagree, score above 0.6 ─
    # Face → "Isaac" (0.80), voice → "Nasrat" (0.70).
    # Synchronizer returns the higher-confidence name with uncertain=True.
    # Expected: "Identified: Isaac (confidence: 70%) [low confidence]"
    print("\n[6] Uncertain — modalities disagree, score above threshold")
    r = fuse_scores("Isaac", 0.80, "Nasrat", 0.70)
    msg = respond(r)
    assert msg == "Identified: Isaac (confidence: 70%) [low confidence]", (
        f"Unexpected message: {msg!r}"
    )
    print(f"  [PASS] {msg!r}")

    # ── Scenario 7: uncertain result — modalities disagree, score below 0.6 ─
    # Face → "Isaac" (0.75), voice → "Nasrat" (0.50).
    # min score = 0.50, below threshold → name="Unknown", uncertain=True.
    # Expected: "Unknown person detected."
    print("\n[7] Uncertain + below threshold — returns Unknown")
    r = fuse_scores("Isaac", 0.75, "Nasrat", 0.50)
    msg = respond(r)
    assert msg == "Unknown person detected.", (
        f"Unexpected message: {msg!r}"
    )
    print(f"  [PASS] {msg!r}")

    # ── Scenario 8: format_result does not print (pure formatting) ─────────
    # Verify that format_result returns the string without side effects.
    print("\n[8] format_result returns string without printing")
    r = fuse_scores("Isaac", 0.91, "Isaac", 0.75)
    msg = format_result(r)
    assert isinstance(msg, str) and msg.startswith("Identified:"), (
        f"Unexpected return from format_result: {msg!r}"
    )
    print(f"  [PASS] format_result returned {msg!r} (no extra print)")

    print("\n" + "=" * 60)
    print("All responder tests passed.")
    print("=" * 60)
