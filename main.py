"""
IDOLL AV Sync — entry point for enrollment and continuous identification.

Two operating modes, selected with --mode:

  python main.py --mode enroll --name "Isaac"
      Runs the enrollment pipeline once: captures face + voice from the
      webcam and microphone, saves 512-d + 256-d embeddings to the database,
      then exits.  Omit --name to be prompted interactively.

  python main.py --mode identify
      Opens the camera and microphone and runs an identification loop
      indefinitely.  Each cycle records a few seconds of audio, captures one
      camera frame, fuses the face and voice scores, and prints the result.
      Press Ctrl+C to stop cleanly.

This file wires together the four pipeline modules without duplicating any
camera or microphone logic — every hardware interaction is delegated to the
relevant sub-module:

  enrollment/enroll.py      → enroll_person()
  vision/live_id.py         → open_camera()
  vision/face_id.py         → identify_face()
  audio/live_speaker.py     → record_audio()
  audio/speaker_id.py       → identify_speaker()
  sync/synchronizer.py      → fuse_scores()
  response/responder.py     → respond()
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Project-root sys.path guard (allows `python main.py` from any cwd).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Per-module imports — each module is responsible for its own hardware.
from enrollment.enroll import EnrollmentError, enroll_person          # noqa: E402
from vision.live_id import LiveIDError, open_camera                   # noqa: E402
from vision.face_id import identify_face                              # noqa: E402
from audio.live_speaker import LiveSpeakerError, record_audio         # noqa: E402
from audio.speaker_id import identify_speaker                         # noqa: E402
from sync.synchronizer import fuse_scores                             # noqa: E402
from response.responder import respond                                 # noqa: E402

# ---------------------------------------------------------------------------
# Identification loop constants
# ---------------------------------------------------------------------------

AUDIO_DURATION: float = 3.0
"""Seconds of microphone audio captured each identification cycle.
3 s gives Resemblyzer enough speech without making each cycle feel sluggish."""

CYCLE_PAUSE: float = 2.0
"""Seconds to wait after printing a result before starting the next cycle.
Gives the operator time to read the output and avoids hammering the models
at full CPU speed."""

CAMERA_WARMUP_FRAMES: int = 5
"""Frames discarded at startup so the camera auto-exposure can settle before
the first identification cycle uses a frame."""


# ---------------------------------------------------------------------------
# Enroll mode
# ---------------------------------------------------------------------------

def run_enroll(name: str, camera_index: int = 0) -> int:
    """
    Run the one-shot enrollment pipeline for a single person.

    Delegates entirely to ``enroll_person()`` from ``enrollment/enroll.py``,
    which opens the webcam, captures face embeddings, records voice audio,
    and writes the result to ``enrollment/database.json``.

    Args:
        name:         The person's display name (used as the database key).
        camera_index: OpenCV camera device index.  Defaults to 0.

    Returns:
        0 on success, 1 on any enrollment failure.
    """
    try:
        enroll_person(name, camera_index=camera_index)
    except EnrollmentError as exc:
        print(f"[enroll] ERROR — {exc}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Identify mode — one cycle
# ---------------------------------------------------------------------------

def _run_one_cycle(cap: cv2.VideoCapture) -> None:
    """
    Execute a single face + voice identification cycle and print the result.

    One cycle:
      1. Record ``AUDIO_DURATION`` seconds of audio from the default mic.
      2. Read one BGR frame from the already-open camera.
      3. Identify the speaker from the audio waveform.
      4. Identify the face from the camera frame.
      5. Fuse the two scores with the synchronizer.
      6. Print the result via the responder.

    Audio recording is the blocking step that paces the loop — it occupies
    the full ``AUDIO_DURATION`` seconds before the frame is read.  This means
    the face and voice are captured from roughly the same moment in time,
    keeping the multimodal fusion meaningful.

    Args:
        cap: An already-open ``cv2.VideoCapture`` instance.  The caller owns
             the lifecycle (open/release); this function only reads from it.
    """
    # ── Step 1: record audio (blocking) ─────────────────────────────────────
    # record_audio() is from audio/live_speaker.py — it uses sounddevice and
    # returns a flat float32 waveform at 16 kHz.
    try:
        waveform = record_audio(duration=AUDIO_DURATION)
    except LiveSpeakerError as exc:
        print(f"[identify] Microphone error — {exc}", file=sys.stderr)
        waveform = None

    # ── Step 2: capture a single camera frame ───────────────────────────────
    # Reading the frame after audio finishes means both modalities cover the
    # same physical moment (the end of the audio window).
    ret, frame = cap.read()
    if not ret:
        print("[identify] Camera dropped a frame — skipping cycle.", file=sys.stderr)
        frame = None

    # ── Step 3: voice identification ─────────────────────────────────────────
    # identify_speaker() is from audio/speaker_id.py — it computes a
    # Resemblyzer embedding and matches it against enrolled voice embeddings.
    # Returns ("Unknown", 0.0) when waveform is None or silent.
    import numpy as np
    voice_name, voice_score = identify_speaker(
        waveform if waveform is not None else np.array([], dtype=np.float32)
    )

    # ── Step 4: face identification ──────────────────────────────────────────
    # identify_face() is from vision/face_id.py — it runs InsightFace on the
    # BGR frame and matches the embedding against enrolled face embeddings.
    # Returns ("Unknown", 0.0) when frame is None or contains no face.
    face_name, face_score = identify_face(
        frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    )

    # ── Step 5: fuse scores ──────────────────────────────────────────────────
    # fuse_scores() is from sync/synchronizer.py — applies weighted fusion
    # (face 0.6, voice 0.4) or single-modality fallback as appropriate.
    fusion_result = fuse_scores(face_name, face_score, voice_name, voice_score)

    # ── Step 6: print the result ─────────────────────────────────────────────
    # respond() is from response/responder.py — formats and prints the result.
    respond(fusion_result)


# ---------------------------------------------------------------------------
# Identify mode — continuous loop
# ---------------------------------------------------------------------------

def run_identify(camera_index: int = 0) -> int:
    """
    Open the camera and run the identification loop until Ctrl+C.

    The loop runs indefinitely, cycling through audio recording, frame
    capture, face + voice identification, score fusion, and output.  A brief
    pause after each result gives the operator time to read the output.

    Camera warmup: the first ``CAMERA_WARMUP_FRAMES`` frames are read and
    discarded so the auto-exposure has settled before the first real cycle.

    Ctrl+C (``KeyboardInterrupt``) exits cleanly: the camera is released and
    a final message is printed before the process terminates.

    Args:
        camera_index: OpenCV camera device index.  Defaults to 0.

    Returns:
        0 on clean exit (Ctrl+C), 1 if the camera could not be opened.
    """
    # ── Open camera ──────────────────────────────────────────────────────────
    try:
        cap = open_camera(camera_index)
    except LiveIDError as exc:
        print(f"[identify] Camera error — {exc}", file=sys.stderr)
        return 1

    # ── Warmup: discard frames to let auto-exposure settle ───────────────────
    print(f"[identify] Warming up camera ({CAMERA_WARMUP_FRAMES} frames) …")
    for _ in range(CAMERA_WARMUP_FRAMES):
        cap.read()

    print("[identify] Starting identification loop.  Press Ctrl+C to stop.\n")

    try:
        while True:
            _run_one_cycle(cap)
            # Pause between cycles so the terminal output is readable.
            time.sleep(CYCLE_PAUSE)

    except KeyboardInterrupt:
        # Ctrl+C is the expected way to stop the loop — not an error.
        print("\n[identify] Stopped by user.")

    finally:
        # Always release the camera, even if an unexpected exception propagates.
        cap.release()
        print("[identify] Camera released.")

    return 0


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for main.py.

    Two required/optional arguments:
    - ``--mode``  (required): ``"enroll"`` or ``"identify"``
    - ``--name``  (enroll only): person's display name; prompted if omitted
    - ``--camera``: OpenCV device index (default 0)

    Args:
        argv: Argument list to parse.  Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed ``argparse.Namespace`` object.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="IDOLL AV Sync — multimodal person identification system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --mode enroll --name \"Isaac\"\n"
            "  python main.py --mode identify\n"
            "  python main.py --mode identify --camera 1"
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["enroll", "identify"],
        help="Operating mode: 'enroll' to register a new person, "
             "'identify' to run the continuous recognition loop.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(enroll mode only) Person's display name. "
             "If omitted you will be prompted interactively.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments and dispatch to the appropriate mode.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).
              Useful for testing: ``main(["--mode", "enroll", "--name", "X"])``.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args(argv)

    if args.mode == "enroll":
        # Resolve name: CLI arg → interactive prompt.
        name = args.name.strip() if args.name else input("Enter name to enroll: ").strip()
        if not name:
            print("ERROR: Name cannot be empty.", file=sys.stderr)
            return 1
        return run_enroll(name, camera_index=args.camera)

    if args.mode == "identify":
        return run_identify(camera_index=args.camera)

    # argparse's choices= constraint makes this unreachable, but guard anyway.
    print(f"ERROR: Unknown mode {args.mode!r}.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
