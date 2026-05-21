"""
Enrollment: capture face and voice from webcam/microphone, save embeddings only.

No raw images or audio are written to disk. Recognition is implemented elsewhere.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import sounddevice as sd
from resemblyzer import VoiceEncoder

try:
    from enrollment.database import (
        FACE_EMBEDDING_DIM,
        VOICE_EMBEDDING_DIM,
        save_person,
    )
except ImportError:
    from database import (  # type: ignore[no-redef]
        FACE_EMBEDDING_DIM,
        VOICE_EMBEDDING_DIM,
        save_person,
    )

CAPTURE_DURATION_SEC = 5.0
VOICE_SAMPLE_RATE = 16000
MIN_VOICE_RMS = 0.001
FACE_SAMPLE_INTERVAL_SEC = 0.2


class EnrollmentError(Exception):
    """Base error for enrollment failures."""


class CameraNotFoundError(EnrollmentError):
    """Webcam could not be opened."""


class MicrophoneError(EnrollmentError):
    """Microphone capture failed."""


class NoFaceDetectedError(EnrollmentError):
    """No usable face embedding could be computed."""


class NoVoiceDetectedError(EnrollmentError):
    """Recorded audio was too quiet to embed."""


def prompt_person_name() -> str:
    """Read and validate a person's name from stdin."""
    name = input("Enter the person's name to enroll: ").strip()
    if not name:
        raise EnrollmentError("Name cannot be empty")
    return name


def open_webcam(device_index: int = 0) -> cv2.VideoCapture:
    """
    Open the default webcam (or ``device_index``) for frame capture.

    Raises:
        CameraNotFoundError: If the device cannot be opened.
    """
    capture = cv2.VideoCapture(device_index)
    if not capture.isOpened():
        raise CameraNotFoundError(
            f"Could not open camera at index {device_index}. "
            "Check that a webcam is connected and not in use by another app."
        )
    return capture


# Initialize InsightFace model once at module level
_face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
_face_app.prepare(ctx_id=0, det_size=(640, 640))


def compute_face_embedding_from_frame(frame_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detect a face in one BGR frame and return its 512-d InsightFace embedding.

    Uses InsightFace buffalo_sc model — no dlib dependency, M2 compatible.
    Returns None if no face is found in the frame.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    faces = _face_app.get(frame_bgr)
    if not faces:
        return None

    # Take the largest face if multiple detected
    largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    return np.asarray(largest_face.embedding, dtype=np.float64)

def capture_face_embedding(
    capture: cv2.VideoCapture,
    duration_sec: float = CAPTURE_DURATION_SEC,
) -> np.ndarray:
    """
    Sample frames for ``duration_sec`` seconds and average face embeddings.

    Averaging several embeddings reduces noise from lighting, pose, and blink.
    """
    embeddings: list[np.ndarray] = []
    deadline = time.monotonic() + duration_sec
    next_sample_at = time.monotonic()

    print(f"Look at the camera for {duration_sec:.0f} seconds...")

    while time.monotonic() < deadline:
        ok, frame = capture.read()
        if not ok:
            raise CameraNotFoundError("Failed to read a frame from the webcam")

        now = time.monotonic()
        if now < next_sample_at:
            continue
        next_sample_at = now + FACE_SAMPLE_INTERVAL_SEC

        embedding = compute_face_embedding_from_frame(frame)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        raise NoFaceDetectedError(
            "No face detected during capture. Face the camera in good lighting."
        )

    mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
    if mean_embedding.shape[0] != FACE_EMBEDDING_DIM:
        raise EnrollmentError(
            f"Expected {FACE_EMBEDDING_DIM}-d face embedding, got {mean_embedding.shape[0]}"
        )
    return mean_embedding


def record_voice_audio(
    duration_sec: float = CAPTURE_DURATION_SEC,
    sample_rate: int = VOICE_SAMPLE_RATE,
) -> np.ndarray:
    """
    Record mono float32 audio from the default microphone.

    Raises:
        MicrophoneError: If ``sounddevice`` cannot access the mic.
    """
    print(f"Speak clearly for {duration_sec:.0f} seconds (say your name)...")
    frame_count = int(duration_sec * sample_rate)

    try:
        audio = sd.rec(
            frame_count,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except sd.PortAudioError as exc:
        raise MicrophoneError(
            "Microphone unavailable. Check system permissions and default input device."
        ) from exc

    waveform = np.squeeze(audio)
    if waveform.ndim != 1:
        raise MicrophoneError("Expected mono audio from microphone")
    return waveform


def compute_voice_embedding(
    waveform: np.ndarray,
    sample_rate: int = VOICE_SAMPLE_RATE,
) -> np.ndarray:
    """
    Compute a 256-d speaker embedding with Resemblyzer's pretrained GE2E model.

    The model expects 16 kHz mono audio. We reject near-silent clips so empty
    rooms are not enrolled by mistake.
    """
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    if rms < MIN_VOICE_RMS:
        raise NoVoiceDetectedError(
            "No voice detected (audio too quiet). Speak closer to the microphone."
        )

    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(waveform)
    vector = np.asarray(embedding, dtype=np.float64).reshape(-1)

    if vector.shape[0] != VOICE_EMBEDDING_DIM:
        raise EnrollmentError(
            f"Expected {VOICE_EMBEDDING_DIM}-d voice embedding, got {vector.shape[0]}"
        )
    return vector


def enroll_person(
    name: str,
    *,
    camera_index: int = 0,
    capture_duration_sec: float = CAPTURE_DURATION_SEC,
) -> None:
    """
    Full enrollment pipeline: face capture, voice capture, save to database.

    Args:
        name: Person's display name (database key).
        camera_index: OpenCV camera device index.
        capture_duration_sec: Seconds for each modality capture.
    """
    capture = open_webcam(camera_index)
    try:
        face_embedding = capture_face_embedding(capture, capture_duration_sec)
        print(f"Face embedding captured ({FACE_EMBEDDING_DIM}-d).")
    finally:
        capture.release()

    waveform = record_voice_audio(capture_duration_sec)
    voice_embedding = compute_voice_embedding(waveform)
    print(f"Voice embedding captured ({VOICE_EMBEDDING_DIM}-d).")
    #save the embeddings to the database.json 
    save_person(name, face_embedding, voice_embedding)
    print(f"Enrolled '{name}' → {Path(__file__).resolve().parent / 'database.json'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for enrollment."""
    parser = argparse.ArgumentParser(
        description="Enroll a person: capture face + voice embeddings (no raw media saved).",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Person's name. If omitted, you will be prompted interactively.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=CAPTURE_DURATION_SEC,
        help=f"Capture duration in seconds for face and voice (default: {CAPTURE_DURATION_SEC}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    args = parse_args(argv)
    try:
        name = args.name.strip() if args.name else prompt_person_name()
        enroll_person(
            name,
            camera_index=args.camera,
            capture_duration_sec=args.duration,
        )
    except EnrollmentError as exc:
        print(f"Enrollment failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
