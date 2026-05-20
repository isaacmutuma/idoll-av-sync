# Research Log — IDOLL AV Sync

Technical decisions and rationale. Newest entries at the top.

---

## 2026-05-19 — Week 1 enrollment (`database.py`, `enroll.py`)

**Decision:** Store all enrollments in a single `enrollment/database.json` keyed by person name. Each entry holds `face_embedding` (128 floats) and `voice_embedding` (256 floats). No raw images or audio on disk.

**Face capture:** OpenCV webcam for 5 s, sample every 0.2 s, `face_recognition` HOG detector + 128-d encodings, mean-pool valid frames. BGR→RGB before `face_recognition` (library requirement).

**Voice capture:** `sounddevice` mono float32 at 16 kHz for 5 s (Resemblyzer’s expected rate). Reject clips with RMS &lt; 0.01 as “no voice”. GE2E embedding via `VoiceEncoder.embed_utterance`.

**Errors:** Custom exceptions for camera, microphone, no face, no voice; terminal-only prints (no GUI).

**Security:** `enrollment/database.json` added to `.gitignore` (biometric data stays local).

**Alternatives considered:** One JSON file per person (rejected for simpler Week 1 matching); storing enrollment WAV/photos (rejected per privacy and project spec).

---

## 2026-05-19 — Project context persisted in repo

**Decision:** Store full project brief in `docs/PROJECT_CONTEXT.md` and load it via Cursor rule `.cursor/rules/idoll-project-context.mdc` (`alwaysApply: true`).

**Rationale:** Ensures every Cursor session has the same goals, structure, libraries, and code standards without re-pasting context. The rule is a short pointer; the markdown file is the single source of truth.

**Alternatives considered:** Pasting context at session start only (fragile); embedding entire brief in the rule file (too long for rule best practices).
