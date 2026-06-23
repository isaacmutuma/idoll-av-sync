# Research Log — IDOLL AV Sync

Technical decisions and rationale. Newest entries at the top.

---

## 2026-06-23 — Score fusion (`sync/synchronizer.py`)

**Decision:** Implement fusion as three explicit branches rather than a single formula:
- **Both agree** → weighted average (face 0.6, voice 0.4). Face gets more weight because InsightFace 512-d embeddings consistently outperform Resemblyzer 256-d GE2E on same-identity cosine similarity in initial tests.
- **One modality only** → use that score at full weight. Blending a 0.0 "no-match" score from the silent modality would artificially penalise single-modality detections (e.g. speaker off-camera or microphone background noise).
- **Names disagree** → return lower of the two scores as a conservative estimate, set `uncertain=True`, prefer the higher-confidence modality's name. Callers can request re-identification on uncertain results.

**Combined threshold:** 0.6 (same for all branches). Chosen to sit above the 0.5 per-modality voice threshold while leaving headroom for face-only and voice-only paths to trigger.

**`FusionResult` dataclass:** Carries `name`, `confidence`, `method` label, and `uncertain` flag. The `method` string makes it easy to trace which branch fired during live demos without adding a logger.

**`identify()` wrapper:** Returns `(name, confidence)` tuple matching the return convention of `face_id.identify_face` and `speaker_id.identify_speaker` so `main.py` can call all three uniformly.

**Hardware dependency:** None. Module is fully headless — accepts plain floats and strings, never touches camera or microphone. Tests run without any hardware.

**Alternatives considered:** Single weighted formula regardless of "Unknown" status (rejected — penalises valid single-modality results); Kalman filter over time (deferred to post-demo, adds statefulness complexity).

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
