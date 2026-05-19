# PROJECT CONTEXT — IDOLL Audio-Visual Person Identification

**Paste this at the start of every Cursor session for this project** (or rely on `.cursor/rules/idoll-project-context.mdc`, which loads automatically).

---

## WHO I AM

Isaac Mutuma — 2nd year CS Engineering undergraduate, Pusan National University (PNU), Busan, South Korea. Korean Government Scholarship (GKS). Computer Vision Intern at IDOLL Robotics.

**Technical background:**
- Python — primary language, comfortable
- PyTorch — trained CNNs from scratch (CIFAR-10, FashionMNIST)
- OpenCV — video processing, optical flow, MediaPipe pose estimation
- No prior experience with: audio processing, speaker identification, face recognition libraries

**GitHub:** github.com/isaacmutuma

---

## THE PROJECT

Build a real-time multimodal person identification system for IDOLL's companion robot. The robot identifies known people by face and voice simultaneously and outputs their name as text.

**Confirmed hardware:**
- Camera: standard RGB webcam
- Compute: laptop CPU, CUDA GPU if available
- ROS: not needed

**Two modes:**

**MODE 1 — ENROLLMENT (run once per person)**
Person looks at camera and speaks their name. System captures face embedding and voice embedding. Saves both to local JSON database under their name.

**MODE 2 — RECOGNITION (runs continuously)**
Camera stream → detect face → compute embedding → match database
Microphone → detect voice → compute embedding → match database
Synchronize by timestamp → fuse confidence scores → output name as text

**Example output:**
```
Identified: Nasrat Shady (face: 91%, voice: 84%, combined: 88%)
Unknown person detected.
```

---

## RESEARCH CONTEXT

This prototype is the foundation of a research paper targeting HRI 2027 (ACM/IEEE International Conference on Human-Robot Interaction, deadline October 2026).

**Research contribution:**
Real-time multimodal person identification combining face and voice embeddings for HRI on standard laptop hardware — no model training required, enrollment-based, CUDA-accelerated when available.

**Key papers:**
- MulT: Multimodal Transformer (Tsai et al., ACL 2019)
- AV-HuBERT (Shi et al., Meta AI 2022)
- Face recognition survey (Guo & Zhang, 2019)

**Collaborator:** Nasrat Shady, founder of IDOLL Robotics, Busan, South Korea.

---

## PROJECT STRUCTURE

```
idoll-av-sync/
├── README.md
├── requirements.txt
├── main.py                    ← entry point: --mode enroll / --mode recognize
├── docs/
│   ├── PROJECT_CONTEXT.md     ← this file
│   └── research_log.md        ← document every technical decision
├── enrollment/
│   ├── __init__.py
│   ├── enroll.py              ← capture face + voice, save embeddings
│   └── database.py            ← store and query embeddings database
├── vision/
│   ├── __init__.py
│   ├── camera.py              ← webcam capture
│   ├── detect.py              ← face detection in frame
│   └── face_id.py             ← face embedding + database matching
├── audio/
│   ├── __init__.py
│   ├── capture.py             ← microphone capture
│   ├── vad.py                 ← voice activity detection
│   └── speaker_id.py          ← voice embedding + database matching
├── sync/
│   ├── __init__.py
│   └── synchronizer.py        ← timestamp alignment, score fusion
└── response/
    ├── __init__.py
    └── responder.py           ← text output of identification result
```

---

## LIBRARIES

```
face_recognition   →  face detection and 128-d face embeddings (dlib-based)
resemblyzer        →  256-d voice embeddings (Google's GE2E model)
sounddevice        →  real-time microphone capture
numpy              →  embedding math
opencv-python      →  camera capture and frame processing
torch              →  CUDA acceleration if available
```

---

## BUILD ORDER

```
Week 1  →  enrollment/enroll.py + enrollment/database.py
Week 2  →  vision/camera.py + vision/detect.py + vision/face_id.py
Week 3  →  audio/capture.py + audio/vad.py + audio/speaker_id.py
Week 4  →  sync/synchronizer.py + response/responder.py + main.py
Demo    →  present to Nasrat Shady, end of June 2026
```

---

## CODE STANDARDS

- Clean `.py` files — no Jupyter notebooks
- Docstring on every function
- Type hints throughout
- Error handling for: camera not found, microphone unavailable, no face detected, no voice detected, unknown person
- `if __name__ == "__main__":` test block in every module
- Every module importable independently
- Log every technical decision in `docs/research_log.md`

---

## HOW TO HELP (for AI assistants)

1. Write production-quality `.py` files matching the structure above
2. Explain every new concept before showing code
3. Build one module at a time — complete and test before moving on
4. Flag decisions Isaac needs to make
5. Keep it real-time capable — robot needs near real-time response
6. Never use Jupyter notebooks
