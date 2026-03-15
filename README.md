# Drowsiness Detector 😴

A real-time drowsiness detection system that monitors your eyes while you study
and automatically pauses your YouTube lecture when you fall asleep; then wakes
you up with an alert.

> Built from a genuine need: falling asleep during online lectures at 6am in the hostel.


---

## How it works

1. **Webcam** captures your face in real time using OpenCV
2. **MediaPipe Face Mesh** detects 468 facial landmarks per frame
3. **Eye Aspect Ratio (EAR)** is calculated from 6 key points around each eye
4. If EAR stays below **0.25 for 20+ consecutive frames** → drowsiness confirmed
5. YouTube is **paused automatically** via a keyboard shortcut
6. An **audio alert** plays to wake you up
7. Video **resumes** once your eyes are open again

### EAR Formula

```
EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 × ‖p1−p4‖)
```

- When eye is **open**: EAR ≈ 0.30
- When eye is **closed**: EAR ≈ 0.10
- **Threshold**: EAR < 0.25 for 20+ frames = drowsy

---

## Tech Stack

| Tool      | Purpose                             |
| --------- | ----------------------------------- |
| Python    | Core language                       |
| OpenCV    | Webcam capture and frame processing |
| MediaPipe | Real-time face landmark detection   |
| NumPy     | EAR distance calculations           |
| PyAutoGUI | YouTube pause/resume via keyboard   |
| Pygame    | Alert sound playback                |

---

## Project Structure

```
drowsiness-detector/
│
├── DrowsinessDetector_week1.py   # Webcam + face mesh (foundation)
├── DrowsinessDetector_week2.py   # EAR calculation + drowsy status
├── DrowsinessDetector_week3.py   # Frame counter (coming soon)
├── DrowsinessDetector_week4.py   # YouTube pause + audio alert (coming soon)
│
└── README.md
```

---

## Setup & Run

**1. Clone the repository**

```bash
git clone https://github.com/scintillating-Shruti-Pandey/drowsiness-detector.git
cd drowsiness-detector
```

**2. Install dependencies**

```bash
pip install opencv-python mediapipe==0.10.21 numpy pyautogui pygame
pip install protobuf==4.25.3
```

**3. Run the latest version**

```bash
python DrowsinessDetector_week2.py
```

Press **Q** to quit.

---

## Current Status

- [x] 1 — Webcam feed + MediaPipe face mesh
- [x] 2 — EAR calculation + real-time drowsy detection
- [x] 3 — Frame counter to eliminate false positives
- [x] 4 — YouTube auto-pause + audio alert
- [x] 5 — Polish, demo video, deployment

---

## Known Limitations (worked upon)

- Head tilt causes false drowsy readings (EAR drops when looking up) -> fixed ✔️
- Single-frame threshold triggers too quickly — frame counter -> fixed ✔️

---

## Why I built this

I'm a BTech CSE (AI/ML) student at Bennett University. I kept falling asleep
during early-morning YouTube lectures and losing my place in the video. Instead of
just accepting it, I decided to build a solution. This project taught me real-time
computer vision, facial geometry, and how to connect a Python script to browser
behaviour.

---

## Author

**Shruti Pandey**
B.Tech CSE (AI/ML) — Bennett University, Greater Noida (2024–2028)

[LinkedIn](https://linkedin.com/in/shrutipandey-7b5541211) • [GitHub](https://github.com/scintillating-Shruti-Pandey)
