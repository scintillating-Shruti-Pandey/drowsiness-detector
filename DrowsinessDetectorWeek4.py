# ============================================================
#  DROWSINESS DETECTOR — Week 4
#  New this week:
#  1. Pauses YouTube automatically when drowsy
#  2. Plays a Hare Krishna kirtan to wake you up 🌸
#  3. Resumes YouTube when your eyes open again
#  4. Cooldown timer — prevents repeated alerts
#
#  SETUP: Put a kirtan audio file named "alert_sound.mp3"
#         in the same folder as this file.
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
import pyautogui   # for sending keyboard shortcuts to the browser
import pygame      # for playing the kirtan audio
import time        # for cooldown tracking

# ── Constants ────────────────────────────────────────────────
EAR_THRESHOLD = 0.21
FRAME_THRESHOLD = 20

# Cooldown: after an alert, wait this many seconds before
# triggering again — prevents the alarm looping every second
COOLDOWN_SECONDS = 10

# Path to your kirtan audio file
# Make sure this file is in the same folder as your script!
ALERT_SOUND_PATH = "alert_sound.mp3"

# ── Eye landmark indices ──────────────────────────────────────
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


# ── EAR calculation (same as before) ─────────────────────────
def calculate_ear(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
    vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# ── NEW: Pause / Resume YouTube ───────────────────────────────
#
# YouTube's keyboard shortcut 'K' toggles pause/play.
# pyautogui.press('k') simulates pressing that key.
# This works as long as your browser is the active window —
# which it will be, since you're watching a lecture!
#
def pause_youtube():
    pyautogui.press('k')
    print("⏸  YouTube paused")


def resume_youtube():
    pyautogui.press('k')
    print("▶️  YouTube resumed")


# ── NEW: Play kirtan alert ────────────────────────────────────
#
# pygame.mixer handles audio playback.
# We initialise it once, then call play() when needed.
#
def play_alert():
    try:
        pygame.mixer.music.load(ALERT_SOUND_PATH)
        pygame.mixer.music.play()
        print("🌸 Hare Krishna! Playing kirtan alert...")
    except Exception as e:
        # If audio file not found, just print a message
        # so the rest of the program still works
        print(f"Audio error: {e}")
        print("Make sure alert_sound.mp3 is in your project folder!")


def stop_alert():
    pygame.mixer.music.stop()
    print("🔇 Alert stopped")


# ── Setup ─────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialise pygame audio
pygame.mixer.init()

cap = cv2.VideoCapture(0)

# ── State variables ───────────────────────────────────────────
drowsy_counter = 0
blink_counter = 0
eye_was_closed = False

# NEW state variables
is_paused = False   # is YouTube currently paused by us?
last_alert_time = 0       # when did we last trigger an alert?

print("Week 4 running! Open YouTube and start a lecture, then switch focus")
print("back to this terminal window is NOT needed — just keep YouTube open.")
print("Press Q to quit.")

# ── Main Loop ─────────────────────────────────────────────────
while True:

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_eye_points = [
                (int(face_landmarks.landmark[i].x * w),
                 int(face_landmarks.landmark[i].y * h))
                for i in LEFT_EYE
            ]
            right_eye_points = [
                (int(face_landmarks.landmark[i].x * w),
                 int(face_landmarks.landmark[i].y * h))
                for i in RIGHT_EYE
            ]

            avg_ear = (calculate_ear(left_eye_points) +
                       calculate_ear(right_eye_points)) / 2.0

            # ── Frame counter logic ───────────────────────────
            if avg_ear < EAR_THRESHOLD:
                drowsy_counter += 1
                eye_was_closed = True
            else:
                if eye_was_closed:
                    blink_counter += 1
                drowsy_counter = 0
                eye_was_closed = False

            # ── Drowsiness action ─────────────────────────────
            current_time = time.time()

            if drowsy_counter >= FRAME_THRESHOLD:

                # Only trigger if we are not already in alert mode
                # and cooldown has passed
                if not is_paused and (current_time - last_alert_time > COOLDOWN_SECONDS):
                    pause_youtube()
                    play_alert()
                    is_paused = True
                    last_alert_time = current_time

            else:
                # Eyes are open — if we had paused, resume now
                if is_paused:
                    stop_alert()
                    resume_youtube()
                    is_paused = False

            # ── Status display ────────────────────────────────
            if drowsy_counter >= FRAME_THRESHOLD:
                status = "DROWSY! Hare Krishna! 🌸"
                colour = (0, 0, 255)
            elif avg_ear < EAR_THRESHOLD:
                status = "Eyes closing..."
                colour = (0, 165, 255)
            else:
                status = "Awake"
                colour = (0, 255, 0)

            # Draw eye dots
            for point in left_eye_points:
                cv2.circle(frame, point, 3, (0, 255, 0), -1)
            for point in right_eye_points:
                cv2.circle(frame, point, 3, (255, 0, 0), -1)

            # Display on screen
            cv2.putText(frame, f"EAR: {avg_ear:.2f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(frame, status,
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            cv2.putText(frame, f"Drowsy frames: {drowsy_counter}/{FRAME_THRESHOLD}",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Blinks: {blink_counter}",
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Show paused indicator
            if is_paused:
                cv2.putText(frame, "[ YouTube Paused ]",
                            (20, 175), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

    else:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detector — Week 4", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print(f"Session ended. Total blinks: {blink_counter}")
