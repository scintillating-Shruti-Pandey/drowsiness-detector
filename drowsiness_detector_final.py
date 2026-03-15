# ============================================================
#  DROWSINESS DETECTOR
#  Author : Shruti Pandey
#  GitHub : github.com/scintillating-Shruti-Pandey
#
#  What it does:
#  Monitors your eyes in real time while you study.
#  If you fall asleep, it pauses your YouTube lecture
#  and plays an alert to wake you up.
#  Resumes automatically when your eyes open again.
#
#  Tech: Python · OpenCV · MediaPipe · PyAutoGUI · Pygame
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pygame
import time

# ── Configuration ─────────────────────────────────────────────
# You can change these values to tune the detector for your eyes

EAR_THRESHOLD = 0.21   # below this = eye closing
# calibrated from real testing:
# open eyes ~0.26, closed ~0.17

FRAME_THRESHOLD = 20     # consecutive low-EAR frames before alert
# at ~30fps this is about 0.66 seconds
# long enough to ignore normal blinks

COOLDOWN_SECONDS = 10     # seconds to wait before triggering again
# prevents the alert looping every second

ALERT_SOUND_PATH = "alert_sound.mp3"   # put this file in same folder

# ── Eye landmark indices (MediaPipe Face Mesh) ────────────────
#
#         p2  p3
#    p1             p4
#         p6  p5
#
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


# ═════════════════════════════════════════════════════════════
#  FUNCTIONS
# ═════════════════════════════════════════════════════════════

def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR) for one eye.

    EAR = (vertical_1 + vertical_2) / (2 * horizontal)

    When eye is open  → EAR is high (~0.30)
    When eye is closed → EAR drops (~0.10)
    """
    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
    vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def extract_eye_points(face_landmarks, eye_indices, frame_w, frame_h):
    """
    Extract pixel coordinates for given landmark indices.

    MediaPipe gives positions as fractions (0.0 to 1.0).
    We multiply by frame width/height to get pixel positions.
    """
    return [
        (int(face_landmarks.landmark[i].x * frame_w),
         int(face_landmarks.landmark[i].y * frame_h))
        for i in eye_indices
    ]


def pause_youtube():
    """ Press K to pause YouTube (YouTube keyboard shortcut) """
    pyautogui.press('k')
    print("⏸  YouTube paused")


def resume_youtube():
    """ Press K again to resume YouTube """
    pyautogui.press('k')
    print("▶️  YouTube resumed")


def play_alert(sound_path):
    """ Play the kirtan alert sound """
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        print("🌸 Hare Krishna! Playing alert...")
    except Exception as e:
        print(f"Audio error: {e}")
        print(f"Make sure {sound_path} is in your project folder!")


def stop_alert():
    """ Stop the alert sound """
    pygame.mixer.music.stop()


def draw_ui(frame, avg_ear, drowsy_counter, blink_counter, is_paused,
            left_eye_points, right_eye_points):
    """
    Draw all UI elements on the frame:
    eye dots, EAR value, status, counters.
    """
    # Decide colour and status text
    if drowsy_counter >= FRAME_THRESHOLD:
        colour = (0, 0, 255)        # red
        status = "DROWSY! Wake up!"
    elif avg_ear < EAR_THRESHOLD:
        colour = (0, 165, 255)      # orange — warning
        status = "Eyes closing..."
    else:
        colour = (0, 255, 0)        # green
        status = "Awake"

    # Draw eye landmark dots
    for point in left_eye_points:
        cv2.circle(frame, point, 3, (0, 255, 0), -1)    # green = left
    for point in right_eye_points:
        cv2.circle(frame, point, 3, (255, 0, 0), -1)    # blue  = right

    # Status and EAR
    cv2.putText(frame, f"EAR: {avg_ear:.2f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(frame, status,
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

    # Counters
    cv2.putText(frame, f"Drowsy frames: {drowsy_counter}/{FRAME_THRESHOLD}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Blinks: {blink_counter}",
                (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Paused indicator
    if is_paused:
        cv2.putText(frame, "[ YouTube Paused ]",
                    (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════

def main():

    # ── Initialise MediaPipe ──────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ── Initialise audio ─────────────────────────────────────
    pygame.mixer.init()

    # ── Initialise webcam ─────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # ── State variables ───────────────────────────────────────
    drowsy_counter = 0
    blink_counter = 0
    eye_was_closed = False
    is_paused = False
    last_alert_time = 0

    print("=" * 50)
    print("  Drowsiness Detector — by Shruti Pandey")
    print("=" * 50)
    print("Open YouTube and start a lecture.")
    print("Press Q to quit.\n")

    # ── Main loop ─────────────────────────────────────────────
    while True:

        success, frame = cap.read()
        if not success:
            print("Webcam read failed.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape

            # Extract eye coordinates
            left_eye_points = extract_eye_points(
                face_landmarks, LEFT_EYE,  w, h)
            right_eye_points = extract_eye_points(
                face_landmarks, RIGHT_EYE, w, h)

            # Calculate average EAR
            avg_ear = (calculate_ear(left_eye_points) +
                       calculate_ear(right_eye_points)) / 2.0

            # ── Update frame counter ──────────────────────────
            if avg_ear < EAR_THRESHOLD:
                drowsy_counter += 1
                eye_was_closed = True
            else:
                if eye_was_closed:
                    blink_counter += 1       # completed blink
                drowsy_counter = 0
                eye_was_closed = False

            # ── Trigger or clear alert ────────────────────────
            current_time = time.time()

            if drowsy_counter >= FRAME_THRESHOLD:
                if not is_paused and (current_time - last_alert_time > COOLDOWN_SECONDS):
                    pause_youtube()
                    play_alert(ALERT_SOUND_PATH)
                    is_paused = True
                    last_alert_time = current_time
            else:
                if is_paused:
                    stop_alert()
                    resume_youtube()
                    is_paused = False

            # ── Draw UI ───────────────────────────────────────
            frame = draw_ui(frame, avg_ear, drowsy_counter, blink_counter,
                            is_paused, left_eye_points, right_eye_points)

        else:
            cv2.putText(frame, "No face detected — adjust your camera",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Drowsiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Cleanup ───────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    face_mesh.close()

    print(f"\nSession ended.")
    print(f"Total blinks counted: {blink_counter}")


# ── Entry point ───────────────────────────────────────────────
# This means: only run main() if we run THIS file directly.
# If someone imports this file into another project,
# main() won't run automatically.
if __name__ == "__main__":
    main()
