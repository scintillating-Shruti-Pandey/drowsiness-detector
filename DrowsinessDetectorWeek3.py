# ============================================================
#  DROWSINESS DETECTOR — Week 3
#  New this week:
#  1. Frame counter — must be drowsy for 20+ frames to trigger
#  2. Calibrated threshold — set to 0.21 based on real testing
#  3. Blink counter — just for fun, counts your blinks!
# ============================================================

import cv2
import mediapipe as mp
import numpy as np

# ── Constants ────────────────────────────────────────────────
#
# EAR_THRESHOLD: below this = eye is closing
# Calibrated for Shruti's eyes — open EAR ~0.26, closed ~0.17
# We pick 0.21 as the midpoint
#
EAR_THRESHOLD = 0.21

#
# FRAME_THRESHOLD: how many consecutive low-EAR frames
# before we say "this is drowsiness, not a blink"
# At 30fps, 20 frames = about 0.66 seconds
#
FRAME_THRESHOLD = 20

# ── Eye landmark indices (same as Week 2) ────────────────────
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


# ── EAR calculation function (same as Week 2) ────────────────
def calculate_ear(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
    vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


# ── Setup ────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# ── NEW: Counters ─────────────────────────────────────────────
#
# drowsy_counter: counts consecutive frames where EAR < threshold
# When this hits FRAME_THRESHOLD, we trigger the alert
# When EAR goes back up, this resets to 0
#
drowsy_counter = 0

#
# blink_counter: counts completed blinks
# A blink = EAR dips low (< threshold) then comes back up
# We detect the "came back up" moment to count it
#
blink_counter = 0
eye_was_closed = False   # tracks whether eye was closed last frame

print("Week 3 running! Press Q to quit.")
print(f"Threshold: {EAR_THRESHOLD} | Frames needed: {FRAME_THRESHOLD}")

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

            # Extract eye points
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

            # Calculate EAR
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0

            # ── Frame counter logic ───────────────────────────
            if avg_ear < EAR_THRESHOLD:
                # Eye is closing this frame — increment counter
                drowsy_counter += 1
                eye_was_closed = True
            else:
                # Eye is open this frame

                # If eye just opened after being closed = completed blink
                if eye_was_closed:
                    blink_counter += 1

                # Reset the drowsy counter — streak is broken
                drowsy_counter = 0
                eye_was_closed = False

            # ── Decide status based on counter ────────────────
            if drowsy_counter >= FRAME_THRESHOLD:
                status = "DROWSY!"
                colour = (0, 0, 255)    # red
            elif avg_ear < EAR_THRESHOLD:
                status = "Eyes closing..."
                colour = (0, 165, 255)  # orange — warning
            else:
                status = "Awake"
                colour = (0, 255, 0)    # green

            # ── Draw eye dots ─────────────────────────────────
            for point in left_eye_points:
                cv2.circle(frame, point, 3, (0, 255, 0), -1)
            for point in right_eye_points:
                cv2.circle(frame, point, 3, (255, 0, 0), -1)

            # ── Display info on screen ────────────────────────
            cv2.putText(frame, f"EAR: {avg_ear:.2f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

            cv2.putText(frame, status,
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

            cv2.putText(frame, f"Drowsy frames: {drowsy_counter}/{FRAME_THRESHOLD}",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)

            cv2.putText(frame, f"Blinks: {blink_counter}",
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)

    else:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detector — Week 3", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print(f"Session ended. Total blinks: {blink_counter}")
