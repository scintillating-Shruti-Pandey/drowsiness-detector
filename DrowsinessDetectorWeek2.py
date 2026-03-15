# ============================================================
#  DROWSINESS DETECTOR — Week 2
#  New this week: Extract eye landmarks, calculate EAR,
#  and print the value live so you can watch it drop
#  when you close your eyes!
# ============================================================

import cv2
import mediapipe as mp
import numpy as np    # new! needed for distance calculations

# ── MediaPipe landmark index numbers for each eye ────────────
#
# Out of 468 landmarks, these are the 6 that matter for each eye.
# These numbers come from MediaPipe's official face map.
# Think of them like seat numbers in a class — fixed positions.
#
#         p2  p3
#    p1             p4       <- left eye
#         p6  p5
#
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
#            p1   p2   p3   p4   p5   p6


# ── Function: calculate EAR for one eye ──────────────────────
#
# We pass in the 6 landmark points and it returns a number.
# Higher number = eye more open. Lower = eye more closed.
#
def calculate_ear(eye_points):

    # eye_points is a list of 6 (x, y) coordinates
    # We label them p1 through p6 to match the diagram

    p1, p2, p3, p4, p5, p6 = eye_points

    # Calculate the two vertical distances
    # np.linalg.norm = straight-line distance between two points
    # (like using a ruler between two dots on paper)
    vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))  # p2 to p6
    vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))  # p3 to p5

    # Calculate the horizontal distance
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))  # p1 to p4

    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


# ── Setup (same as Week 1) ───────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
print("Webcam started! Watch the EAR value — close your eyes and see it drop!")
print("Press Q to quit.")

# ── Main Loop ────────────────────────────────────────────────
while True:

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Get the actual pixel size of the frame
            # We need this to convert landmark positions to pixel coordinates
            h, w, _ = frame.shape
            # h = height in pixels, w = width in pixels
            # _ = colour channels (3 for BGR) — we don't need this

            # ── Extract eye landmark coordinates ─────────────
            #
            # MediaPipe gives positions as fractions (0.0 to 1.0)
            # e.g. x=0.5 means "halfway across the image"
            # We multiply by w and h to get actual pixel positions
            #
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

            # ── Calculate EAR for both eyes ───────────────────
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)

            # Average of both eyes gives a more stable reading
            avg_ear = (left_ear + right_ear) / 2.0

            # ── Draw the 6 eye points on screen ──────────────
            # Green dots on left eye, blue dots on right eye
            for point in left_eye_points:
                cv2.circle(frame, point, 3, (0, 255, 0), -1)   # green

            for point in right_eye_points:
                cv2.circle(frame, point, 3, (255, 0, 0), -1)   # blue

            # ── Show EAR value on screen ──────────────────────
            # Round to 2 decimal places so it's readable
            ear_text = f"EAR: {avg_ear:.2f}"

            # Choose colour based on whether EAR is below threshold
            if avg_ear < 0.25:
                colour = (0, 0, 255)    # red = drowsy!
                status = "DROWSY!"
            else:
                colour = (0, 255, 0)    # green = awake
                status = "Awake"

            # Put text on the frame
            cv2.putText(frame, ear_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

            cv2.putText(frame, status, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

            # Also print to terminal so you can see live values
            print(f"EAR: {avg_ear:.3f}  |  Status: {status}")

    else:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detector — Week 2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("Done.")
