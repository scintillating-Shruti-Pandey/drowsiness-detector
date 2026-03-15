# ============================================================
#  DROWSINESS DETECTOR — Week 1
#  What this does: Opens your webcam and draws face landmarks
#  on your face in real time. No drowsiness logic yet —
#  just making sure everything is working first.
# ============================================================

import cv2                      # for webcam access and drawing on screen
import mediapipe as mp          # for detecting your face and landmarks

# ── Setup MediaPipe ──────────────────────────────────────────
# MediaPipe's Face Mesh finds 468 points on your face
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils   # helper to draw the dots

# Create the face mesh detector
# min_detection_confidence: how sure it needs to be before saying "yes, a face"
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── Open Webcam ───────────────────────────────────────────────
# 0 means "use the first (default) webcam on your laptop"
cap = cv2.VideoCapture(0)

print("Webcam started! Press Q to quit.")

# ── Main Loop ─────────────────────────────────────────────────
# This loop runs continuously, reading one frame at a time
while True:

    # Read one frame from the webcam
    # success = True if frame was read correctly
    # frame   = the actual image (like a photo)
    success, frame = cap.read()

    if not success:
        print("Could not read from webcam. Check if it is connected.")
        break

    # Flip the frame horizontally so it acts like a mirror
    # (feels more natural — your left hand appears on left side)
    frame = cv2.flip(frame, 1)

    # MediaPipe needs RGB color, but OpenCV gives BGR
    # So we convert the color format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run face mesh detection on this frame
    results = face_mesh.process(rgb_frame)

    # ── Draw landmarks if a face was found ────────────────────
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Draw all 468 landmark dots on the face
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,   # mesh lines
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),   # green dots
                    thickness=1,
                    circle_radius=1
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 180, 0),   # slightly darker green lines
                    thickness=1
                )
            )

        # Show a small status message on screen
        cv2.putText(
            frame,
            "Face detected!",          # text to display
            (20, 40),                  # position on screen (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # font style
            0.8,                       # font size
            (0, 255, 0),               # color (green in BGR)
            2                          # thickness
        )
    else:
        # No face found — maybe you moved away or lighting is bad
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ── Show the frame on screen ──────────────────────────────
    cv2.imshow("Drowsiness Detector — Week 1", frame)

    # Wait 1ms for a key press. If Q is pressed, quit the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# ── Cleanup ───────────────────────────────────────────────────
# Always release the webcam and close windows when done
cap.release()
cv2.destroyAllWindows()
print("Done.")
