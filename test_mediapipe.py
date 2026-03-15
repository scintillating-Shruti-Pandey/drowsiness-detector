import mediapipe as mp
print("MediaPipe version:", mp.__version__)
print("Available attributes:", dir(mp))
try:
    print("solutions:", hasattr(mp, 'solutions'))
    if hasattr(mp, 'solutions'):
        print("solutions contents:", dir(mp.solutions))
except Exception as e:
    print("Error accessing solutions:", e)
