import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from sklearn.svm import SVC  # Using SVM for better accuracy
from collections import Counter

# Load MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

GESTURES = ["Hi", "I Love You", "Bye", "Please", "Sorry"]
SAMPLES_PER_GESTURE = 100  # More samples for better accuracy

data, labels = [], []

cap = cv2.VideoCapture(0)

print("Show gestures one by one and press 'c' to capture data.")

for gesture_id, gesture in enumerate(GESTURES):
    print(f"Show gesture: {gesture} and press 'c' to capture, 'q' to quit.")
    captured_count = 0

    while captured_count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract and flatten hand landmark positions
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                
                cv2.putText(frame, f"Gesture: {gesture} ({captured_count}/{SAMPLES_PER_GESTURE})",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Collecting Data", frame)
        key = cv2.waitKey(10)

        if key == ord("c") and result.multi_hand_landmarks:
            data.append(landmarks)
            labels.append(gesture_id)
            captured_count += 1
            print(f"Captured {captured_count}/{SAMPLES_PER_GESTURE} for: {gesture}")

        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

# Ensure consistent data shape
data = [sample for sample in data if len(sample) == 63]
labels = labels[:len(data)]

if len(data) == 0:
    print("Error: No valid data collected! Please try again.")
    exit()

# Print collected data info
gesture_counts = Counter(labels)
for gesture_id, count in gesture_counts.items():
    print(f"Samples collected for {GESTURES[gesture_id]}: {count}")

# Train an SVM model
model = SVC(kernel="linear")  # Linear kernel works well for gestures
model.fit(data, labels)

# Save the trained model
model_path = r"C:\Users\ShaliniN\Documents\gesture_model.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Training complete. Model saved at {model_path}.")
