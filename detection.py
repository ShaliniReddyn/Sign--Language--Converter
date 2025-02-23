import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Load trained model
model_path = r"C:\Users\ShaliniN\Documents\gesture_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✅ Loaded trained model.")
else:
    print("❌ No trained model found. Train first!")
    model = None

# Labels for gestures
GESTURE_LABELS = {
    0: "Hii",
    1: "I Love You",
    2: "Bye",
    3: "Please",
    4: "Sorry",
}

# Start video capture
cap = cv2.VideoCapture(0)
last_spoken = ""  # Prevent repeated speech

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(rgb_frame)
    text = "Waiting for hand gesture..."

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract and flatten hand landmark positions
            landmarks = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_landmarks.landmark]).flatten()

            if landmarks.shape[0] == 63 and model is not None:
                try:
                    gesture_id = model.predict([landmarks])[0]
                    text = GESTURE_LABELS.get(gesture_id, "Unknown")
                    
                    # Speak the detected gesture (if changed)
                    if text != last_spoken:
                        speak(text)
                        last_spoken = text
                except Exception as e:
                    text = "Model Error"
                    print(f"Error: {e}")

    # Display detected gesture
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()