import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import torch
import time

# -----------------------------
# 1. TensorFlow Model Setup with a Custom LSTM
# -----------------------------
# Define a custom LSTM class that removes the unsupported 'time_major' keyword.
class CustomLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super(CustomLSTM, self).__init__(*args, **kwargs)

# Load the TensorFlow model using the custom LSTM.
tf_model = tf.keras.models.load_model("action.h5", custom_objects={'LSTM': CustomLSTM})
# Define the class labels for the sequence-based model.
actions = np.array(['hello', 'thanks', 'iloveyou'])

# -----------------------------
# 2. PyTorch Model Setup for Hand Landmark Classification
# -----------------------------
# Define the PyTorch model architecture.
class LandmarkClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(63, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 29)
        )
    
    def forward(self, x):
        return self.model(x)

# Set device: use GPU if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pt_model = LandmarkClassifier().to(device)
pt_model.load_state_dict(torch.load("asl_landmark_model.pth", map_location=device))
pt_model.eval()

# Define class labels for the PyTorch model (ASL alphabet and extra commands).
CLASS_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]
# Confidence threshold to accept predictions.
CONFIDENCE_THRESHOLD = 0.7

# -----------------------------
# 3. MediaPipe Setup for Holistic Detection
# -----------------------------
# Initialize MediaPipe Holistic for face, pose, and hand detection.
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# 4. Helper Function to Extract Keypoints
# -----------------------------
def extract_keypoints(results):
    """
    Extract keypoints for pose, face, left hand, and right hand.
    If landmarks are not detected, returns an array of zeros.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                      for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] 
                      for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] 
                    for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] 
                    for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# -----------------------------
# 5. Initialize Variables and Webcam
# -----------------------------
# Store the last 30 frames' keypoints for sequence prediction.
sequence = []
sequence_length = 30

# For FPS calculation.
prev_time = time.time()

# Open the webcam.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

# -----------------------------
# 6. Main Loop: Process Each Frame
# -----------------------------
with mp_holistic.Holistic(min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        # Mirror the frame horizontally.
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB for MediaPipe processing.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Improve performance.
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw detected landmarks on the frame.
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # -----------------------------
        # Sequence-Based Prediction using TensorFlow Model
        # -----------------------------
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]  # Keep only the latest 30 frames.

        if len(sequence) == sequence_length:
            input_data = np.expand_dims(sequence, axis=0)  # Shape: (1, 30, number_of_keypoints)
            predictions = tf_model.predict(input_data)[0]
            tf_confidence = np.max(predictions)
            tf_label = actions[np.argmax(predictions)]
            tf_prediction_text = f"{tf_label.upper()} ({tf_confidence:.2f})"
            cv2.putText(image, f"Seq: {tf_prediction_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # -----------------------------
        # Hand Landmark Prediction using PyTorch Model
        # -----------------------------
        pt_prediction_text = "No hand detected"
        hand_landmarks = None
        flip_landmarks = False  # Flag to indicate whether to mirror landmarks

        # Prefer the left hand if available; if not, use the right hand.
        if results.left_hand_landmarks:
            hand_landmarks = results.left_hand_landmarks
        elif results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks
            flip_landmarks = True  # We'll flip x-coordinates for the right hand.

        if hand_landmarks is not None:
            landmarks = []
            # Process each landmark.
            for lm in hand_landmarks.landmark:
                # If using the right hand, flip the x-coordinate (mirror horizontally).
                x_val = 1.0 - lm.x if flip_landmarks else lm.x
                landmarks.extend([x_val, lm.y, lm.z])
            # Check if we have exactly 63 values (21 landmarks * 3 coordinates).
            if len(landmarks) == 63:
                landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).to(device)
                with torch.no_grad():
                    outputs = pt_model(landmarks_tensor.unsqueeze(0))
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                if confidence.item() >= CONFIDENCE_THRESHOLD:
                    class_label = CLASS_LABELS[predicted_idx.item()]
                    if class_label == 'space':
                        pt_prediction_text = "SPACE"
                    elif class_label == 'del':
                        pt_prediction_text = "DELETE"
                    elif class_label == 'nothing':
                        pt_prediction_text = "NO SIGN"
                    else:
                        pt_prediction_text = class_label
                    pt_prediction_text += f" ({confidence.item():.2f})"
                else:
                    pt_prediction_text = "Low confidence"
                # Optionally, redraw hand landmarks.
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        cv2.putText(image, f"Hand: {pt_prediction_text}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # -----------------------------
        # FPS Calculation and Display
        # -----------------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(image, f"FPS: {int(fps)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the final output.
        cv2.imshow("Sign Language & ASL Recognition", image)

        # Exit loop when 'q' is pressed.
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()
print("[INFO] Program finished successfully.")
