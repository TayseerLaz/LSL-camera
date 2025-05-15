import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

def normalize_landmarks(landmarks):
    lm_array = np.array(landmarks, dtype=np.float32).reshape(21, 3)
    wrist = lm_array[0]
    lm_centered = lm_array - wrist
    dists = np.linalg.norm(lm_centered, axis=1)
    scale = np.max(dists)
    if scale > 0:
        lm_normalized = lm_centered / scale
    else:
        lm_normalized = lm_centered
    return lm_normalized.flatten()

class ASLRecognizerArabic:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path="model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.CLASS_LABELS = [
            "NA",  # Index 0 = Not used
            "أ", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر",
            "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف",
            "ق", "ك", "ل", "م", "ن", "هـ", "و", "ي",
        ]
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            if len(landmarks) == 63:
                normalized = normalize_landmarks(landmarks)
                input_data = np.expand_dims(normalized, axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                pred_label = int(np.argmax(output_data))
                confidence = float(np.max(output_data))
                return self.CLASS_LABELS[pred_label], confidence
        return "لا يد", 0.0


# Keep your main() webcam logic untouched
def main():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded.", flush=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.", flush=True)
        return

    print("Starting webcam stream...", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame.", flush=True)
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        prediction_text = "No hand detected"
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == 63:
                normalized = normalize_landmarks(landmarks)
                input_data = np.expand_dims(normalized, axis=0)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                pred_label = int(np.argmax(output_data))
                confidence = np.max(output_data)
                prediction_text = f"Predicted: {pred_label} ({confidence:.2f})"
            else:
                prediction_text = "Incomplete landmarks"

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stream closed.", flush=True)

if __name__ == "__main__":
    main()
