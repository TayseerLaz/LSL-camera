import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf


class CustomLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

phrase_model = tf.keras.models.load_model("action.h5", custom_objects={'LSTM': CustomLSTM})
phrase_labels = np.array(['hello', 'thanks', 'iloveyou'])

sequence = []
sequence_length = 30

class ASLRecognizer:
    def __init__(self):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path="asl_model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.CLASS_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
        self.CONFIDENCE_THRESHOLD = 0.7
        self.last_prediction_time = time.time()
        self.fps = 0

    def extract_keypoints(self, results):
        pose = np.zeros(33 * 4)
        face = np.zeros(468 * 3)
        lh = np.zeros(21 * 3)
        rh = np.zeros(21 * 3)

        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        if results.face_landmarks:
            face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()

        return np.concatenate([pose, face, lh, rh])

    def process_frame(self, frame):
        global sequence
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Predict with TFLite hand model
            results_hand = self.hands.process(rgb)
            label_asl, conf_asl = "No hand", 0.0
            if results_hand.multi_hand_landmarks:
                hand_landmarks = results_hand.multi_hand_landmarks[0]
                landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
                if len(landmarks) == 63:
                    input_data = np.array(landmarks, dtype=np.float32).reshape(1, 63)
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                    self.interpreter.invoke()
                    output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                    predicted_idx = np.argmax(output_data)
                    conf_asl = output_data[predicted_idx]
                    label_asl = self.CLASS_LABELS[predicted_idx] if conf_asl >= self.CONFIDENCE_THRESHOLD else "Low confidence"

            # Predict with LSTM phrase model
            results_holistic = self.holistic.process(rgb)
            keypoints = self.extract_keypoints(results_holistic)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]
            label_phrase, conf_phrase = "Waiting", 0.0
            if len(sequence) == sequence_length:
                prediction = phrase_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                conf_phrase = np.max(prediction)
                label_phrase = phrase_labels[np.argmax(prediction)] if conf_phrase >= self.CONFIDENCE_THRESHOLD else "Low confidence"


            if conf_phrase > conf_asl and conf_phrase >= self.CONFIDENCE_THRESHOLD:
                return label_phrase, conf_phrase
            elif conf_asl >= self.CONFIDENCE_THRESHOLD:
                return label_asl, conf_asl
            else:
                return "Low confidence", 0.0
        except Exception as e:
            return "Error", 0.0
