# conda activate tf2.5, mediapipe installed
# https://google.github.io/mediapipe/solutions/holistic

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Image is nor longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


# cap = cv2.VideoCapture(0)
# with mp_holistic.Holistic(
#     min_detection_confidence=0.5, min_tracking_confidence=0.5
# ) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()

#         image, results = mediapipe_detection(frame, holistic)
#         # print(results.face_landmarks)

#         draw_styled_landmarks(image, results)

#         cv2.imshow("OpenCV feed", image)

#         if cv2.waitKey(10) & 0xFF == ord("q"):
#             break

# cap.release()
# cv2.destroyAllWindows()


def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(132)
    )

    face = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.face_landmarks.landmark
            ]
        ).flatten()
        if results.face_landmarks
        else np.zeros(1872)
    )

    lh = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.left_hand_landmarks.landmark
            ]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(84)
    )

    rh = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.right_hand_landmarks.landmark
            ]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(84)
    )

    # print(pose.shape, face.shape, lh.shape, rh.shape)
    return np.concatenate([pose, face, lh, rh])


DATA_PATH = os.path.join("MP_Data")
actions = np.array(["hello", "thanks", "iloveyou"])  # Actions that we try to detect
no_sequences = 30  # 30 videos worth of data
sequence_length = 30  # videos are going to be 30 frames in length

# for action in actions:
#     for sequence in range(no_sequences):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass


# cap = cv2.VideoCapture(0)
# with mp_holistic.Holistic(
#     min_detection_confidence=0.5, min_tracking_confidence=0.5
# ) as holistic:
#     for action in actions:
#         for sequence in range(no_sequences):
#             for frame_num in range(sequence_length):
#                 ret, frame = cap.read()

#                 image, results = mediapipe_detection(frame, holistic)
#                 # print(results.face_landmarks)

#                 draw_styled_landmarks(image, results)

#                 if frame_num == 0:
#                     cv2.putText(
#                         image,
#                         "STARTING COLLECTION",
#                         (120, 200),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1,
#                         (0, 255, 0),
#                         4,
#                         cv2.LINE_AA,
#                     )
#                     cv2.putText(
#                         image,
#                         f"Collecting frames for {action} Video Number {sequence}",
#                         (15, 12),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         (0, 0, 255),
#                         1,
#                         cv2.LINE_AA,
#                     )
#                     cv2.waitKey(2000)
#                 else:
#                     cv2.putText(
#                         image,
#                         f"Collecting frames for {action} Video Number {sequence}",
#                         (15, 12),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         (0, 0, 255),
#                         1,
#                         cv2.LINE_AA,
#                     )

#                 keypoints = extract_keypoints(results)
#                 npy_path = os.path.join(
#                     DATA_PATH, action, str(sequence), str(frame_num)
#                 )
#                 np.save(npy_path, keypoints)

#                 cv2.imshow("OpenCV Feed", image)

#                 if cv2.waitKey(10) & 0xFF == ord("q"):
#                     break

# cap.release()
# cv2.destroyAllWindows()


label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(
                os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            )
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# print(np.array(sequences).shape) 90, 30, 2172
# print(np.array(labels).shape) 90,

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 2172)))
model.add(LSTM(128, return_sequences=True, activation="relu"))
model.add(LSTM(64, return_sequences=False, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))

model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

# print(model.summary())

model.fit(X_train, y_train, epochs=1000)


# model.save("action.h5")

# loaded_model = tf.keras.models.load_model("action.h5")

# res = loaded_model.predict(X_test)
# print(actions[np.argmax(res[0])])
# print(actions[np.argmax(y_test[0])])

# yhat = loaded_model.predict(X_test)

# ytrue = np.argmax(y_test, axis=1).tolist()
# yhat = np.argmax(yhat, axis=1).tolist()

# print(multilabel_confusion_matrix(ytrue, yhat))
# print(accuracy_score(ytrue, yhat))

sequence = []
sentence = []
threshold = 0.8

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(
            output_frame,
            (0, 60 + num * 40),
            (int(prob * 100), 90 + num * 40),
            colors[num],
            -1,
        )
        cv2.putText(
            output_frame,
            actions[num],
            (0, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output_frame


cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)

        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = loaded_model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])

            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            image = prob_viz(res, actions, image, colors)

        if len(sentence) > 5:
            sentence = sentence[-5:]

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(
            image,
            " ".join(sentence),
            (3, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("OpenCV Feed", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
