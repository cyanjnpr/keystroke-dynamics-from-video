import cv2
import numpy as np
from cv2.typing import MatLike
from typing import Tuple
from math import sqrt
import cbb
import ibb
from resnet import load_model, predict, prediction_to_char

def main(font_size: int = 12, save_video: bool = False):
    src = cv2.VideoCapture("res/video.mp4")

    frame_width = int(src.get(3))
    frame_height = int(src.get(4))
    fps = src.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    if (save_video): out = cv2.VideoWriter("res/tracking.mp4", codec, fps, (frame_width, frame_height))

    cursor_positions = []
    status, frame_p = src.read()
    if not status: raise Exception()

    status, frame = src.read()
    i = 1
    while status:
        contour = cbb.cursor_detection(frame, frame_p, cursor_positions[-1][2] if len(cursor_positions) > 0 else None)
        if (contour is not None):
            if (len(cursor_positions) == 0 or cbb.contour_distance(contour, cursor_positions[-1][2]) > 1):
                cursor_positions.append((i / float(fps), frame.copy(), contour))
        for _, _, contour in cursor_positions:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_p, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.imshow('image', frame_p)
        cv2.waitKey(1)
        # out.write(frame_p)
        frame_p = frame.copy()
        status, frame = src.read()
        i += 1

    cursor_positions = cbb.clear_anomalies(cursor_positions)
    for _, _, contour in cursor_positions:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame_p, (x - h, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.rectangle(frame_p, (x, y), (x + w, y + h), (0, 0, 255), 1)
    out.write(frame_p)
    cv2.imshow('image', frame_p)
    cv2.waitKey(100)

    # model = load_model("models/202512071647/model.keras")
    model = load_model("models/202512081642/model.keras")

    for pos in cursor_positions:
        x_pos, y_pos, w_pos, h_pos = cv2.boundingRect(pos[2])
        frame_p2 = frame_p.copy()
        rcc = ibb.extract_rc(*pos)
        if (rcc is not None):
            rcc_copy = cv2.cvtColor(rcc, cv2.COLOR_GRAY2BGR)
            x, y, w, h = cv2.boundingRect(rcc)
            w *= 5
            h *= 5
            rcc = cv2.resize(rcc, (w, h))
            x, y, w, h = cv2.boundingRect(rcc)
            rcc = cv2.cvtColor(rcc, cv2.COLOR_GRAY2BGR)
            frame_p2[0:h, 0:w] = rcc[y:y+h, x:x+w]
            image_path = "res/{}.png".format(round(pos[0], 3))
            cv2.imwrite(image_path, rcc_copy)
            prediction = prediction_to_char(predict(image_path, model))
            cv2.rectangle(frame_p2, (x_pos, y_pos), (x_pos + w_pos, y_pos + h_pos), (0, 255, 0), 3)
            cv2.putText(frame_p2, "Predicted char: {}, Time: {}s".format(prediction, round(pos[0], 2)), (10, 10 + h + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow('image', frame_p2)
            print("Predicting -------------")
            print("Time: {}s".format(round(pos[0], 3)))
            print(f'Prediction: {prediction}')
            print("-------------")

            for i in range(20): out.write(frame_p2)
            cv2.waitKey(10)

    if (save_video): out.release()
    src.release()
        
if __name__ == "__main__":
    main(16, True)