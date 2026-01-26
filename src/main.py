from .config import ConfigManager
ConfigManager.set_config_path("default.conf")

import click
from .cmds import (train_command, analyze_command, 
    cbb_command, ibb_command, kunit_command)

@click.group()
@click.version_option("0.0.1")
@click.pass_context
def cli(ctx):
    pass

@click.command("train")
@click.option("-f", "--fallback", is_flag=True, help="Use archive link for dataset")
@click.pass_context
def train(ctx, fallback: bool):
    """
    download dataset and train the ResNet model with hyperparameters matching the one from the paper
    """
    train_command(fallback)

@click.command("analyze")
@click.argument("filename")
@click.argument("dest")
@click.pass_context
def analyze(ctx, filename: str, dest: str):
    """
    extract keystroke dynamics from existing video file
    """
    analyze_command(filename, dest)

@click.command("cbb")
@click.argument("filename")
@click.argument("dest")
@click.pass_context
def cbb(ctx, filename: str, dest: str):
    """
    detect cursor bounding box in each frame and save to a directory dest
    """
    cbb_command(filename, dest)

@click.command("ibb")
@click.argument("filename")
@click.argument("dest")
@click.pass_context
def ibb(ctx, filename: str, dest: str):
    """
    detect isolation bounding box in each frame and save to a directory dest
    """
    ibb_command(filename, dest)


@click.command("kunit")
@click.argument("filename")
@click.argument("dest")
@click.option("-c", "--convexity", is_flag=True, help="Draw convexity of the character")
@click.pass_context
def kunit(ctx, filename: str, dest: str, convexity: bool):
    """
    detect rightmost character in each isolation bounding box and save to a directory dest
    """
    kunit_command(filename, dest, convexity)


cli.add_command(train)
cli.add_command(analyze)
cli.add_command(cbb)
cli.add_command(ibb)
cli.add_command(kunit)

# def main(save_video: bool = False):
#     model = get_model()

#     src = cv2.VideoCapture("res/video.mp4")

#     frame_width = int(src.get(3))
#     frame_height = int(src.get(4))
#     fps = src.get(cv2.CAP_PROP_FPS)
#     codec = cv2.VideoWriter_fourcc(*'mp4v')
#     if (save_video): out = cv2.VideoWriter("res/tracking.mp4", codec, fps, (frame_width, frame_height))

#     cursor_positions = []
#     status, frame_p = src.read()
#     if not status: raise Exception()

#     status, frame = src.read()
#     i = 1
#     while status:
#         contour = cbb.cursor_detection(frame, frame_p, cursor_positions[-1][2] if len(cursor_positions) > 0 else None)
#         if (contour is not None):
#             if (len(cursor_positions) == 0 or cbb.contour_distance(contour, cursor_positions[-1][2]) > 1):
#                 cursor_positions.append((i, frame.copy(), contour))
#         for _, _, contour in cursor_positions:
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(frame_p, (x, y), (x + w, y + h), (0, 0, 255), 1)
#         frame_p = frame.copy()
#         status, frame = src.read()
#         i += 1

#     cursor_positions = cbb.clear_anomalies(cursor_positions)
#     for _, _, contour in cursor_positions:
#         # frame_pp = frame_p.copy()
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(frame_p, (x - h, y), (x + w, y + h), (255, 0, 0), 1)
#         cv2.rectangle(frame_p, (x, y), (x + w, y + h), (0, 0, 255), 1)
#         # cv2.imshow('image', frame_pp)
#         # cv2.waitKey(0)
#     out.write(frame_p)
#     cv2.imshow('image', frame_p)
#     cv2.waitKey(100)

#     text = ""
#     previous_kunit = None
#     for pos in cursor_positions:
#         x_pos, y_pos, w_pos, h_pos = cv2.boundingRect(pos[2])
#         frame_p2 = frame_p.copy()
#         kunit = ibb.extract_rc(*pos)
#         rcc = kunit.image
#         if (rcc is not None and (previous_kunit is None or not previous_kunit.is_the_same(kunit))):
#             rcc_copy = cv2.cvtColor(rcc, cv2.COLOR_GRAY2BGR)
#             x, y, w, h = cv2.boundingRect(rcc)
#             w, h = 128, 128
#             rcc = cv2.resize(rcc, (w, h))
#             x, y, w, h = cv2.boundingRect(rcc)
#             rcc = cv2.cvtColor(rcc, cv2.COLOR_GRAY2BGR)
#             cv2.rectangle(frame_p2, (0, 0), (512 + 256, 256), (0, 0, 0), -1)
#             frame_p2[0:h, 0:w] = rcc[y:y+h, x:x+w]
#             image_path = "res/{}.png".format(round(pos[0], 3))
#             cv2.imwrite(image_path, rcc_copy)
#             prediction = predict(image_path, model)
#             cv2.rectangle(frame_p2, (x_pos, y_pos), (x_pos + w_pos, y_pos + h_pos), (0, 255, 0), 3)
#             cv2.putText(frame_p2, "Predicted char: {}, Acc: {}%, Time: {}s".format(prediction.character, round(100 * prediction.accuracy), round(pos[0], 2)), (10, 10 + h + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
#             cv2.imshow('image', frame_p2)
#             print("Predicting -------------")
#             print("Time: {}s".format(round(pos[0], 3)))
#             print(f'Prediction: {prediction.character}')
#             print("-------------")

#             if (prediction.accuracy >= 0.7): text = text + prediction.character
#             cv2.putText(frame_p2, "{}".format(text), (10, frame_height - 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
#             for i in range(15): out.write(frame_p2)
#             # cv2.waitKey(10)
#             # cv2.waitKey(0)
#             previous_kunit = kunit
#     print("Here's the recovered text: ", text)

#     if (save_video): out.release()
#     src.release()
        