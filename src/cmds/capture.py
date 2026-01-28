from ..util import save_location
from pynput import keyboard
from typing import Tuple
import csv
import time
import string
import click

def key_to_char(key: keyboard.Key) -> Tuple[bool, str]:
    k = str(key).strip("'")
    if k in (string.digits + string.ascii_uppercase + string.ascii_lowercase):
        return True, k
    return False, k

def time_ms():
    return int(time.time_ns() / 1000_000)

# on_release is not used
# to match how data is collected from the video
# where the release event is detected when another character appears
def capture_command(dest: str):
    dest_path = save_location(dest, "capture")

    start_time = time_ms()

    with open(str(dest_path / "biometry.csv"), "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["PressTime", "ReleaseTime", "KeyDelay", "KeyText"])

        last_press = time_ms()
        last_char = ''

        def on_press(key: keyboard.Key):
            nonlocal last_press
            nonlocal last_char
            pt = time_ms()
            if key == keyboard.Key.esc:
                return False
            s, c = key_to_char(key)
            if s:
                writer.writerow([last_press - start_time, pt - start_time, pt - last_press, last_char])
                last_press = pt
                last_char = c


        with keyboard.Listener(
            on_press=on_press
        ) as listener:
            click.echo("Stop the capture by pressing esc...")
            listener.join()
