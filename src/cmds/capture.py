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

def capture_command(dest: str):
    dest_path = save_location(dest, "capture")

    start_time = int(time.time_ns() / 1000_000)

    with open(str(dest_path / "biometry.csv"), "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["PressTime", "ReleaseTime", "KeyDelay", "KeyText"])

        press_events = {}

        def on_press(key: keyboard.Key):
            pt = int(time.time_ns() / 1000_000)
            if key == keyboard.Key.esc:
                return False
            s, c = key_to_char(key)
            if s:
                press_events[c] = pt

        def on_release(key):
            rt = int(time.time_ns() / 1000_000)
            s, c = key_to_char(key)
            if s:
                pt = press_events.pop(c, 0)
                if pt:
                    writer.writerow([pt - start_time, rt - start_time, rt - pt, c])

        with keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        ) as listener:
            click.echo("Stop the capture by pressing esc...")
            listener.join()
