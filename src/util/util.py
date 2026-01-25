from typing import Tuple
from pathlib import Path
import time

def cbb_to_ibb(x, y, w, h) -> Tuple[int, int, int, int]:
    diff = max(2*h - w, 0)
    return (x-diff, y, w+diff, h)

def save_location(dest: str, category: str) -> Path:
        timestamp = time.strftime("%Y%m%d%H%M%S")
        path = Path(dest) / f"{category}-{timestamp}"
        path.mkdir(parents=True, exist_ok=False)
        return path
