from dataclasses import dataclass
from typing import Tuple, Self
import yaml
import sys

@dataclass()
class MainConfig():
    ppi: int
    # in pt
    font_size: int
    # recorded document is zoomed in/out affecting perceived font size
    zoom: float

    @staticmethod
    def defaults() -> Self:
        return MainConfig(100, 12, 1)
    
    def get_font_height(self) -> int:
        return int((self.font_size / 72.) * self.ppi * self.zoom)


class ConfigManager():
    path: str = ""

    @staticmethod
    def set_config_path(path: str):
        ConfigManager.path = path

    @staticmethod
    def read_main_config() -> Tuple[bool, MainConfig]:
        with open(ConfigManager.path, "r") as handle:
            raw_conf = yaml.safe_load(handle)
            try:
                main_conf = MainConfig(raw_conf["ppi"], 
                                    raw_conf["font_size"], 
                                    raw_conf["zoom"])
                return True, main_conf
            except:
                print("Main Configuration file is malformed", file=sys.stderr)
        return False, MainConfig.defaults()
