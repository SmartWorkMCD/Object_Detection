from classes.Config import Config
from dataclasses import fields
from dotenv import load_dotenv
import os

load_dotenv(override=True)


# color: class_id
COLOR_CLASSES = {
    "blue": 0,
    "green": 1,
    "orange": 2,
    "red": 3,
    "yellow": 4,
}

# color: (red, green, blue)
COLOR_VALUES = {
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "orange": (255, 165, 0),
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
}

# video_filename: (x1, y1, x2, y2, mask_color)
OBJECTS_CONFIG = {
    "1742573138": [
        (305, 306, 320, 320, "blue"),
        (294, 332, 314, 347, "blue"),
        (301, 273, 321, 287, "green"),
        (336, 327, 350, 341, "green"),
        (265, 253, 286, 268, "orange"),
        (337, 311, 351, 329, "orange"),
        (273, 271, 289, 286, "red"),
        (323, 299, 342, 314, "red"),
        (349, 246, 365, 261, "yellow"),
        (352, 276, 367, 295, "yellow"),
    ],
    "1742573249": [
        (290, 265, 308, 279, "blue"),
        (277, 294, 291, 310, "blue"),
        (308, 282, 319, 297, "green"),
        (314, 309, 335, 324, "green"),
        (336, 289, 350, 306, "orange"),
        (337, 329, 351, 351, "orange"),
        (255, 314, 271, 328, "red"),
        (358, 323, 373, 341, "red"),
        (329, 211, 344, 228, "yellow"),
        (349, 216, 367, 231, "yellow"),
    ],
}


# =========================
# MQTT Configuration below
# =========================


def string_to_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in ["true", "1"]:
        return True
    elif s in ["false", "0"]:
        return False
    else:
        raise ValueError(
            f"Cannot convert '{s}' to boolean. Expected 'true' or 'false'."
        )


def init_config():
    config = Config()
    for field in fields(Config):
        value = os.getenv(field.name)
        if value:
            try:
                if field.type == bool:
                    value = string_to_bool(value)
                else:
                    value = field.type(value)
                setattr(config, field.name, value)
            except Exception as e:
                raise ValueError(f"Error setting {field.name} to {value}: {e}")
    return config


def validate_config(config: Config):
    for field in fields(config):
        value = getattr(config, field.name)
        expected_type = field.type
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Config value {field.name} must be of type {expected_type}, but got {type(value)}"
            )
    return config


MQTT_CONFIG = validate_config(init_config())
