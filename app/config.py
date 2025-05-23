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
    "1747248814": [
        (704, 289, 725, 305, "blue"),
        (681, 322, 702, 344, "blue"),
        (595, 290, 615, 306, "green"),
        (568, 321, 588, 344, "green"),
        (489, 292, 510, 309, "orange"),
        (461, 321, 482, 345, "orange"),
        (274, 298, 296, 315, "red"),
        (238, 328, 261, 352, "red"),
        (381, 289, 403, 314, "yellow"),
        (344, 328, 367, 353, "yellow"),
    ],
    "1747249260": [
        (452, 246, 472, 270, "blue"),
        (454, 439, 474, 470, "blue"),
        (329, 318, 351, 339, "green"),
        (534, 372, 555, 403, "green"),
        (575, 312, 600, 335, "orange"),
        (378, 376, 399, 407, "orange"),
        (551, 459, 573, 481, "red"),
        (399, 298, 420, 330, "red"),
        (360, 467, 382, 490, "yellow"),
        (505, 300, 527, 331, "yellow"),
    ],
    "1747249573": [
        (539, 302, 559, 325, "blue"),
        (647, 302, 673, 329, "blue"),
        (332, 381, 356, 401, "green"),
        (565, 393, 589, 421, "green"),
        (503, 466, 526, 497, "orange"),
        (345, 208, 368, 229, "orange"),
        (589, 220, 610, 243, "red"),
        (430, 240, 458, 265, "red"),
        (266, 339, 290, 359, "yellow"),
        (500, 258, 527, 281, "yellow"),
    ],
    "1747249871": [
        (332, 326, 351, 349, "blue"),
        (382, 289, 406, 313, "blue"),
        (495, 325, 518, 346, "green"),
        (400, 426, 426, 447, "green"),
        (526, 280, 549, 298, "orange"),
        (452, 256, 475, 283, "orange"),
        (316, 273, 337, 298, "red"),
        (535, 234, 555, 259, "red"),
        (392, 240, 414, 263, "yellow"),
        (551, 348, 572, 372, "yellow"),
    ],
    "1747250118": [
        (686, 487, 707, 509, "blue"),
        (764, 351, 793, 374, "blue"),
        (409, 318, 430, 341, "green"),
        (151, 485, 180, 511, "green"),
        (629, 354, 652, 375, "orange"),
        (299, 396, 327, 418, "orange"),
        (462, 517, 482, 539, "red"),
        (451, 205, 474, 236, "red"),
        (568, 447, 588, 463, "yellow"),
        (523, 266, 545, 288, "yellow"),
    ],
    "1747250366": [
        (334, 463, 356, 485, "blue"),
        (537, 476, 556, 505, "blue"),
        (225, 239, 250, 270, "green"),
        (401, 288, 422, 318, "orange"),
        (536, 206, 558, 224, "yellow"),
        (676, 472, 709, 495, "yellow"),
    ],
    "1747250719": [
        (464, 231, 484, 255, "green"),
        (319, 468, 342, 497, "green"),
        (154, 432, 176, 456, "orange"),
        (354, 292, 373, 313, "red"),
        (487, 430, 507, 458, "red"),
        (246, 409, 270, 438, "yellow"),
    ],
    "1747250942": [
        (434, 190, 454, 218, "blue"),
        (610, 567, 630, 593, "blue"),
        (569, 231, 591, 251, "orange"),
        (538, 553, 563, 575, "orange"),
        (413, 413, 439, 441, "red"),
        (435, 482, 455, 507, "yellow"),
    ],
    "1747251155": [
        (643, 249, 672, 269, "blue"),
        (691, 318, 715, 335, "green"),
        (604, 512, 627, 532, "green"),
        (391, 274, 411, 299, "red"),
        (423, 219, 444, 242, "yellow"),
        (362, 355, 389, 376, "yellow"),
    ],
    "1747251450": [
        (480, 188, 499, 213, "blue"),
        (348, 553, 368, 580, "green"),
        (558, 257, 579, 280, "orange"),
        (438, 477, 462, 500, "orange"),
        (375, 318, 395, 341, "red"),
        (671, 436, 695, 464, "red"),
    ],
    "1747251763": [
        (622, 487, 646, 506, "blue"),
        (263, 505, 291, 527, "blue"),
        (532, 261, 554, 281, "green"),
        (262, 328, 284, 355, "green"),
        (616, 496, 643, 520, "orange"),
        (391, 292, 411, 311, "orange"),
        (650, 280, 675, 299, "red"),
        (653, 263, 679, 283, "red"),
        (604, 359, 625, 382, "yellow"),
        (384, 276, 413, 298, "yellow"),
    ],
    "1747251979": [
        (512, 227, 530, 247, "blue"),
        (325, 460, 351, 480, "blue"),
        (649, 455, 666, 475, "green"),
        (394, 288, 418, 309, "green"),
        (699, 363, 719, 384, "orange"),
        (524, 378, 553, 398, "orange"),
        (533, 362, 555, 384, "red"),
        (671, 268, 700, 289, "red"),
        (676, 285, 698, 303, "yellow"),
        (662, 453, 683, 476, "yellow"),
    ],
    "1747252147": [
        (663, 223, 686, 244, "blue"),
        (385, 380, 405, 408, "blue"),
        (395, 393, 415, 412, "green"),
        (553, 248, 580, 268, "green"),
        (628, 392, 646, 413, "orange"),
        (212, 404, 234, 428, "orange"),
        (676, 208, 696, 237, "red"),
        (121, 528, 146, 550, "red"),
        (612, 382, 634, 410, "yellow"),
        (401, 220, 425, 239, "yellow"),
    ],
    "1747252273": [
        (459, 157, 488, 179, "blue"),
        (725, 452, 747, 470, "blue"),
        (305, 334, 326, 354, "green"),
        (507, 338, 526, 365, "green"),
        (539, 510, 562, 531, "orange"),
        (188, 392, 210, 415, "orange"),
        (736, 437, 757, 460, "red"),
        (494, 331, 518, 352, "red"),
        (714, 462, 745, 485, "yellow"),
        (768, 256, 793, 276, "yellow"),
    ],
    "1747252514": [
        (467, 368, 486, 388, "blue"),
        (634, 484, 655, 506, "blue"),
        (482, 368, 497, 388, "green"),
        (568, 544, 594, 566, "green"),
        (493, 371, 513, 389, "orange"),
        (524, 192, 544, 215, "orange"),
        (493, 357, 513, 373, "red"),
        (452, 560, 476, 582, "red"),
        (404, 190, 428, 210, "yellow"),
        (662, 399, 686, 428, "yellow"),
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
