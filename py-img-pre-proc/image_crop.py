import sys
from typing import Tuple, Dict, TextIO, Optional

import numpy as np
from PIL import Image
import face_recognition as fr

from common import CSV_SEP


def find_face_avg(image_array: np.array) -> Optional[Tuple[float, float]]:
    face_locations = fr.face_locations(image_array)
    face_locations_len = len(face_locations)

    if face_locations_len == 0:
        return None

    c_x, c_y = 0.0, 0.0

    for top, right, bottom, left in face_locations:
        c_x += (left + right) / 2
        c_y += (top + bottom) / 2

    return c_x / face_locations_len, c_y / face_locations_len


def remove_bars(image: Image, black_threshold: int = 10, white_threshold: int = 245) -> Image:
    gray_image = image.convert("L")

    mask = gray_image.point(lambda x: 255 if black_threshold < x < white_threshold else 0, '1')
    bbox = mask.getbbox()

    if bbox:
        return image.crop(bbox)

    return image


def crop(img: Image, image_file_name: str, point_cache: Dict[str, Tuple[float, float]], cache_file: TextIO):
    img = remove_bars(img)

    width, height = img.size

    if image_file_name in point_cache:
        x, y = point_cache.get(image_file_name)
    else:
        x, y = width // 2, height // 2
        point = None

        try:
            point = find_face_avg(np.array(img))
        except Exception as e:
            print(e, file=sys.stderr)

        if point is not None:
            x = point[0]
            y = point[1]

    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Point is out of image bounds, {image_file_name} | {x}:{y}")

    cache_file.write(f"{image_file_name}{CSV_SEP}{x}:{y}\n")

    # Ensure square size doesn't exceed image dimensions
    crop_size = min(width, height)

    # Calculate crop bounds
    left = max(0, x - crop_size // 2)
    top = max(0, y - crop_size // 2)
    right = min(width, left + crop_size)
    bottom = min(height, top + crop_size)

    # Adjust crop to maintain size
    if right - left < crop_size:
        left = max(0, right - crop_size)
    if bottom - top < crop_size:
        top = max(0, bottom - crop_size)

    crop_area = (left, top, right, bottom)
    return img.crop(crop_area)
