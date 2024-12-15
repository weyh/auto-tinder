import argparse
import os
import sys
import random
import shutil
import tempfile
import time
from typing import List, Tuple, Dict, TextIO, Union
import re
from PIL import Image
import zipfile
import multiprocessing as mp
import face_recognition as fr
import numpy as np

RND_MAX = 2 ** 64
CSV_SEP = ';'


def progress_bar(percent, bar_length=30, suffix=""):
    bar = '#' * int(bar_length * percent) + '-' * (bar_length - int(bar_length * percent))
    sys.stdout.write(f'\r[{bar}] {percent * 100:.2f}%{suffix}')
    sys.stdout.flush()


def qprint(text: str):
    sys.stdout.write('\r')
    sys.stdout.write(text)
    sys.stdout.flush()


def get_files(in_dir: str, file_filter: List[str]) -> List[str]:
    files = []

    for dir_path, _, filenames in os.walk(in_dir):
        for filename in filenames:
            for ff in file_filter:
                if filename.lower().endswith(ff):
                    files.append(os.path.join(dir_path, filename))
                    break

    return files


def extract_zips(in_dir: str, output: str):
    zips = get_files(in_dir, [".zip"])

    zips_len = len(zips)
    for i, zip_file in enumerate(zips):
        progress_bar(i / zips_len,
                     suffix=f" Extracting: {os.path.basename(zip_file)}({os.path.getsize(zip_file) / float(10 ** 6):.2f}MB)")

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output)
    progress_bar(1)
    print()


def find_face_avg(img: Image) -> Tuple[float, float] | None:
    image_array = np.array(img)
    face_locations = fr.face_locations(image_array)
    face_locations_len = len(face_locations)

    if face_locations_len == 0:
        return None

    c_x, c_y = 0, 0

    for top, right, bottom, left in face_locations:
        x, y = (left + right) / 2, (top + bottom) / 2
        c_x += x
        c_y += y

    return c_x / face_locations_len, c_y / face_locations_len


def crop_center(img: Image, image_file_name: str, point_cache: Dict[str, Tuple[int, int]], cache_file: TextIO):
    width, height = img.size

    if image_file_name in point_cache:
        x, y = point_cache.get(image_file_name)
    else:
        center_x, center_y = width // 2, height // 2
        point = None

        try:
            point = find_face_avg(img)
        except Exception as e:
            print(e, file=sys.stderr)

        x, y = center_x, center_y

        if point is not None:
            x = (point[0] + center_x) // 2
            y = (point[1] + center_y) // 2

    # Validate bounding box
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Point is out of image bounds")

    cache_file.write(f"{image_file_name}{CSV_SEP}{x}:{y}\n")

    # Ensure square size doesn't exceed image dimensions
    square_size = min(width, height)

    # Calculate the crop area
    crop_x_min = max(0, x - square_size // 2)
    crop_y_min = max(0, y - square_size // 2)
    crop_x_max = crop_x_min + square_size
    crop_y_max = crop_y_min + square_size

    # Adjust if the crop area exceeds image dimensions
    if crop_x_max > width:
        crop_x_min = width - square_size
        crop_x_max = width

    if crop_y_max > height:
        crop_y_min = height - square_size
        crop_y_max = height

    # Perform the crop
    crop_area = (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
    return img.crop(crop_area)


def process_pngs(dir_struct, files_png, train_ratio, val_ratio, eval_ratio,
                 point_cache: Dict[str, Tuple[int, int]], cache_file_path):
    with open(cache_file_path, 'w') as f:
        for i, png_path in enumerate(files_png):
            rnd = random.randint(0, RND_MAX - 1)
            file_name = f"{os.path.basename(png_path)[:-4]}_{rnd}.png"

            if rnd < RND_MAX * train_ratio:
                working_dir = dir_struct["train"]
            elif rnd < RND_MAX * (train_ratio + val_ratio):
                working_dir = dir_struct["validation"]
            else:
                working_dir = dir_struct["evaluation"]

            if re.match(".*(ok)_.+.((png)|(PNG))$", png_path):
                path = os.path.join(working_dir["ok"], file_name)
            elif re.match(".*(x)_.+.((png)|(PNG))$", png_path):
                path = os.path.join(working_dir["x"], file_name)
            else:
                continue

            path = path.replace("png", "jpg")

            with Image.open(png_path) as img:
                rgb_img = img.convert('RGB')
                cropped = crop_center(rgb_img, png_path, point_cache, f)
                cropped.save(path, 'JPEG', quality=90)
            qprint(f"DONE: {png_path} -> {path}")


def process_jpgs(dir_struct, files_jpg, train_ratio, val_ratio, eval_ratio,
                 point_cache: Dict[str, Tuple[int, int]], cache_file_path):
    with open(cache_file_path, 'w') as f:
        for i, jpg_path in enumerate(files_jpg):
            rnd = random.randint(0, RND_MAX - 1)
            file_name = f"{os.path.basename(jpg_path)[:-4]}_{rnd}.jpg"

            if rnd < RND_MAX * train_ratio:
                working_dir = dir_struct["train"]
            elif rnd < RND_MAX * (train_ratio + val_ratio):
                working_dir = dir_struct["validation"]
            else:
                working_dir = dir_struct["evaluation"]

            if re.match(".*(ok)(_.+|).((jp(e|)g)|(JP(E|)G))$", jpg_path):
                path = os.path.join(working_dir["ok"], file_name)
            elif re.match(".*(x)(_.+|).((jp(e|)g)|(JP(E|)G))$", jpg_path):
                path = os.path.join(working_dir["x"], file_name)
            else:
                continue

            try:
                with Image.open(jpg_path) as img:
                    cropped = crop_center(img, jpg_path, point_cache, f)
                    cropped.save(path, 'JPEG', quality=90)
                qprint(f"DONE: {jpg_path} -> {path}")
            except OSError:
                print(f"{jpg_path} is bad, SKIP", file=sys.stderr)


def parse_cache_csv(file_path: str) -> Dict[str, Tuple[int, int]]:
    ret = dict()

    with open(file_path, mode='r') as f:
        count = 0
        for line in f:
            count += 1

            if len(line) < 5:
                continue

            parts = line.strip().split(CSV_SEP)
            if len(parts) < 2:
                print(f"Skipping invalid line ({count})", file=sys.stderr)
                continue

            point = parts[1].split(':')
            if len(parts) < 2:
                print(f"Skipping invalid line ({count})", file=sys.stderr)
                continue

            ret[parts[0]] = (float(point[0]), float(point[1]))

    return ret


def main(args: argparse.Namespace):
    seed: int = int(args.seed) if args.seed is not None else random.randint(0, RND_MAX - 1)
    random.seed(seed)
    print("Seed:", seed)

    if args.zip:
        print("ZIP mode")
        temp_folder = os.path.join(args.temp_dir, "ic_ipe")
        os.makedirs(temp_folder, exist_ok=True)
        extract_zips(args.input, temp_folder)
        args.input = temp_folder

    try:
        files_jpg = get_files(args.input, [".jpg", ".jpeg"])
        files_png = get_files(args.input, [".png"])

        train_dir = os.path.join(args.output, "train")
        val_dir = os.path.join(args.output, "val")
        eval_dir = os.path.join(args.output, "eva")

        train_ratio = 0.80
        val_ratio = 0.15
        eval_ratio = 0.05

        assert abs(train_ratio + val_ratio + eval_ratio - 1) < 0.0001

        dir_struct = {
            "train": {
                "ok": os.path.join(train_dir, "ok"),
                "x": os.path.join(train_dir, "x")
            },
            "validation": {
                "ok": os.path.join(val_dir, "ok"),
                "x": os.path.join(val_dir, "x")
            },
            "evaluation": {
                "ok": os.path.join(eval_dir, "ok"),
                "x": os.path.join(eval_dir, "x")
            }
        }
        print(f"train: {train_ratio * 100}%, validation: {val_ratio * 100}%, evaluation: {eval_ratio * 100}%")

        if args.clean_output_dir:
            print("Cleaning output dir")
            shutil.rmtree(args.output)
            os.makedirs(args.output, exist_ok=False)

        with open(os.path.join(args.output, "seed.txt"), "a") as f:
            f.write(f"{seed}\n")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

        for key in dir_struct:
            os.makedirs(dir_struct[key]["ok"], exist_ok=True)
            os.makedirs(dir_struct[key]["x"], exist_ok=True)

        print("Loading cache")
        point_cache = dict()
        try:
            point_cache = parse_cache_csv(args.cache)
        except FileNotFoundError:
            pass

        cpu_count = mp.cpu_count()
        worker_count = int(max(1, cpu_count * 0.99))
        print(f"CPU core count: {cpu_count}, Worker count: {worker_count}")

        print("Starting png files:")
        files_png_len = len(files_png)

        if files_png_len < 32:
            print(f"Not enough images to multiprocess ({files_png_len})")
            process_pngs(dir_struct, files_png, train_ratio, val_ratio, eval_ratio,
                         point_cache, os.path.join(os.getcwd(), "cache_png_0.tmp"))
        else:
            part_size = files_png_len // worker_count
            remainder = files_png_len % worker_count

            procs = []

            for i in range(worker_count):
                start = i * part_size + min(i, remainder)
                end = start + part_size + (1 if i < remainder else 0)
                part = files_png[start:end]

                p = mp.Process(target=process_pngs,
                               name=f"png_proc_worker_{i}",
                               args=(dir_struct, part, train_ratio, val_ratio, eval_ratio,
                                     point_cache, os.path.join(os.getcwd(), f"cache_png_{i}.tmp")))
                procs.append(p)
                p.start()

            for i, p in enumerate(procs):
                p.join()

        print("\nStarting jpg files:")
        files_jpg_len = len(files_jpg)

        if files_jpg_len < 32:
            print(f"Not enough images to multiprocess ({files_jpg_len})")
            process_jpgs(dir_struct, files_jpg, train_ratio, val_ratio, eval_ratio,
                         point_cache, os.path.join(os.getcwd(), "cache_jpg_0.tmp"))
        else:
            part_size = files_jpg_len // worker_count
            remainder = files_jpg_len % worker_count

            procs = []

            for i in range(worker_count):
                start = i * part_size + min(i, remainder)
                end = start + part_size + (1 if i < remainder else 0)
                part = files_jpg[start:end]

                p = mp.Process(target=process_jpgs,
                               name=f"jpg_proc_worker_{i}",
                               args=(dir_struct, part, train_ratio, val_ratio, eval_ratio,
                                     point_cache, os.path.join(os.getcwd(), f"cache_jpg_{i}.tmp")))
                procs.append(p)
                p.start()

            for p in procs:
                p.join()

        print("\nMerge csv cache files")
        tmp_csvs = get_files(os.getcwd(), [".tmp"])
        with open(args.cache, "wb") as out:
            for tmp_csv in tmp_csvs:
                with open(tmp_csv, "rb") as tmp:
                    shutil.copyfileobj(tmp, out)
                os.remove(tmp_csv)
    finally:
        if args.zip:
            print("\nCleanup zip artifacts")
            shutil.rmtree(temp_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=False, default=os.path.join(os.getcwd(), "cache.csv"),
                        help='path to face location cache csv file')
    parser.add_argument('-s', '--seed',  default=None,
                        help='seed used for sorting images to training, validation, evaluation folder')
    parser.add_argument('-t', '--temp-dir', default=os.path.join(tempfile.gettempdir(), "pipp"),
                        help='folder where files are temporarily extracted to')
    parser.add_argument('-z', '--zip', action='store_true', default=False,
                        help='if set in dir is treated as a folder that contains zips')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-c', '--clean-output-dir', action='store_true', default=False,
                        help='if set out dir is cleaned before new data is written to it')

    args = parser.parse_args()

    main(args)
