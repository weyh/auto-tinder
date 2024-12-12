import argparse
import os
import sys
import random
import shutil
import tempfile
from typing import List, Union
import re
from PIL import Image
import zipfile
import multiprocessing as mp

RND_MAX = 2 ** 64

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


def crop_center(img: Image, offset=100):
    width, height = img.size

    square_size = min(width, height)

    left = (width - square_size) // 2
    top = (height - square_size) // 2 - offset
    right = left + square_size
    bottom = top + square_size

    # Ensure the crop box is within bounds
    top = max(0, top)
    bottom = min(height, bottom)

    return img.crop((left, top, right, bottom))


def process_pngs(dir_struct, files_png, train_ratio, val_ratio, eval_ratio):
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
            cropped = crop_center(img)
            rgb_img = cropped.convert('RGB')
            rgb_img.save(path, 'JPEG', quality=90)
        qprint(f"DONE: {png_path} -> {path}")


def process_jpgs(dir_struct, files_jpg, train_ratio, val_ratio, eval_ratio):
    rnd_max = 2 ** 64
    for i, jpg_path in enumerate(files_jpg):
        rnd = random.randint(0, rnd_max - 1)
        file_name = f"{os.path.basename(jpg_path)[:-4]}_{rnd}.jpg"

        if rnd < rnd_max * train_ratio:
            working_dir = dir_struct["train"]
        elif rnd < rnd_max * (train_ratio + val_ratio):
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
                cropped = crop_center(img)
                cropped.save(path, 'JPEG', quality=90)
            qprint(f"DONE: {jpg_path} -> {path}")
        except OSError:
            print(f"{jpg_path} is bad, SKIP", file=sys.stderr)


def main(args: argparse.Namespace):
    seed: int = args.seed if args.seed is not None else random.randint(0, RND_MAX - 1)
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
            f.write(f"{seed}")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

        for key in dir_struct:
            os.makedirs(dir_struct[key]["ok"], exist_ok=True)
            os.makedirs(dir_struct[key]["x"], exist_ok=True)

        cpu_count = mp.cpu_count()
        worker_count = int(max(1, cpu_count * 0.8))
        print(f"CPU core count: {cpu_count}, Worker count: {worker_count}")

        print("Starting png files:")
        files_png_len = len(files_png)

        if files_png_len < 20:
            print(f"Not enough images to multiprocess ({files_png_len})")
            process_pngs(dir_struct, files_png, train_ratio, val_ratio, eval_ratio)
        else:
            part_size = files_png_len // worker_count
            remainder = files_png_len % worker_count

            procs = []

            for i in range(worker_count):
                start = i * part_size + min(i, remainder)
                end = start + part_size + (1 if i < remainder else 0)
                part = files_png[start:end]

                p = mp.Process(target=process_pngs,
                               args=(dir_struct, part, train_ratio, val_ratio, eval_ratio))
                procs.append(p)
                p.start()

            for i, p in enumerate(procs):
                p.join()

        print("\nStarting jpg files:")
        files_jpg_len = len(files_jpg)

        if files_jpg_len < 20:
            print(f"Not enough images to multiprocess ({files_jpg_len})")
            process_jpgs(dir_struct, files_jpg, train_ratio, val_ratio, eval_ratio)
        else:
            part_size = files_jpg_len // worker_count
            remainder = files_jpg_len % worker_count

            procs = []

            for i in range(worker_count):
                start = i * part_size + min(i, remainder)
                end = start + part_size + (1 if i < remainder else 0)
                part = files_jpg[start:end]

                p = mp.Process(target=process_jpgs,
                               args=(dir_struct, part, train_ratio, val_ratio, eval_ratio))
                procs.append(p)
                p.start()

            for i, p in enumerate(procs):
                p.join()
    finally:
        if args.zip:
            print("\nCleanup zip artifacts")
            shutil.rmtree(temp_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=Union[int | None], default=None,
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
