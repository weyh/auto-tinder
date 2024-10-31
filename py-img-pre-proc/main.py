import argparse
import os
import sys
import random
import shutil
import tempfile
from typing import List
import re
from PIL import Image
import zipfile
import multiprocessing as mp


def progress_bar(percent, bar_length=30, suffix=""):
    bar = '#' * int(bar_length * percent) + '-' * (bar_length - int(bar_length * percent))
    sys.stdout.write(f'\r[{bar}] {percent * 100:.2f}%{suffix}')
    sys.stdout.flush()


def qprint(text: str):
    sys.stdout.write('\r')
    sys.stdout.write(text)
    sys.stdout.flush()


def convert_png_to_jpg(png_path, jpg_path, quality):
    with Image.open(png_path) as img:
        # Convert image to RGB (PNG might have transparency channel which JPG does not support)
        rgb_img = img.convert('RGB')
        rgb_img.save(jpg_path, 'JPEG', quality=quality)


def resize_image(input_path, scale):
    with Image.open(input_path) as img:
        # Calculate the new size
        new_size = (int(img.width * scale), int(img.height * scale))
        # Resize the image
        resized_img = img.resize(new_size)
        return resized_img


def get_files(in_dir: str, file_filter: List[str]) -> List[str]:
    files = []

    for dir_path, _, filenames in os.walk(in_dir):
        for filename in filenames:
            for ff in file_filter:
                if filename.lower().endswith(ff):
                    files.append(os.path.join(dir_path, filename))
                    break

    return files


def extract_zips(in_dir: str, out_dir: str):
    zips = get_files(in_dir, [".zip"])

    zips_len = len(zips)
    for i, zip_file in enumerate(zips):
        progress_bar(i / zips_len,
                     suffix=f" Extracting: {os.path.basename(zip_file)}({os.path.getsize(zip_file) / float(10 ** 6):.2f}MB)")

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(out_dir)
    progress_bar(1)
    print()


def process_pngs(dir_struct, files_png):
    for i, png_path in enumerate(files_png):
        rnd = random.randint(0, 2 ** 64 - 1)
        file_name = f"{os.path.basename(png_path)[:-4]}_{rnd}.png"

        # There is ~33% chance that img will be copied to val
        val_idx = int(rnd % 3 == 0)
        working_dir = dir_struct[val_idx]

        if re.match(".*(ok)_.+.((png)|(PNG))$", png_path):
            path = os.path.join(working_dir["ok"], file_name)
        elif re.match(".*(x)_.+.((png)|(PNG))$", png_path):
            path = os.path.join(working_dir["x"], file_name)
        else:
            continue

        convert_png_to_jpg(png_path, path.replace("png", "jpg"), 80)
        qprint(f"DONE: {png_path}")


def process_jpgs(dir_struct, files_jpg):
    for i, jpg_path in enumerate(files_jpg):
        rnd = random.randint(0, 2 ** 64 - 1)
        file_name = f"{os.path.basename(jpg_path)[:-4]}_{rnd}.jpg"

        val_idx = int(rnd % 3 == 0)
        working_dir = dir_struct[val_idx]

        if re.match(".*(ok)(_.+|).((jp(e|)g)|(JP(E|)G))$", jpg_path):
            path = os.path.join(working_dir["ok"], file_name)
        elif re.match(".*(x)(_.+|).((jp(e|)g)|(JP(E|)G))$", jpg_path):
            path = os.path.join(working_dir["x"], file_name)
        else:
            continue

        try:
            img = resize_image(jpg_path, 0.6)
            img.save(path, 'JPEG', quality=85)
            qprint(f"DONE: {jpg_path}")
        except OSError:
            print(f"{jpg_path} is bad, SKIP", file=sys.stderr)


def main(args: argparse.Namespace):
    if args.zip:
        print("ZIP mode")
        temp_folder = os.path.join(tempfile.gettempdir(), "pipp")
        os.makedirs(temp_folder, exist_ok=True)
        extract_zips(args.in_dir, temp_folder)
        args.in_dir = temp_folder

    try:
        files_jpg = get_files(args.in_dir, [".jpg", ".jpeg"])
        files_png = get_files(args.in_dir, [".png"])

        train_dir = os.path.join(args.out_dir, "train")
        val_dir = os.path.join(args.out_dir, "val")

        dir_struct = [
            {
                "ok": os.path.join(train_dir, "ok"),
                "x": os.path.join(train_dir, "x")
            },
            {
                "ok": os.path.join(val_dir, "ok"),
                "x": os.path.join(val_dir, "x")
            }
        ]

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for item in dir_struct:
            os.makedirs(item["ok"], exist_ok=True)
            os.makedirs(item["x"], exist_ok=True)

        cpu_count = mp.cpu_count()
        worker_count = int(max(1, cpu_count * 0.8))
        print(f"CPU core count: {cpu_count}, Worker count: {worker_count}")

        print("Starting png files:")
        files_png_len = len(files_png)

        if files_png_len < 20:
            print(f"Not enough images to multiprocess ({files_png_len})")
            process_pngs(dir_struct, files_png)
        else:
            part_size = files_png_len // worker_count
            remainder = files_png_len % worker_count

            procs = []

            for i in range(worker_count):
                start = i * part_size + min(i, remainder)
                end = start + part_size + (1 if i < remainder else 0)
                part = files_png[start:end]

                p = mp.Process(target=process_pngs, args=(dir_struct, part))
                procs.append(p)
                p.start()

            for i, p in enumerate(procs):
                p.join()

        print("\nStarting jpg files:")
        files_jpg_len = len(files_jpg)

        if files_png_len < 20:
            print(f"Not enough images to multiprocess ({files_jpg_len})")
            process_jpgs(dir_struct, files_jpg)
        else:
            part_size = files_jpg_len // worker_count
            remainder = files_jpg_len % worker_count

            procs = []

            for i in range(worker_count):
                start = i * part_size + min(i, remainder)
                end = start + part_size + (1 if i < remainder else 0)
                part = files_jpg[start:end]

                p = mp.Process(target=process_jpgs, args=(dir_struct, part))
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
    parser.add_argument('-z', '--zip', action='store_true', default=False,
                        help='if set in dir is treated as a folder that contains zips')
    parser.add_argument('-i', '--in-dir', required=True)
    parser.add_argument('-o', '--out-dir', required=True)

    args = parser.parse_args()

    main(args)
