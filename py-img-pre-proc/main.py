import argparse
import os
import sys
import random
import threading
import shutil
import tempfile
from typing import List
import re
from PIL import Image
import zipfile


def progress_bar(percent, bar_length=30, suffix=""):
    bar = '#' * int(bar_length * percent) + '-' * (bar_length - int(bar_length * percent))
    sys.stdout.write(f'\r[{bar}] {percent * 100:.2f}%{suffix}')
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

        ok_dir = os.path.join(args.out_dir, "ok")
        x_dir = os.path.join(args.out_dir, "x")

        os.makedirs(ok_dir, exist_ok=True)
        os.makedirs(x_dir, exist_ok=True)

        print("Starting png files:")

        files_png_len = len(files_png)
        for i, png_path in enumerate(files_png):
            rnd = random.randint(0, 2 ** 64 - 1)
            file_name = f"{os.path.basename(png_path)[:-4]}_{rnd}.png"

            if re.match(".*(ok)_.+.png$", png_path):
                path = os.path.join(ok_dir, file_name)
            elif re.match(".*(x)_.+.png$", png_path):
                path = os.path.join(x_dir, file_name)
            else:
                continue

            progress_bar((i + 1) / files_png_len)

            convert_png_to_jpg(png_path, path.replace("png", "jpg"), 80)

        print("\nStarting jpg files:")

        def process_jpg(jpg_path: str):
            rnd = random.randint(0, 2 ** 64 - 1)
            file_name = f"{os.path.basename(jpg_path)[:-4]}_{rnd}.jpg"

            if re.match(".*(ok)_.+.jp(e|)g$", jpg_path):
                path = os.path.join(ok_dir, file_name)
            elif re.match(".*(x)_.+.jp(e|)g$", jpg_path):
                path = os.path.join(x_dir, file_name)
            else:
                return

            try:
                img = resize_image(jpg_path, 0.6)
                img.save(path, 'JPEG', quality=85)
            except OSError:
                print(f"{jpg_path} is bad, SKIP", file=sys.stderr)

        files_jpg_len = len(files_jpg)
        it_files_jpg = iter(files_jpg)

        for i, (a, b, c) in enumerate(zip(it_files_jpg, it_files_jpg, it_files_jpg)):
            progress_bar((i + 1) / (files_jpg_len / 3))

            t1 = threading.Thread(target=process_jpg, args=(a,))
            t2 = threading.Thread(target=process_jpg, args=(b,))
            t3 = threading.Thread(target=process_jpg, args=(c,))

            t1.start()
            t2.start()
            t3.start()

            t1.join()
            t2.join()
            t3.join()

        for i, jpg_path in enumerate(files_jpg[files_jpg_len - files_jpg_len % 3:]):
            progress_bar(((files_jpg_len - files_jpg_len % 3) + i + 1) / files_jpg_len)
            process_jpg(jpg_path)

        print()
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
