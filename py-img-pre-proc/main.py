import argparse
import os
import random
import re
import shutil
import sys
import tempfile
import time
import zipfile
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional

from PIL import Image

from image_crop import crop
from common import CSV_SEP

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


def parse_cache_csv(file_path: str) -> Dict[str, Tuple[float, float]]:
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


def worker(job_queue: mp.Queue, dir_struct: Dict[str, Dict[str, str]],
           train_ratio: float, val_ratio: float, eval_ratio: float,
           point_cache: Dict[str, Tuple[float, float]], cache_file_path: str):
    with open(cache_file_path, 'w') as f:
        while True:
            file_path = job_queue.get()
            if file_path is None:  # Signal to terminate
                break

            rnd = random.randint(0, RND_MAX - 1)
            file_name = os.path.basename(file_path)

            if rnd < RND_MAX * train_ratio:
                working_dir = dir_struct["train"]
            elif rnd < RND_MAX * (train_ratio + val_ratio):
                working_dir = dir_struct["validation"]
            else:
                working_dir = dir_struct["evaluation"]

            is_png = False
            if re.match(".*(ok)(_.+|).((jp(e|)g)|(JP(E|)G))$", file_path):
                new_file_path = os.path.join(working_dir["ok"], file_name)
            elif re.match(".*(x)(_.+|).((jp(e|)g)|(JP(E|)G))$", file_path):
                new_file_path = os.path.join(working_dir["x"], file_name)
            elif re.match(".*(ok)_.+.((png)|(PNG))$", file_path):
                new_file_path = os.path.join(working_dir["ok"], file_name)
                new_file_path = new_file_path.replace("png", "jpg")
                is_png = True
            elif re.match(".*(x)_.+.((png)|(PNG))$", file_path):
                new_file_path = os.path.join(working_dir["x"], file_name)
                new_file_path = new_file_path.replace("png", "jpg")
                is_png = True
            else:
                continue

            try:
                with Image.open(file_path) as img:
                    if is_png:
                        rgb_img = img.convert('RGB')
                    else:
                        rgb_img = img
                    cropped = crop(rgb_img, file_path, point_cache, f)
                    cropped.save(new_file_path, 'JPEG', quality=90)
                qprint(f"DONE: {file_path} -> {new_file_path}")
            except OSError:
                print(f"{file_path} is bad, SKIP", file=sys.stderr)


def main(args: argparse.Namespace):
    start_time = time.time()

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
        files = get_files(args.input, [".jpg", ".jpeg", ".png"])

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

        job_queue: mp.Queue[Optional[str]] = mp.Queue()
        cpu_count = mp.cpu_count()
        worker_count = int(max(1, cpu_count * 0.99))
        print(f"CPU core count: {cpu_count}, Worker count: {worker_count}")

        print("Starting workers")
        procs = []
        for i in range(worker_count):
            p = mp.Process(target=worker,
                           name=f"prp_worker_{i}",
                           args=(job_queue,
                                 dir_struct, train_ratio, val_ratio, eval_ratio,
                                 point_cache, os.path.join(os.getcwd(), f"csv_cache_{i}.tmp")))
            procs.append(p)
            p.start()

        print("Adding jobs")
        for file_path in files:
            job_queue.put(file_path)

        for _ in range(worker_count):
            job_queue.put(None)

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

    end_time = time.time()
    print(f"Executed in {end_time - start_time:.4f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=False, default=os.path.join(os.getcwd(), "cache.csv"),
                        help='path to face location cache csv file')
    parser.add_argument('-s', '--seed', default=None,
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
