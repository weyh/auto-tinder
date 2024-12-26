import os
import sys
from typing import List


def get_files(folder: str, file_filter: List[str]) -> List[str]:
    files = []

    for dir_path, _, filenames in os.walk(folder):
        for filename in filenames:
            for ff in file_filter:
                if filename.lower().endswith(ff):
                    files.append(os.path.join(dir_path, filename))
                    break

    return files


def progress_bar(percent: float, bar_length: int = 30, suffix: str = '', prefix: str = ''):
    bar = '#' * int(bar_length * percent) + '-' * (bar_length - int(bar_length * percent))
    sys.stdout.write(f'\r{prefix}[{bar}] {percent * 100:.2f}%{suffix}')
    sys.stdout.flush()
