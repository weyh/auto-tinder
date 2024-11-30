import argparse
import os
import random
import tempfile
import time
from typing import List
import re
import socket
from functools import lru_cache

import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from PIL import Image

import torch
from torchvision import transforms

import common


KEY = "4-KEY_for_this+s3rveR"
CHECK_DATA_BASE = "<0_w_0>"

CLASS_NAMES = ['o', 'x']

TEMP_FILE = "TMP_pred_img.jpg"

@lru_cache(maxsize=2)
def derive_key(key_str: str) -> bytes:
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(key_str.encode('utf-8'))
    return digest.finalize()[:16]  # AES-128 bit key


def decrypt(encrypted_str: str, key: str) -> str:
    encrypted_data = base64.b64decode(encrypted_str)

    # Derive the AES key
    derived_key = derive_key(key)[:16]  # AES-128 bit key

    cipher = Cipher(algorithms.AES(derived_key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()

    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()

    # Remove padding (PKCS7 padding)
    padding_len = decrypted_data[-1]
    decrypted_data = decrypted_data[:-padding_len]

    return decrypted_data.decode('utf-8')


def generate_check_data(port: int) -> str:
    return f"{CHECK_DATA_BASE}_{port}+{int(time.time() / 3600)}"


def generate_check_data_prev(port: int) -> str:
    return f"{CHECK_DATA_BASE}_{port}+{int((time.time() - 3600) / 3600)}"


def get_files(in_dir: str, file_filter: List[str]) -> List[str]:
    files = []

    for dir_path, _, filenames in os.walk(in_dir):
        for filename in filenames:
            for ff in file_filter:
                if filename.lower().endswith(ff):
                    files.append(os.path.join(dir_path, filename))
                    break

    return files


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


def preprocess_image(image_path: str):
    preprocess = transforms.Compose([
        transforms.Resize((common.IMG_HEIGHT, common.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=common.NORM_VECS[0], std=common.NORM_VECS[1]),
    ])

    img = crop_center(Image.open(image_path).convert("RGB"))
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor


def predict(model: torch.nn.Module, ref_path: str) -> (str, float):
    img_tensor = preprocess_image(ref_path)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    predicted_class = CLASS_NAMES[probabilities.argmax()]
    confidence = 100 * probabilities.max().item()

    return predicted_class, confidence


def load_model(model_dir: str) -> torch.nn.Module:
    models = get_files(model_dir, [".pt", ".pth"])

    def extract_number(s):
        match = re.search(r'\d+', s)
        return int(match.group()) if match else None

    selected_model = max(models, key=extract_number)

    print("Selected model: " + selected_model)

    model = common.MyModel(len(CLASS_NAMES))
    tmp = torch.load(selected_model, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(tmp)
    return model


def start_server(ip: str, port: int, model_dir: str):
    """
    Msg:
     - in: 4 byte, KEY (enc)
     - out: 1 byte, 'o' or ('n' -> close socket)
     - in: 4 bytes, img size
     - in: data of size img_size
     - out: 1 byte, 'o' or 'x'
     - out: 2 bytes, percent
    """
    temp_img_file = os.path.join(tempfile.gettempdir(), TEMP_FILE)

    model = load_model(model_dir)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, port))
    s.listen(1)

    try:
        print(f"Listening on {ip}:{port}")

        while True:
            conn, addr = s.accept()
            print(f"Connected by {addr}")

            # check data
            data = conn.recv(4)
            next_len: int = int.from_bytes(data, byteorder='little')
            enc_data = conn.recv(next_len)
            data = decrypt(enc_data.decode("utf-8"), KEY)

            if generate_check_data(port) != data:
                if generate_check_data_prev(port) != data:
                    print("Handshake failed, incorrect CHECK_DATA")
                    time.sleep(random.randint(0, 2))
                    conn.sendall("n".encode("utf-8"))
                    conn.close()
                    continue
                else:
                    print("Using prev CHECK_DATA")

            conn.sendall("o".encode("utf-8"))

            data = conn.recv(4)
            img_size: int = int.from_bytes(data, byteorder='little')
            written_size = 0
            with open(temp_img_file, "wb") as f:
                while written_size < img_size:
                    img_bytes = conn.recv(1024)
                    if not img_bytes:
                        break

                    f.write(img_bytes)
                    written_size += len(img_bytes)

            class_name, score = predict(model, temp_img_file)
            print(f"Rating: {class_name} {score}")

            conn.sendall(class_name.encode("utf-8"))

            # max num is 100*100 so 2 bytes is good
            data = int(score * 100).to_bytes(2, byteorder='little')
            conn.sendall(data)

            # Close the connection
            conn.close()
    finally:
        s.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ip', required=True)
    parser.add_argument('-p', '--port', required=True, type=int)
    parser.add_argument('-m', '--model-dir', required=True)

    args = parser.parse_args()
    start_server(args.ip, args.port, args.model_dir)
