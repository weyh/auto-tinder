import argparse
import os
import random
import tempfile
import time
from typing import List
import re
import socket
from functools import lru_cache

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import tensorflow as tf
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

KEY = "4-KEY_for_this+s3rveR"
CHECK_DATA_BASE = "<0_w_0>"
CLASS_NAMES = ['o', 'x']
IMG_HEIGHT, IMG_WIDTH = 180, 180

TEMP_FILE = "TMP_pred_img.jpg"


@lru_cache(maxsize=2)
def derive_key(key_str: str) -> bytes:
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(key_str.encode('utf-8'))
    return digest.finalize()[:16]  # AES-128 bit key


def decrypt(encrypted_str: str, key: str) -> str:
    # Decode the base64-encoded encrypted string
    encrypted_data = base64.b64decode(encrypted_str)

    # Derive the AES key
    derived_key = derive_key(key)[:16]  # AES-128 bit key

    # Initialize the cipher for AES decryption with ECB mode
    cipher = Cipher(algorithms.AES(derived_key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt and remove padding
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()

    # Remove padding (PKCS7 padding)
    padding_len = decrypted_data[-1]
    decrypted_data = decrypted_data[:-padding_len]

    return decrypted_data.decode('utf-8')


def generate_check_data(port: int) -> str:
    return f"{CHECK_DATA_BASE}_{port}+{int(time.time() // 60)}"


def generate_check_data_prev(port: int) -> str:
    return f"{CHECK_DATA_BASE}_{port}+{int((time.time() - 60) // 60)}"


def get_files(in_dir: str, file_filter: List[str]) -> List[str]:
    files = []

    for dir_path, _, filenames in os.walk(in_dir):
        for filename in filenames:
            for ff in file_filter:
                if filename.lower().endswith(ff):
                    files.append(os.path.join(dir_path, filename))
                    break

    return files


def predict(interpreter: Interpreter, ref_path: str) -> (str, float):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image
    img = tf.keras.utils.load_img(ref_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array = img_array.astype(np.float32)  # Ensure the data type matches the model's input

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    score = tf.nn.softmax(output_data[0])

    return CLASS_NAMES[np.argmax(score)], 100 * np.max(score)


def load_model(model_dir: str) -> Interpreter:
    models = get_files(model_dir, [".tflite"])

    def extract_number(s):
        match = re.search(r'\d+', s)
        return int(match.group()) if match else None

    selected_model = max(models, key=extract_number)

    print("Selected model: " + selected_model)

    interpreter = Interpreter(model_path=selected_model)
    interpreter.allocate_tensors()
    return interpreter


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

    interpreter = load_model(model_dir)
    tf.function(jit_compile=True)
    print("JIT ON")

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

            class_name, score = predict(interpreter, temp_img_file)
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
