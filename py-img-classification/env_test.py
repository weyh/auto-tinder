#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import platform
print(platform.python_version())

