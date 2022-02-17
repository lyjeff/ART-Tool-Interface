import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm
from art.utils import load_mnist
from PIL import Image
import cv2

"""
path = './cifar-100-python/test/'

files = [
    os.path.join(path, f) for f in os.listdir(path)
    if os.path.isfile(os.path.join(path, f))
]

for i in range(100):
    if not os.path.exists(os.path.join(path, f"./{i}")):
        os.mkdir(os.path.join(path, f"./{i}"))

for file in tqdm(files):
    base_name = os.path.basename(file)
    label = base_name.split('.')[0].split('_')[1]
    shutil.move(file, os.path.join(path, f'./{label}/', base_name))
"""


print("success!!!")
os._exit(0)
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test), _, _ = load_mnist()
x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_train = np.swapaxes(x_train, 2, 3)
x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 2, 3)

path = './mnist-dataset/test/'
len = len(x_test)

for i in range(10):
    if not os.path.exists(os.path.join(path, f"./{i}")):
        os.mkdir(os.path.join(path, f"./{i}"))

for idx in tqdm(range(len)):
    for i in range(10):
        if y_test[idx][i] == 1:
            label = i
            break

    # print(label)
    img = Image.fromarray(x_test[idx][0]*255).convert('RGB')

    img.save(os.path.join(path, f'./{label}/', f'./test_{label}_{idx}.png'))
    # cv2.imwrite(os.path.join(
    #     path, f'./{label}/', f'./train_{label}_{idx}.png'), img*255)
    # break
