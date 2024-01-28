import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

dir, _, files = next(os.walk(f"../data/base"))
img_idxs = list(
    map(lambda name: int(name.replace("struc", "").split(".")[0]),
        files))
sorted_files = sorted(files, key=lambda name: int(
    name.replace("struc", "").split(".")[0]))

imgs = []
for file in sorted_files:
    img = Image.open(f"{dir}/{file}")
    img = img.rotate(10)
    img.save(f'../data/base_rot10/{file}')