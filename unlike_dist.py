import os

import numpy as np
from PIL import Image

from utils.utils import calc_fhd
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import pandas as pd


dir, _, files = next(os.walk(f"data/20160816_good_bunch"))
dir, _, files = next(os.walk(f"data/20160617_carpet_2"))
imgs = []

for file in files:
    img = Image.open(f"{dir}/{file}").convert("L")
    imgs.append(np.array(img))

responses = np.stack(imgs)

results = []
for r1_idx, r1 in enumerate(responses):
    mean_pc = 0
    for r2_idx, r2 in enumerate(responses):
        #fig, axs = plt.subplots(1, 2)
        #axs[0].imshow(r1)
        #axs[1].imshow(r2)
        fhd = calc_fhd(r1, r2)
        pc, _ = pearsonr(r1.flatten(), r2.flatten())
        mean_pc += pc
        #plt.title(fhd)
        #plt.show()
        #print(fhd)
        #print(pc)
        #print(r1_idx, r1_idx +1 + r2_idx)
        results.append((r1_idx, r2_idx, fhd))
    print(mean_pc / 100)
    mean_pc = 0

df = pd.DataFrame(results, columns=["r1", "r2", "fhd"])
df.to_csv("unlike.csv", index=False)
