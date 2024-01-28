import argparse
import os

import numpy as np
import pandas as pd
import scipy.stats
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.utils import (
    get_corr_center, shift_image, center_crop, shift_and_crop_image_and_ref,
    calc_fhd, gabor_transform
)


def inspect_correlations(all_images):
    for r1_idx, r1 in enumerate(all_images):
        for r2_idx, r2 in enumerate(all_images):
            if r1_idx == r2_idx:
                continue
            x_shift, y_shift, corr = get_corr_center(r1, r2)
            r1_shifted = shift_image(r1, x_shift, y_shift)
            x_shift_new, y_shift_new, corr_new = get_corr_center(
                r1_shifted, r2
            )
            print(f"Old: ({x_shift},{y_shift}) - "
                  f"NEW: {x_shift_new},{y_shift_new} - "
                  f"CORR: {corr:.2f}")

            if np.abs(x_shift) > 30 or np.abs(y_shift) > 30:
                print(img_idxs[r1_idx], "-", img_idxs[r2_idx])
                fig, axs = plt.subplots(2, 2)
                axs[0][0].imshow(r1)
                axs[0][1].imshow(r1_shifted)
                axs[1][0].imshow(r2)
                axs[1][1].imshow(r2)
                plt.show()


def create_corr_files(all_images, folder, crop_size=None):
    file = f"results/{folder}/corr{f'_crop{crop_size}' if crop_size is not None else ''}.csv"
    if not os.path.exists(file):
        print(f'Creating file {file} ...')
        all_corrs = []
        for r1_idx, r1 in tqdm(enumerate(all_images)):
            corrs = []
            for r2_idx, r2 in enumerate(all_images):
                x_shift, y_shift, corr = get_corr_center(r1, r2)

                if crop_size is not None:
                    r1_shifted, r2_shifted = shift_and_crop_image_and_ref(
                        r1, r2, x_shift, y_shift
                    )
                    r1_shifted_cropped = center_crop(r1_shifted, crop_size)
                    r2_shifted_cropped = center_crop(r2_shifted, crop_size)
                    corr = scipy.stats.pearsonr(
                        r1_shifted_cropped.flatten(),
                        r2_shifted_cropped.flatten()
                    )[0]

                corrs.append(corr)
            all_corrs.append(corrs)

        df = pd.DataFrame(all_corrs)
        df.to_csv(file, index=False, index_label=False, header=False)


def create_fhd_files(all_images, folder, crop_size=None):
    file = f"results/{folder}/fhd{f'_crop{crop_size}' if crop_size is not None else ''}.csv"
    if not os.path.exists(file):
        print(f'Creating file {file} ...')
        all_fhds = []
        for r1_idx, r1 in tqdm(enumerate(all_images)):
            fhds = []
            for r2_idx, r2 in enumerate(all_images):
                x_shift, y_shift, corr = get_corr_center(r1, r2)

                r1_shifted, r2_shifted = shift_and_crop_image_and_ref(
                    r1, r2, x_shift, y_shift
                )

                if crop_size is not None:
                    r1_shifted = center_crop(r1_shifted, crop_size)
                    r2_shifted = center_crop(r2_shifted, crop_size)

                fhd = calc_fhd(r1_shifted, r2_shifted)
                fhds.append(fhd)

                '''
                fig, axs = plt.subplots(2, 4)
                axs[0][0].imshow(r1_shifted)
                axs[0][1].imshow(gabor_transform(r1_shifted))
                axs[0][2].imshow(r1_shifted_cropped)
                axs[0][3].imshow(gabor_transform(r1_shifted_cropped))
                axs[1][0].imshow(r2_shifted)
                axs[1][1].imshow(gabor_transform(r2_shifted))
                axs[1][2].imshow(r2_shifted_cropped)
                axs[1][3].imshow(gabor_transform(r2_shifted_cropped))
                plt.show()
                '''

            all_fhds.append(fhds)

        df = pd.DataFrame(all_fhds)
        df.to_csv(file, index=False, index_label=False, header=False)


def create_unlike_fhd_files(r_puf1, r_puf2, folder1, folder2, crop_size=None):
    file = f"results/{folder1}_vs_{folder2}/fhd{f'_crop{crop_size}' if crop_size is not None else ''}.csv"
    if not os.path.exists(file):
        print(f'Creating file {file} ...')
        all_fhds = []
        for r1_idx, r1 in tqdm(enumerate(r_puf1)):
            fhds = []
            for r2_idx, r2 in enumerate(r_puf2):
                x_shift, y_shift, corr = get_corr_center(r1, r2)

                r1_shifted, r2_shifted = shift_and_crop_image_and_ref(
                    r1, r2, x_shift, y_shift
                )

                if crop_size is not None:
                    r1_shifted = center_crop(r1_shifted, crop_size)
                    r2_shifted = center_crop(r2_shifted, crop_size)

                fhd = calc_fhd(r1_shifted, r2_shifted)
                fhds.append(fhd)

                '''fig, axs = plt.subplots(2, 4)
                axs[0][0].imshow(r1_shifted)
                axs[0][1].imshow(gabor_transform(r1_shifted))
                #axs[0][2].imshow(r1_shifted_cropped)
                #axs[0][3].imshow(gabor_transform(r1_shifted_cropped))
                axs[1][0].imshow(r2_shifted)
                axs[1][1].imshow(gabor_transform(r2_shifted))
                #axs[1][2].imshow(r2_shifted_cropped)
                #axs[1][3].imshow(gabor_transform(r2_shifted_cropped))
                plt.show()'''

            all_fhds.append(fhds)

        df = pd.DataFrame(all_fhds)
        df.to_csv(file, index=False, index_label=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int)
    start = parser.parse_args().start

    '''
    # data_idxs = [2, 3, 4]
    data_idxs = ["good_bunch"]
    all_images = []
    for r1_idx in data_idxs:
        dir, _, files = next(os.walk(f"data/20160816_{r1_idx}"))
        img_idxs = list(
            map(lambda name: int(name.replace("struc", "").split(".")[0]),
                files))
        sorted_files = sorted(files, key=lambda name: int(
            name.replace("struc", "").split(".")[0]))
        imgs = []
        for file in sorted_files:
            img = Image.open(f"{dir}/{file}") \
                # .convert("L")
            imgs.append(np.array(img))
        all_images.append(np.stack(imgs))
        '''

    data_folders = ["base", 'base_rot10']
    for folder in data_folders:
        dir, _, files = next(os.walk(f"data/{folder}"))
        img_idxs = list(
            map(lambda name: int(name.replace("struc", "").split(".")[0]),
                files))
        sorted_files = sorted(files, key=lambda name: int(
            name.replace("struc", "").split(".")[0]))

        imgs = []
        for file in sorted_files:
            img = Image.open(f"{dir}/{file}")
            imgs.append(np.array(img))

        if not os.path.exists(f'results/{folder}'):
            os.mkdir(f'results/{folder}')

        for crop_size in [None, 150, 125, 100, 75, 50]:
            create_corr_files(imgs, folder, crop_size=crop_size)
            create_fhd_files(imgs, folder, crop_size=crop_size)

    data_folders_unlike = ['20160617_carpet_2', '20160617_carpet_3', '20160617_carpet_4']
    for folder1 in data_folders:
        dir, _, files = next(os.walk(f"data/{folder1}"))
        img_idxs = list(
            map(lambda name: int(name.replace("struc", "").split(".")[0]),
                files)
        )
        sorted_files = sorted(files, key=lambda name: int(
            name.replace("struc", "").split(".")[0]))

        imgs_f1 = []
        for file in sorted_files:
            img = Image.open(f"{dir}/{file}")
            imgs_f1.append(np.array(img))

        for folder2 in data_folders_unlike:
            dir, _, files = next(os.walk(f"data/{folder2}"))
            img_idxs = list(
                map(lambda name: int(
                    name.replace("struc", "").split(".")[0]),
                    files))
            sorted_files = sorted(files, key=lambda name: int(
                name.replace("struc", "").split(".")[0]))

            imgs_f2 = []
            for file in sorted_files:
                img = Image.open(f"{dir}/{file}")
                imgs_f2.append(np.array(img))

            if not os.path.exists(f'results/{folder1}_vs_{folder2}'):
                os.mkdir(f'results/{folder1}_vs_{folder2}')

            for crop_size in [None, 150, 125, 100, 75, 50]:
                create_unlike_fhd_files(
                    imgs_f1, imgs_f2, folder1, folder2, crop_size=crop_size
                )
