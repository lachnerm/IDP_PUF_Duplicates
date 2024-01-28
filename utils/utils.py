import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
from scipy.spatial.distance import hamming

from utils.gabor import gabor_hash
from PIL import Image
import seaborn as sns

def calc_fhd(r1, r2):
    r1_bitstring = gabor_transform(r1.squeeze()).flatten()
    r2_bitstring = gabor_transform(r2.squeeze()).flatten()
    return hamming(r1_bitstring, r2_bitstring)


def gabor_transform(r, k_scale=1):
    r_bit_img = gabor_hash(r, k_scale=k_scale)
    r_bit_img = r_bit_img.astype(np.uint8)
    return r_bit_img


def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X[:dy, :] = 0
    elif dy < 0:
        X[dy:, :] = 0
    if dx > 0:
        X[:, :dx] = 0
    elif dx < 0:
        X[:, dx:] = 0
    return X


def shift_and_crop_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X = np.delete(X, list(range(dy)), axis=0)
    elif dy < 0:
        X = np.delete(X, list(range(X.shape[0] + dy, X.shape[0])), axis=0)
    if dx > 0:
        X = np.delete(X, list(range(dx)), axis=1)
    elif dx < 0:
        X = np.delete(X, list(range(X.shape[1] + dx, X.shape[1])), axis=1)
    return X


def shift_and_crop_image_and_ref(X, Y, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X = np.delete(X, list(range(dy)), axis=0)
        Y = np.delete(Y, list(range(dy)), axis=0)
    elif dy < 0:
        cut = X.shape[0]
        X = np.delete(X, list(range(cut + dy, cut)), axis=0)
        Y = np.delete(Y, list(range(cut + dy, cut)), axis=0)
    if dx > 0:
        X = np.delete(X, list(range(dx)), axis=1)
        Y = np.delete(Y, list(range(dx)), axis=1)
    elif dx < 0:
        cut = X.shape[1]
        X = np.delete(X, list(range(cut + dx, cut)), axis=1)
        Y = np.delete(Y, list(range(cut + dx, cut)), axis=1)
    return X, Y


def get_fhd_center(template, image, crop_size=None):
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if
                 template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    r1 = gabor_transform(image)
    r2 = gabor_transform(template)

    fhds = []
    shifts = []
    for dx in range(-124, 124):
        for dy in range(-124, 124):
            r1c, r2c = r1, r2
            if crop_size is not None:
                r1c = center_crop(r1c, crop_size)
                r2c = center_crop(r2c, crop_size)
            r1_shifted, r2_shifted = shift_and_crop_image_and_ref(r1c, r2c, dx, dy)

            fhd = np.sum(r1_shifted != r2_shifted) / r1_shifted.shape[0] ** 2
            fhds.append(fhd)
            shifts.append((dx, dy))
    fhd = np.min(fhds)
    shift = shifts[np.argmin(fhds).item()]
    '''fig, axs = plt.subplots(3, 2)
    axs[0][0].imshow(r1)

    img = shift_image(r1, shift[0], shift[1])
    axs[1][0].imshow(img)

    shift = shifts[np.argmin(fhds).item()]
    img = shift_and_crop_image(r1, shift[0], shift[1])
    axs[2][0].imshow(img)

    axs[0][1].imshow(r2)

    img = shift_image(r2, - shift[0], - shift[1])
    axs[1][1].imshow(img)

    shift = shifts[np.argmin(fhds).item()]
    img = shift_and_crop_image(r2, - shift[0], - shift[1])
    axs[2][1].imshow(img)
    plt.show()'''
    #sns.heatmap(np.array(fhds).reshape(-100, 100))
    #plt.show()
    #print(np.array(shifts)[np.argsort(fhds)][:10])
    return shift[0], shift[1], fhd




def get_corr_center(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if
                 template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (
                np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    xc, yc = np.unravel_index(out.argmax(), out.shape)
    return yc - out.shape[0] // 2, xc - out.shape[0] // 2, np.max(out)



def center_crop(img, crop_size=512):
    y_size, x_size = img.shape[-2:]
    x_start = x_size // 2 - (crop_size // 2)
    y_start = y_size // 2 - (crop_size // 2)

    x_start = max(0, x_start)
    y_start = max(0, y_start)

    if len(img.shape) == 2:
        return img[y_start:y_start + crop_size, x_start:x_start + crop_size]
    else:
        return img[:, y_start:y_start + crop_size, x_start:x_start + crop_size]