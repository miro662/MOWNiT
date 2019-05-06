import sys

from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt, patches

def load_image(img_name: str, inverted: bool = False):
    img = Image.open(img_name).convert('L')
    if inverted:
        img = ImageOps.invert(img)
    return np.array(img)


def analyse(image, pattern):
    image_fft = np.fft.fft2(image)
    w, h = image.shape
    pattern_fft = np.fft.fft2(np.rot90(pattern, 2), s=image.shape)
    splot = np.fft.ifft2(np.multiply(image_fft, pattern_fft))
    return np.abs(np.real(splot))


def cutoff(result, ratio=0.9, pattern_max=None):
    points = []
    min_value = np.amax(result) * ratio
    if pattern_max is not None:
        min_value = pattern_max * ratio

    for y, row in enumerate(result):
        for x, data in enumerate(row):
            if data > min_value:
                points.append((x, y, data))

    return points


def draw_with_rects(image, rects, rect_size):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    for p in rects:
        w, h = rect_size
        x, y = p
        p = (x - w, y - h)
        rect = patches.Rectangle(p,h,w,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    

def fft_analyse(image, pattern, invert=True, cutoff_v=0.9):
    base = load_image(image)
    image = load_image(image, inverted=invert)
    pattern = load_image(pattern, inverted=invert)

    result = analyse(image, pattern)
    points = cutoff(result, cutoff_v)
    draw_with_rects(base, points, pattern.shape)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("syntax: {} image pattern".format(sys.argv[0]))
    fft_analyse("samples/galia.png", "samples/galia_e.png")