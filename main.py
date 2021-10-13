import numpy as np
import os
from skimage import io
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from PIL import Image


def show_image(image, title):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(title)
    plt.show()


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float64)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


# Загрузка изображения в массив
# image = io.imread(os.path.join('C:/Users/Wqano/PycharmProjects/coi1/images', 'I23.BMP'))
image = Image.open('C:/Users/Wqano/PycharmProjects/coi1/images/I23.BMP').convert('RGB')
image = np.array(image)
show_image(image, 'Начальное изображение')

# Получение полутоновых изображений
R = np.uint8(np.zeros(image.shape))
G = np.uint8(np.zeros(image.shape))
B = np.uint8(np.zeros(image.shape))
pxR = image[..., 0]
pxG = image[..., 1]
pxB = image[..., 2]
R[..., 0] = np.uint8(pxR)
G[..., 1] = np.uint8(pxG)
B[..., 2] = np.uint8(pxB)
show_image(R, 'R')
show_image(G, 'G')
show_image(B, 'B')

# Y = 0,299R+0,587G+0,114B,
# Cb = 0,564(B-Y),
# Cr = 0,713(R-Y);

# R = Y+1,402Cr,
# G = Y-0,344Cb-0,714Cr,
# B = Y+1,772Cb.

Y = np.uint8(0.299 * R + 0.587 * G + 0.114 * B)
Cb = np.uint8(0.564 * (B - Y))
Cr = np.uint8(0.713 * (R - Y))
# print(Y.shape)
# YcbCr = np.zeros(image.shape)
# YcbCr[..., 0] = np.uint8(Y)
# YcbCr[..., 1] = np.uint8(Cb)
# YcbCr[..., 2] = np.uint8(Cr)
# YcbCr = np.array((Y,Cb,Cr))
# print(YcbCr.shape)
# show_image(np.uint8(YcbCr), 'YcbCr')

# YcbCr = rgb2ycbcr(image)

show_image(Y, "Y")
show_image(Cb, "Cb")
show_image(Cr, "Cr")
