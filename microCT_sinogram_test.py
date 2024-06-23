import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy as sp
import cv2
import sys
import seaborn as sns

def hflip(sinogram):
    return sinogram[::-1, :]

def vshift(sinogram, shift):
    return np.roll(sinogram, shift, axis=0)

sinogram_0_360 = np.load("sinogram_0_360.npy")

# Complete sinogram (i-th column corresponds to the i-th projection restricted to the slice we seek to reconstruct.)
# fig, ax = plt.subplots(figsize=(8,4), dpi=150)
# ax.imshow(sinogram_0_360, cmap=plt.cm.Greys_r, aspect='auto')
# ax.set_xlabel("Slice index")
# ax.set_ylabel(r"Projection $x$ axis")
# ax.set_title("Complete sinogram")
# plt.show()

for i in range(0,50,5):
    half_index = sinogram_0_360.shape[1]//2 + i # = 900
    # print(f"Total amount of projections = {sinogram_0_360.shape[1]}")
    print(f"Half index for 0-180° truncation = {half_index}")

    sinogram_0_180 = sinogram_0_360[:, :half_index]

    # fig, ax = plt.subplots(figsize=(6,4), dpi=150)
    # ax.imshow(np.hstack((sinogram_0_180, flip(sinogram_0_180))), cmap=plt.cm.Greys_r, aspect='auto')
    # ax.set_xlabel("Slice index")
    # ax.set_ylabel(r"Projection $x$ axis")
    # ax.set_title("Cut and reflected sinogram")
    # plt.show()
    # shift_value = 80
    # sinogram_0_180_shifted = vshift(sinogram_0_180, shift_value)

    fig, ax = plt.subplots(figsize=(6,4), dpi=150)
    ax.imshow(np.hstack((sinogram_0_180, hflip(sinogram_0_180))), cmap=plt.cm.Greys_r, aspect='auto')
    ax.set_xlabel("Slice index")
    ax.set_ylabel(r"Projection $x$ axis")
    ax.set_title(f"Cut and reflected sinogram with shift, i = {i}")
    plt.show()