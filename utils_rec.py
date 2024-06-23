import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy as sp
import cv2
import sys
import seaborn as sns
import timeit
import gc

from skimage.data import shepp_logan_phantom
from skimage.transform import (radon, rescale, iradon)
from skimage.restoration import (denoise_bilateral, calibrate_denoiser, denoise_nl_means,
                                 denoise_tv_bregman, denoise_tv_chambolle, denoise_wavelet, estimate_sigma)
from skimage.util import (img_as_float, random_noise)
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as peakSNR
from skimage.io import use_plugin, imread, imsave, find_available_plugins, imshow

import algotom.io.loadersaver as losa
import algotom.io.converter as cvr
import algotom.prep.correction as corr
import algotom.prep.calculation as calc
import algotom.prep.removal as remo
import algotom.prep.filtering as filt
import algotom.rec.reconstruction as rec

# Image Rotation=-0.07100
# Optical Axis (line)= 1344
# Camera to Source (mm)=172.31482
# Object to Source (mm)=95.73050
# Scanning position=23.000 mm
# Flat Field Correction=ON
# FF updating interval=258
# Geometrical Correction=ON
# Cone-beam Angle Horiz.(deg)= 12.021693
# Cone-beam Angle Vert.(deg)= 8.030821

class CTReconstructor():
    
    def __init__(self):
        pass

    # def generate_sinogram(self, projections_path, n_imgs = 1800, id_slice = 1200):

    #     # Radon transform of u data (sinogram), noisy
    #     for i in range(n_imgs):
    #         if i % 10 == 0:
    #             print(f"Current slice: {i}")

    #         # Read file            
    #         current_proj = cv2.imread(projections_path + f"{i:04}" + '.tif', cv2.IMREAD_UNCHANGED)

    #         if i == 0:
    #             print(f"Saving first projection at i = {i}")
    #             Ru_T = np.zeros((n_imgs, current_proj.shape[1]))
    #             self.Ru0 = current_proj
    #         if i == n_imgs//2 - 1:
    #             print(f"Saving mid projection at i = {i}")
    #             self.Ru180 = current_proj

    #         Ru_T[i, :] = current_proj[id_slice, :]

    #     self.n_imgs = n_imgs
    #     self.Ru_all = Ru_T

    def generate_sinogram_hdf(self, hdf_path, id_slice = 900, save=True, plot=False):

        proj_obj = losa.load_hdf(hdf_path, "entry/projections")
        (depth, height, width) = proj_obj.shape
        print(f"depth = {depth}")
        print(f"height = {height}")
        print(f"width = {width}")

        # offset_angle = -0.07100 # deg/rad ?????
        offset_angle = 0
        angles = np.deg2rad(offset_angle + np.linspace(0.0, 360.0, depth))

        idx = height // 2
        sinogram_mid = proj_obj[:, idx, :]
        sinogram_mid = remo.remove_all_stripe(sinogram_mid, 2.0, 51, 21)

        center_vo = calc.find_center_vo(sinogram_mid[::sinogram_mid.shape[0]//2, :], start=1750)
        print("Center of rotation vo: {}".format(center_vo))

        center_360, overlap, side, overlap_position = calc.find_center_360(sinogram_mid, 10)
        print("Center of rotation vo: {}".format(center_360))

        del sinogram_mid
        gc.collect()

        sinogram_id = proj_obj[:, id_slice, :]
        out_name = f'./sinograms/sinogram_{id_slice}.tif'
        losa.save_image(out_name, sinogram_id)

        return sinogram_id, angles, center_vo

    def reconstruct_slice_from_sinogram(self, sinogram, angles, cor):

        # sinogram = self.Ru_0_180.copy()
        sinogram = remo.remove_zinger(sinogram, 0.08)
        sinogram = remo.remove_stripe_based_normalization(sinogram, 15)
        sinogram = filt.fresnel_filter(sinogram, 100)

        rec_img = rec.dfi_reconstruction(sinogram, cor, angles=angles,
                                    filter_name = "hamming", apply_log=True, ncore = 8)
        
        return rec_img

        