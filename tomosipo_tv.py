# %%
import torch
import tomosipo as ts
import numpy as np
from ts_algorithms import fbp, sirt, tv_min2d, fdk, nag_ls
import matplotlib.pyplot as plt
from PIL import Image

print(torch.__version__)

# %load_ext autoreload
# %autoreload 2

# %%
dev = torch.device("cuda")
print(torch.cuda.is_available())

# %%
def extract_square(image, corner, side_length):
    return image[corner[0]:corner[0]+side_length, corner[1]:corner[1]+side_length]

# %%
slice = np.array(Image.open(r"1471_P2_M2.bmp"))
slice_shape = slice.shape
print(slice_shape)
corner = [slice.shape[0]//2-200, slice.shape[1]//2]
side_length = 400

# %%
sq = extract_square(slice, corner, side_length)
# fig, ax = plt.subplots(figsize=(3,3), dpi=200)
# ax.imshow(sq, cmap=plt.cm.Greys_r, interpolation='nearest')
# ax.axis('off')
# plt.show()

# %%
square_img = torch.from_numpy(np.pad(sq, pad_width = sq.shape[0], mode='constant', constant_values=0)).to(dev)
# fig, ax = plt.subplots(figsize=(3,3), dpi=200)
# ax.imshow(square_img, cmap=plt.cm.Greys_r, interpolation='nearest')
# ax.axis('off')
# plt.show()

# %%
height, width = square_img.shape

# %%
# Setup up volume and parallel projection geometry
vg = ts.volume(shape=(1, width, width))
pg = ts.parallel(angles=900, shape=(1, width))
A = ts.operator(vg, pg)

# %%
square_img = square_img.reshape(1, *square_img.shape)
y = A(square_img)
fig, ax = plt.subplots(figsize=(3,3), dpi=200)

ax.imshow(y.detach().cpu().numpy().squeeze(), cmap=plt.cm.Greys_r, interpolation='nearest', aspect='auto')
ax.axis('off')
plt.show()

# %%
rec_tv_min = tv_min2d(A, y, 0.001, num_iterations=500, progress_bar=True, plot=True)

# %%


# %%


# reconstructions made with different algorithms
# rec_fbp = fbp(A, sino_slice)
# rec_sirt = sirt(A, sino_slice, num_iterations=100)
# rec_tv_min = tv_min2d(A, sino_slice, 0.0001, num_iterations=100)
# rec_nag_ls = nag_ls(A, sino_slice, num_iterations=100)

# %%
# # Setup up volume and parallel projection geometry
# vg = ts.volume(shape=(1, 256, 256))
# pg = ts.parallel(angles=384, shape=(1, 384))
# A = ts.operator(vg, pg)

# # Create hollow cube phantom
# x = torch.zeros(A.domain_shape)
# x[:, 10:-10, 10:-10] = 1.0
# x[:, 20:-20, 20:-20] = 0.0

# # Forward project
# y = A(x)

# # reconstructions made with different algorithms
# rec_fbp = fbp(A, y)
# rec_sirt = sirt(A, y, num_iterations=100)
# rec_tv_min = tv_min2d(A, y, 0.0001, num_iterations=100)
# rec_nag_ls = nag_ls(A, y, num_iterations=100)

# %%
# plt.imshow(rec_sirt.squeeze())


