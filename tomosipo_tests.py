# %%
import torch
import tomosipo as ts
from ts_algorithms import fbp, sirt, tv_min2d, fdk, nag_ls

# %%
# Setup up volume and parallel projection geometry
vg = ts.volume(shape=(1, 256, 256))
pg = ts.parallel(angles=384, shape=(1, 384))
A = ts.operator(vg, pg)

# Create hollow cube phantom
x = torch.zeros(A.domain_shape)
x[:, 10:-10, 10:-10] = 1.0
x[:, 20:-20, 20:-20] = 0.0

# Forward project
y = A(x)

# reconstructions made with different algorithms
rec_fbp = fbp(A, y)
rec_sirt = sirt(A, y, num_iterations=100)
rec_tv_min = tv_min2d(A, y, 0.0001, num_iterations=100)
rec_nag_ls = nag_ls(A, y, num_iterations=100)

# %%
ts.svg(pg, vg)


