import os, sys
import numpy as np
import imageio
import math

import pvtools as pv

output_dir = os.path.join(os.pardir, 'output')
checkpoint_dir = os.path.join(os.pardir, 'output', 'Checkpoints/')

# get a list of checkpoints, in ascending order
ckpt_list = os.listdir(checkpoint_dir)
ckpt_list.sort()

# create output directory for the weights 
weights_dir = os.path.join(os.pardir, 'Analysis', 'weights')
os.makedirs(weights_dir, exist_ok=True)

weights_movie = []

for ckpt_name in ckpt_list:
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    weights_pvp_path = os.path.join(ckpt_path, 'V1ToInputError_W.pvp')

    weights_header = pv.readpvpheader(weights_pvp_path)
    weights_data = np.squeeze(pv.readpvpfile(weights_pvp_path)['values'])

    # dimensions of each patch
    nyp, nxp, nfp = weights_header['nyp'], weights_header['nxp'], weights_header['nfp']
    num_patches = weights_header['numpatches']

    # rows and cols of the weights grid
    num_rows = math.ceil(math.sqrt(num_patches))
    num_cols = math.ceil(num_patches / num_rows)

    weights_grid = np.zeros(shape=(nyp * num_rows, nxp * num_cols, nfp), dtype=np.uint8)

    for i, patch_weights in enumerate(weights_data):
        # normalize weights
        weights_min = np.min(patch_weights)
        weights_max = np.max(patch_weights)
        patch_weights /= max(math.fabs(weights_min), math.fabs(weights_max))

        #patch_weights /= max(math.fabs(weights_header['wMin']), math.fabs(weights_header['wMin']))

        # to uint8, the math is to take care of both + and - values
        patch_weights = np.uint8(127.5 * patch_weights + 127.500001)

        # get (row, col) for the patch in the weight grid
        patch_row, patch_col = np.unravel_index(i, shape=(num_rows, num_cols))

        # set the patch weights into the grid
        y1 = patch_row * nyp
        y2 = (patch_row + 1) * nyp
        x1 = patch_col * nxp 
        x2 = (patch_col + 1) * nxp
        weights_grid[y1:y2, x1:x2] = patch_weights

    weights_movie.append(weights_grid)

    # save weights
    out_name = f'weights_{ckpt_name}.png'
    out_path = os.path.join(weights_dir, out_name)
    imageio.imsave(out_path, weights_grid)

movie_path = os.path.join(weights_dir, 'movie.gif')
imageio.mimwrite(movie_path, weights_movie, fps=3)
