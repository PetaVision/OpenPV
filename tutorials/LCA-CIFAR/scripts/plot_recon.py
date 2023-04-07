import os, sys
import numpy as np
import imageio

import pvtools

# paths to pvp files
output_dir = os.path.join(os.pardir, 'output')

input_pvp_path = os.path.join(output_dir, 'Input.pvp')
recon_pvp_path = os.path.join(output_dir, 'InputRecon.pvp')

# read number of frames
input_pvp_header = pvtools.readpvpheader(input_pvp_path)
num_frames = input_pvp_header['nbands']

# create out dir for the recons
recon_dir = os.path.join(os.pardir, 'Analysis', 'Recons')
os.makedirs(recon_dir, exist_ok=True)

frame_skip = 1000

for i in range(0, num_frames, frame_skip):
    # no need to read all the images into memory, read one by one
    input_frame = pvtools.readpvpfile(input_pvp_path, startFrame=i, lastFrame=i+1)
    recon_frame = pvtools.readpvpfile(recon_pvp_path, startFrame=i, lastFrame=i+1)
    
    time = input_frame['time'][0]
    assert time == recon_frame['time'][0]
    batch = i % input_frame['header']['nbatch']
    
    # delete the frame dimension
    input_image = np.squeeze(input_frame['values'])
    recon_image = np.squeeze(recon_frame['values'])

    # normalize
    input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image))
    recon_image = (recon_image - np.min(recon_image)) / (np.max(recon_image) - np.min(recon_image))

    # to 8-bit
    input_image = np.uint8(input_image * 255)
    recon_image = np.uint8(recon_image * 255)

    # concat input and recon
    output_image = np.vstack([input_image, recon_image])

    # save to disk
    out_name = f'recon_{int(time):06d}_{batch:03d}.png'
    out_path = os.path.join(recon_dir, out_name)
    print(out_path)
    imageio.imwrite(out_path, output_image)
