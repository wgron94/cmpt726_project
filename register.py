"""
Register two RMLI chips with a VoxelMorph model.
The source and target input images are expected to be affinely registered.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import sarlab.gammax as gx
from .generators import scale_rmli


def register_chip(moving_name, fixed_name, model_name, crop, center,
                  warp_name=None, moved_name=None, gpu='0', debug=False):
    """
    Registers a chip from the two RMLI scenes and writes the results to npz
    files.

    Args:
        moving_name (str): Moving image (source) filename (.rmli file)
        fixed_name (str): Fixed image (target) filename (.rmli file)
        moved_name (str): Warped image output filename (.npz file)
        warp_name (str): Output warp deformation filename (.npz file)
        model_name (str): Pytorch Voxelmorph model
        crop (Tuple[int, int]): rg, az size of the chip
        center (Tuple[int, int]): rg, az location of the chip in the RMLI
        gpu (str): GPU number to use
        debug (bool): Plot debug images
    """
    if os.getenv('HOSTNAME') != 'ensc-sarserv-03.research.sfu.ca':
        raise ValueError('This script assumes we are running on the Nemo server')
    code_dir = os.path.dirname(__file__)
    voxelmorph_path = os.path.join(code_dir, 'voxelmorph')
    sys.path.append(voxelmorph_path)

    # Import voxelmorph with pytorch backend
    os.environ['VXM_BACKEND'] = 'pytorch'
    import voxelmorph as vxm

    # Device handling
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    print('Reading all RMLI data ...')
    if (not moving_name.endswith('.rmli')) or (not fixed_name.endswith('.rmli')):
        raise ValueError('Input files need to be RMLI format')
    moving_full = gx.MLI(moving_name,
                         par=gx.MLI_Par(moving_name + '.par')).array
    fixed_full = gx.MLI(fixed_name,
                        par=gx.MLI_Par(fixed_name + '.par')).array
    print('... Done reading all RMLI data')

    print('Cropping and pre-processing ...')
    rg_crop = crop[0]
    az_crop = crop[1]
    rg_cen = center[0]
    az_cen = center[1]
    if (rg_cen < rg_crop // 2) or (az_cen < az_crop // 2)\
            or (fixed_full.shape[0] - rg_cen < rg_crop // 2)\
            or (fixed_full.shape[1] - az_cen < az_crop // 2):
        raise ValueError('Center coordinates are too close to the edge')
    moving = moving_full[rg_cen - rg_crop // 2:rg_cen + rg_crop // 2,
                         az_cen - az_crop // 2:az_cen + az_crop // 2]
    fixed = fixed_full[rg_cen - rg_crop // 2:rg_cen + rg_crop // 2,
                       az_cen - az_crop // 2:az_cen + az_crop // 2]
    moving = scale_rmli(moving)
    fixed = scale_rmli(fixed)

    if debug:
        plt.figure()
        plt.imshow(moving)
        plt.colorbar()
        plt.savefig('moving.png')
        plt.figure()
        plt.imshow(fixed)
        plt.colorbar()
        plt.savefig('fixed.png')

    moving = moving[np.newaxis, ...] # Batch axis
    fixed = fixed[np.newaxis, ...]
    moving = moving[..., np.newaxis] # Channel axis
    fixed = fixed[..., np.newaxis]
    print('... Done cropping and pre-processing ...')

    # Load and set up model
    model = vxm.networks.VxmDense.load(model_name, device)
    model.to(device)
    model.eval()

    # Set up tensors and permute
    input_moving = torch.from_numpy(moving).to(device).float().permute(0, 3, 1, 2)
    input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 3, 1, 2)

    # Predict
    moved, warp = model(input_moving, input_fixed, registration=True)

    # Save moved image
    moved = moved.detach().cpu().numpy().squeeze()
    if moved_name is not None:
        if not moved_name.endswith('.npz'):
            raise ValueError('Data must be saved to an npz file.')
        np.savez_compressed(moved_name, scene=moved)

    # Save warp
    warp = warp.detach().cpu().numpy().squeeze()
    if warp_name is not None:
        if not warp_name.endswith('.npz'):
            raise ValueError('Data must be saved to an npz file.')
        np.savez_compressed(warp_name, offs=warp)

    return moved, warp
