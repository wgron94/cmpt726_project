"""
This module contains functions for creating a Voxelmorph pair-wise dataset
given Gamma format RMLI files.

DEPRECATED: I am keeping this here in case it is useful in the future.
"""
import os.path
import glob
import numpy as np
import sarlab.gammax as gx
import progressbar  # Note: install this using pip install progressbar2


def create_rmli_dataset(data_dir, num_pairs, subset_dim):
    """
    Reads in RMLI files, chooses random pairs, and then crops a random subset
    out of those pairs.

    Args:
        data_dir (str): Name of the project directory
        num_pairs (int): Number of pairs to generate
        subset_dim (Tuple[int, int]): (Range, azimuth) size of the subsets
    """
    rmli_dir = os.path.join(data_dir, 'rmli_fr')
    subset_dir = os.path.join(data_dir, 'subsets')
    gx.ensureDir(subset_dir)
    rmli_filenames = glob.glob(os.path.join(rmli_dir, '*.rmli'))

    if len(rmli_filenames) == 0:
        raise ValueError('No RMLI files found in {}'.format(rmli_dir))

    # All the RMLI files can fit in memory fairly easily, so read them in once
    # instead of in a loop
    rmli_data = list()
    for name in rmli_filenames:
        rmli_data.append(gx.MLI(name, par=gx.MLI_Par(name + '.par')).array)

    for i in progressbar.progressbar(np.arange(num_pairs)):
        ref_outname = os.path.join(subset_dir, 'ref_{}.npz'.format(i))
        mov_outname = os.path.join(subset_dir, 'mov_{}.npz'.format(i))

        # Choose two random RMLI scenes
        idxs = np.random.randint(len(rmli_filenames), size=2)
        ref_data = rmli_data[idxs[0]]
        mov_data = rmli_data[idxs[1]]
        ref_subset, mov_subset = grab_random_subset(ref_data, mov_data,
                                                    subset_dim, orient=True)

        # Finally save the data to an npz file
        np.savez(ref_outname, vol=ref_subset)
        np.savez(mov_outname, vol=mov_subset)


def grab_random_subset(arr1, arr2, subset_dim, orient=True):
    """
    Grab a random subset from arr1 and arr2. The subset is from the same area
    in both arr1 and arr2.

    Args:
        arr1 (np.ndarray): Input array 1
        arr2 (np.ndarray): Input array 2
        subset_dim (Tuple[int, int]): (Range, azimuth) size of the subsets
        orient (bool): If true, the subsets are also randomly oriented

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the cropped versions
            of arr1, arr2
    """
    if arr1.shape != arr2.shape:
        raise ValueError('Array shapes have to be the same')

    # Choose a random subset
    row_start = np.random.randint(0, arr1.shape[0] - subset_dim[0])
    col_start = np.random.randint(0, arr1.shape[1] - subset_dim[1])

    # Crop and orient the data
    arr1_subset = arr1[row_start:row_start + subset_dim[0],
                       col_start:col_start + subset_dim[1]]
    arr2_subset = arr2[row_start:row_start + subset_dim[0],
                       col_start:col_start + subset_dim[1]]

    if orient:
        # Choose a random orientation
        rot_num = np.random.randint(0, 4)
        arr1_subset = np.rot90(arr1_subset, rot_num)
        arr2_subset = np.rot90(arr2_subset, rot_num)

    return arr1_subset, arr2_subset
