"""
Generators for creating training data for the Voxelmorph network. We could not
use their generators out-of-the-box because they did not have an option to
specify exact pairs.
"""
import os.path
import glob
import numpy as np
import sarlab.gammax as gx


def rmli_pair_gen(rmli_dir, chip_dim, batch_size=1, scene_subset=None,
                  random_orient=False):
    """
    Reads in RMLI files, chooses random pairs, and then crops a random subset
    out of those pairs.

    The data is preprocessed by using the power-scaling method Gamma uses in
    raspwr, and then normalizing to be between 0-1.

    Because of the pre-processing, this generator is very slow the first time
    it is called. After that though it is very fast.

    Args:
        rmli_dir (str): Name of the directory containing Gamma RMLI files
        chip_dim (Tuple[int, int]): (Range, azimuth) size of the training image
            chips
        batch_size (int): The batch size to use for training
        scene_subset (Tuple[int, int, int, int]): Area to grab training data
            out of the RMLI files. Format: (rg_start, rg_end, az_start, az_end)
            The default value uses the entire scene.
        random_orient (bool): Randomly rotate the pair of scenes to improve
            data augmentation. Does not work for rectangular chip sizes.
    """
    rmli_filenames = glob.glob(os.path.join(rmli_dir, '*.rmli'))
    if len(rmli_filenames) == 0:
        raise ValueError('No RMLI files found in {}'.format(rmli_dir))

    # All the RMLI files can fit in memory fairly easily, so read them in once
    # at the beginning
    if scene_subset is None:
        scene_subset = (0, None, 0, None)
    print('Reading all RMLI data ...')
    rmli_data = [gx.MLI(name, par=gx.MLI_Par(name + '.par')).array[
        scene_subset[0]:scene_subset[1], scene_subset[2]:scene_subset[3]]
                 for name in rmli_filenames]
    print('... Done reading all RMLI data')

    while True:
        chip0_batches = []
        chip1_batches = []
        for _ in range(batch_size):
            # Choose two random RMLI scenes
            idxs = np.random.randint(len(rmli_filenames), size=2)
            scene0 = rmli_data[idxs[0]]
            scene1 = rmli_data[idxs[1]]
            chip0, chip1 = grab_random_subset(scene0, scene1, chip_dim,
                                              orient=random_orient)

            # Do some preprocessing so pixel values are between 0-1
            chip0 = scale_rmli(chip0)
            chip1 = scale_rmli(chip1)

            # To follow along with Voxelmorphs data dimensions, we add a
            # leading batch dimension, and a trailing channel dimension
            chip0 = chip0[np.newaxis, ...] # Batch axis
            chip1 = chip1[np.newaxis, ...]
            chip0 = chip0[..., np.newaxis] # Channel axis
            chip1 = chip1[..., np.newaxis]

            chip0_batches.append(chip0)
            chip1_batches.append(chip1)

        chip0_total = np.concatenate(chip0_batches, axis=0)
        chip1_total = np.concatenate(chip1_batches, axis=0)

        # Put empty warp tensor in output
        zeros_shape = chip0_total.shape[1:-1]
        zeros = np.zeros((batch_size, *zeros_shape, len(zeros_shape)))
        in_chips = [chip0_total, chip1_total]
        out_chips = [chip1_total, zeros]

        yield (in_chips, out_chips)


def scale_rmli(data, debug=False):
    """
    For the data preprocessing we use the power-scaling exponent that Gamma
    uses to display RMLI data (using the raspwr program).

    Then the data is normalized, outliers are removed, and the data is scaled
    from 0 to 1.
    """
    abs_pwr = np.abs(data)**0.35
    pwr_mean = np.mean(abs_pwr)
    pwr_std = np.std(abs_pwr)

    # Z score normalization
    data_norm = (abs_pwr - pwr_mean) / pwr_std

    # Cull outliers
    # Data has approx Rayleigh distribution so the positive tail is heavier
    outlier_low = -2
    outlier_high = 3.5
    data_norm[data_norm > outlier_high] = outlier_high
    data_norm[data_norm < outlier_low] = outlier_low

    # Map the data to the range 0-1
    scaled_scene = (data_norm - outlier_low) / (outlier_high - outlier_low)

    # Original scaling method from Gamma
    # Does not work that well for machine learning
    # scale_factor = 150 * np.mean(abs_pwr)
    # scaled_scene = scale_factor * abs_pwr
    # scaled_scene = np.minimum(scaled_scene, 255.0) / 255.0
    # scaled_scene = scaled_scene / (np.amax(scaled_scene) + 1e-3)

    if debug:
        import matplotlib.pyplot as plt
        print('mean(data): {}'.format(np.mean(data)))
        print('stddev(data): {}'.format(np.std(data)))
        print('mean(abs_pwr): {}'.format(np.mean(abs_pwr)))
        print('stddev(abs_pwr): {}'.format(np.std(abs_pwr)))
        print('mean(scaled_scene): {}'.format(np.mean(scaled_scene)))
        print('stddev(scaled_scene): {}'.format(np.std(scaled_scene)))
        plt.figure()
        plt.hist(data.flatten(), bins=100, density=True, histtype='step')
        plt.title('data')
        plt.figure()
        plt.hist(abs_pwr.flatten(), bins=100, density=True, histtype='step')
        plt.title('abs_pwr')
        plt.figure()
        plt.hist(scaled_scene.flatten(), bins=100, density=True, histtype='step')
        plt.title('scaled_scene')

    return scaled_scene


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
