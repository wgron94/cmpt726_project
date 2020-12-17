import os.path
import sarlab.gammax as gx
import sarlab.speckle_tracking.machine_learning as speckle


def main():
    # Build up all the filenames required
    test_data_dir = '/datadisk1/cmpt726_speckletracking/rsat2_sla19d/test/rmli_lr/'
    # model_dir = '/datadisk1/cmpt726_speckletracking/rsat2_sla19d/model/20201203_3/'
    # model_dir = '/datadisk1/cmpt726_speckletracking/rsat2_sla19d/tsx_trained_model/'
    model_dir = '/datadisk1/cmpt726_speckletracking/wg_scratch/'
    speckle_dir = '/datadisk1/cmpt726_speckletracking/rsat2_sla19d/test/speckle/'

    fixed_base = '20100714'
    moving_base = '20150712'
    offs_base = fixed_base + '_' + moving_base

    # model_name = '0266.pt'
    # model_name = '0200.pt'
    model_name = '0024.pt'

    moving_rmli_fname = os.path.join(test_data_dir, moving_base + '.rmli')
    fixed_rmli_fname = os.path.join(test_data_dir, fixed_base + '.rmli')
    moved_npz_fname = os.path.join(model_dir, moving_base + '_moved.npz')
    warp_npz_fname = os.path.join(model_dir, offs_base + '_warp.npz')
    model_fname = os.path.join(model_dir, model_name)
    off_par_fname = os.path.join(speckle_dir, offs_base + '.off_par')

    #crop_center = (1900, 1000)
    #crop_center = (1650, 600)
    crop_center = (3100, 950)  # Good for LR data
    #crop_center = (9200, 3000)  # Good for HR data
    crop_sz = (512, 512)
    gpu = '2'
    reg_weight = 0.01
    ncc_win = 16

    # Create an RMLI object to grab its dimensions
    rmli = gx.MLI(fixed_rmli_fname, par=gx.MLI_Par(fixed_rmli_fname + '.par'))
    multilook = (rmli.par['range_looks'], rmli.par['azimuth_looks'])

    speckle.register_chip(
        moving_rmli_fname,
        fixed_rmli_fname,
        model_fname,
        crop_sz,
        crop_center,
        warp_npz_fname,
        moved_npz_fname,
        gpu
    )

    speckle.plot_warp(warp_npz_fname, show=False)
    speckle.plot_warped_image(moving_rmli_fname, moved_npz_fname,
                              crop_center, crop_sz)
    speckle.plot_gamma_comparison(warp_npz_fname, off_par_fname, crop_center,
                                  crop_sz, rmli.dim, multilook, show=False)
    speckle.loss_comparison(warp_npz_fname, off_par_fname, moving_rmli_fname,
                            fixed_rmli_fname, moved_npz_fname, crop_center,
                            crop_sz, reg_weight, ncc_win, debug=False)


if __name__ == '__main__':
    main()
