import sarlab.speckle_tracking.machine_learning as speckle
import os.path
import sarlab.gammax as gx


import matplotlib.pyplot as plt


def main():
    # Build up all the filenames required
    train_data_dir = '/datadisk1/cmpt726_speckletracking/rsat2_sla19d/train/rmli_lr/'
    model_dir = '/datadisk1/cmpt726_speckletracking/rsat2_sla19d/model/20201119/'
    speckle_dir = '/datadisk1/cmpt726_speckletracking/rsat2_sla19d/test/speckle/'

    data_gen = speckle.rmli_pair_gen(train_data_dir, (512, 512), random_orient=True,
                                     scene_subset=(900, 2100, 200, 1400))
    while True:
        #import pdb
        #pdb.set_trace()
        inputs, _  = next(data_gen)
        fixed_img = inputs[0]
        moving_img = inputs[1]
        fixed_img = fixed_img[0, ..., 0]
        moving_img = moving_img[0, ..., 0]

        fig, (fix_ax, mov_ax) = plt.subplots(1, 2, figsize=(8, 4))
        fix_im = fix_ax.imshow(fixed_img)
        fix_ax.set_title('Fixed image')
        fig.colorbar(fix_im, ax=fix_ax)
        mov_im = mov_ax.imshow(moving_img)
        mov_ax.set_title('Moving image')
        fig.colorbar(mov_im, ax=mov_ax)
        plt.show()


if __name__ == '__main__':
    main()
