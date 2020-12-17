import os
import sys
import torch
import sarlab.speckle_tracking.machine_learning as speckle
import random
import numpy as np
import matplotlib.pyplot as plt

# independent variable
lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
lambda_strs = ['0001/', '0005/', '0010/', '0050/','0100/', '0500/']

lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
lr_strs = ['1e-5/','5e-5/', '1e-4/', '5e-4/', '1e-3/', '5e-3/']

gpu = '1'
moving = '20160720'
fixed = '20170604'
model_dir_lambda = '/tsx/lambda'
model_dir_lr = '/tsx/lr'
rmli_dir = '/TSX/validation/rmli_lr/'

rg_n = 1420
az_n = 790
chip_size = 512

moving_name = rmli_dir + moving + '.rmli'
fixed_name = rmli_dir + fixed + '.rmli'

ncc_window = 16
border = chip_size // 2 + 2

rg_center = random.choice(range(border, rg_n - border))
az_center = random.choice(range(border, az_n - border))
# rg_center = rg_n // 2
# az_center = az_n // 2

# center = [rg_center, az_center]
center = [1000, 515]
crop = [512, 512]

if __name__ == '__main__':

    if os.getenv('HOSTNAME') != 'ensc-sarserv-03.research.sfu.ca':
        raise ValueError('This script assumes we are running on the Nemo server')
    code_dir = os.path.dirname(__file__)
    voxelmorph_path = os.path.join(code_dir, 'voxelmorph')
    sys.path.append(voxelmorph_path)

    # Import voxelmorph with pytorch backend
    os.environ['VXM_BACKEND'] = 'pytorch'
    import voxelmorph as vxm

    loss_lambda = np.zeros((3,len(lambdas)))
    loss_lr = np.zeros((3,len(lrs)))

    # process lambdas
    for i, ld in enumerate(lambdas):
        model_name = model_dir_lambda + lambda_strs[i] + '0100.pt'
        warp_name = model_dir_lambda + lambda_strs[i] + moving + '_' + fixed +'_warp.npz'
        moved_name = model_dir_lambda + lambda_strs[i] + moving + '_moved.npz'

        # speckle.register_chip(moving_name, fixed_name, model_name, crop, center, warp_name, moved_name, gpu, debug=False)

        # loss_lambdas[0, i], loss_lambdas[1, i], loss_lambdas[2, i] = speckle.loss_vxm(warp_name, moving_name, fixed_name, moved_name, center, crop, ld, ncc_window)
        loss_lambda[:, i] = speckle.loss_vxm(warp_name, moving_name, fixed_name, moved_name, center, crop, ld, ncc_window)
    
    # process learning rates
    for i, lr in enumerate(lrs):
        model_name = model_dir_lr + lr_strs[i] + '0100.pt' #0032
        warp_name = model_dir_lr + lr_strs[i] + moving + '_' + fixed +'_warp.npz'
        moved_name = model_dir_lr + lr_strs[i] + moving + '_moved.npz'

        # speckle.register_chip(moving_name, fixed_name, model_name, crop, center, warp_name, moved_name, gpu, debug=False)

        loss_lr[:,i] = speckle.loss_vxm(warp_name, moving_name, fixed_name, moved_name, center, crop, 0.01, ncc_window)
    
    plt.figure(figsize=(12,5))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    linestyles = ['-', '-.', '--']
    # data = losses[:, 1:]
    # plt.plot(lambdas[1:], data.transpose())
    labels = ['Similarity', 'Weighted smoothness', 'Weighted total']
    for i in [0,2]:
        ax1.semilogx(lambdas, loss_lambda[i], label = labels[i], linestyle=linestyles[i], marker='o', fillstyle = 'none', color = 'k')
        plt.semilogx(lrs, loss_lr[i], label = labels[i], linestyle=linestyles[i], marker='o', fillstyle = 'none', color = 'k')
    
    ylim = [-0.76, -0.6]
    ax1.set(xlabel = r'$\lambda$', ylabel = 'Loss', title = 'Regularization Loss', ylim=ylim)

    ax2.set(xlabel = 'Learning rate', title = 'Learning Rate Loss', ylim=ylim)
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower left')

    label_size = 13
    ax1.xaxis.label.set_size(label_size)
    ax1.yaxis.label.set_size(label_size)
    ax2.xaxis.label.set_size(label_size)
    ax2.yaxis.label.set_size(label_size)


    plt.savefig('losses.png')





