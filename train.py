"""
Command line script for training a Voxelmorph network on SAR data.

This was modified from the train.py script in the Voxelmorph repository
(https://github.com/voxelmorph/voxelmorph).
"""
import os
import argparse
import time
import sys
import logging
import datetime
import torch
import sarlab.speckle_tracking.machine_learning as speckle


def setup_logging(log_filename):
    """
    Sets up the logging module to log to both the console and a file.
    Based on the Logging Cookbook
    (https://docs.python.org/3/howto/logging-cookbook.html#logging-cookbook)
    """
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_filename,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def main():
    # Manually set the location of the Voxelmorph package
    # Don't install via pip. This is so we can customize their code easily
    if os.getenv('HOSTNAME') != 'ensc-sarserv-03.research.sfu.ca':
        raise ValueError('This script assumes we are running on the Nemo server')
    code_dir = os.path.dirname(__file__)
    voxelmorph_path = os.path.join(code_dir, 'voxelmorph')
    sys.path.append(voxelmorph_path)

    # Import voxelmorph with pytorch backend
    os.environ['VXM_BACKEND'] = 'pytorch'
    import voxelmorph as vxm

    # Parse the commandline
    parser = argparse.ArgumentParser()

    # Data organization parameters
    parser.add_argument('datadir', help='base data directory')
    parser.add_argument('--model-dir', default='models',
                        help='model output directory (default: models)')
    parser.add_argument('--scene-subset', type=int, nargs='+', help='Subset of'
                        ' scene to use for training. rg_start rg_end az_start'
                        ' az_end')
    parser.add_argument('--chip-size', type=int, nargs='+',
                        help='Chip size of training data, specified as '
                        'rg_size az_size. Default is 512 512.')
    parser.add_argument('--random-orient', action='store_true',
                        help='Increase data augmentation by randomly '
                        'orienting input pairs (only valid for square chip '
                        'sizes)')

    # Training parameters
    parser.add_argument('--gpu', default='0',
                        help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100,
                        help='frequency of model saves (default: 100)')
    parser.add_argument('--load-model', help='optional model file to initialize with')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--cudnn-nondet',  action='store_true',
                        help='disable cudnn determinism - might slow down training')

    # Network architecture parameters
    parser.add_argument('--enc', type=int, nargs='+',
                        help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters'
                             '(default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--stride', type=int, default=2,
                        help='Stride in the down-arm of the U-net (default: 2)')
    parser.add_argument('--bidir', action='store_true',
                        help='Not used, kept for voxelmorph compatibility')

    # Loss hyperparameters
    parser.add_argument('--image-loss', default='mse',
                        help='image reconstruction loss - can be mse or ncc'
                             '(default: mse)')
    parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                        help='weight of deformation loss (default: 0.01)')
    parser.add_argument('--ncc-window', type=int, default=9,
                        help='NCC window size (default: 9)')
    args = parser.parse_args()
    bidir = args.bidir

    # Prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # Log file Preparation
    train_log_fname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")\
        + '_train.log'
    log_filename = os.path.join(args.model_dir, train_log_fname)
    setup_logging(log_filename)
    training_log = logging.getLogger('training')
    # Log all the arguments
    training_log.info('Argument list:')
    for arg, value in sorted(vars(args).items()):
        training_log.info("{}: {}".format(arg, value))

    # Setup the data generator
    training_size = args.chip_size if args.chip_size else [512, 512]
    scene_subset = args.scene_subset if args.scene_subset else None
    if training_size[0] == training_size[1]:
        # Size is a square, use the specified flag
        random_orient = args.random_orient
    else:
        # Size is a rectangle, do not randomly orient
        random_orient = False

    generator = speckle.rmli_pair_gen(args.datadir, training_size, args.batch_size,
                                      scene_subset, random_orient)

    # Extract shape from sampled input
    inshape = next(generator)[0][0].shape[1:-1]

    # Device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.batch_size >= nb_gpus, 'Batch size ({}) should be no less than' + \
        ' the number of gpus ({})'.format(args.batch_size, nb_gpus)

    # Enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = True

    # unet architecture
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    if args.load_model:
        # load initial model (if specified)
        model = vxm.networks.VxmDense.load(args.load_model, device)
    else:
        # otherwise configure new model
        model = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize,
            stride=args.stride
        )

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # prepare image loss
    if args.image_loss == 'ncc':
        print('NCC window size: {}'.format(args.ncc_window))
        ndims = len(inshape)
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        ncc_win = [args.ncc_window] * ndims  # Build a square window
        image_loss_func = vxm.losses.NCC(ncc_win).loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    # Not using bidirectional losses here
    losses  = [image_loss_func]
    weights = [1]

    # Prepare deformation loss
    losses  += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
    weights += [args.weight]

    # Training loops
    for epoch in range(args.initial_epoch, args.epochs):
        # save model checkpoint
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

        for step in range(args.steps_per_epoch):
            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(generator)
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in y_true]

            # run inputs through the model to produce a warped image and flow field
            y_pred = model(*inputs)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append('%.6f' % curr_loss.item())
                loss += curr_loss

            loss_info = 'loss: %.6f  (%s)' % (loss.item(), ', '.join(loss_list))

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print step info
            epoch_info = 'epoch: %04d' % (epoch + 1)
            step_info = ('step: %d/%d' % (step + 1, args.steps_per_epoch)).ljust(14)
            time_info = 'time: %.2f sec' % (time.time() - step_start_time)
            training_log.debug('  '.join((epoch_info, step_info, time_info, loss_info)))

    # final model save
    model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))


if __name__ == '__main__':
    main()
