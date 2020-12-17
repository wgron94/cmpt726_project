import argparse
import sarlab.gammax as gx
import sarlab.speckle_tracking.machine_learning as speckle


def training_set():
    data_dir = '/datadisk1/cmpt726_speckletracking/rsat2_sla19d/train/'
    ingest_cfg = {'polarizations': 'HH'}
    stack = gx.SLC_stack(dirname=data_dir, ingest_cfg=ingest_cfg,
                         looks_hr=[1,5], looks_lr=[3,15], multiprocess=False,
                         skipmode='exists')
    stack.mk_mli_all(looks=('fr', 'hr', 'lr'))


def testing_set():
    data_dir = '/datadisk1/cmpt726_speckletracking/rsat2_sla19d/test/'
    ingest_cfg = {'polarizations': 'HH'}
    stack = gx.SLC_stack(dirname=data_dir, ingest_cfg=ingest_cfg,
                         looks_hr=[1,5], looks_lr=[3,15],
                         multiprocess=False, skipmode='exists')
    stack.mk_mli_all(looks=('fr', 'hr', 'lr'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create datasets')
    parser.add_argument('--train', action='store_true', help='Create training data')
    parser.add_argument('--test', action='store_true', help='Create test data')
    args = parser.parse_args()

    if args.train:
        training_set()

    if args.test:
        testing_set()
