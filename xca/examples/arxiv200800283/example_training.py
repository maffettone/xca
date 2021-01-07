from xca.ml.tf_data_proc import dir2TFR
from xca.ml.tf_models import CNN_training as training
from xca.ml.tf_parameters import load_hyperparameters
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params',
                        help='hyperparameters file (.json)')
    parser.add_argument('--npy',
                        help='npy directory for conversion',
                        default=None)
    args = vars(parser.parse_args())
    params = load_hyperparameters(params_file = args['params'])
    if args['npy']:
        dir2TFR(args['npy'], params['dataset_path'])
    res = training(params=params)
    print(res)