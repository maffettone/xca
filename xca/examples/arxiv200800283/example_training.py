from xca.ml.tf_data_proc import np_dir_to_record
from xca.ml.tf_models import model_training, build_fusion_ensemble_model, build_CNN_model
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
        np_dir_to_record(args['npy'], params['dataset_path'])
    model = build_fusion_ensemble_model(params["ensemble_size"], model_builder=build_CNN_model, **params)
    res = model_training(model, **params)
    print(res)