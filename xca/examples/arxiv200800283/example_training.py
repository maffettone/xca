from xca.ml.tf.data_proc import np_dir_to_record
from xca.ml.tf.cnn import build_CNN_model, build_fusion_ensemble_model, training
from xca.ml.tf.utils import load_hyperparameters
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", help="hyperparameters file (.json)")
    parser.add_argument("--npy", help="npy directory for conversion", default=None)
    args = vars(parser.parse_args())
    params = load_hyperparameters(params_file=args["params"])
    if args["npy"]:
        np_dir_to_record(args["npy"], params["dataset_path"])
    model = build_fusion_ensemble_model(
        params["ensemble_size"], model_builder=build_CNN_model, **params
    )
    res = training(model, **params)
    print(res)
