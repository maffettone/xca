import time
from pathlib import Path
from tensorflow.python.keras.optimizer_v2.adam import Adam
import numpy as np
import tensorflow as tf


def _default_CNN_hyperparameters():
    hyperparams = {
        # Keras Params
        "ReLU_alpha": 0.2,
        "lr": 0.0002,
        "beta_1": 0.5,
        "beta_2": 0.999,
        # Overall Training Params
        "seed": None,
        "n_classes": 5,
        "batch_size": 16,
        "data_shape": [2894, 1],
        "dataset_paths": None,
        "verbose": True,
        "multiprocessing": 8,
        "n_epochs": 10,
        "out_dir": None,
        "checkpoint_rate": 2,
        "continue": False,
        "val_split": 0.15,
        "ensemble_size": 1,
        # Classifier params
        "filters": [8, 8, 4],
        "kernel_sizes": [5, 5, 5],
        "strides": [2, 2, 2],
        "pool_sizes": [1, 1, 1],
        "batchnorm": False,
        "dense_dims": [],
        "dense_dropout": 0.4,
        "categorical": True,
    }
    return hyperparams


def load_hyperparameters(params_file=None, params_dict=None):
    """
    load hyperparameters for a single run from a json or a dictionary
    """
    import json

    if params_file:
        with open(params_file, "r") as f:
            params_file = json.load(f)

    params = _default_CNN_hyperparameters()

    if params_file:
        params.update(params_file)
    if params_dict:
        params.update(params_dict)
    return params


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_training(
    *, model, seed, out_dir, optimizer, verbose, data_shape, learning_rate
):
    """Common initialization across TF training"""
    start_time = time.time()
    set_seed(seed)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    if verbose:
        model.build((None, *data_shape))
        model.summary()
    if optimizer is None:
        optimizer = Adam(learning_rate=learning_rate)
    # Checkpoints
    checkpoint_dir = str(Path(out_dir) / "training_checkpoints")
    checkpoint_prefix = str(Path(checkpoint_dir) / "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    return start_time, checkpoint_prefix, checkpoint, optimizer


def breakdown_training(*, verbose, start_time, results, out_dir):
    if verbose:
        print()
        print("Time for full training is {} sec".format(time.time() - start_time))

    for key in results:
        with open(Path(out_dir) / (key + ".txt"), "w") as f:
            for result in results[key]:
                f.write(str(result))
                f.write("\n")
