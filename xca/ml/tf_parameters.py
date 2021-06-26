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
