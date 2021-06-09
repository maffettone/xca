from xca.examples.arxiv200800283.example_synthesis import pattern_simulation
from xca.ml.tf_data_proc import np_dir_to_record
from xca.ml.tf_models import CNN_training as training
from xca.ml.tf_parameters import load_hyperparameters
from pathlib import Path


def main():
    for system in ("BaTiO", "ADTA", "NiCoAl"):
        pattern_simulation(100, system)
        np_dir_to_record(Path("tmp") / f"{system}", Path("tmp") / f"{system}.tfrecords")
        params = load_hyperparameters(params_file=f"{system}_training.json")
        res, model = training(params=params)
        print(f"Results for {system}")
        print(res)
        model.save(str(Path("tmp") / f"{system}"))


if __name__ == '__main__':
    main()
