from xca.examples.arxiv200800283.example_synthesis import pattern_simulation
from xca.ml.tf_data_proc import dir2TFR
from xca.ml.tf_models import CNN_training as training
from xca.ml.tf_parameters import load_hyperparameters


def main():
    for system in ("BaTiO", "ADTA", "NiCoAl"):
        pattern_simulation(100, system)
        dir2TFR(f"tmp/{system}", f"tmp/{system}.tfrecords")
        params = load_hyperparameters(params_file=f"{system}_training.json")
        res, model = training(params=params)
        print(f"Results for {system}")
        print(res)
        model.save(f"tmp/{system}_model")


if __name__ == '__main__':
    main()
