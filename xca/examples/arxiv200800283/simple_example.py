from xca.examples.arxiv200800283.example_synthesis import pattern_simulation
from xca.ml.tf_data_proc import xr_dir_to_record, _int64_feature
from xca.ml.tf_models import model_training, build_fusion_ensemble_model, build_CNN_model
from xca.ml.tf_parameters import load_hyperparameters
from pathlib import Path


def main():
    for system in ("BaTiO", "ADTA", "NiCoAl"):
        mapping = pattern_simulation(10, system)
        xr_dir_to_record(Path("tmp") / f"{system}",
                         Path("tmp") / f"{system}.tfrecords",
                         attrs_key="input_cif",
                         transform=lambda x: _int64_feature(mapping[x]))
        params = load_hyperparameters(params_file=f"{system}_training.json")
        model = build_fusion_ensemble_model(params.pop("ensemble_size", 1),
                                            build_CNN_model,
                                            **params)
        res = model_training(model, **params)
        print(f"Results for {system}")
        print(res)
        model.save(str(Path("tmp") / f"{system}"))


if __name__ == '__main__':
    main()
