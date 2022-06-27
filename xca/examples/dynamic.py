from pathlib import Path
from xca.ml.torch.training import dynamic_training, ClassificationTrainer
from xca.ml.torch.cnn import EnsembleCNN

# BEGIN XRD PARAMETERS #
param_dict = {
    "wavelength": 0.1671,
    "noise_std": 5e-4,
    "instrument_radius": 1065.8822732979447,
    "theta_m": 0.0,
    "2theta_min": 0.011231808788013649,
    "2theta_max": 24.853167100343246,
    "n_datapoints": 3488,
}
kwargs = {
    "bkg_1": (-1e-4, 1e-4),
    "bkg_0": (0, 1e-3),
    "sample_height": (-0.2, 0.2),
    "march_range": (0.8, 1.0),
}
cif_paths = list(
    (Path(__file__).parent / "arxiv200800283" / "cifs-BaTiO/").glob("*.cif")
)
shape_limit = 1e-1
# END XRD PARAMETERS #


def training():
    #  Construct CNN model
    model = EnsembleCNN(
        ensemble_size=100,
        filters=[8, 8, 4],
        kernel_sizes=[5, 5, 5],
        strides=[2, 2, 2],
        pool_sizes=[1, 1, 1],
        n_classes=4,
        ReLU_alpha=0.2,
        dense_dropout=0.4,
    )

    # Use helper dynamic_training function, passing kwargs for Dataset
    pl_module = ClassificationTrainer(model, lr=2e-4)
    metrics = dynamic_training(
        pl_module,
        max_epochs=5,
        batch_size=4,
        num_workers=32,
        prefetch_factor=8,
        cif_paths=cif_paths,
        param_dict=param_dict,
        shape_limit=shape_limit,
        **kwargs,
    )
    return metrics


def vae_main():
    from xca.ml.torch.vae import CNNDecoder, CNNEncoder, VAE
    from xca.ml.torch.training import dynamic_training, VAETrainer

    encoder = CNNEncoder(
        input_length=3488,
        latent_dim=2,
        filters=(8, 4),
        kernel_sizes=(5, 5),
        strides=(2, 1),
        pool_sizes=(2, 2),
    )
    decoder = CNNDecoder.from_encoder(encoder)

    vae = VAE(encoder, decoder)
    trainer = VAETrainer(vae, kl_weight=0.5)

    metrics = dynamic_training(
        trainer,
        2,
        batch_size=4,
        num_workers=4,
        batch_per_train_epoch=10,
        batch_per_val_epoch=1,
        cif_paths=cif_paths,
        param_dict=param_dict,
        shape_limit=shape_limit,
        **kwargs,
    )
    return metrics


if __name__ == "__main__":
    metrics = training()
