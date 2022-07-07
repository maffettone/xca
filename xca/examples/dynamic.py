from pathlib import Path
from xca.ml.torch.training import (
    dynamic_training,
    ClassificationModule,
    VAEModule,
    JointVAEClassifierModule,
)
from xca.ml.torch.cnn import EnsembleCNN
from xca.ml.torch.vae import CNNDecoder, CNNEncoder, VAE

# BEGIN XRD PARAMETERS #
param_dict = {
    "wavelength": 0.1671,
    "noise_std": 5e-4,
    "instrument_radius": 1065.8822732979447,
    "theta_m": 0.0,
    "tth_min": 0.011231808788013649,
    "tth_max": 24.853167100343246,
    "n_datapoints": 3488,
}
kwargs = {
    "bkg_1": (-1e-4, 1e-4),
    "bkg_0": (0, 1e-3),
    "sample_height": (-0.2, 0.2),
    "march_range": (0.8, 1.0),
}
# Forcing Order with respect to T
cif_paths = [
    Path(__file__).parent / "arxiv200800283" / "cifs-BaTiO" / "rhomb.cif",
    Path(__file__).parent / "arxiv200800283" / "cifs-BaTiO" / "ortho.cif",
    Path(__file__).parent / "arxiv200800283" / "cifs-BaTiO" / "tetra.cif",
    Path(__file__).parent / "arxiv200800283" / "cifs-BaTiO" / "cubic.cif",
]
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
    pl_module = ClassificationModule(model, lr=2e-4)
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
    pl_module = VAEModule(vae, kl_weight=0.5)

    metrics = dynamic_training(
        pl_module,
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


def joint_vae_class_main():
    cnn = EnsembleCNN(
        ensemble_size=10,
        filters=[8, 8, 4],
        kernel_sizes=[5, 5, 5],
        strides=[2, 2, 2],
        pool_sizes=[1, 1, 1],
        n_classes=4,
        ReLU_alpha=0.2,
        dense_dropout=0.4,
    )
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
    pl_module = JointVAEClassifierModule(
        cnn, vae, classification_lr=1e-3, vae_lr=3e-4, kl_weight=1e-2
    )

    metrics = dynamic_training(
        pl_module,
        max_epochs=3,
        batch_size=8,
        num_workers=8,
        batch_per_train_epoch=10,
        batch_per_val_epoch=1,
        cif_paths=cif_paths,
        param_dict=param_dict,
        shape_limit=shape_limit,
        **kwargs,
    )
    return pl_module, metrics


def joint_bto_main(checkpoint=None):
    """Train and val (sim) and test (real)"""
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import torch

    class ExpDataset(Dataset):
        def __init__(self, path=None):
            if path is None:
                path = Path(__file__).parent / "data"
            self.x = torch.tensor(
                np.load(path / "bto_spectra.npy") * 2 - 1, dtype=torch.float
            )
            self.y = torch.tensor(np.load(path / "bto_classes.npy"), dtype=torch.long)

        def __len__(self):
            return self.y.shape[0]

        def __getitem__(self, idx):
            return self.x[idx, ...][None, ...], self.y[idx]

    test_dataset = ExpDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

    if checkpoint is None:
        cnn = EnsembleCNN(
            ensemble_size=10,
            filters=[8, 8, 4],
            kernel_sizes=[5, 5, 5],
            strides=[2, 2, 2],
            pool_sizes=[1, 1, 1],
            n_classes=4,
            ReLU_alpha=0.2,
            dense_dropout=0.4,
        )
        encoder = CNNEncoder(
            input_length=3488,
            latent_dim=2,
            filters=(8, 8, 4),
            kernel_sizes=(5, 5, 5),
            strides=(2, 2, 2),
            pool_sizes=(1, 1, 1),
        )
        decoder = CNNDecoder.from_encoder(encoder)

        vae = VAE(encoder, decoder)
        # torch.save(vae, "/tmp/vae.torch")
        pl_module = JointVAEClassifierModule(
            cnn, vae, classification_lr=0.0002, vae_lr=0.0002, kl_weight=1e-4
        )
    else:
        ckpt = torch.load(checkpoint)
        cnn = EnsembleCNN(**ckpt["hyper_parameters"]["classifier_hparams"])
        encoder = CNNEncoder(**ckpt["hyper_parameters"]["encoder_hparams"])
        decoder = CNNDecoder(**ckpt["hyper_parameters"]["decoder_hparams"])
        vae = VAE(encoder, decoder)
        pl_module = JointVAEClassifierModule.load_from_checkpoint(
            checkpoint, classification_model=cnn, vae_model=vae
        )

    metrics = dynamic_training(
        pl_module,
        max_epochs=100,
        gpus=[0],
        metric_monitor="val_classification_loss",
        batch_size=16,
        num_workers=32,
        prefetch_factor=8,
        batch_per_train_epoch=100,
        cif_paths=cif_paths,
        param_dict=param_dict,
        shape_limit=shape_limit,
        exp_dataloader=test_dataloader,
        **kwargs,
    )
    return pl_module, metrics


if __name__ == "__main__":
    metrics = training()
