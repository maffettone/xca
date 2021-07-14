from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import time
import os
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Reshape,
    Input,
    LeakyReLU,
    Conv1D,
    Conv1DTranspose,
    AveragePooling1D,
    Average,
    Lambda,
)
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from .tf_data_proc import build_dataset
from pathlib import Path
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_consistent_vae(
    *,
    data_shape,
    latent_dim,
    encoder_dense_dims,
    encoder_filters,
    encoder_kernel_sizes,
    encoder_pool_sizes,
    encoder_strides,
    verbose=False,
    **kwargs
):

    """
    Builds Convolutional VAE network that has consistent shapes across layers and encoder/decoder models.

    ** Note: Padding is assumed to be 'same' for all convolutional layers **

    Parameters
    ----------
    data_shape: tuple of int
        shape of input data
    latent_dim: int
        number of variables defining the latent space
    encoder_dense_dims: list of int
        numbers of neurons in each of the encoder's dense, fully connected layers
    encoder_filters: list of int
        number of filters in each convolutional layer of the encoder model. consistent with length of encoder_kernel_sizes, encoder_pool_sizes, encoder_strides, and encoder_paddings
    encoder_kernel_sizes: list of int
        kernel sizes for each convolutional layer of the encoder model. consistent with length of other input lists
    encoder_strides: list of int
        number of strides in each convolutional layer of the encoder model. consistent with length of other input lists
    encoder_pool_sizes: list of int
        pool sizes for each convolutional layer of the encoder model. consistent with length of other input lists
    verbose: bool
        if True, displays model summaries for the encoder and decoder


    Returns
    -------
    tf.keras.Model (VAE model)
    """

    # Build the encoder as specified by the user
    encoder, last_conv_layer_shape = build_CNN_encoder_model(
        data_shape=data_shape,
        latent_dim=latent_dim,
        dense_dims=encoder_dense_dims,
        filters=encoder_filters,
        kernel_sizes=encoder_kernel_sizes,
        strides=encoder_strides,
        pool_sizes=encoder_pool_sizes,
        paddings=["same"] * len(encoder_filters),
        verbose=verbose,
    )

    # Build stride vector such that output dimensionality matches data_shape
    shrink_factors = encoder_strides + encoder_pool_sizes
    decoder_strides = shrink_factors + [1]
    decoder_filters = [8] * (len(decoder_strides) - 1) + [1]
    decoder_kernel_sizes = [5] * (len(decoder_strides) - 1) + [1]

    # With stride size configured, build the decoder
    decoder = build_CNN_decoder_model(
        latent_dim=latent_dim,
        last_conv_layer_shape=last_conv_layer_shape,
        filters=decoder_filters,
        kernel_sizes=decoder_kernel_sizes,
        strides=decoder_strides,
        paddings=["same"] * len(decoder_strides),
        verbose=verbose,
    )

    kl_loss_factor = 1 / 2048
    vae = VAE(encoder, decoder, kl_loss_factor)
    return vae


def build_CNN_encoder_model(
    *,
    data_shape,
    latent_dim,
    dense_dims,
    filters,
    kernel_sizes,
    strides,
    pool_sizes,
    paddings,
    dense_dropout=0.0,
    verbose=False,
    **kwargs
):
    """
    Builds Convolutional encoder network for downsampling of 1-d xrd data

    Parameters
    ----------
    data_shape: tuple of int
        shape of input XRD pattern
    latent_dim: int
        number of variables defining the latent space
    dense_dims: list of int
        dimensions of dense layers in encoder model
    filters: list of int
        consistent length with other items, number of filters for each Conv1D layer
    kernel_sizes: list of int
        consistent length with other items, size of kernel for each Conv1D layer
    strides: list of int
        consistent length with other items, size of stride for each Conv1D layer
    pool_sizes: list of int
        consistent length with other items, size of pooling kernel for each Conv1D layer
    paddings: list of str
        consistent length with other items, type of padding for each Conv1D layer
        see tensorflow/keras documentation for valid entries
    dense_dropout: float
        percentage of dropout for final layer
    verbose: bool
        if True, prints out model summary (default is False)

    Returns
    -------
    model: Model
    last_conv_layer_shape: tuple
        Final shape prior to dense layers. Retained for building effective decoder.
    """

    encoder_inputs = Input(shape=data_shape, name="X")
    x = encoder_inputs
    # Downsampling
    for i in range(len(filters)):
        x = Conv1D(
            filters[i],
            kernel_sizes[i],
            strides=strides[i],
            padding=paddings[i],
            name="conv_{}".format(i),
        )(x)
        x = AveragePooling1D(pool_size=pool_sizes[i], strides=None, padding="same")(x)

    last_conv_layer_shape = x.shape
    # Flatten and output
    x = Flatten()(x)
    x = Dropout(dense_dropout)(x)
    for i, dim in enumerate(dense_dims):
        x = Dense(dim, activation="relu", name="dense_{}".format(i))(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_sigma = Dense(latent_dim, name="z_log_sig")(x)

    # Create model and display summary if verbose
    model = Model(encoder_inputs, [z_mean, z_log_sigma], name="CNN_encoder")
    if verbose:
        model.summary()
    return model, last_conv_layer_shape


def build_CNN_decoder_model(
    *,
    latent_dim,
    last_conv_layer_shape,
    filters,
    kernel_sizes,
    strides,
    paddings,
    verbose,
    **kwargs
):
    """
    Builds Convolutional decoder network for upsampling of 1-d xrd data

    Parameters
    ----------
    latent_dim: int
        number of variables defining the latent space
    last_conv_layer_shape: tuple of int
        shape of the output from the last convolutional layer in the encoder; used to calculate dimensionality of the decoder's initial dense layer
    filters: list of int
        consistent length with other items, number of filters for each Conv1DTranspose layer
    kernel_sizes: list of int
        consistent length with other items, size of kernel for each Conv1DTranspose layer
    strides: list of int
        consistent length with other items, size of stride for each Conv1DTranspose layer
    pool_sizes: list of int
        consistent length with other items, size of pooling kernel for each Conv1DTranspose layer
    paddings: list of str
        consistent length with other items, type of padding for each Conv1DTranspose layer
        see tensorflow/keras documentation for valid entries
    verbose: bool
        if True, prints out model summary (default is False).

    Returns
    -------
    model: Model
    """

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(last_conv_layer_shape[1] * last_conv_layer_shape[2], activation="relu")(
        latent_inputs
    )
    x = Reshape((last_conv_layer_shape[1], last_conv_layer_shape[2]))(x)

    # Upsampling
    for i in range(len(filters)):
        x = Conv1DTranspose(
            filters[i],
            kernel_sizes[i],
            strides=strides[i],
            padding=paddings[i],
            name="conv_transpose{}".format(i),
        )(x)

    # Decoder output
    decoder = Model(latent_inputs, x, name="CNN_decoder")

    if verbose:
        decoder.summary()

    return decoder


def build_dense_encoder_model(
    *, data_shape, latent_dim, dense_dims, activation="relu", verbose=False, **kwargs
):
    """
    Builds dense encoder network for 1-d xrd data

    Parameters
    ----------
    data_shape: tuple of int
        shape of input data XRD pattern (2D array)
    latent_dim: int
        number of variables defining the latent space
    activation: string
        specifies activation function for hidden layers
    dense_dims: list of int
        dimensions of hidden layers in encoder model (default is [256, 128])
    verbose: bool
        if True, prints out model summary (default is False)
    """
    inputs = Input(shape=data_shape)
    h_1 = Dense(dense_dims[0], activation=activation, name="enc_dense_1")(inputs)
    h_2 = Dense(dense_dims[1], activation=activation, name="enc_dense_2")(h_1)
    z_mean = Dense(latent_dim, name="z_mean_sample")(h_2)
    z_log_sigma = Dense(latent_dim, name="z_log_var_sample")(h_2)

    model = Model(inputs, [z_mean, z_log_sigma], name="encoder")
    if verbose:
        model.summary()
    return model


def build_dense_decoder_model(
    *, original_dim, latent_dim, dense_dims, activation="relu", verbose=False, **kwargs
):
    """
    Builds dense decoder network for 1-d xrd data

    Parameters
    ----------
    original_dim: int
        original dimensionality of xrd data (i.e. the length of any one sample)
    latent_dim: int
        number of variables defining the latent distribution
    activation: string
        specifies activation function for hidden layers
    dense_dims: list of int
        dimensions of hidden layers in encoder model (default is [256, 128])
    verbose: bool
        if True, prints out model summary (default is False)
    """

    latent_inputs = Input(shape=(latent_dim,), name="z_sample")
    h_1 = Dense(dense_dims[0], activation=activation, name="dec_dense_1")(latent_inputs)
    h_2 = Dense(dense_dims[1], activation=activation, name="dec_dense_2")(h_1)
    outputs = Dense(original_dim, name="output")(h_2)

    model = Model(latent_inputs, outputs, name="decoder")
    if verbose:
        model.summary()
    return model


class VAE(Model):
    def __init__(self, encoder, decoder, kl_loss_weight=1.0, **kwargs):
        """
        Complete variational autoencoder
        Parameters
        ----------
        encoder: Model
        decoder: Model
        kl_loss_weight: float
        kwargs: dict
        """
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_loss_weight = kl_loss_weight

    @staticmethod
    def kl_loss(z_mean, z_log_sigma):
        """Computes the KL-divergence loss with a batchwise sum as a reduction"""
        kl_loss = 1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss = tf.reduce_sum(kl_loss, axis=0)
        return kl_loss

    @staticmethod
    def reconstruction_loss(data, reconstruction):
        """Computes the Mean squared error loss with a batchwise sum as a reduction"""
        reconstruction_loss = tf.reduce_sum(
            tf.keras.losses.mean_squared_error(
                tf.squeeze(data), tf.squeeze(reconstruction)
            ),
            axis=0,
        )
        return reconstruction_loss

    def loss(self, x, z_mean, z_log_sigma, reconstruction):
        return self.kl_loss_weight * self.kl_loss(
            z_mean, z_log_sigma
        ) + self.reconstruction_loss(x, reconstruction)

    @staticmethod
    def sample(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.0, stddev=1
        )
        return z_mean + K.exp(z_log_sigma) * epsilon

    def encode(self, x, *args, **kwargs):
        mean, log_var = self.encoder(x, *args, **kwargs)
        return mean, log_var

    def decode(self, z, *args, **kwargs):
        logits = self.decoder(z, *args, **kwargs)
        return logits

    def __call__(self, x, *args, **kwargs):
        z_mean, z_log_sigma = self.encode(x, *args, **kwargs)
        z = Lambda(self.sample)([z_mean, z_log_sigma], *args, **kwargs)
        reconstruction = self.decode(z, *args, **kwargs)
        return {
            "z_mean": z_mean,
            "z_log_sigma": z_log_sigma,
            "reconstruction": reconstruction,
        }


def VAE_training(
    model,
    *,
    dataset_paths,
    out_dir,
    batch_size,
    lr,
    multiprocessing,
    categorical,
    data_shape,
    n_epochs,
    checkpoint_rate=1,
    verbose=False,
    seed=None,
    **kwargs
):
    """

    Parameters
    ----------
    model: VAE
    dataset_paths: list
        List of paths for import using tf_data_proc.
        Note that even though the VAE does not need labels to train, all datasets require labels.
        See the build_dataset() docs.
    out_dir
    batch_size
    lr
    multiprocessing
    categorical
    data_shape
    n_epochs
    checkpoint_rate
    verbose
    seed
    kwargs

    Returns
    -------

    """
    # Setup
    start_time = time.time()
    set_seed(seed)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    if verbose:
        model.summary()
    optimizer = Adam(lr=lr, **kwargs)
    # Checkpoints
    checkpoint_dir = str(Path(out_dir) / "training_checkpoints")
    checkpoint_prefix = str(Path(checkpoint_dir) / "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # Build dataset
    dataset, _ = build_dataset(
        dataset_paths=dataset_paths,
        batch_size=batch_size,
        multiprocessing=multiprocessing,
        categorical=categorical,
        val_split=0.0,
        data_shape=data_shape,
        preprocess=lambda data, label: {
            "X": tf.cast(data, tf.float32),
            "label": label,
        },
    )

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            output = model(batch["X"], training=True)
            reconstruction_loss = model.reconstruction_loss(
                batch["X"], output["reconstruction"]
            )
            kl_loss = model.kl_loss(output["z_mean"], output["z_log_sigma"])
            loss = reconstruction_loss + model.kl_loss_weight * kl_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return (
            dict(reconstruction_loss=reconstruction_loss, kl_loss=kl_loss, loss=loss),
            output,
        )

    # Actual training
    results = defaultdict(list)
    for epoch in range(n_epochs):
        results["loss"].append(0.0)
        results["kl_loss"].append(0.0)
        results["reconstruction_loss"].append(0.0)
        start = time.time()
        for batch in dataset:
            loss, output = train_step(batch)
            for key in loss:
                results[key][epoch] += loss[key]

        # Save the model every set epochs
        if (epoch + 1) % checkpoint_rate == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        if verbose:
            print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

    if verbose:
        print()
        print("Time for full training is {} sec".format(time.time() - start_time))

    for key in results:
        with open(Path(out_dir) / (key + ".txt"), "w") as f:
            for result in results[key]:
                f.write(str(result))
                f.write("\n")

    return results


def build_CNN_model(
    *,
    data_shape,
    filters,
    kernel_sizes,
    strides,
    ReLU_alpha,
    pool_sizes,
    batchnorm,
    n_classes,
    dense_dims=(),
    dense_dropout=0.0,
    **kwargs
):
    """
    Builds single feed forward convolutional neural network for 1-d xrd data

    Parameters
    ----------
    data_shape: tuple
        shape of input data XRD pattern (should be 2-d)
    filters: list of int
        Consistent length with other items, number of filters for each Conv1D layer
    kernel_sizes: list of int
        Consistent length with other items, size of kernel for each Conv1D layer
    strides: list of int
        Consistent length with other items, size of stride for each Conv1D layer
    ReLU_alpha: float
        [0.0, 1.0) decay for ReLU function
    pool_sizes: list of int
        Consistent length with other items, size of pooling for each AveragePool layer
    batchnorm: bool
        Turn batch normalization on or off
    n_classes: int
        Number of classes for classification
    dense_dims: list of int
        Dimensions of dense layers to follow convolutional model
    dense_dropout: float
        Dropout rate for final layer and all dense layers
    kwargs:
        Dummy dict for using convenience methods that pass larger **kwargs to both model and training

    Returns
    -------
    model: Model

    """
    x_in = Input(shape=data_shape, name="X")
    x = tf.identity(x_in)
    # Downsampling
    for i in range(len(filters)):
        x = Conv1D(
            filters[i],
            kernel_sizes[i],
            strides=strides[i],
            padding="valid",
            kernel_initializer=RandomNormal(0, 0.02),
            name="conv_{}".format(i),
        )(x)
        x = LeakyReLU(alpha=ReLU_alpha)(x)
        x = AveragePooling1D(pool_size=pool_sizes[i], strides=None, padding="same")(x)
        if batchnorm:
            x = BatchNormalization(axis=-1, name="batchnorm_{}".format(i))(x)

    # Flatten and output
    x = Flatten()(x)
    x = Dropout(dense_dropout)(x)
    for i, dim in enumerate(dense_dims):
        x = Dense(dim, activation="relu", name="dense_{}".format(i))(x)
        x = Dropout(dense_dropout)(x)
    x = Dense(n_classes, activation="softmax", name="Discriminator")(x)

    model = Model(x_in, x)
    return model


def build_fusion_ensemble_model(ensemble_size, model_builder, *, data_shape, **kwargs):
    """
    Build's a simple fusion ensemble that connects multiple models by an averaging layer.
    The output of a fusion ensemble is the averaged output from all base estimators.

    Parameters
    ----------
    ensemble_size: int
        Size of the ensemble
    model_builder: Callable
    data_shape: tuple
    kwargs: dict
        Keyword arguments for model_builder

    Returns
    -------
    model: Model
        Ensemble model
    """
    x_in = Input(shape=data_shape, name="X")
    members = []
    for i in range(ensemble_size):
        m = model_builder(data_shape=data_shape, **kwargs)
        members.append(m(x_in))
    outputs = Average()(members)
    model = Model(x_in, outputs)
    return model


def model_training(
    model,
    *,
    dataset_paths,
    out_dir,
    batch_size,
    lr,
    multiprocessing,
    categorical,
    data_shape,
    n_epochs,
    n_classes=0,
    val_split=0.2,
    checkpoint_rate=1,
    beta_1=0.5,
    beta_2=0.999,
    verbose=False,
    seed=None,
    **kwargs
):
    """

    Parameters
    ----------
    model: Model
        Tensorflow model
        Inputs:  'X'
        Outputs: y_pred
    dataset_paths: list of str, list of Path
    out_dir: str, Path
    batch_size: int
    lr: float
    multiprocessing: int
    categorical: bool
    data_shape: tuple
    n_epochs: int
    n_classes: int
    val_split: float
    checkpoint_rate: int
    beta_1: float
    beta_2: float
    verbose: bool
    seed: int
    kwargs:
        Dummy dict for using convenience methods that pass larger **kwargs to both model and training

    Returns
    -------

    """
    start_time = time.time()
    set_seed(seed)

    Path(out_dir).mkdir(exist_ok=True, parents=True)

    if verbose:
        model.summary()

    # Build dataset
    dataset, val_dataset = build_dataset(
        dataset_paths=dataset_paths,
        batch_size=batch_size,
        multiprocessing=multiprocessing,
        categorical=categorical,
        val_split=val_split,
        data_shape=data_shape,
        n_classes=n_classes,
    )

    cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2)

    # Checkpoints
    checkpoint_dir = str(Path(out_dir) / "training_checkpoints")
    checkpoint_prefix = str(Path(checkpoint_dir) / "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            y_pred = model({"X": batch["X"]}, training=True)
            loss = cross_entropy(batch["label"], y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, y_pred

    @tf.function
    def val_step(batch):
        y_pred = model({"X": batch["X"]}, training=False)
        return y_pred

    # Actual training
    results = {"loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(n_epochs):
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        val_accuracy = tf.keras.metrics.CategoricalAccuracy()
        results["loss"].append(0.0)
        start = time.time()
        for batch in dataset:
            loss, y_pred = train_step(batch)
            results["loss"][epoch] += loss
            train_accuracy(batch["label"], y_pred)

        # Validation and results update
        for batch in val_dataset:
            val_pred = val_step(batch)
            val_accuracy(batch["label"], val_pred)
        results["train_acc"].append(train_accuracy.result().numpy())
        results["val_acc"].append(val_accuracy.result().numpy())

        # Save the model every set epochs
        if (epoch + 1) % checkpoint_rate == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        if verbose:
            print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))
    if verbose:
        print()
        print("Time for full training is {} sec".format(time.time() - start_time))

    for key in results:
        with open(Path(out_dir) / (key + ".txt"), "w") as f:
            for result in results[key]:
                f.write(str(result))
                f.write("\n")

    return results
