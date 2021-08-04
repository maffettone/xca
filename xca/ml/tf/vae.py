import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import (
    Dense,
    Reshape,
    Conv1DTranspose,
    UpSampling1D,
    Conv1D,
    AveragePooling1D,
    Flatten,
    Dropout,
    Lambda,
)

from xca.ml.tf.data_proc import build_dataset
from xca.ml.tf.utils import setup_training, breakdown_training


def calculate_transpose_output_size(input_size, kernel_size, stride):
    return (input_size - 1) * stride + kernel_size


def build_consistent_vae(
    *,
    data_shape,
    latent_dim,
    encoder_dense_dims,
    encoder_filters,
    encoder_kernel_sizes,
    encoder_pool_sizes,
    encoder_strides,
    kl_loss_weight,
    verbose=False,
    **kwargs
):

    """
    Given a set of encoder hyperparameters, this builds a decoder which produces output of the correct shape and returns
    a consistent vae model.

    ** Note: Padding is assumed to be 'valid' for all convolutional layers **

    Parameters
    ----------
    data_shape: tuple of int
        shape of input data
    latent_dim: int
        number of variables defining the latent space
    encoder_dense_dims: list of int
        numbers of neurons in each of the encoder's dense, fully connected layers
    encoder_filters: list of int
        number of filters in each convolutional layer of the encoder model. consistent with length of
        encoder_kernel_sizes, encoder_pool_sizes, encoder_strides, and encoder_paddings
    encoder_kernel_sizes: list of int
        kernel sizes for each convolutional layer of the encoder model. consistent with length of other input lists
    encoder_strides: list of int
        number of strides in each convolutional layer of the encoder model. consistent with length of other input lists
    encoder_pool_sizes: list of int
        pool sizes for each convolutional layer of the encoder model. consistent with length of other input lists
    kl_loss_weight: float
        scaler for kl-divergence loss
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
        paddings=["valid"] * len(encoder_filters),
        verbose=verbose,
    )

    decoder_filters = encoder_filters[::-1]
    decoder_kernel_sizes = encoder_kernel_sizes[::-1]
    decoder_strides = encoder_strides[::-1]
    upsampling_sizes = encoder_pool_sizes[::-1]

    # With stride size configured, build the decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(last_conv_layer_shape[1] * last_conv_layer_shape[2], activation="relu")(
        latent_inputs
    )
    x = Reshape((last_conv_layer_shape[1], last_conv_layer_shape[2]))(x)

    x_shape = x.shape[1]
    test_shape = x_shape
    last_valid_index = 0

    # Calculate the final output size with current parameters
    for i in range(len(decoder_filters)):
        test_shape = calculate_transpose_output_size(
            test_shape, decoder_kernel_sizes[i], decoder_strides[i]
        )
        # Make sure that we don't overshoot the original input size
        if test_shape > data_shape[0]:
            last_valid_index = i
            break
        test_shape *= upsampling_sizes[i]
        # Do it again after factoring in upsampling
        if test_shape > data_shape[0]:
            last_valid_index = i
            break

    # Upsampling
    for i in range(last_valid_index):
        x = Conv1DTranspose(
            decoder_filters[i],
            decoder_kernel_sizes[i],
            strides=decoder_strides[i],
            padding="valid",
            name="conv_transpose{}".format(i),
        )(x)
        x_shape = calculate_transpose_output_size(
            x_shape, decoder_kernel_sizes[i], decoder_strides[i]
        )
        x = UpSampling1D(size=upsampling_sizes[i])(x)
        x_shape *= upsampling_sizes[i]

    # produce the final output layer such that the output size equals the encoder input size
    last_kernel_size = data_shape[0] - x_shape + 1
    x = Conv1DTranspose(1, last_kernel_size, strides=1, padding="valid")(x)

    # Decoder output
    decoder = Model(latent_inputs, x, name="CNN_decoder")

    if verbose:
        decoder.summary()

    vae = VAE(encoder, decoder, kl_loss_weight)
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
    data_shape,
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
        shape of the output from the last convolutional layer in the encoder;
        used to calculate dimensionality of the decoder's initial dense layer
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

    x = Flatten()(x)
    x = Dense(data_shape[0], activation="relu", name="dense_out")(x)
    x = Reshape(data_shape)(x)
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
                tf.reshape(data, (tf.shape(data)[0], tf.shape(data)[1])),
                tf.reshape(
                    reconstruction,
                    (tf.shape(reconstruction)[0], tf.shape(reconstruction)[1]),
                ),
            ),
            axis=0,
        )
        return reconstruction_loss

    def loss(self, x, z_mean, z_log_sigma, reconstruction, *args):
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

    def call(self, x, *args, **kwargs):
        z_mean, z_log_sigma = self.encode(x, *args, **kwargs)
        z = Lambda(self.sample)([z_mean, z_log_sigma], *args, **kwargs)
        reconstruction = self.decode(z, *args, **kwargs)
        return {
            "z_mean": z_mean,
            "z_log_sigma": z_log_sigma,
            "reconstruction": reconstruction,
        }

    def __call__(self, x, *args, **kwargs):
        return self.call(x, *args, **kwargs)


class PredictiveVAE(VAE):
    def __init__(
        self,
        *,
        encoder,
        decoder,
        predictor,
        mode,
        kl_loss_weight=1.0,
        predictive_loss_weight=1.0,
        **kwargs
    ):
        """
        VAE with predictive module for classification or regression

        Parameters
        ----------
        encoder: Model
        decoder: Model
        predictor: Model
        mode: str
            Operational mode, one of {"classification", "regression"}
        kl_loss_weight: float
        predictive_loss_weight: float
        kwargs: dict
        """
        super(PredictiveVAE, self).__init__(
            encoder, decoder, kl_loss_weight=kl_loss_weight, **kwargs
        )
        self.predictor = predictor
        self.predictive_loss_weight = predictive_loss_weight
        self.predictive_loss = {
            "classification": tf.keras.losses.CategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM
            ),
            "regression": tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM
            ),
        }[mode.lower()]

    def loss(self, *x, z_mean, z_log_sigma, reconstruction, y_true, y_pred):
        return (
            self.kl_loss_weight * self.kl_loss(z_mean, z_log_sigma)
            + self.predictive_loss_weight * self.predictive_loss(y_true, y_pred)
            + self.reconstruction_loss(x, reconstruction)
        )

    def call(self, x, *args, **kwargs):
        z_mean, z_log_sigma = self.encode(x, *args, **kwargs)
        z = Lambda(self.sample)([z_mean, z_log_sigma], *args, **kwargs)
        y_pred = self.predictor(z)
        reconstruction = self.decode(z, *args, **kwargs)
        return {
            "z_mean": z_mean,
            "z_log_sigma": z_log_sigma,
            "reconstruction": reconstruction,
            "y_pred": y_pred,
        }


def _run_epoch(
    *,
    epoch,
    train_step,
    dataset,
    results_dict,
    checkpoint,
    checkpoint_rate,
    checkpoint_prefix,
    verbose
):
    """
    Convenience function for refactorization of running internal epoch for VAEs and like methods
    Parameters
    ----------
    epoch
    train_step
    dataset
    results_dict
    checkpoint
    checkpoint_rate
    checkpoint_prefix
    verbose

    Returns
    -------

    """
    start = time.time()
    for batch in dataset:
        loss, output = train_step(batch)
        for key in loss:
            results_dict[key][epoch] += loss[key]

    # Save the model every set epochs
    if (epoch + 1) % checkpoint_rate == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    if verbose:
        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))


def training(
    model,
    *,
    dataset_paths,
    out_dir,
    batch_size,
    multiprocessing,
    categorical,
    data_shape,
    n_epochs,
    optimizer=None,
    learning_rate=0.001,
    checkpoint_rate=1,
    verbose=False,
    seed=None
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
    multiprocessing
    categorical
    data_shape
    n_epochs
    checkpoint_rate
    learning_rate: float
        Learning rate if no optimizer is given
    optimizer: Optimizer, None
        Default to Adam with learning rate
    verbose
    seed

    Returns
    -------

    """
    # Setup
    start_time, checkpoint_prefix, checkpoint, optimizer = setup_training(
        model=model,
        seed=seed,
        out_dir=out_dir,
        optimizer=optimizer,
        verbose=verbose,
        data_shape=data_shape,
        learning_rate=learning_rate,
    )

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
        _run_epoch(
            epoch=epoch,
            train_step=train_step,
            dataset=dataset,
            results_dict=results,
            checkpoint=checkpoint,
            checkpoint_rate=checkpoint_rate,
            checkpoint_prefix=checkpoint_prefix,
            verbose=verbose,
        )

    # Closeout and save
    breakdown_training(
        verbose=verbose, start_time=start_time, results=results, out_dir=out_dir
    )

    return results


def denoiser_training(
    model,
    *,
    dataset_paths,
    log_noise_min,
    log_noise_max,
    out_dir,
    batch_size,
    multiprocessing,
    categorical,
    data_shape,
    n_epochs,
    optimizer=None,
    learning_rate=0.001,
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
    log_noise_min: float
        Minimum log of noise to sample. Noise added is normally distributed with a standard deviation of 10**noise_log
    log_noise_max: float
        Minimum log of noise to sample. Noise added is normally distributed with a standard deviation of 10**noise_log
    out_dir
    batch_size
    learning_rate: float
        Learning rate if no optimizer is given
    optimizer: Optimizer, None
        Default to Adam with learning rate
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
    start_time, checkpoint_prefix, checkpoint, optimizer = setup_training(
        model=model,
        seed=seed,
        out_dir=out_dir,
        optimizer=optimizer,
        verbose=verbose,
        data_shape=data_shape,
        learning_rate=learning_rate,
    )

    # Build dataset
    def preprocess(data, label):
        X = tf.cast(data, tf.float32)
        X = (X - tf.math.reduce_min(X, axis=0, keepdims=True)) / (
            tf.math.reduce_max(X, axis=0, keepdims=True)
            - tf.math.reduce_min(X, axis=0, keepdims=True)
        )
        noisy = tf.cast(data, tf.float32) + tf.random.normal(
            data.shape,
            stddev=10 ** np.random.uniform(log_noise_min, log_noise_max),
            dtype=tf.float32,
        )
        noisy = (noisy - tf.math.reduce_min(noisy, axis=0, keepdims=True)) / (
            tf.math.reduce_max(noisy, axis=0, keepdims=True)
            - tf.math.reduce_min(noisy, axis=0, keepdims=True)
        )
        return {"X": X, "X_noisy": noisy, "label": label}

    dataset, _ = build_dataset(
        dataset_paths=dataset_paths,
        batch_size=batch_size,
        multiprocessing=multiprocessing,
        categorical=categorical,
        val_split=0.0,
        data_shape=data_shape,
        # Preprocessing step adding noise and assuming probabilities needed on [0,1] and not on [-1,1]
        preprocess=preprocess,
    )

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            output = model(batch["X_noisy"], training=True)
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
        _run_epoch(
            epoch=epoch,
            train_step=train_step,
            dataset=dataset,
            results_dict=results,
            checkpoint=checkpoint,
            checkpoint_rate=checkpoint_rate,
            checkpoint_prefix=checkpoint_prefix,
            verbose=verbose,
        )

    # Closeout and save
    breakdown_training(
        verbose=verbose, start_time=start_time, results=results, out_dir=out_dir
    )

    return results


def predictive_training(
    model,
    *,
    dataset_paths,
    out_dir,
    batch_size,
    multiprocessing,
    categorical,
    data_shape,
    n_epochs,
    optimizer=None,
    learning_rate=0.001,
    n_classes=0,
    checkpoint_rate=1,
    verbose=False,
    seed=None
):
    # Setup
    start_time, checkpoint_prefix, checkpoint, optimizer = setup_training(
        model=model,
        seed=seed,
        out_dir=out_dir,
        optimizer=optimizer,
        verbose=verbose,
        data_shape=data_shape,
        learning_rate=learning_rate,
    )

    # Build dataset
    dataset, _ = build_dataset(
        dataset_paths=dataset_paths,
        batch_size=batch_size,
        multiprocessing=multiprocessing,
        categorical=categorical,
        val_split=0.0,
        data_shape=data_shape,
        n_classes=n_classes,
    )

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            output = model(batch["X"], training=True)
            reconstruction_loss = model.reconstruction_loss(
                batch["X"], output["reconstruction"]
            )
            kl_loss = model.kl_loss(output["z_mean"], output["z_log_sigma"])
            predictive_loss = model.predictive_loss(batch["label"], output["y_pred"])
            loss = (
                reconstruction_loss
                + model.kl_loss_weight * kl_loss
                + model.predictive_loss_weight * predictive_loss
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return (
            dict(
                reconstruction_loss=reconstruction_loss,
                kl_loss=kl_loss,
                predictive_loss=predictive_loss,
                loss=loss,
            ),
            output,
        )

    # Actual training
    results = defaultdict(list)
    for epoch in range(n_epochs):
        results["loss"].append(0.0)
        results["kl_loss"].append(0.0)
        results["reconstruction_loss"].append(0.0)
        results["predictive_loss"].append(0.0)
        _run_epoch(
            epoch=epoch,
            train_step=train_step,
            dataset=dataset,
            results_dict=results,
            checkpoint=checkpoint,
            checkpoint_rate=checkpoint_rate,
            checkpoint_prefix=checkpoint_prefix,
            verbose=verbose,
        )

    # Closeout and save
    breakdown_training(
        verbose=verbose, start_time=start_time, results=results, out_dir=out_dir
    )

    return results
