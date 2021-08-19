import time
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.initializers.initializers_v2 import RandomNormal
from tensorflow.python.keras.layers import (
    Conv1D,
    LeakyReLU,
    AveragePooling1D,
    BatchNormalization,
    Flatten,
    Dropout,
    Dense,
    Average,
)

from xca.ml.tf.data_proc import build_dataset
from xca.ml.tf.utils import setup_training, breakdown_training


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


def training(  # noqa: C901
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
    val_split=0.2,
    checkpoint_rate=1,
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
    multiprocessing: int
    categorical: bool
    data_shape: tuple
    n_epochs: int
    n_classes: int
    learning_rate: float
        Learning rate if no optimizer is given
    optimizer: Optimizer, None
        Default to Adam with learning rate
    val_split: float
    checkpoint_rate: int
    verbose: bool
    seed: int
    kwargs:
        Dummy dict for using convenience methods that pass larger **kwargs to both model and training

    Returns
    -------

    """
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
    dataset, val_dataset = build_dataset(
        dataset_paths=dataset_paths,
        batch_size=batch_size,
        multiprocessing=multiprocessing,
        categorical=categorical,
        val_split=val_split,
        data_shape=data_shape,
        n_classes=n_classes,
    )

    if categorical:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    else:
        loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            y_pred = model({"X": batch["X"]}, training=True)
            loss = loss_fn(batch["label"], y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, y_pred

    @tf.function
    def val_step(batch):
        y_pred = model({"X": batch["X"]}, training=False)
        loss = loss_fn(batch["label"], y_pred)
        return loss, y_pred

    # Actual training
    results = defaultdict(list)
    for epoch in range(n_epochs):
        if categorical:
            train_metrics = {"train_acc": tf.keras.metrics.CategoricalAccuracy()}
            val_metrics = {"val_acc": tf.keras.metrics.CategoricalAccuracy()}
        else:
            train_metrics = {"train_loss": tf.keras.metrics.MeanSquaredError()}
            val_metrics = {"val_loss": tf.keras.metrics.MeanSquaredError()}

        start = time.time()
        results["train_loss"].append(0.0)
        count = 0
        for batch in dataset:
            count += len(batch)
            loss, y_pred = train_step(batch)
            results["train_loss"][epoch] += loss.numpy()
            for _, metric in train_metrics.items():
                metric(batch["label"], y_pred)
        results["train_loss"][epoch] /= count

        # Validation and results update
        results["val_loss"].append(0.0)
        count = 0
        for batch in val_dataset:
            count += len(batch)
            loss, val_pred = val_step(batch)
            results["val_loss"][epoch] += loss.numpy()
            for _, metric in val_metrics.items():
                metric(batch["label"], val_pred)
        results["val_loss"][epoch] /= count

        for key, metric in train_metrics.items():
            results[key].append(metric.result().numpy())
        for key, metric in val_metrics.items():
            results[key].append(metric.result().numpy())

        # Save the model every set epochs
        if (epoch + 1) % checkpoint_rate == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        if verbose:
            print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

    # Closeout and save
    breakdown_training(
        verbose=verbose, start_time=start_time, results=results, out_dir=out_dir
    )

    return results
