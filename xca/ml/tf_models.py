from __future__ import absolute_import, division, print_function, unicode_literals
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout, Flatten, Input, LeakyReLU, Conv1D,
                                     AveragePooling1D, Average)
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import time

from .tf_data_proc import build_dataset
from pathlib import Path


def build_CNN_model(*,
                    data_shape,
                    filters,
                    kernel_sizes,
                    strides,
                    ReLU_alpha,
                    pool_sizes,
                    batchnorm,
                    n_classes,
                    dense_dims=(),
                    dense_dropout=0.,
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

    Returns
    -------
    model: Model

    """
    x_in = Input(shape=data_shape, name='X')
    x = tf.identity(x_in)
    # Downsampling
    for i in range(len(filters)):
        x = Conv1D(filters[i],
                   kernel_sizes[i],
                   strides=strides[i],
                   padding='valid',
                   kernel_initializer=RandomNormal(0, 0.02),
                   name="conv_{}".format(i))(x)
        x = LeakyReLU(alpha=ReLU_alpha)(x)
        x = AveragePooling1D(pool_size=pool_sizes[i],
                             strides=None,
                             padding='same')(x)
        if batchnorm:
            x = BatchNormalization(axis=-1, name="batchnorm_{}".format(i))(x)

    # Flatten and output
    x = Flatten()(x)
    x = Dropout(dense_dropout)(x)
    for i, dim in enumerate(dense_dims):
        x = Dense(dim, activation='relu', name='dense_{}'.format(i))(x)
        x = Dropout(dense_dropout)(x)
    x = Dense(n_classes, activation='softmax', name="Discriminator")(x)

    model = Model(x_in, x)
    return model


def build_fusion_ensemble_model(ensemble_size,
                                model_builder,
                                *,
                                data_shape,
                                **kwargs):
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
    x_in = Input(shape=data_shape, name='X')
    members = []
    for i in range(ensemble_size):
        m = model_builder(data_shape=data_shape,
                          **kwargs)
        members.append(m(x_in))
    outputs = Average()(members)
    model = Model(x_in, outputs)
    return model


def model_training(model,
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
                   seed=None):
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

    Returns
    -------

    """
    start_time = time.time()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    Path(out_dir).mkdir(exist_ok=True, parents=True)

    if verbose:
        model.summary()

    # Build dataset
    dataset, val_dataset = build_dataset(dataset_paths=dataset_paths,
                                         batch_size=batch_size,
                                         multiprocessing=multiprocessing,
                                         categorical=categorical,
                                         val_split=val_split,
                                         data_shape=data_shape,
                                         n_classes=n_classes
                                         )

    cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2)

    # Checkpoints
    checkpoint_dir = str(Path(out_dir) / 'training_checkpoints')
    checkpoint_prefix = str(Path(checkpoint_dir) / "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model)

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            y_pred = model({'X': batch['X']}, training=True)
            loss = cross_entropy(batch['label'], y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, y_pred

    @tf.function
    def val_step(batch):
        y_pred = model({'X': batch['X']}, training=False)
        return y_pred

    # Actual training
    results = {'loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(n_epochs):
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        val_accuracy = tf.keras.metrics.CategoricalAccuracy()
        results['loss'].append(0.)
        start = time.time()
        for batch in dataset:
            loss, y_pred = train_step(batch)
            results['loss'][epoch] += loss
            train_accuracy(batch['label'], y_pred)

        # Validation and results update
        for batch in val_dataset:
            val_pred = val_step(batch)
            val_accuracy(batch['label'], val_pred)
        results['train_acc'].append(train_accuracy.result().numpy())
        results['val_acc'].append(val_accuracy.result().numpy())

        # Save the model every set epochs
        if (epoch + 1) % checkpoint_rate == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        if verbose:
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    if verbose:
        print()
        print('Time for full training is {} sec'.format(time.time() - start_time))

    for key in results:
        with open(Path(out_dir) / (key + '.txt'), 'w') as f:
            for result in results[key]:
                f.write(str(result))
                f.write('\n')

    return results
