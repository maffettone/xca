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


def build_CNN_model(params):
    """
    Builds single feed forward convolutional neural network for 1-d xrd data
    Parameters
    ----------
    params : dict
        complete hyper parameter dictionary for model generation and training

    Required Parameters
    ----------
    n_classes : integer
        number of classes or phases
    data_shape : tuple
        shape of input data XRD pattern (should be 2-d)
    filters : list of integers
        Consistent length with other items, number of filters for each Conv1D layer
    kernel_size : list of integers
        Consistent length with other items, size of kernel for each Conv1D layer
    strides : list of integers
        Consistent length with other items, size of stride for each Conv1D layer
    pool_sizes : list of integers
        Consistent length with other items, size of pooling for each Conv1D layer
    batchnorm : bool
        Whether or not to include batchnorm after each convolutional downsample
    dense_dropout : float
        Dropout rate for final layer and all dense layers
    dense_dims : list of integers
        Dimensions of dense layers to follow convolutional model
    Refer to Keras documentation for below.
    ReLU_alpha: float [0, 1.0)
    lr : float [0, 1.0)
    beta_1: float [0, 1.0)
    beta_2: float [0, 1.0)
    Returns
    -------
    model
    """

    x_in = Input(shape=params['data_shape'], name='X')
    x = tf.identity(x_in)
    # Downsampling
    for i in range(len(params['filters'])):
        x = Conv1D(params['filters'][i],
                   params['kernel_size'][i],
                   strides=params['strides'][i],
                   padding='valid',
                   kernel_initializer=RandomNormal(0, 0.02),
                   name="conv_{}".format(i))(x)
        x = LeakyReLU(alpha=params['ReLU_alpha'])(x)
        x = AveragePooling1D(pool_size=params['pool_sizes'][i],
                             strides=None,
                             padding='same')(x)
        if params['batchnorm']:
            x = BatchNormalization(axis=-1, name="batchnorm_{}".format(i))(x)

    # Flatten and output
    x = Flatten()(x)
    x = Dropout(params['dense_dropout'])(x)
    for i, dim in enumerate(params['dense_dims']):
        x = Dense(dim, activation='relu', name='dense_{}'.format(i))(x)
        x = Dropout(params['dense_dropout'])(x)
    x = Dense(params['n_classes'], activation='softmax', name="Discriminator")(x)

    model = Model(x_in, x)
    return model


def build_CNN_ensemble_model(params):
    """
    Builds ensemble of FFCNN's
    Parameters
    ----------
    params : dict
        complete hyper parameter dictionary for model generation and training

    Required Parameters
    ----------
    ensemble_size : integer
        Number of feed forward CNNs based on params to include in ensemble

    Returns
    -------
    model

    """
    x_in = Input(shape=params['data_shape'], name='X')
    members = []
    for i in range(params['ensemble_size']):
        m = build_CNN_model(params)
        members.append(m(x_in))
    outputs = Average()(members)
    model = Model(x_in, outputs)
    return model


def CNN_training(params):
    start_time = time.time()
    if 'seed' in params and params['seed']:
        np.random.seed(params['seed'])
        tf.random.set_seed(params['seed'])

    if not os.path.isdir(params['out_dir']):
        os.mkdir(params['out_dir'])

    if params['ensemble_size'] > 1:
        CNN = build_CNN_ensemble_model(params)
    else:
        CNN = build_CNN_model(params)
    if params['verbose']:
        CNN.summary()

    # Build dataset
    dataset, val_dataset = build_dataset(params)

    cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    optimizer = Adam(lr=params['lr'], beta_1=params['beta_1'], beta_2=params['beta_2'])

    # Checkpoints
    checkpoint_dir = os.path.join(params['out_dir'], 'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     CNN=CNN)

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            y_pred = CNN({'X': batch['X']}, training=True)
            loss = cross_entropy(batch['label'], y_pred)
        gradients = tape.gradient(loss, CNN.trainable_variables)
        optimizer.apply_gradients(zip(gradients, CNN.trainable_variables))
        return loss, y_pred

    @tf.function
    def val_step(batch):
        y_pred = CNN({'X': batch['X']}, training=False)
        return y_pred

    # Actual training
    results = {'loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(params['epochs']):
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
        results['train_acc'].append(train_accuracy.result())
        results['val_acc'].append(val_accuracy.result())

        # Save the model every set epochs
        if (epoch + 1) % params['checkpoint_rate'] == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        if params['verbose']:
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    if params['verbose']:
        print()
        print('Time for full training is {} sec'.format(time.time() - start_time))

    for key in results:
        with open(os.path.join(params['out_dir'], key + '.txt'), 'w') as f:
            for tensor in results[key]:
                f.write(str(tensor.numpy()))
                f.write('\n')

    return results, CNN
