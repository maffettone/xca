from __future__ import absolute_import, division, print_function, unicode_literals
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import xarray as xr


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def np_dir_to_record(npy_path, tfr_path, shuffle=True):
    """Depreceated categorial dir to tfrecord for numpy data"""
    import random
    fnames = list(Path(npy_path).glob("*/*.npy"))
    if not fnames:
        raise RuntimeError(f"Path is empty: {npy_path.absolute()}")
    if shuffle:
        random.shuffle(fnames)

    writer = tf.io.TFRecordWriter(str(tfr_path))

    for fname in fnames:
        label = fname.parent.name
        X = np.load(fname)
        dim = X.shape[0]
        X_raw = X.astype(np.float32).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'X_raw': _bytes_feature(X_raw),
            'dim': _int64_feature(dim),
            'label': _int64_feature(int(label))}))
        writer.write(example.SerializeToString())
    writer.close()


def xr_dir_to_record(xr_path, tfr_path, attrs_key, transform=_float_feature, shuffle=True):
    """
    Consumes a directory of xarrays to produce a tfrecord.
    The 'label' of the dataset will correspond to the attribute.
    An optional callable transform can be used to featurize the label. The default will use
    a float feature. This callable can be used to cast categorical strings as integer labels.

    Parameters
    ----------
    xr_path : Path, str
        Directory of xarray files
    tfr_path : Path, str
        Path to output tfrecord
    attrs_key : str
        dataarray.attrs key to use for label
    transform : callable
        transform of attribute to tensorflow feature
    shuffle : bool

    Returns
    -------

    """
    import random
    fnames = list(Path(xr_path).glob("*.nc"))
    if not fnames:
        raise RuntimeError(f"Path is empty: {xr_path.absolute()}")
    if shuffle:
        random.shuffle(fnames)

    writer = tf.io.TFRecordWriter(str(tfr_path))

    for fname in fnames:
        X = xr.open_dataarray(fname)
        dim = X.shape[0]
        X_raw = X.data.astype(np.float32).tobytes()
        label = X.attrs[attrs_key]
        example = tf.train.Example(features=tf.train.Features(feature={
            'X_raw': _bytes_feature(X_raw),
            'dim': _int64_feature(dim),
            'label': transform(label)}))
        writer.write(example.SerializeToString())
    writer.close()


def exp2TFR(fnames, tfr_path):
    writer = tf.io.TFRecordWriter(tfr_path)

    for fname in fnames:
        fname = Path(fname)
        label = fname.parent.name
        source_fname = fname.name
        X = np.load(fname)
        dim = X.shape[0]
        X_raw = X.astype(np.float32).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'X_raw': _bytes_feature(X_raw),
            'dim': _int64_feature(dim),
            'str_label': _bytes_feature(str.encode(label)),
            'source_fname': _bytes_feature(str.encode(source_fname))}))
        writer.write(example.SerializeToString())
    writer.close()


def parse_categorical_TFR(example_proto, params):
    '''
    Reads in Tensorflow Record object
    Parameters
    ----------
    example_proto
    params

    Returns
    -------

    '''
    feature_description = {
        'X_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'dim': tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    data = tf.reshape(tf.io.decode_raw(features['X_raw'], tf.float32),
                      params['data_shape'])
    label = tf.cast(features['label'], tf.int32)
    return data, label


def parse_continuous_TFR(example_proto, params):
    '''
    Reads in Tensorflow Record object
    Parameters
    ----------
    example_proto
    params

    Returns
    -------

    '''
    feature_description = {
        'X_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32),
        'dim': tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    data = tf.reshape(tf.io.decode_raw(features['X_raw'], tf.float32),
                      params['data_shape'])
    label = tf.cast(features['label'], tf.float32)
    return data, label


def categorical_preprocess(data, label, params):
    """
    Preprocesses data and label according to parameters for categorical data
    Parameters
    ----------
    data
    label
    params

    Returns
    -------
    Dataset dictionary to be passed to training
    """
    # Preprocess XRD - input should be on 0-1
    num_classes = params['n_classes']
    # Moving from -1. to 1.
    data = tf.cast(data, tf.float32) * 2 - 1
    label = tf.one_hot(label, num_classes)

    return {'X': data, 'label': label}

def continuous_preprocess(data, label, params):
    """
    Preprocesses data and label according to parameters for continuous regression tasks
    Parameters
    ----------
    data
    label
    params

    Returns
    -------
    Dataset dictionary to be passed to training
    """
    # Preprocess XRD - input should be on 0-1
    # Moving from -1. to 1.
    data = tf.cast(data, tf.float32) * 2 - 1
    return {'X': data, 'label': label}


def build_dataset(params):
    """

    Parameters
    ----------
    params : dict
    categorical : bool

    Required Parameters in params
    ----------
    num_classes : integer
    dataset_path : string or list of strings
    batch_size : integer
    latent_dim : integer
        input noise dimension
    multiprocessing : integer


    Returns
    -------

    """
    check_paths = all([os.path.isfile(p) for p in params['dataset_path']])
    assert check_paths, "Dataset path is invalid: {}".format(params['dataset_path'])
    dataset = tf.data.TFRecordDataset(filenames=params['dataset_path'], num_parallel_reads=len(params['dataset_path']))
    dataset_size = sum(1 for _ in dataset)
    train_size = int(dataset_size * (1-params['val_split']))
    dataset = dataset.shuffle(100000, reshuffle_each_iteration=False)
    if params['categorical']:
        dataset = dataset.map(lambda e: parse_categorical_TFR(e, params),
                              num_parallel_calls=params['multiprocessing'])
        dataset = dataset.map(lambda d, l: categorical_preprocess(d, l, params),
                              num_parallel_calls=params['multiprocessing'])
    else:
        dataset = dataset.map(lambda e: parse_continuous_TFR(e, params),
                              num_parallel_calls=params['multiprocessing'])
        dataset = dataset.map(lambda d, l: continuous_preprocess(d, l, params),
                              num_parallel_calls=params['multiprocessing'])
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    train_dataset = train_dataset.batch(params['batch_size'])
    val_dataset = val_dataset.batch(params['batch_size'])
    train_dataset = train_dataset.prefetch(1)
    val_dataset = val_dataset.prefetch(1)

    return train_dataset, val_dataset


def parse_exp_TFR(example_proto, params):
    """
    Reads in Tensorflow Record object
    Parameters
    ----------
    example_proto
    params

    Returns
    -------

    """
    feature_description = {
        'X_raw': tf.io.FixedLenFeature([], tf.string),
        'str_label': tf.io.FixedLenFeature([], tf.string),
        'dim': tf.io.FixedLenFeature([], tf.int64),
        'source_fname': tf.io.FixedLenFeature([], tf.string)
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    data = tf.reshape(tf.io.decode_raw(features['X_raw'], tf.float32),
                      params['data_shape'])
    label = tf.cast(features['str_label'], tf.string)
    fname = tf.cast(features['source_fname'], tf.string)
    return data, label, fname


def test_preprocess(data, label, fname, params):
    """
    Preprocesses data and label according to parameters
    Parameters
    ----------
    data
    label
    params

    Returns
    -------
    Dataset dictionary to be passed to training
    """
    # Preprocess XRD - input should be on 0-1
    # Moving to -1. to 1.
    data = tf.cast(data, tf.float32) / tf.math.reduce_max(data) * 2 - 1
    return {'X': data, 'label': label, 'fname': fname}


def build_test_dataset(params):
    """

    Parameters
    ----------
    params

    Required Parameters in params
    ----------
    num_classes : integer
    dataset_path : string or list of strings
    batch_size : integer
    latent_dim : integer
        input noise dimension
    multiprocessing : integer


    Returns
    -------

    """
    assert os.path.isfile(params['dataset_path']), "Dataset path is invalid: {}".format(params['dataset_path'])
    dataset = tf.data.TFRecordDataset(filenames=params['dataset_path'])
    dataset = dataset.map(lambda e: parse_exp_TFR(e, params),
                          num_parallel_calls=params['multiprocessing'])
    dataset = dataset.map(lambda d, l, f: test_preprocess(d, l, f, params),
                          num_parallel_calls=params['multiprocessing'])
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(1)

    return dataset
