"""
@author: maffettone
"""

import json

default_params = {
    'input_cif': None,
    'wavelength': [(1.54060, 0.5), (1.54439, 0.5)],  # List of tuples (lambda, fraction), or single value
    # Aditional Debye-Waller factors
    'debye_waller_factors': None,
    # SHELX extinction correction
    'extinction_correction_x': None,
    # Preferred orientaiton and march parameter
    'preferred': None,
    'march_parameter': 1,
    # LP factor
    'theta_m': 26.6,
    # Radius and Sample offset
    'instrument_radius': 174.8,
    'offset_height': 0.001,
    # Pattern linspace
    '2theta_min': 10.,
    '2theta_max': 110.,
    'n_datapoints': 5000,
    # Peak Profile
    'U': 0.1,
    'V': 0.1,
    'W': 0.1,
    'X': 0.1,
    'Y': 0.1,
    'Z': 0.0,
    # Background
    'bkg_6': 0.,
    'bkg_5': 0.,
    'bkg_4': 0.,
    'bkg_3': 0.,
    'bkg_2': 0.,
    'bkg_1': 0.,
    'bkg_0': 0.,
    'bkg_-1': 0.,
    'bkg_-2': 0.,
    # Noise
    'noise_std': 0,
    # Extras`
    'verbose': False
}


def load_params(input_params=None):
    """
    Loads complete set of hyperparameters for pattern synthesis.
    Updates default dictionary by loading a json file or using a dictionary.

    Parameters
    ----------
    input_params: basestring, Path, or dict
        Path to json file or dictionary of hyperparameters

    Returns
    -------
    parameters: dict
        Complete dictionary of hyperparameters
    """
    parameters = {}
    parameters.update(default_params)
    try:
        with open(input_params, 'r') as f:
            d = json.load(f)
            parameters.update(d)
    except TypeError:
        if input_params:
            parameters.update(input_params)
    return parameters


def cycle_params(n_profiles, output_path, input_params=None, shape_limit=0.,
                 march_range=(0., 1.), preferred_axes=None,
                 sample_height=None, noise_exp=None, n_jobs=1, **kwargs):
    """
    Generates n_profiles of profiles for a single cif.
    Outputs can be a directory containing many individual numpy files, a bulk numpy file, or a bulk csv.

    Uses default dictionary, and overriden by any changes in the input params.

    Any params which are to be cycled inside a range should be passed as kwargs,
    with key:tuple pairs of matching parameters for the uniform random ranges.

    Parameters
    ----------
    n_profiles: int
        number of profiles to generate for cif
    output_path: basestring, path
        path for output
        directory for collection of .npy files
        .npy for complete dataset as .npy
        .csv for complete dataset as .csv
    input_params: basestring, Path, or dict
        Path to json file or dictionary of hyperparameters
    shape_limit: float
        Limit for random variations to caglioti parameters. If 0, defaults are used from input dict
    march_range: tuple of floats
        range of march parameters to choose on uniform dist
    preferred_axes: list of tuples
        HKL for preferred axes to randomly select from
    sample_height: tuple
        range of sample heights to choose on a uniform dist
    noise_exp: tuple
        range of exponents noise to choose on uniform dist (log noise)
    n_jobs: int
        Number of jobs for multiproc
    kwargs: dict
        key:tuple pairs of matching parameters for the uniform random ranges

    Returns
    -------

    """

    from pathlib2 import Path
    from cctbx import create_complete_profile, sum_multi_wavelength_profiles, multi_phase_profile
    import numpy as np
    import random

    # Checks for output availability
    path = Path(output_path)
    if not (path.is_dir or path.suffix in ['.npy', '.csv']):
        raise TypeError("Output path type not implemented: {}".format(path))

    # Load default dictionary and linspace
    _default = load_params(input_params)
    _x = np.linspace(_default['2theta_min'], _default['2theta_max'], num=_default['n_datapoints'])
    data = np.zeros((_x.shape[0], n_profiles))

    # Assemble list of parameters
    params_list = []
    for _ in range(n_profiles):
        parameters = {}
        parameters.update(_default)
        if shape_limit:
            test_y = np.zeros_like(_x) - 1
            a = shape_limit
            while any(test_y < 0):
                parameters['U'] = np.random.uniform(-a, a)
                parameters['V'] = np.random.uniform(-a, a)
                parameters['W'] = np.random.uniform(-a, a)
                parameters['X'] = np.random.uniform(-a, a)
                parameters['Y'] = np.random.uniform(np.abs(parameters['X']), a)
                test_y = parameters['U'] * (np.tan(_x) ** 2) + parameters['V'] * np.tan(_x) + parameters['W']
        parameters['march_parameter'] = np.random.uniform(*march_range)
        if sample_height:
            parameters['offset_height'] = np.random.uniform(*sample_height)
        if preferred_axes:
            parameters['preferred'] = random.choice(preferred_axes)
        if noise_exp:
            parameters['noise_std'] = 10 ** np.random.uniform(*noise_exp)
        for i in range(-2, 7):
            parameters['bkg_{}'.format(i)] = np.random.uniform(0, _default['bkg_{}'.format(i)])
        for key in kwargs:
            parameters[key] = np.random.uniform(*kwargs[key])
        params_list.append(parameters)

    if n_jobs > 1:
        from multiprocessing import Pool
        pool = Pool(n_jobs)
        if type(parameters['input_cif']) == type([]):
            results = list(pool.imap_unordered(multi_phase_profile, params_list))
        elif type(parameters['wavelength']) == type([]):
            results = list(pool.imap_unordered(sum_multi_wavelength_profiles, params_list))
        else:
            results = list(pool.imap_unordered(create_complete_profile, params_list))

        for idx in range(n_profiles):
            data[:, idx] = results[idx][1]
        pool.close()
        pool.join()

    else:
        for idx, parameters in enumerate(params_list):
            if type(parameters['input_cif']) == type([]):
                _, data[:, idx] = multi_phase_profile(parameters)
            elif type(parameters['wavelength']) == type([]):
                _, data[:, idx] = sum_multi_wavelength_profiles(parameters)
            else:
                _, data[:, idx] = create_complete_profile(parameters)

    if path.is_dir():
        for idx in range(n_profiles):
            np.save(str(path / "{}.npy".format(idx)), data[:, idx])
    elif path.suffix == '.npy':
        np.save(str(output_path), data)
    elif path.suffix == '.csv':
        cols = ["Intensity {}".format(idx) for idx in range(n_profiles)]
        np.savetxt(str(output_path), data, delimiter=',', header=",".join(cols), comments='')
    else:
        raise ValueError("Path {} is invalid (doesn't exist or improper extension)".format(path))
    return


def single_pattern(input_params, shape_limit=0., **kwargs):
    """
    Tool for generating a single pattern for exploration.
    The input_params should be fully specified, with optional randomization available in kwargs and shape limit.
    Any params which are to be cycled inside a range should be passed as kwargs,
    with key:tuple pairs of matching parameters for the uniform random ranges.

    Parameters
    ----------
    input_params: basestring, Path, or dict
        Path to json file or dictionary of hyperparameters
    shape_limit: float
        Limit for random variations to caglioti parameters. If 0, defaults are used from input dict
    kwargs: dict
        key:tuple pairs of matching parameters for the uniform random ranges

    Returns
    -------
    x: two theta values
    y: intensity values
    """
    import numpy as np
    from cctbx import create_complete_profile, sum_multi_wavelength_profiles, multi_phase_profile
    parameters = load_params(input_params)
    _x = np.linspace(parameters['2theta_min'], parameters['2theta_max'], num=parameters['n_datapoints'])
    test_y = np.zeros_like(_x) - 1

    if shape_limit:
        while any(test_y < 0):
            parameters['U'] = np.random.uniform(-shape_limit, shape_limit)
            parameters['V'] = np.random.uniform(-shape_limit, shape_limit)
            parameters['W'] = np.random.uniform(-shape_limit, shape_limit)
            parameters['X'] = np.random.uniform(-shape_limit, shape_limit)
            parameters['Y'] = np.random.uniform(np.abs(parameters['X']), shape_limit)
            test_y = parameters['U'] * (np.tan(_x) ** 2) + parameters['V'] * np.tan(_x) + parameters['W']

    for key in kwargs:
        parameters[key] = np.random.uniform(*kwargs[key])
    if type(parameters['input_cif']) == type([]):
        x, y = multi_phase_profile(parameters)
    elif type(parameters['wavelength']) == type([]):
        x, y = sum_multi_wavelength_profiles(parameters)
    else:
        x, y = create_complete_profile(parameters)

    return x, y
