"""
A set of convenience functions to aid in generating XRD patterns from dictionaries of parameters using the cctbx.
@author: maffettone
"""
import os
from pathlib import Path
import json
import uuid
import numpy as np
import xarray as xa
from xca.data_synthesis.cctbx import (
    create_complete_profile,
    sum_multi_wavelength_profiles,
    multi_phase_profile,
)

default_params = {
    "input_cif": None,
    "wavelength": [
        (1.54060, 0.5),
        (1.54439, 0.5),
    ],  # List of tuples (lambda, fraction), or single value
    # Aditional Debye-Waller factors
    "debye_waller_factors": (),
    # SHELX extinction correction
    "extinction_correction_x": 0,
    # Preferred orientaiton and march parameter
    "preferred": [],
    "march_parameter": 1,
    # LP factor
    "theta_m": 26.6,
    # Radius and Sample offset
    "instrument_radius": 174.8,
    "offset_height": 0.001,
    # Pattern linspace
    "tth_min": 10.0,
    "tth_max": 110.0,
    "n_datapoints": 5000,
    # Peak Profile
    "U": 0.1,
    "V": 0.1,
    "W": 0.1,
    "X": 0.1,
    "Y": 0.1,
    "Z": 0.0,
    # Background
    "bkg_6": 0.0,
    "bkg_5": 0.0,
    "bkg_4": 0.0,
    "bkg_3": 0.0,
    "bkg_2": 0.0,
    "bkg_1": 0.0,
    "bkg_0": 0.0,
    "bkg_n1": 0.0,
    "bkg_n2": 0.0,
    "bkg_ea": 0.0,
    "bkg_eb": 0.0,
    # Noise
    "noise_std": 0,
    # Extras`
    "verbose": False,
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
        with open(input_params, "r") as f:
            d = json.load(f)
            parameters.update(d)
    except TypeError:
        if input_params:
            parameters.update(input_params)
    return parameters


def metadata_adjustments(da):
    """Helper to change metadata of known conflict to appropriate type"""
    da.attrs["input_cif"] = Path(da.attrs["input_cif"]).stem
    da.attrs["is_chiral"] = int(da.attrs["is_chiral"])
    da.attrs["is_centric"] = int(da.attrs["is_centric"])
    if "verbose" in da.attrs:
        del da.attrs["verbose"]
    if isinstance(da.attrs["wavelength"], list):
        da.attrs["wavelength"], da.attrs["wavelength_weight"] = zip(
            *da.attrs["wavelength"]
        )
    # Clean up any others to fix type for netcdf
    for key, value in da.attrs.items():
        if not isinstance(value, (str, np.ndarray, np.number, list, tuple)):
            da.attrs[key] = str(value)
    return


def concat_and_clean(da_list):
    """Concatenate and clean the attributes of a list of datarrays for output"""
    da = xa.concat(da_list, dim="idx", combine_attrs="drop_conflicts")
    metadata_adjustments(da)
    return da


def complete_profile_wrapper(kwargs):
    return create_complete_profile(**kwargs)


def multi_wavelength_wrapper(kwargs):
    wavelengths = kwargs.pop("wavelength")
    return sum_multi_wavelength_profiles(wavelengths, **kwargs)


def multi_phase_wrapper(kwargs):
    input_cifs = kwargs.pop("input_cif")
    return multi_phase_profile(input_cifs, **kwargs)


def complete_profile_wrapper_io(path, kwargs):
    da = create_complete_profile(**kwargs)
    metadata_adjustments(da)
    da.to_netcdf(path / f"{uuid.uuid4().hex}.nc")


def multi_wavelength_wrapper_io(path, kwargs):
    wavelengths = kwargs.pop("wavelength")
    da = sum_multi_wavelength_profiles(wavelengths, **kwargs)
    metadata_adjustments(da)
    da.to_netcdf(path / f"{uuid.uuid4().hex}.nc")


def multi_phase_wrapper_io(path, kwargs):
    input_cifs = kwargs.pop("input_cif")
    da = multi_phase_profile(input_cifs, **kwargs)
    metadata_adjustments(da)
    da.to_netcdf(path / f"{uuid.uuid4().hex}.nc")


def cycle_params(
    n_profiles,
    output_path,
    input_params=None,
    shape_limit=0.0,
    march_range=(0.0, 1.0),
    preferred_axes=None,
    noise_exp=None,
    n_jobs=1,
    **kwargs,
):
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
        directory for collection of .pkl files of complete datarrays
        All other options produce lossy outputs where metadata is lost partially or completely
        .npy for complete dataset as .npy
        .csv for complete dataset as .csv
        .nc for complete dataset as netcdf datarray
    input_params: basestring, Path, or dict
        Path to json file or dictionary of hyperparameters
    shape_limit: float
        Limit for random variations to caglioti parameters. If 0, defaults are used from input dict
    march_range: tuple of floats
        range of march parameters to choose on uniform dist
    preferred_axes: list of tuples
        HKL for preferred axes to randomly select from
    noise_exp: tuple
        range of exponents noise to choose on uniform dist (log noise)
    n_jobs: int
        Number of jobs for multiproc
    start_idx : int
        Optional value to start the indexing of outputs
    kwargs: dict
        key:tuple pairs of matching parameters for the uniform random ranges

    Returns
    -------

    """

    from pathlib import Path
    import random
    from multiprocessing import Pool

    # Checks for output availability
    path = Path(output_path)
    if not (path.is_dir or path.suffix in [".npy", ".csv", ".nc"]):
        raise TypeError("Output path type not implemented: {}".format(path))

    # Load default dictionary and linspace
    _default = load_params(input_params)
    _x = np.linspace(
        _default["tth_min"], _default["tth_max"], num=_default["n_datapoints"]
    )

    # Assemble list of parameters
    params_list = []
    for _ in range(n_profiles):
        parameters = {}
        parameters.update(_default)
        if shape_limit:
            test_y = np.zeros_like(_x) - 1
            a = shape_limit
            while any(test_y < 0):
                parameters["U"] = np.random.uniform(-a, a)
                parameters["V"] = np.random.uniform(-a, a)
                parameters["W"] = np.random.uniform(-a, a)
                parameters["X"] = np.random.uniform(-a, a)
                parameters["Y"] = np.random.uniform(np.abs(parameters["X"]), a)
                test_y = (
                    parameters["U"] * (np.tan(_x) ** 2)
                    + parameters["V"] * np.tan(_x)
                    + parameters["W"]
                )
        parameters["march_parameter"] = np.random.uniform(*march_range)
        if preferred_axes:
            parameters["preferred"] = random.choice(preferred_axes)
        if noise_exp:
            parameters["noise_std"] = 10 ** np.random.uniform(*noise_exp)
        for i in range(7):
            parameters["bkg_{}".format(i)] = np.random.uniform(
                0, _default["bkg_{}".format(i)]
            )
        for i in range(-2, 0):
            parameters["bkg_n{}".format(abs(i))] = np.random.uniform(
                0, _default["bkg_n{}".format(abs(i))]
            )
        for key in kwargs:
            parameters[key] = np.random.uniform(*kwargs[key])
        params_list.append(parameters)

    if n_jobs <= 0:
        n_jobs = os.cpu_count()
    pool = Pool(n_jobs)

    if path.is_dir():
        """Fill directory with UIDs"""
        if isinstance(_default["input_cif"], list):
            pool.starmap(
                multi_phase_wrapper_io,
                zip([path for _ in range(len(params_list))], params_list),
            )
        elif isinstance(_default["wavelength"], list):
            pool.starmap(
                multi_wavelength_wrapper_io,
                zip([path for _ in range(len(params_list))], params_list),
            )
        else:
            pool.starmap(
                complete_profile_wrapper_io,
                zip([path for _ in range(len(params_list))], params_list),
            )
        pool.close()
        pool.join()
        return
    else:
        if isinstance(_default["input_cif"], list):
            results = list(pool.imap_unordered(multi_phase_wrapper, params_list))
        elif isinstance(_default["wavelength"], list):
            results = list(pool.imap_unordered(multi_wavelength_wrapper, params_list))
        else:
            results = list(pool.imap_unordered(complete_profile_wrapper, params_list))
        pool.close()
        pool.join()
    return result_output(results, path)


def result_output(results, path):
    if path.suffix == ".npy":
        np.save(str(path), np.stack([da.data for da in results], axis=-1))
    elif path.suffix == ".csv":
        cols = ["Intensity {}".format(idx) for idx in range(len(results))]
        np.savetxt(
            str(path),
            np.stack([da.data for da in results], axis=-1),
            delimiter=",",
            header=",".join(cols),
            comments="",
        )
    elif path.suffix == ".nc":
        da = concat_and_clean(results)
        da.to_netcdf(path)
    else:
        raise ValueError(
            "Path {} is invalid (doesn't exist or improper extension)".format(path)
        )
    return


def single_pattern(input_params, shape_limit=0.0, **kwargs):
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
    da : DataArray
    """
    parameters = load_params(input_params)
    _x = np.linspace(
        parameters["tth_min"], parameters["tth_max"], num=parameters["n_datapoints"]
    )
    test_y = np.zeros_like(_x) - 1

    if shape_limit:
        while any(test_y < 0):
            parameters["U"] = np.random.uniform(-shape_limit, shape_limit)
            parameters["V"] = np.random.uniform(-shape_limit, shape_limit)
            parameters["W"] = np.random.uniform(-shape_limit, shape_limit)
            parameters["X"] = np.random.uniform(-shape_limit, shape_limit)
            parameters["Y"] = np.random.uniform(np.abs(parameters["X"]), shape_limit)
            test_y = (
                parameters["U"] * (np.tan(_x) ** 2)
                + parameters["V"] * np.tan(_x)
                + parameters["W"]
            )

    for key in kwargs:
        parameters[key] = np.random.uniform(*kwargs[key])
    if isinstance(parameters["input_cif"], list):
        da = multi_phase_wrapper(parameters)
    elif isinstance(parameters["wavelength"], list):
        da = multi_wavelength_wrapper(parameters)
    else:
        da = complete_profile_wrapper(parameters)

    metadata_adjustments(da)
    return da
