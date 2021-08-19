import json
from pathlib import Path
from xca.data_synthesis.builder import cycle_params
from xca.data_synthesis.cctbx import load_cif, calc_structure_factor, convert_to_numpy


def get_reflections(cif_path, tth_min, tth_max, wavelength):
    """Checks relevant reflections that occur between tth_min and tth_max at a given wavelength"""
    data = load_cif(cif_path)
    sf = calc_structure_factor(data["structure"])
    scattering = convert_to_numpy(
        sf, wavelength=wavelength, tth_max=tth_max, tth_min=tth_min
    )
    reflections = zip(scattering.hkl, scattering.tth, scattering.I)
    keep = []
    for reflection in reflections:
        if reflection[1] < tth_max and reflection[1] > tth_min and reflection[2] > 1:
            keep.append(tuple(reflection[0]))
    return keep


def log_reflections(cif_paths, tth_min, tth_max, wavelength, outpath=None):
    """Itterates over list of cifs and puts relevant reflections into json file"""
    dic = {}
    for cif_path in cif_paths:
        if not isinstance(cif_path, Path):
            path = Path(cif_path)
        else:
            path = cif_path
        dic[path.stem] = get_reflections(path, tth_min, tth_max, wavelength)
    if outpath:
        with open(outpath, "w") as f:
            json.dump(dic, f)

    return dic


def pattern_simulation(n_patterns, system="BaTiO"):
    """Example pattern simulation as reported in arXiv:2008.00283"""
    if system == "BaTiO":
        wavelength = 0.1671
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
        }
        cif_paths = list((Path(__file__).parent / "cifs-BaTiO/").glob("*.cif"))
        march_range = (0.8, 1.0)
        shape_limit = 1e-1
        reflections = log_reflections(
            cif_paths, param_dict["2theta_min"], param_dict["2theta_max"], wavelength
        )
    elif system == "ADTA":
        wavelength = 1.54060
        param_dict = {
            "noise_std": 2e-3,
            "instrument_radius": 240.00,
            "2theta_min": 2.00756514,
            "2theta_max": 39.99347292,
            "n_datapoints": 2894,
        }
        kwargs = {
            "bkg_-1": (0.0, 0.5),
            "bkg_-2": (0.0, 1.0),
            "sample_height": (-2.0, 2.0),
        }
        march_range = (0.05, 1)
        shape_limit = 0.05
        cif_paths = list((Path(__file__).parent / "cifs-ADTA/").glob("*.cif"))
        reflections = log_reflections(
            cif_paths, param_dict["2theta_min"], param_dict["2theta_max"], wavelength
        )
    elif system == "NiCoAl":
        wavelength = [(1.54060, 0.5), (1.54439, 0.5)]
        param_dict = {
            "noise_std": 5e-3,
            "instrument_radius": 240.00,
            "theta_m": 26.6,
            "2theta_min": 20.0,
            "2theta_max": 89.93999843671914,
            "n_datapoints": 3498,
        }
        kwargs = {"bkg_0": (0.0, 0.05), "sample_height": (-2.0, 2.0)}
        march_range = (0.05, 1)
        shape_limit = 0.05
        cif_paths = list((Path(__file__).parent / "cifs-NiCoAl/").glob("*.cif"))
        reflections = log_reflections(
            cif_paths,
            param_dict["2theta_min"],
            param_dict["2theta_max"],
            wavelength[0][0],
        )
    else:
        raise ValueError(
            "Unknown system for example pattern simulation {}".format(system)
        )

    d = {}
    for idx, cif in enumerate(cif_paths):
        print(cif)
        phase = cif.stem
        d[phase] = idx
        param_dict["input_cif"] = cif
        output_path = Path("tmp") / f"{system}"
        output_path.mkdir(parents=True, exist_ok=True)
        cycle_params(
            n_patterns,
            output_path,
            input_params=param_dict,
            march_range=march_range,
            shape_limit=shape_limit,
            preferred_axes=reflections[phase],
            **kwargs,
        )
    return d
