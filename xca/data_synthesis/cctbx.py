"""
Python wrapping for the mass production of synthetic XRD patterns.
This module utilizes the cctbx and its features and nomenclature.
"""
import numpy as np
import xarray as xr
from math import sqrt, cos, sin, radians
from ast import literal_eval as make_tuple
import iotbx.cif
from collections import namedtuple
from dataclasses import dataclass

Lattice = namedtuple("Lattice", "a b c alpha beta gamma")


@dataclass
class Data:
    """Numpy array storage of structure factor calculation"""

    Q: np.ndarray
    FoQ: np.ndarray
    I: np.ndarray
    Q_mag: np.ndarray
    hkl: np.ndarray
    tth: np.ndarray  # 2-theta
    mult: np.ndarray  # multiplicities
    R: np.ndarray  # Reciprocal lattice

    def apply_mask(self, mask):
        for attr, value in self.__dict__.items():
            if attr in ("Q", "FoQ", "I", "Q_mag", "hkl", "tth", "mult"):
                setattr(self, attr, value[mask])


def load_cif(fname):
    """
    Loads cif file using cctbx and returns dictionary of key data.

    Parameters
    ----------
    fname : str, Path

    Returns
    -------
    data: dict
        Dictionary of key structural data from cif file

    """
    data = {}
    struct = list(
        iotbx.cif.reader(file_path=str(fname)).build_crystal_structures().values()
    )[0]
    data["structure"] = struct
    data["point_group"] = "{0!s}".format(struct.space_group().point_group_type())
    data["is_chiral"] = struct.space_group().is_chiral()
    data["is_centric"] = struct.space_group().is_centric()
    data["laue_group"] = "{0!s}".format(struct.space_group().laue_group_type())
    data["crystal_system"] = "{0!s}".format(struct.space_group().crystal_system())
    data["space_group"] = "{0!s}".format(struct.space_group().info())
    data["space_group_number"] = "{0!s}".format(struct.space_group().type().number())
    return data


def get_lattice(unit_cell):
    """
    Creates matrix of lattice vectors
    from the cctbx.xray.structure.unit_cell()
    """
    uc = list(unit_cell)
    lat = Lattice(
        a=uc[0],
        b=uc[1],
        c=uc[2],
        alpha=radians(uc[3]),
        beta=radians(uc[4]),
        gamma=radians(uc[5]),
    )
    af2c = np.zeros((3, 3))
    v = sqrt(
        1
        - cos(lat.alpha) ** 2
        - cos(lat.beta) ** 2
        - cos(lat.gamma) ** 2
        + 2 * cos(lat.alpha) * cos(lat.beta) * cos(lat.gamma)
    )
    af2c[0, 0] = lat.a
    af2c[0, 1] = lat.b * cos(lat.gamma)
    af2c[1, 1] = lat.b * sin(lat.gamma)
    af2c[0, 2] = lat.c * cos(lat.beta)
    af2c[1, 2] = (
        lat.c * (cos(lat.alpha) - cos(lat.beta) * cos(lat.gamma)) / sin(lat.gamma)
    )
    af2c[2, 2] = lat.c * v / sin(lat.gamma)
    return af2c


def Q_from_hkl(HKL, R):
    """Converts array of HKL vectors into Q space"""
    hkl = np.array(HKL)
    Q = np.matmul(R, hkl.T)
    return Q.T


def calc_structure_factor(struct, wavelength=1.5406):
    """
    Calculates a structure factor using cctbx functionality.

    Parameters
    -----------
    struct : cctbx.xray.structure as generated from structure file
    wavelength : angstroms wavelength of source, defaults to copper

    Returns
    ----------
    f_calc : cctbx.miller.array of hkl, and structure factor values
    """

    f_calc = struct.structure_factors(d_min=wavelength / 2).f_calc()
    return f_calc


def apply_additional_debye_waller(f_calc, debye_waller_factors):
    """
    Routine to apply additional debye waller factors that
    were not included in initial cif or structure.

    Parameters
    -----------
    f_calc : cctbx.miller.array
        Array of hkl, and structure factor values
    debye_waller_factors: dict
        Keyword arguments for additional debye_waller_factors (not included in struct)

    Returns
    ----------
    f_calc : cctbx.miller.array
        Updated array of hkl, and structure factor values
    """
    if debye_waller_factors:
        f_calc.apply_debye_waller_factors(**debye_waller_factors)
    return f_calc


def apply_extinction_correction(f_calc, wavelength, extinction_correction_x=0):
    """
    Routine to apply additional extinction correction
    Parameters
    -----------
    f_calc : cctbx.miller.array
        Array of hkl, and structure factor values
    wavelength: float
        Measurement wavelength in angstroms
    extinction_correction_x: float
        value for shelx extinction correction

    Returns
    ----------
    f_calc : cctbx.miller.array
        Updated array of hkl, and structure factor values
    """
    if extinction_correction_x:
        f_calc.apply_shelxl_extinction_correction(extinction_correction_x, wavelength)
    return f_calc


def convert_to_numpy(f_calc, wavelength=1.5406, tth_min=0.0, tth_max=179.0):
    """
    Converts a structure factor intensity into a dicitonary
    of numpy arrays in terms of hkl, Q, and 2theta.
    Importantly, the Q vector is calculated by the physics convention of reciprocal lattice.
    I.E., the without the 2pi factor.

    Parameters
    -----------
    f_calc : cctbx.miller.array
        Array of hkl, and structure factor values
    wavelength : float
        angstroms wavelength of source, defaults to copper
    tth_min: float
        two theta value at minimum, for removing reflections out of range
    tth_max: float
        two theta value at maximum, for removing reflections out of range

    Returns
    ----------
    data: Data
        Dataclass of numpy arrays for processing
    """

    hkl = np.array(f_calc.intensities().indices())

    R = get_lattice(make_tuple("{0!s}".format(f_calc.unit_cell().reciprocal())))
    Q = Q_from_hkl(hkl, R)

    data = Data(
        Q=Q,
        FoQ=f_calc.data().as_numpy_array(),
        I=f_calc.intensities().data().as_numpy_array(),
        Q_mag=np.linalg.norm(Q, axis=1),
        hkl=hkl,
        tth=np.degrees(f_calc.two_theta(wavelength).data().as_numpy_array()),
        mult=f_calc.multiplicities().data().as_numpy_array(),
        R=R,
    )
    data.apply_mask((data.tth > tth_min) & (data.tth < tth_max))
    return data


def apply_multiplicities(data):
    """
    Takes in the Data class of scattering data returned by convert_to_numpy()
    and applies the multiplicity to each intensity.

    Sets the multiplicity to 1 to avoid accidental double-use.
    Parameters
    ----------
    data : Data

    Returns
    -------
    data: Data
        Updated with multiplicities

    """
    for i in range(len(data.tth)):
        data.I[i] *= data.mult[i]
        data.mult[i] = 1
    return data


def apply_texturing(data, preferred=(), march=1.0):
    """
    Takes in scattering data as a dictionary of vectors from convert_to_numpy() and applies preffered orientation
    augmentation to the intensities according tyo March-Dollase approach.

    Importantly, this does not regard symmetry of the plane, and only considers the reflections as labeled.
    https://journals.iucr.org/j/issues/2009/03/00/ks5199/

    Parameters
    ----------
    data : Data
        scattering data as a dataclass of vectors from convert_to_numpy()
    preferred : tuple, list
        miller index normal to the preferred plane
    march : float
        The March parameter, 0 < r <=1.
        Greater than 1 would indicate needles not plates.
        1 is completely random orientaiton.
        0 would be uniaxial orientation.

    Returns
    -------
    data : Data
        scattering data as a dictionary of vectors from convert_to_numpy()

    """
    if not preferred:
        return data
    if march <= 0 or march > 1:
        raise ValueError(
            "Invalid value for march. Should be between 0 and 1. Is: {}".format(march)
        )

    R = data.R
    H = Q_from_hkl(preferred, R)
    for i in range(len(data.Q)):
        h = np.array(data.Q[i])
        # arccos is forced to 1,-1 for precision errors of parallel vectors
        alpha = np.arccos(
            max(min(np.dot(H, h) / np.linalg.norm(H) / np.linalg.norm(h), 1.0), -1.0)
        )
        W = (march ** 2 * (np.cos(alpha) ** 2) + (1 / march) * np.sin(alpha) ** 2) ** (
            -3.0 / 2
        )
        data.I[i] *= W

    return data


def apply_polarization(Th_I_pairs, theta_m=0.0):
    """
    Lorentz-polarization correction
    reference.iucr.org/dictionary/Lorentz-polarization_correction

    Parameters
    -----------
    Th_I_pairs : List of tupes
        scattering intensities as a function of 2theta
    theta_m : float
        experimental Bragg angle of the monochromator crystal

    Returns
    -------
    Th_I_pairs : scattering intensities as a function of 2theta. List of tuples
    """

    Th, intensity = [list(x) for x in zip(*Th_I_pairs)]
    # Form from CrystalDiffract manual (Correction for with no monochromator)
    for i in range(len(Th)):
        intensity[i] *= (1 + np.cos(np.radians(Th[i])) ** 2) / (
            2 * np.sin(np.radians(Th[i] / 2)) * np.sin(np.radians(Th[i]))
        )

    Th_I_pairs = zip(Th, intensity)

    return Th_I_pairs


def apply_sample_offset(Th_I_pairs, s=0.0, R=200):
    """
    Sample offset for peak shift
    delta 2-theta = (2*s*cos(theta))/R * (180/pi)
    where s is the sample offset height
    and R is the instrument radius

    Parameters
    -----------
    Th_I_pairs : List of tuples
        scattering intensities as a function of 2theta.
    s : float
        sample offset height [mm]
    R : float
        instrument radius [mm]

    Returns
    -----------
    Th_I_pairs : Th_I_pairs : List of tuples
        scattering intensities as a function of 2theta.
    """
    Th, intensity = [list(x) for x in zip(*Th_I_pairs)]
    for i in range(len(Th)):
        Th[i] -= 2 * s * np.cos(radians(Th[i] / 2)) / R * (180 / np.pi)

    Th_I_pairs = zip(Th, intensity)

    return Th_I_pairs


def apply_background(
    da,
    *,
    bkg_6=0.0,
    bkg_5=0.0,
    bkg_4=0.0,
    bkg_3=0.0,
    bkg_2=0.0,
    bkg_1=0.0,
    bkg_0=0.0,
    bkg_n1=0.0,
    bkg_n2=0.0,
    bkg_ea=0.0,
    bkg_eb=0.0,
    **kwargs
):
    """
    Applies a 6-term polynomial addition to the background.

    Parameters
    -----------
    da : DataArray
        1-dimensional DataArray with coordinates of 2theta values and intensity values
    bkg_6 : float
        6th term polynomial coefficient
    bkg_5 : float
        5th term polynomial coefficient
    bkg_4 : float
        4th term polynomial coefficient
    bkg_3 : float
        3th term polynomial coefficient
    bkg_2 : float
        2th term polynomial coefficient
    bkg_1 : float
        1th term polynomial coefficient
    bkg_0 : float
        constant term in polynomial
    bkg_n1: float
        1/x term polynomial coefficient
    bkg_n2: float
        1/x^2 polynomial coefficient
    bkg_ea:
        A*e^(bx), a factor in exponential
    bkg_eb:
        A*e^(bx), b term in coeffcient
    """
    da += (
        bkg_6 * da["2theta"] ** 6
        + bkg_5 * da["2theta"] ** 5
        + bkg_4 * da["2theta"] ** 4
        + bkg_3 * da["2theta"] ** 3
        + bkg_2 * da["2theta"] ** 2
        + bkg_1 * da["2theta"] ** 1
        + bkg_0
        + bkg_n1 * da["2theta"] ** -1
        + bkg_n2 * da["2theta"] ** -2
        + bkg_ea * np.exp(bkg_eb * da["2theta"])
    )
    return da


def apply_peak_profile(
    Th_I_pairs,
    *,
    tth_min,
    tth_max,
    n_datapoints,
    U=0.1,
    V=0.1,
    W=0.1,
    X=0.1,
    Y=0.1,
    Z=0.0,
    **kwargs
):
    """
    Takes in I vs 2theta data and applies a profiling function.

    Parameters
    -----------
    Th_I_pairs: tuple
        two_theta and intensity paris
    tth_min: float
        2-theta min
    tth_max: float
        2-theta max
    n_datapoints: int
        number of datapoints in linspace
    U: float
    V: float
    W: float
    X: float
    Y: float
    Z: float

    Returns
    -----------
    x : ndarray
        2theta values over range
    y : ndarray
        intensity as a function of 2theta

    PMM Notes:
    Cagliotti, Paoletti, & Ricci is the gaussian case of eta=0 and Z=0.
    """
    np.seterr(invalid="raise")
    A = 2.69269
    B = 2.42843
    C = 4.47163
    D = 0.07842

    x = np.linspace(tth_min, tth_max, n_datapoints)
    y = np.zeros_like(x)

    # The slowest component of the process is calculating the gauss and lorentz curves.
    # Attempted vectorization offered no speedups.
    # The slowest operation is the exponential
    for Th, I in Th_I_pairs:
        if Th < 0 or Th > 180:
            continue
        try:
            theta = np.radians(Th / 2)
            Gamma_G = np.sqrt(
                U * (np.tan(theta) ** 2)
                + V * np.tan(theta)
                + W
                + Z / (np.cos(theta) ** 2)
            )
            Gamma_L = X * np.tan(theta) + Y / np.cos(theta)
            Gamma = (
                Gamma_G ** 5
                + A * (Gamma_G ** 4) * Gamma_L
                + B * (Gamma_G ** 3) * (Gamma_L ** 2)
                + C * (Gamma_G ** 2) * (Gamma_L ** 3)
                + D * (Gamma_G) * (Gamma_L ** 4)
                + Gamma_L ** 5
            ) ** 0.2
            q = Gamma_L / Gamma
            eta = 1.36603 * q - 0.47719 * q ** 2 + 0.1116 * q ** 3

            gauss = (
                2
                * np.sqrt(np.log(2) / np.pi)
                / Gamma
                * np.exp(-4 * np.log(2) * ((x - Th) ** 2) / (Gamma ** 2))
            )
            lorentz = (2 / np.pi / Gamma) / ((1 + 4 * ((x - Th) ** 2)) / (Gamma ** 2))
            y = y + I * (lorentz * eta + gauss * (1 - eta))
        except (FloatingPointError, ValueError):
            raise ValueError(
                "Bad values for 2theta: {}\
                             \nor bad values for peak shape:\
                             \nU={}\nV={}\nW={}\nX={}\nY={}\nZ={}".format(
                    Th, U, V, W, X, Y, Z
                )
            )
    return x, y


def create_complete_profile(
    *,
    input_cif,
    wavelength,
    tth_min,
    tth_max,
    n_datapoints,
    instrument_radius,
    offset_height=0.0,
    preferred=(),
    march_parameter=1.0,
    theta_m=26.6,
    debye_waller_factors=(),
    extinction_correction_x=0.0,
    noise_std=0.0,
    normalize=True,
    verbose=False,
    **kwargs
):
    """
    Final function that takes input dictionary of parameters and generates numpy array
    of complete PXRD profile.
    Parameters
    -----------
    input_cif: str, Path
        Cit file to read and create pattern from
    wavelength: float
        Instrument wavelength in angstroms
    tth_min: float
        2-theta min
    tth_max: float
        2-theta max
    n_datapoints: int
        number of datapoints in linspace
    instrument_radius: float
        instrument radius (measurement distance) in millimeters
    offset_height: float
        Offset height for sample offset in millimeters. Can be used as a proxy for isotropic expansion/contraction
    preferred: list, tuple
        hkl of preferred axis
    march_parameter: float
        [0,1] parameter desribing degree of uniform orientation.
        1 is completely random orientaiton.
        0 would be uniaxial orientation.
        Greater than 1 would indicate needles not plates, and should not be used.
    noise_std: float
        Normally distributed noise is applied with this standard deviation if greater than 0.
    theta_m: float
    debye_waller_factors: dict
    extinction_correction_x: float
    normalize: bool
        Whether to perform max normalization of profile AND add background
    verbose: bool
    kwargs: dict
        Keyword arguments for apply_background and apply_peak_profile

    Returns
    --------
    da : DataArray


    """
    params = locals()
    del params["kwargs"]
    params.update(kwargs)

    # Load cif
    data = load_cif(str(input_cif))
    # Calculate structure factor as cctbx miller array
    sf = calc_structure_factor(data["structure"], wavelength=wavelength)
    # Apply additional Debeye Waller factor and extinction correction to miller array
    sf = apply_additional_debye_waller(sf, debye_waller_factors=debye_waller_factors)
    sf = apply_extinction_correction(
        sf, wavelength=wavelength, extinction_correction_x=extinction_correction_x
    )
    # Convert to dictionary of numpy arrays
    scattering = convert_to_numpy(
        sf, wavelength=wavelength, tth_min=tth_min, tth_max=tth_max
    )
    # Apply multiplicites from symmetry
    scattering = apply_multiplicities(scattering)
    # Apply texturing from preffered orientation
    scattering = apply_texturing(scattering, preferred=preferred, march=march_parameter)

    # Cast as data pairing 2-theta and Intensity and applies polarization
    Th_I_pairs = zip(scattering.tth, scattering.I)
    Th_I_pairs = apply_polarization(Th_I_pairs, theta_m=theta_m)
    # peak position augmentation to shift peaks based on offset height and instrument radius
    Th_I_pairs = apply_sample_offset(Th_I_pairs, s=offset_height, R=instrument_radius)

    # Apply's peak shape to intensities and returns xy-like spectrum
    x, y = apply_peak_profile(
        Th_I_pairs,
        tth_min=tth_min,
        tth_max=tth_max,
        n_datapoints=n_datapoints,
        **kwargs
    )

    # Convert to xarray
    params.update(data)
    params.update(
        dict(
            zip(
                ["L_a", "L_b", "L_c", "alpha", "beta", "gamma"],
                (data["structure"].unit_cell().parameters()),
            )
        )
    )
    del params["structure"]
    da = xr.DataArray(y, coords={"2theta": x}, dims=["2theta"], attrs=params)

    # Normalizes to maximum intensity and adds background w.r.t normalized prof then renormalize
    if normalize:
        da /= np.max(da)
        da = apply_background(da, **kwargs)
        da /= np.max(da)

    # Applied noise sampled from a normal dist to the normalized spectra.
    if noise_std > 0:
        with xr.set_options(keep_attrs=True):
            noise = np.random.normal(0, noise_std, da.shape[-1])
            da = da + noise

    if verbose:
        _Th, _I = [list(_x) for _x in zip(*sorted(Th_I_pairs))]
        print("{:12s}{:12s}".format("2-theta", "Intensity"))
        for i in range(len(_I)):
            print("{:12.2f}{:12.3f}".format(_Th[i], _I[i] / max(_I) * 100))

    return da


def sum_multi_wavelength_profiles(
    wavelengths, *, normalize=True, noise_std=0.0, **kwargs
):
    """
    Applies the complete profile generation to systems with muiltiple wavelengths.

    Parameters
    -----------
    wavelengths: list of tuples
        [(wavelength, weight)]
    normalize: bool
        Whether to perform max normalization of profile AND add background
    noise_std: float
        Normally distributed noise is applied with this standard deviation if greater than 0.
    kwargs: dict
        Keyword arguments for create_complete_profile

    Returns
    -----------
    da : DataArray
    """
    first = True

    with xr.set_options(keep_attrs=True):
        for wavelength, weight in wavelengths:
            da_tmp = create_complete_profile(
                wavelength=wavelength, noise_std=0.0, normalize=False, **kwargs
            )
            if first:
                da = da_tmp * weight
                first = False
            else:
                da += da_tmp * weight

    if normalize:
        da /= np.max(da)
        da = apply_background(da, **kwargs)
        da /= np.max(da)

    if noise_std > 0:
        with xr.set_options(keep_attrs=True):
            noise = np.random.normal(0, noise_std, da.shape[-1])
            da = da + noise

    # Update attrs to local params
    da.attrs["wavelength"] = wavelengths
    da.attrs["normalize"] = normalize
    da.attrs["noise_std"] = noise_std
    return da


def multi_phase_profile(
    input_cifs, *, wavelength, normalize=True, noise_std=0.0, **kwargs
):
    """
    Applies the complete profile generation to systems with multiple phases.

    Parameters
    -----------
    input_cifs :
        list of tuples of (phases, weights)
    wavelength: list, float
        Either list of tuples for multiple wavelengths or single float of wavelength in angstroms
    noise_std: float
        Normally distributed noise is applied with this standard deviation if greater than 0.
    normalize: bool
        Whether to perform max normalization of profile AND add background
    Returns
    -----------
    da : DataArray
    """
    from collections import defaultdict

    first = True

    update_dict = defaultdict(list)

    with xr.set_options(keep_attrs=True):
        for cif, weight in input_cifs:
            if isinstance(wavelength, list):
                da_tmp = sum_multi_wavelength_profiles(
                    noise_std=0.0, normalize=False, input_cif=cif, **kwargs
                )
            else:
                da_tmp = create_complete_profile(
                    input_cif=cif, normalize=False, **kwargs
                )

            for key in [
                "L_a",
                "L_b",
                "L_c",
                "alpha",
                "beta",
                "gamma",
                "point_group",
                "is_chiral",
                "is_centric",
                "laue_group",
                "crystal_system",
                "space_group",
                "point_group",
            ]:
                update_dict[key].append(da_tmp.attrs[key])

            if first:
                da = da_tmp * weight
                first = False
            else:
                da += da_tmp * weight

    if normalize:
        da /= np.max(da)
        da = apply_background(da, **kwargs)
        da /= np.max(da)

    if noise_std > 0:
        with xr.set_options(keep_attrs=True):
            noise = np.random.normal(0, noise_std, da.shape[-1])
            da = da + noise

    # Update attrs to full dict, and list of lattice parameters
    da.attrs["input_cif"] = input_cifs
    da.attrs["wavelength"] = wavelength
    da.attrs["normalize"] = normalize
    da.attrs["noise_std"] = noise_std
    da.attrs.update(update_dict)
    return da
