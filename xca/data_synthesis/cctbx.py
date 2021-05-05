"""
Python wrapping for the mass production of synthetic XRD patterns.
This package depends on the cctbx, so requires a specific python2 kernel or interpreter.

@author: maffettone
"""
import numpy as np
import xarray as xr
from math import sqrt, cos, sin, radians
from ast import literal_eval as make_tuple
import iotbx.cif


def load_cif(fname):
    data = {}
    struct = list(iotbx.cif.reader(file_path=str(fname)).build_crystal_structures().values())[0]
    data['structure'] = struct
    data['point_group'] = '{0!s}'.format(struct.space_group().point_group_type())
    data['is_chiral'] = struct.space_group().is_chiral()
    data['is_centric'] = struct.space_group().is_centric()
    data['laue_group'] = '{0!s}'.format(struct.space_group().laue_group_type())
    data['crystal_system'] = '{0!s}'.format(struct.space_group().crystal_system())
    data['space_group'] = '{0!s}'.format(struct.space_group().info())
    data['space_group_number'] = '{0!s}'.format(struct.space_group().type().number())
    return data


def get_lattice(unit_cell):
    '''
    Creates matrix of lattice vectors
    from the cctbx.xray.structure.unit_cell()
    '''
    uc = list(unit_cell)
    lat = {}
    lat['L_a'] = uc[0]
    lat['L_b'] = uc[1]
    lat['L_c'] = uc[2]
    lat['alpha'] = radians(uc[3])
    lat['beta'] = radians(uc[4])
    lat['gamma'] = radians(uc[5])
    af2c = np.zeros((3, 3))
    v = sqrt(1 - cos(lat['alpha']) ** 2 - cos(lat['beta']) ** 2 - cos(lat['gamma']) ** 2 +
             2 * cos(lat['alpha']) * cos(lat['beta']) * cos(lat['gamma']))
    af2c[0, 0] = lat['L_a']
    af2c[0, 1] = lat['L_b'] * cos(lat['gamma'])
    af2c[1, 1] = lat['L_b'] * sin(lat['gamma'])
    af2c[0, 2] = lat['L_c'] * cos(lat['beta'])
    af2c[1, 2] = lat['L_c'] * (cos(lat['alpha']) - cos(lat['beta']) * cos(lat['gamma'])) / sin(lat['gamma'])
    af2c[2, 2] = lat['L_c'] * v / sin(lat['gamma'])
    return af2c


def Q_from_hkl(HKL, R):
    hkl = np.array(HKL)
    Q = np.matmul(R, hkl.T)
    return Q.T


def calc_structure_factor(struct, wavelength=1.5406):
    '''
    Calculates a structure factor using cctbx functionality.

    Returns
    ----------
    f_calc : cctbx.miller.array of hkl, and structure factor values

    Parameters
    -----------
    struct : cctbx.xray.structure as generated from structure file
    wavelength : angstroms wavelength of source, defaults to copper
    '''

    f_calc = struct.structure_factors(d_min=wavelength / 2).f_calc()
    return f_calc


def apply_additional_debye_waller(f_calc, debye_waller_factors=None):
    '''
    Routine to apply additional debye waller factors that
    were not included in initial cif or structure.

    Returns
    ----------
    f_calc : cctbx.miller.array of hkl, and structure factor values

    Parameters
    -----------
    f_calc : cctbx.miller.array of hkl, and structure factor values
    debye_waller_factors: kwargs, additional debye_waller_factors (not included in struct)
    '''
    if debye_waller_factors:
        f_calc.apply_debye_waller_factors(**debye_waller_factors)
    return f_calc


def apply_extinction_correction(f_calc, extinction_correction_x=None, wavelength=1.5406):
    '''
    Routine to apply additional extinction correction

    Returns
    ----------
    f_calc : cctbx.miller.array of hkl, and structure factor values

    Parameters
    -----------
    f_calc : cctbx.miller.array of hkl, and structure factor values
    extinction_correction_x: value for shelx extinction correction
    '''
    if extinction_correction_x is not None:
        f_calc.apply_shelxl_extinction_correction(extinction_correction_x, wavelength)
    return f_calc


def convert_to_numpy(f_calc, wavelength=1.5406, tth_min=0., tth_max=179.):
    '''
    Converts a structure factor intensity into a dicitonary
    of numpy arrays in terms of hkl, Q, and 2theta.

    Returns
    ----------
    data : dictionary containing numpy arrays...
        Q : Q vector
        I : Intensity
        Q_mag: magnitude of Q
        hkl : tuples of hkl values
        2theta : two theta (degrees)
        mult : reflection multiplicities
        R : reciprocal n(dlattice vectors

    Parameters
    -----------
    struct : cctbx.xray.structure as generated from structure file
    wavelength : angstroms wavelength of source, defaults to copper
    tth_min: two theta value at minimum, for removing reflections out of range
    tth_max: two theta value at maximum, for removing reflections out of range

    Importantly, the Q vector is calculated by the physics convention of reciprocal lattice.
    I.E., the without the 2pi factor.
    '''

    hkl = np.array(f_calc.intensities().indices())

    R = get_lattice(make_tuple('{0!s}'.format(f_calc.unit_cell().reciprocal())))
    Q = Q_from_hkl(hkl, R)

    data = {}
    data['Q'] = Q
    data['FoQ'] = f_calc.data().as_numpy_array()
    data['I'] = f_calc.intensities().data().as_numpy_array()
    data['Q_mag'] = np.linalg.norm(Q, axis=1)
    data['hkl'] = hkl
    data['2theta'] = np.degrees(f_calc.two_theta(wavelength).data().as_numpy_array())
    data['mult'] = f_calc.multiplicities().data().as_numpy_array()
    data['R'] = R

    # Removing data that is outside of range
    non_array_keys = ['R']
    mask = (data['2theta'] > tth_min) & (data['2theta'] < tth_max)
    for key in data:
        if key in non_array_keys:
            continue
        data[key] = data[key][mask]
    return data

def apply_multiplicities(data):
    '''
    Takes in the dictionary of scattering data returned by convert_to_numpy()
    and applies the multiplicity to each intensity.

    Sets the multiplicity to 1 to avoid accidental double-use.

    Returns
    -----------
    data : scattering data as a dictionary of vectors from convert_to_numpy()


    '''
    for i in range(len(data['2theta'])):
        data['I'][i] *= data['mult'][i]
        data['mult'][i] = 1
    return data


def apply_texturing(data, preferred=None, march=None):
    """
    Takes in scattering data as a dictionary of vectors from convert_to_numpy() and applies preffered orientation
    augmentation to the intensities according tyo March-Dollase approach.

    Importantly, this does not regard symmetry of the plane, and only considers the reflections as labeled.
    https://journals.iucr.org/j/issues/2009/03/00/ks5199/

    Returns
    -----------
    data : scattering data as a dictionary of vectors from convert_to_numpy()

    Parameters
    -----------
    data : scattering data as a dictionary of vectors from convert_to_numpy()
    preferred : miller index normal to the preferred plane
    march : The March parameter, 0 < r <=1.
        Greater than 1 would indicate needles not plates.
        1 is completely random orientaiton.
        0 would be uniaxial orientation.
    """
    if preferred == None:
        return data
    if march <= 0 or march > 1:
        raise ValueError("Invalid value for march. Should be between 0 and 1. Is: {}".format(march))

    R = data['R']
    H = Q_from_hkl(preferred, R)
    for i in range(len(data['Q'])):
        h = np.array(data['Q'][i])
        # arccos is forced to 1,-1 for precision errors of parallel vectors
        alpha = np.arccos(max(min(np.dot(H, h) / np.linalg.norm(H) / np.linalg.norm(h), 1.0), -1.0))
        W = (march ** 2 * (np.cos(alpha) ** 2) + (1 / march) * np.sin(alpha) ** 2) ** (-3. / 2)
        data['I'][i] *= W

    return data


def apply_polarization(Th_I_pairs, theta_m=0.):
    """
    Lorentz-polarization correction
    reference.iucr.org/dictionary/Lorentz-polarization_correction

    Returns
    -----------
    Th_I_pairs : scattering intensities as a function of 2theta. List of tuples

    Parameters
    -----------
    Th_I_pairs : scattering intensities as a function of 2theta. List of tuples
    theta_m : experimental Bragg angle of the monochromator crystal

    """

    Th, I = [list(x) for x in zip(*Th_I_pairs)]
    # Form from CrystalDiffract manual (Correction for with no monochromator)
    for i in range(len(Th)):
        I[i] *= (1 + np.cos(np.radians(Th[i])) ** 2) / (2 * np.sin(np.radians(Th[i] / 2)) * np.sin(np.radians(Th[i])))

    Th_I_pairs = zip(Th, I)

    return Th_I_pairs


def apply_sample_offset(Th_I_pairs, s=0., R=200):
    """
    Sample offset for peak shift
    delta 2-theta = (2*s*cos(theta))/R * (180/pi)
    where s is the sample offset height
    and R is the instrument radius

    Returns
    -----------
    Th_I_pairs : scattering intensities as a function of 2theta. List of tuples

    Parameters
    -----------
    Th_I_pairs : scattering intensities as a function of 2theta. List of tuples
    s : sample offset height [mm]
    R : instrument radius [mm]
    """
    Th, I = [list(x) for x in zip(*Th_I_pairs)]
    for i in range(len(Th)):
        Th[i] -= 2 * s * np.cos(radians(Th[i] / 2)) / R * (180 / np.pi)

    Th_I_pairs = zip(Th, I)

    return Th_I_pairs


def apply_background(da, params=None):
    """
    Applies a 6-term polynomial addition to the background.

    Parameters
    -----------
    da : DataArray
        1-dimensional DataArray with coordinates of 2theta values and intensity values
    params : dictionary of parameters containing ....
        bkg_6 : 6th term polynomial coefficient
        bkg_5 : 5th term polynomial coefficient
        bkg_4 : 4th term polynomial coefficient
        bkg_3 : 3th term polynomial coefficient
        bkg_2 : 2th term polynomial coefficient
        bkg_1 : 1th term polynomial coefficient
        bkg_0 : constant term in polynomial
        bkg_-1: 1/x term polynomial coefficient
        bkg_-1: 1/x^2 polynomial coefficient
        bkg_ea: A*e^(bx), a factor in exponential
        bkg_eb: A*e^(bx), C term in coeffcient
    """
    da += (params['bkg_6'] * da['2theta'] ** 6 +
           params['bkg_5'] * da['2theta'] ** 5 +
           params['bkg_4'] * da['2theta'] ** 4 +
           params['bkg_3'] * da['2theta'] ** 3 +
           params['bkg_2'] * da['2theta'] ** 2 +
           params['bkg_1'] * da['2theta'] ** 1 +
           params['bkg_0'] +
           params['bkg_-1'] * da['2theta'] ** -1 +
           params['bkg_-2'] * da['2theta'] ** -2 +
           params['bkg_ea'] * np.exp(params['bkg_eb'] * da['2theta'])
           )
    return da


def apply_peak_profile(Th_I_pairs, parameters):
    """
    Takes in I vs 2theta data and applies a profiling function.

    Parameters
    -----------
    Th_I_pairs: two_theta and intensity paris
    parameters: dictionary containing...
        2theta_min
        2theta_max
        n_datapoints
        U, V, W, X, Y, Z.

    Returns
    -----------
    x : 2theta values over range
    y : intensity as a function of 2theta

    PMM Notes:
    Cagliotti, Paoletti, & Ricci is the gaussian case of eta=0 and Z=0.
    """
    np.seterr(invalid='raise')
    A = 2.69269
    B = 2.42843
    C = 4.47163
    D = 0.07842
    U = parameters['U']
    V = parameters['V']
    W = parameters['W']
    X = parameters['X']
    Y = parameters['Y']
    Z = parameters['Z']

    x = np.linspace(parameters['2theta_min'], parameters['2theta_max'], num=parameters['n_datapoints'])
    y = np.zeros_like(x)

    # The slowest component of the process is calculating the gauss and lorentz curves.
    # Attempted vectorization offered no speedups.
    # The slowest operation is the exponential
    for Th, I in Th_I_pairs:
        if Th < 0 or Th > 180:
            continue
        try:
            theta = np.radians(Th / 2)
            Gamma_G = np.sqrt(U * (np.tan(theta) ** 2) + V * np.tan(theta) +
                              W + Z / (np.cos(theta) ** 2))
            Gamma_L = X * np.tan(theta) + Y / np.cos(theta)
            Gamma = (Gamma_G ** 5 + A * (Gamma_G ** 4) * Gamma_L + B * (Gamma_G ** 3) * (Gamma_L ** 2) +
                     C * (Gamma_G ** 2) * (Gamma_L ** 3) + D * (Gamma_G) * (Gamma_L ** 4) + Gamma_L ** 5) ** 0.2
            q = Gamma_L / Gamma
            eta = 1.36603 * q - 0.47719 * q ** 2 + 0.1116 * q ** 3

            gauss = 2 * np.sqrt(np.log(2) / np.pi) / Gamma * np.exp(-4 * np.log(2) * ((x - Th) ** 2) / (Gamma ** 2))
            lorentz = (2 / np.pi / Gamma) / ((1 + 4 * ((x - Th) ** 2)) / (Gamma ** 2))
            y = y + I * (lorentz * eta + gauss * (1 - eta))
        except:
            raise ValueError('Bad values for 2theta: {}\
                             \nor bad values for peak shape:\
                             \nU={}\nV={}\nW={}\nX={}\nY={}\nZ={}'.format(Th, U, V, W, X, Y, Z))
    return x, y


def create_complete_profile(params, normalize=True):
    """
    Final function that takes input dictionary of parameters and generates numpy array
    of complete PXRD profile.
    Parameters
    -----------
    params : dictionary containing...
    normalize: bool for max normalization of profile

    Returns
    -----------
    x : two theta values over the range
    y : normalized PXRD profile


    """

    # Load cif
    data = load_cif(str(params['input_cif']))
    # Calculate structure factor as cctbx miller array
    sf = calc_structure_factor(data['structure'], wavelength=params['wavelength'])
    # Apply additional Debeye Waller factor and extinction correction to miller array
    sf = apply_additional_debye_waller(sf, debye_waller_factors=params['debye_waller_factors'])
    sf = apply_extinction_correction(sf,
                                     extinction_correction_x=params['extinction_correction_x'],
                                     wavelength=params['wavelength'])
    # Convert to dictionary of numpy arrays
    scattering = convert_to_numpy(sf,
                                  wavelength=params['wavelength'],
                                  tth_min=params['2theta_min'],
                                  tth_max=params['2theta_max'])
    # Apply multiplicites from symmetry
    scattering = apply_multiplicities(scattering)
    # Apply texturing from preffered orientation
    scattering = apply_texturing(scattering,
                                 preferred=params['preferred'],
                                 march=params['march_parameter'])

    # Cast as data pairing 2-theta and Intensity and applies polarization
    Th_I_pairs = zip(scattering['2theta'], scattering['I'])
    Th_I_pairs = apply_polarization(Th_I_pairs, theta_m=params['theta_m'])
    # peak position augmentation to shift peaks based on offset height and instrument radius
    Th_I_pairs = apply_sample_offset(Th_I_pairs, s=params['offset_height'], R=params['instrument_radius'])

    # Apply's peak shape to intensities and returns xy-like spectrum
    x, y = apply_peak_profile(Th_I_pairs, params)

    # Convert to xarray
    params.update(data)
    params.update(
        dict(zip(["L_a", "L_b", "L_c", "alpha", "beta", "gamma"],
                 (data['structure'].unit_cell().parameters())))
    )
    del params["structure"]
    da = xr.DataArray(
        y,
        coords={"2theta": x},
        dims=["2theta"],
        attrs=params
    )

    # Normalizes to maximum intensity and adds background w.r.t normalized prof then renormalize
    if normalize:
        da /= np.max(da)
        da = apply_background(da, params=params)
        da /= np.max(da)

    # Applied noise sampled from a normal dist to the normalized spectra.
    if params['noise_std'] > 0:
        noise = np.random.normal(0, params['noise_std'], da.shape[-1])
        da = da + noise

    if params['verbose']:
        _Th, _I = [list(_x) for _x in zip(*sorted(Th_I_pairs))]
        print("{:12s}{:12s}".format('2-theta', 'Intensity'))
        for i in range(len(_I)):
            print("{:12.2f}{:12.3f}".format(_Th[i], _I[i] / max(_I) * 100))

    return da


def sum_multi_wavelength_profiles(params, normalize=True):
    """
    Applies the complete profile generation to systems with muiltiple wavelengths.

    Parameters
    -----------
    params : dictionary containing...
        wavelength : list of tuples of wavelegth and weight
    normalize: bool for max normalization of profile

    Returns
    -----------
    x : two theta values over the range
    y : normalized PXRD profile
    """
    first = True
    single_params = {x: params[x] for x in params if x != 'wavelength'}

    for wavelength, weight in params['wavelength']:
        single_params['wavelength'] = wavelength
        x_tmp, y_tmp = create_complete_profile(single_params, normalize=False)
        if first:
            x = x_tmp
            y = y_tmp * weight
            first = False
        else:
            assert x[0] == x_tmp[0]
            y += y_tmp * weight

    if normalize:
        y /= np.max(y)
        y = apply_background(x, y, params)
        y /= np.max(y)

    if params['noise_std'] > 0:
        noise = np.random.normal(0, params['noise_std'], len(y))
        y = y + noise

    return x, y


def multi_phase_profile(params, normalize=True):
    """
    Applies the complete profile generation to systems with multiple phases.

    Parameters
    -----------
    params : dictionary containing...
        input_cif : list of tuples of (phases, weights)
    normalize: bool for max normalization of profile
    Returns
    -----------
    x : two theta values over the range
    y : normalized PXRD profile
    """

    first = True
    single_params = {x: params[x] for x in params if x != 'input_cif'}

    for cif, weight in params['input_cif']:
        single_params['input_cif'] = cif
        if type(single_params['wavelength']) == type([]):
            x_tmp, y_tmp = sum_multi_wavelength_profiles(single_params, normalize=False)
        else:
            x_tmp, y_tmp = create_complete_profile(single_params, normalize=False)

        if first:
            x = x_tmp
            y = y_tmp * weight
            first = False
        else:
            assert x[0] == x_tmp[0]
            y += y_tmp * weight

    if normalize:
        y /= np.max(y)
        y = apply_background(x, y, params)
        y /= np.max(y)
    if params['noise_std'] > 0:
        noise = np.random.normal(0, params['noise_std'], len(y))
        y = y + noise
    return x, y
