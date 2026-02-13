# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,line-too-long,too-many-lines,too-many-arguments,too-many-locals,too-many-statements
"""
Created on 13 Feb 2026

@author: Ivica Stevanovic
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import warnings


# Constants class
class Const:
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
    a_0_km = 6371.0              # Earth radius, in km
    a_e_km = 9257.0              # Effective Earth radius, in km
    N_s = 341
    epsilon_r = 15.0
    sigma = 0.005
    LOS_EPSILON = 0.00001
    THIRD = 1.0 / 3.0
    
    # Modes
    CONST_MODE__SEARCH = 0
    CONST_MODE__DIFFRACTION = 1
    CONST_MODE__SCATTERING = 2
    
    # Cases
    CASE_1 = 1
    CASE_2 = 2
    
    # Propagation modes
    PROP_MODE__NOT_SET = 0
    PROP_MODE__LOS = 1
    PROP_MODE__DIFFRACTION = 2
    PROP_MODE__SCATTERING = 3
    
    # Polarizations
    POLARIZATION__HORIZONTAL = 0
    POLARIZATION__VERTICAL = 1
    
    Y_pi_99_INDEX = 16
    
    # Return codes
    SUCCESS = 0
    ERROR_VALIDATION__D_KM = 1
    ERROR_VALIDATION__H_1 = 2
    ERROR_VALIDATION__H_2 = 3
    ERROR_VALIDATION__TERM_GEO = 4
    ERROR_VALIDATION__F_MHZ_LOW = 5
    ERROR_VALIDATION__F_MHZ_HIGH = 6
    ERROR_VALIDATION__PERCENT_LOW = 7
    ERROR_VALIDATION__PERCENT_HIGH = 8
    ERROR_VALIDATION__POLARIZATION = 9
    ERROR_HEIGHT_AND_DISTANCE = 10
    WARNING__DFRAC_TROPO_REGION = 20
    
    # P835 related
    RHO_0__M_KG = 7.5
    ERROR_HEIGHT_TOO_SMALL = -1
    ERROR_HEIGHT_TOO_LARGE = -2


@dataclass
class Result:
    A_fs_db: float = 0.0
    A_a_db: float = 0.0
    A_db: float = 0.0
    d_km: float = 0.0
    theta_h1_rad: float = 0.0
    propagation_mode: int = Const.PROP_MODE__NOT_SET
    rtn: int = 0


@dataclass
class Terminal:
    h_r_km: float = 0.0
    theta_rad: float = 0.0
    A_a_db: float = 0.0
    a_km: float = 0.0
    d_r_km: float = 0.0
    phi_rad: float = 0.0
    h_e_km: float = 0.0
    delta_h_km: float = 0.0


@dataclass
class Path:
    d_ML_km: float = 0.0
    d_d_km: float = 0.0
    d_0_km: float = 0.0


@dataclass
class LineOfSightParams:
    # Heights
    z_km: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # Distances
    d_km: float = 0.0           # Path distance between terminals
    r_0_km: float = 0.0         # Direct ray length
    r_12_km: float = 0.0        # Indirect ray length
    D_km: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # Angles
    theta_h1_rad: float = 0.0   # Take-off angle from low terminal to high terminal
    theta_h2_rad: float = 0.0   # Take-off angle from high terminal to low terminal
    theta: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # Misc
    a_a_km: float = 0.0         # Adjusted earth radius
    delta_r_km: float = 0.0     # Ray length path difference
    A_LOS_db: float = 0.0       # Loss due to LOS path


@dataclass
class TropoParams:
    """Troposcatter parameters structure"""
    A_s_db: float = 0.0
    theta_s: float = 0.0
    h_v_km: float = 0.0
    d_s_km: float = 0.0
    d_z_km: float = 0.0
    theta_A: float = 0.0

@dataclass
class SlantPathResult:
    """Result structure for slant path attenuation calculations"""
    angle_rad: float = 0.0        # Arrival angle in radians
    A_gas_db: float = 0.0         # Gaseous attenuation in dB
    a_km: float = 0.0             # Path length in km
    bending_rad: float = 0.0      # Ray bending in radians
    delta_L_km: float = 0.0       # Excess path length in km



"""
bt_loss - Computes basic transmission loss according to Recommendation
ITU-R P.528-5 for aeronautical mobile and radionavigation services.

Description:  Annex 2, Section 3 of Recommendation ITU-R
              P.528-5, "Propagation curves for aeronautical mobile and
              radionavigation services using the VHF, UHF and SHF bands"

Translated from MATLAB to Python
Original MATLAB translation by Ivica Stevanovic (OFCOM CH)
from C++ code by William Kozma Jr (NTIA, USA)
Latest version: 13FEB2026

Usage:
    result = bt_loss(d_km, h_1_meter, h_2_meter, f_mhz, T_pol, p)
    result = bt_loss(d_km, h_1_meter, h_2_meter, f_mhz, T_pol, use_reflection = False)

Input:
    d_km            - Path distance, in km
    h_1_meter       - Height of the low terminal, in meters
    h_2_meter       - Height of the high terminal, in meters
    f_mhz           - Frequency, in MHz
    T_pol           - Polarization (0: horizontal, 1: vertical)
    p               - Time percentage
    use_reflection  - Optional boolean flag, set to False when not including the reflected ray

Output:
    result          - Dictionary containing computed parameters
"""


def bt_loss(d_km: float, h_1_meter: float, h_2_meter: float, f_mhz: float,
            T_pol: int, p: float, use_reflection: bool = True) -> Result:
    """
    Compute basic transmission loss according to ITU-R P.528-5
    
    Parameters:
    -----------
    d_km : float
        Path distance, in km
    h_1_meter : float
        Height of the low terminal, in meters
    h_2_meter : float
        Height of the high terminal, in meters
    f_mhz : float
        Frequency, in MHz
    T_pol : int
        Polarization (0: horizontal, 1: vertical)
    p : float
        Time percentage
    use_reflection : Optional bool (default: True)
        Whether to include the reflected ray
    
    Returns:
    --------
    Result
        Result object containing computed parameters
    """
    # Reset Results
    result = Result()
    
    # Validate inputs
    err = validate_inputs(d_km, h_1_meter, h_2_meter, f_mhz, T_pol, p)
    
    if err != Const.SUCCESS:
        if err == Const.ERROR_HEIGHT_AND_DISTANCE:
            result.rtn = Const.SUCCESS
            return result
        else:
            result.rtn = err
            return result
    
    # Compute terminal geometries
    # Step 1 for low terminal
    terminal_1 = Terminal()
    terminal_1.h_r_km = h_1_meter / 1000.0
    terminal_1 = terminal_geometry(f_mhz, terminal_1)
    
    # Step 1 for high terminal
    terminal_2 = Terminal()
    terminal_2.h_r_km = h_2_meter / 1000.0
    terminal_2 = terminal_geometry(f_mhz, terminal_2)
    
    # Step 2
    path = Path()
    path.d_ML_km = terminal_1.d_r_km + terminal_2.d_r_km  # [Eqn 3-1]
    
    # Smooth earth diffraction line calculations
    # Step 3.1
    d_3_km = path.d_ML_km + 0.5 * (Const.a_e_km**2 / f_mhz)**Const.THIRD  # [Eqn 3-2]
    d_4_km = path.d_ML_km + 1.5 * (Const.a_e_km**2 / f_mhz)**Const.THIRD  # [Eqn 3-3]
    
    # Step 3.2
    A_3_db = smooth_earth_diffraction(terminal_1.d_r_km, terminal_2.d_r_km, 
                                      f_mhz, d_3_km, T_pol)
    A_4_db = smooth_earth_diffraction(terminal_1.d_r_km, terminal_2.d_r_km, 
                                      f_mhz, d_4_km, T_pol)
    
    # Step 3.3
    M_d = (A_4_db - A_3_db) / (d_4_km - d_3_km)  # [Eqn 3-4]
    A_d0 = A_4_db - M_d * d_4_km                  # [Eqn 3-5]
    
    # Step 3.4
    A_dML_db = (M_d * path.d_ML_km) + A_d0      # [Eqn 3-6]
    path.d_d_km = -(A_d0 / M_d)                 # [Eqn 3-7]
    
    K_LOS = 0
    
    # Step 4. If the path is in the Line-of-Sight range, call LOS and then exit
    if path.d_ML_km - d_km > 0.001:
        result.propagation_mode = Const.PROP_MODE__LOS
        result, los_params, K_LOS = line_of_sight(path, terminal_1, terminal_2, f_mhz,
                                                   -A_dML_db, p, d_km, T_pol, 
                                                   use_reflection)
        result.rtn = Const.SUCCESS
        return result
    else:
        # Get K_LOS
        result, los_params, K_LOS = line_of_sight(path, terminal_1, terminal_2, f_mhz,
                                                   -A_dML_db, p, path.d_ML_km - 1, 
                                                   T_pol, use_reflection)
        
        # Step 6. Search past horizon to find crossover point
        rtn, M_d, A_d0, d_crx_km, CASE = transhorizon_search(path, terminal_1, 
                                                              terminal_2, f_mhz, 
                                                              A_dML_db, M_d, A_d0)
        
        # Compute terrain attenuation, A_T_db
        # Step 7.1
        A_d_db = M_d * d_km + A_d0  # [Eqn 3-14]
        
        # Step 7.2
        tropo = troposcatter(terminal_1, terminal_2, d_km, f_mhz)
        
        # Step 7.3
        if d_km < d_crx_km:
            # Always in diffraction if less than d_crx
            A_T_db = A_d_db
            result.propagation_mode = Const.PROP_MODE__DIFFRACTION
        else:
            if CASE == Const.CASE_1:
                # Select the lower loss mode of propagation
                if tropo.A_s_db <= A_d_db:
                    A_T_db = tropo.A_s_db
                    result.propagation_mode = Const.PROP_MODE__SCATTERING
                else:
                    A_T_db = A_d_db
                    result.propagation_mode = Const.PROP_MODE__DIFFRACTION
            else:  # CASE_2
                A_T_db = tropo.A_s_db
                result.propagation_mode = Const.PROP_MODE__SCATTERING
        
        # Compute variability
        # f_theta_h is unity for transhorizon paths
        f_theta_h = 1
        
        # Compute the 50% and p% of the long-term variability distribution
        Y_e_db, _ = long_term_variability(terminal_1.d_r_km, terminal_2.d_r_km, 
                                          d_km, f_mhz, p, f_theta_h, -A_T_db)
        Y_e_50_db, _ = long_term_variability(terminal_1.d_r_km, terminal_2.d_r_km,
                                             d_km, f_mhz, 50, f_theta_h, -A_T_db)
        
        # Compute the 50% and p% of the Nakagami-Rice distribution
        ANGLE = 0.02617993878  # 1.5 deg
        
        if tropo.theta_s >= ANGLE:  # theta_s > 1.5 deg
            K_t_db = 20
        elif tropo.theta_s <= 0.0:
            K_t_db = K_LOS
        else:
            K_t_db = (tropo.theta_s * (20.0 - K_LOS) / ANGLE) + K_LOS
        
        Y_pi_50_db = 0.0  # zero mean
        Y_pi_db = nakagami_rice(K_t_db, p)
        
        # Combine the long-term and Nakagami-Rice distributions
        Y_total_db = combine_distributions(Y_e_50_db, Y_e_db, Y_pi_50_db, 
                                           Y_pi_db, p)
        
        # Atmospheric absorption for transhorizon path
        result_v = slant_path_attenuation(f_mhz / 1000.0, 0, tropo.h_v_km, 
                                          np.pi / 2)
        
        result.A_a_db = (terminal_1.A_a_db + terminal_2.A_a_db + 
                        2 * result_v.A_gas_db)  # [Eqn 3-17]
        
        # Compute free-space loss
        r_fs_km = terminal_1.a_km + terminal_2.a_km + 2 * result_v.a_km  # [Eqn 3-18]
        result.A_fs_db = (20.0 * np.log10(f_mhz) + 20.0 * np.log10(r_fs_km) + 
                         32.45)  # [Eqn 3-19]
        
        result.d_km = d_km
        result.A_db = (result.A_fs_db + result.A_a_db + A_T_db - 
                      Y_total_db)  # [Eqn 3-20]
        
        result.theta_h1_rad = -terminal_1.theta_rad
        result.rtn = rtn
        return result


def validate_inputs(d_km: float, h_1_meter: float, h_2_meter: float, 
                   f_mhz: float, T_pol: int, p: float) -> int:
    """Validate the model input values"""
    if d_km < 0:
        return Const.ERROR_VALIDATION__D_KM
    
    if h_1_meter < 1.5 or h_1_meter > 20000:
        return Const.ERROR_VALIDATION__H_1
    
    if h_2_meter < 1.5 or h_2_meter > 20000:
        return Const.ERROR_VALIDATION__H_2
    
    if h_1_meter > h_2_meter:
        return Const.ERROR_VALIDATION__TERM_GEO
    
    if f_mhz < 100:
        return Const.ERROR_VALIDATION__F_MHZ_LOW
    
    if f_mhz > 30000:
        return Const.ERROR_VALIDATION__F_MHZ_HIGH
    
    if T_pol not in [Const.POLARIZATION__HORIZONTAL, Const.POLARIZATION__VERTICAL]:
        return Const.ERROR_VALIDATION__POLARIZATION
    
    if p < 1:
        return Const.ERROR_VALIDATION__PERCENT_LOW
    
    if p > 99:
        return Const.ERROR_VALIDATION__PERCENT_HIGH
    
    if h_1_meter == h_2_meter and d_km == 0:
        return Const.ERROR_HEIGHT_AND_DISTANCE
    
    return Const.SUCCESS


def terminal_geometry(f_mhz: float, terminal: Terminal) -> Terminal:
    """
    Compute the terminal geometry as described in Annex 2, Section 4 
    of Recommendation ITU-R P.528-5
    """
    theta_tx_rad = 0
    
    result = slant_path_attenuation(f_mhz / 1000.0, 0, terminal.h_r_km, 
                                    np.pi / 2 - theta_tx_rad)
    terminal.theta_rad = np.pi / 2 - result.angle_rad
    terminal.A_a_db = result.A_gas_db
    terminal.a_km = result.a_km
    
    # Compute arc distance
    central_angle = ((np.pi / 2 - result.angle_rad) - theta_tx_rad + 
                    result.bending_rad)
    terminal.d_r_km = Const.a_0_km * central_angle
    
    terminal.phi_rad = terminal.d_r_km / Const.a_e_km  # [Eqn 4-1]
    terminal.h_e_km = (Const.a_e_km / np.cos(terminal.phi_rad) - 
                      Const.a_e_km)  # [Eqn 4-2]
    
    terminal.delta_h_km = terminal.h_r_km - terminal.h_e_km  # [Eqn 4-3]
    
    return terminal


def smooth_earth_diffraction(d_1_km: float, d_2_km: float, f_mhz: float, 
                            d_0_km: float, T_pol: int) -> float:
    """
    Compute the smooth earth diffraction loss as described in 
    Annex 2, Section 10 of Recommendation ITU-R P.528-5
    """
    s = 18000 * Const.sigma / f_mhz
    
    if T_pol == Const.POLARIZATION__HORIZONTAL:
        K = (0.01778 * f_mhz**(-Const.THIRD) * 
             ((Const.epsilon_r - 1)**2 + s**2)**(-0.25))
    else:
        K = (0.01778 * f_mhz**(-Const.THIRD) * 
             ((Const.epsilon_r**2 + s**2) / 
              ((Const.epsilon_r - 1)**2 + s**2)**0.5)**0.5)
    
    B_0 = 1.607
    
    # [Vogler 1964, Equ 2] with C_0 = 1 due to "4/3" Earth assumption
    x_0_km = (B_0 - K) * (f_mhz**Const.THIRD) * d_0_km
    x_1_km = (B_0 - K) * (f_mhz**Const.THIRD) * d_1_km
    x_2_km = (B_0 - K) * (f_mhz**Const.THIRD) * d_2_km
    
    # Compute the distance function for the path
    G_x_db = distance_function(x_0_km)
    
    # Compute the height functions for the two terminals
    F_x1_db = height_function(x_1_km, K)
    F_x2_db = height_function(x_2_km, K)
    
    # [Vogler 1964, Equ 1]
    A_d_db = G_x_db - F_x1_db - F_x2_db - 20.0
    
    return A_d_db


def distance_function(x_km: float) -> float:
    """[Vogler 1964, Equ 13]"""
    G_x_db = 0.05751 * x_km - 10.0 * np.log10(x_km)
    return G_x_db


def height_function(x_km: float, K: float) -> float:
    """Compute height function for diffraction calculations"""
    # [FAA-ES-83-3, Equ 73]
    y_db = 40.0 * np.log10(x_km) - 117.0
    
    # [Vogler 1964, Equ 13]
    G_x_db = distance_function(x_km)
    
    if x_km <= 200.0:
        x_t_km = 450 / (-(np.log10(K))**3)  # [Eqn 109]
        
        # [Eqn 110]
        if x_km >= x_t_km:
            if abs(y_db) < 117:
                F_x_db = y_db
            else:
                F_x_db = -117
        else:
            F_x_db = 20 * np.log10(K) - 15 + (0.000025 * (x_km**2.0) / K)
    elif x_km > 2000.0:
        # [Vogler 1964] F_x ~= G_x for large x
        F_x_db = G_x_db
    else:  # Blend y_db with G_x_db for 200 < x_km < 2000
        # [FAA-ES-83-3, Equ 72] weighting variable
        W = 0.0134 * x_km * np.exp(-0.005 * x_km)
        
        # [FAA-ES-83-3, Equ 75]
        F_x_db = W * y_db + (1.0 - W) * G_x_db
    
    return F_x_db


def line_of_sight(path, terminal_1, terminal_2, f_mhz: float, A_dML_db: float,
                  p: float, d_km: float, T_pol: int, use_reflection: bool
                   ) -> Tuple[Result, LineOfSightParams, float]:
    """
    Compute the total loss in the line-of-sight region.
    
    Parameters:
    -----------
    path : Path
        Struct containing path parameters
    terminal_1 : Terminal
        Struct containing low terminal parameters
    terminal_2 : Terminal
        Struct containing high terminal parameters
    f_mhz : float
        Frequency, in MHz
    A_dML_db : float
        Diffraction loss at d_ML, in dB
    p : float
        Time percentage
    d_km : float
        Path length, in km
    T_pol : int
        Code indicating polarization
        + 0 : POLARIZATION__HORIZONTAL
        + 1 : POLARIZATION__VERTICAL
    use_reflection : bool
        False - do not account for the reflected ray
    Const : Const
        Constants class
    
    Returns:
    --------
    result : Result
        Struct containing P.528 results
    los_params : LineOfSightParams
        Struct containing LOS parameters
    K_LOS : float
        K-value
    """
    
    # 0.2997925 = speed of light, gigameters per sec
    lambda_km = 0.2997925 / f_mhz  # [Eqn 6-1]
    terminate = lambda_km / 1e6
    
    # Determine psi_limit, where you switch from free space to 2-ray model
    # lambda / 2 is the start of the lobe closest to d_ML
    # TODO: remove Cost from the list of arguments
    psi_limit = find_psi_at_delta_r(lambda_km / 2, path, terminal_1, terminal_2, 
                                    terminate)
    
    # "[d_y6_km] is the largest distance at which a free-space value is obtained 
    # in a two-ray model of reflection from a smooth earth with a reflection 
    # coefficient of -1" [ES-83-3, page 44]
    # TODO: remove Cost from the list of arguments
    d_y6_km = find_distance_at_delta_r(lambda_km / 6, path, terminal_1, terminal_2,
                                       terminate)
    
    ############################
    # Determine d_0_km distance  
    # In IF-73, the values for d_0 (d_d in IF-77) were found to be too small 
    # when both antennas are low, so this "heuristic" was developed to fix that
    # [Eqns 8-2 and 8-3]
    if terminal_1.d_r_km >= path.d_d_km or path.d_d_km >= path.d_ML_km:
        if terminal_1.d_r_km > d_y6_km or d_y6_km > path.d_ML_km:
            path.d_0_km = terminal_1.d_r_km
        else:
            path.d_0_km = d_y6_km
    elif path.d_d_km < d_y6_km and d_y6_km < path.d_ML_km:
        path.d_0_km = d_y6_km
    else:
        path.d_0_km = path.d_d_km
    #
    # Determine d_0_km distance
    ############################
    
    ########################
    # Tune d_0_km distance
    # Now that we have d_0, lets carefully walk it forward, 1 meter at a time, 
    # to tune it to as precise as possible without going beyond the LOS region 
    # (ie, beyond d_ML)
    d_temp_km = path.d_0_km
    
    los_result = LineOfSightParams()
    
    while True:
        #TODO: Remove Const from the list of arguments
        psi = find_psi_at_distance(d_temp_km, path, terminal_1, terminal_2)
        
        los_result = ray_optics(terminal_1, terminal_2, psi, los_result)
        
        # If the resulting distance is beyond d_0 OR if we incremented again 
        # we'd be outside of LOS...
        if (los_result.d_km >= path.d_0_km or 
            (d_temp_km + 0.001) >= path.d_ML_km):
            # Use the resulting distance as d_0
            path.d_0_km = los_result.d_km
            break
        
        d_temp_km = d_temp_km + 0.001 
    #
    # Tune d_0_km distance
    #######################
    
    ##########################
    # Compute loss at d_0_km
    #
    # TODO: Remove Const from the list of arguments
    psi_d0 = find_psi_at_distance(path.d_0_km, path, terminal_1, terminal_2)
    
    los_params = LineOfSightParams()
    
    los_params = ray_optics(terminal_1, terminal_2, psi_d0, los_params)
    
    los_params, R_Tg = get_path_loss(psi_d0, path, f_mhz, psi_limit, A_dML_db, 
                                     0, T_pol, los_params, use_reflection)
    
    #
    # Compute loss at d_0_km
    ##########################
    
    # Tune psi for the desired distance
    # TODO: Remove Const from the list of arguments
    psi = find_psi_at_distance(d_km, path, terminal_1, terminal_2)
    
    los_params = ray_optics(terminal_1, terminal_2, psi, los_params)
    
    los_params, R_Tg = get_path_loss(psi, path, f_mhz, psi_limit, A_dML_db,
                                     los_params.A_LOS_db, T_pol, los_params,
                                     use_reflection)
    
    ##################################
    # Compute atmospheric absorption
    #
    # TODO: Remove Const from the list of arguments
    result_slant = slant_path_attenuation(f_mhz / 1000, terminal_1.h_r_km,
                                         terminal_2.h_r_km, 
                                         np.pi / 2 - los_params.theta_h1_rad)
    
    result = Result()
    result.A_a_db = result_slant.A_gas_db
    
    #
    # Compute atmospheric absorption
    ################################
    
    ##########################
    # Compute free-space loss
    #
    
    result.A_fs_db = (20.0 * np.log10(los_params.r_0_km) + 
                      20.0 * np.log10(f_mhz) + 32.45)  # [Eqn 6-4]
    
    #
    # Compute free-space loss
    #########################
    
    #######################
    # Compute variability
    #
    
    # [Eqn 13-1]
    if los_params.theta_h1_rad <= 0.0:
        f_theta_h = 1.0
    elif los_params.theta_h1_rad >= 1.0:
        f_theta_h = 0.0
    else:
        f_theta_h = max(0.5 - (1 / np.pi) * 
                       (np.arctan(20.0 * np.log10(32.0 * los_params.theta_h1_rad))), 0)
    #TODO: Remove const from the list of arguments
    Y_e_db, A_Y = long_term_variability(terminal_1.d_r_km, terminal_2.d_r_km, 
                                        d_km, f_mhz, p, f_theta_h, 
                                        los_params.A_LOS_db)
    Y_e_50_db, A_Y = long_term_variability(terminal_1.d_r_km, terminal_2.d_r_km,
                                           d_km, f_mhz, 50, f_theta_h,
                                           los_params.A_LOS_db)
    
    # [Eqn 13-2]
    if A_Y <= 0.0:
        F_AY = 1.0
    elif A_Y >= 9.0:
        F_AY = 0.1
    else:
        F_AY = (1.1 + (0.9 * np.cos((A_Y / 9.0) * np.pi))) / 2.0
    
    # [Eqn 175]
    if los_params.delta_r_km >= (lambda_km / 2.0):
        F_delta_r = 1.0
    elif los_params.delta_r_km <= lambda_km / 6.0:
        F_delta_r = 0.1
    else:
        F_delta_r = 0.5 * (1.1 - (0.9 * np.cos(((3.0 * np.pi) / lambda_km) * 
                                                (los_params.delta_r_km - 
                                                 (lambda_km / 6.0)))))
    
    R_s = R_Tg * F_delta_r * F_AY  # [Eqn 13-4]
    
    Y_pi_99_db = (10.0 * np.log10(f_mhz * (result_slant.a_km**3)) - 
                  84.26)  # [Eqn 13-5]
    
    K_t = find_k_for_ypi_at_99_percent(Y_pi_99_db)
    
    W_a = 10.0**(K_t / 10.0)      # [Eqn 13-6]
    W_R = R_s**2 + 0.01**2        # [Eqn 13-7]
    W = W_R + W_a                 # [Eqn 13-8]
    
    # [Eqn 13-9]
    if W <= 0.0:
        K_LOS = -40.0
    else:
        K_LOS = 10.0 * np.log10(W)
    
    if K_LOS < -40.0:
        K_LOS = -40.0
    
    Y_pi_50_db = 0.0  # zero mean
    
    Y_pi_db = nakagami_rice(K_LOS, p)
    
    Y_total_db = -combine_distributions(Y_e_50_db, Y_e_db, Y_pi_50_db, 
                                        Y_pi_db, p)
    
    #
    # Compute variability
    #####################
    
    result.d_km = los_params.d_km
    result.A_db = (result.A_fs_db + result.A_a_db - los_params.A_LOS_db + 
                   Y_total_db)
    result.theta_h1_rad = los_params.theta_h1_rad
    
    return result, los_params, K_LOS


def find_psi_at_distance(d_km: float, path, terminal_1, terminal_2) -> float:
    """
    Find the psi angle for a given distance.
    
    Uses iterative binary search to find the psi angle that produces the
    desired distance between terminals.
    
    Parameters:
    -----------
    d_km : float
        Desired distance in km
    path : Path
        Path parameters
    terminal_1 : Terminal
        Low terminal parameters
    terminal_2 : Terminal
        High terminal parameters

    
    Returns:
    --------
    psi : float
        Psi angle in radians
    """
    if d_km == 0:
        psi = np.pi / 2
        return psi
    
    # Initialize to start at mid-point
    psi = np.pi / 2
    delta_psi = -np.pi / 4
    
    while True:
        psi = psi + delta_psi  # new psi
        params_temp = LineOfSightParams()
        # TODO: Remove Const from the list of arguments
        params_temp = ray_optics(terminal_1, terminal_2, psi, params_temp)
        
        d_psi_km = params_temp.d_km
        
        # Compute delta
        if d_psi_km > d_km:
            delta_psi = abs(delta_psi) / 2
        else:
            delta_psi = -abs(delta_psi) / 2
        
        # Get within 1 meter of desired delta_r value
        if abs(d_km - d_psi_km) <= 1e-3 or abs(delta_psi) <= 1e-12:
            break
    
    return psi


def find_psi_at_delta_r(delta_r_km: float, path, terminal_1, terminal_2,
                        terminate: float) -> float:
    """
    Find the psi angle for a given delta_r.
    
    Uses iterative binary search to find the psi angle that produces the
    desired ray path difference (delta_r).
    
    Parameters:
    -----------
    delta_r_km : float
        Desired ray path difference in km
    path : Path
        Path parameters
    terminal_1 : Terminal
        Low terminal parameters
    terminal_2 : Terminal
        High terminal parameters
    terminate : float
        Termination tolerance

    
    Returns:
    --------
    psi : float
        Psi angle in radians
    """
    psi = np.pi / 2
    delta_psi = -np.pi / 4
    
    while True:
        psi = psi + delta_psi
        params_temp = LineOfSightParams()
        
        params_temp = ray_optics(terminal_1, terminal_2, psi, params_temp)
        
        if params_temp.delta_r_km > delta_r_km:
            delta_psi = -abs(delta_psi) / 2
        else:
            delta_psi = abs(delta_psi) / 2
        
        if abs(params_temp.delta_r_km - delta_r_km) <= terminate:
            break
    
    return psi


def find_distance_at_delta_r(delta_r_km: float, path, terminal_1, terminal_2,
                             terminate: float) -> float:
    """
    Find the distance for a given delta_r.
    
    Uses iterative binary search to find the distance that produces the
    desired ray path difference (delta_r).
    
    Parameters:
    -----------
    delta_r_km : float
        Desired ray path difference in km
    path : Path
        Path parameters
    terminal_1 : Terminal
        Low terminal parameters
    terminal_2 : Terminal
        High terminal parameters
    terminate : float
        Termination tolerance

    
    Returns:
    --------
    d_km : float
        Distance in km
    """
    psi = np.pi / 2
    delta_psi = -np.pi / 4
    
    while True:
        psi = psi + delta_psi
        params_temp = LineOfSightParams()
        
        params_temp = ray_optics(terminal_1, terminal_2, psi, params_temp)
        
        if params_temp.delta_r_km > delta_r_km:
            delta_psi = -abs(delta_psi) / 2
        else:
            delta_psi = abs(delta_psi) / 2
        
        if abs(params_temp.delta_r_km - delta_r_km) <= terminate:
            break
    
    d_km = params_temp.d_km
    
    return d_km
    

def ray_optics(terminal_1, terminal_2, psi: float, params):
    """
    Compute the line-of-sight ray optics as described in Annex 2, Section 7
    of Recommendation ITU-R P.528-5, "Propagation curves for aeronautical
    mobile and radionavigation services using the VHF, UHF and SHF bands"
    
    Parameters:
    -----------
    terminal_1 : Terminal
        Structure holding low terminal parameters
    terminal_2 : Terminal
        Structure holding high terminal parameters
    psi : float
        Reflection angle, in radians
    params : LineOfSightParams
        Structure holding resulting parameters

    
    Returns:
    --------
    params : LineOfSightParams
        Updated structure with computed ray optics parameters
    """
    
    z = (Const.a_0_km / Const.a_e_km) - 1       # [Eqn 7-1]
    k_a = 1 / (1 + z * np.cos(psi))             # [Eqn 7-2]
    params.a_a_km = Const.a_0_km * k_a          # [Eqn 7-3]
    
    # [Eqn 7-4]
    delta_h_a1_km = (terminal_1.delta_h_km * (params.a_a_km - Const.a_0_km) / 
                     (Const.a_e_km - Const.a_0_km))
    delta_h_a2_km = (terminal_2.delta_h_km * (params.a_a_km - Const.a_0_km) / 
                     (Const.a_e_km - Const.a_0_km))
    
    H_km = np.zeros(2)
    H_km[0] = terminal_1.h_r_km - delta_h_a1_km    # [Eqn 7-5]
    H_km[1] = terminal_2.h_r_km - delta_h_a2_km    # [Eqn 7-5]
    
    Hprime_km = np.zeros(2)
    
    for i in range(2):
        params.z_km[i] = params.a_a_km + H_km[i]                                    # [Eqn 7-6]
        params.theta[i] = (np.arccos(params.a_a_km * np.cos(psi) / params.z_km[i]) 
                          - psi)                                                     # [Eqn 7-7]
        params.D_km[i] = params.z_km[i] * np.sin(params.theta[i])                  # [Eqn 7-8]
        
        # [Eqn 7-9]
        if psi > 1.56:
            Hprime_km[i] = H_km[i]
        else:
            Hprime_km[i] = params.D_km[i] * np.tan(psi)
    
    delta_z = abs(params.z_km[0] - params.z_km[1])   # [Eqn 7-10]
    
    params.d_km = max(params.a_a_km * (params.theta[0] + params.theta[1]), 0)  # [Eqn 7-11]
    
    alpha = np.arctan((Hprime_km[1] - Hprime_km[0]) / 
                      (params.D_km[0] + params.D_km[1]))                        # [Eqn 7-12]
    params.r_0_km = max(delta_z, (params.D_km[0] + params.D_km[1]) / 
                       np.cos(alpha))                                            # [Eqn 7-13]
    params.r_12_km = (params.D_km[0] + params.D_km[1]) / np.cos(psi)           # [Eqn 7-14]
    
    params.delta_r_km = (4.0 * Hprime_km[0] * Hprime_km[1] / 
                         (params.r_0_km + params.r_12_km))                      # [Eqn 7-15]
    
    params.theta_h1_rad = alpha - params.theta[0]                # [Eqn 7-16]
    params.theta_h2_rad = -(alpha + params.theta[1])             # [Eqn 7-17]
    
    return params


def get_path_loss(psi_rad: float, path, f_mhz: float, psi_limit: float,
                  A_dML_db: float, A_d_0_db: float, T_pol: int, params,
                  use_reflection: bool) -> Tuple:
    """
    Compute the line of sight loss as described in Annex 2, Section 8
    of Recommendation ITU-R P.528-5, "Propagation curves for aeronautical
    mobile and radionavigation services using the VHF, UHF and SHF bands"
    
    Parameters:
    -----------
    psi_rad : float
        Reflection angle, in rad
    path : Path
        Struct containing path parameters
    f_mhz : float
        Frequency, in MHz
    psi_limit : float
        Angular limit separating FS and 2-Ray, in rad
    A_dML_db : float
        Diffraction loss at d_ML, in dB
    A_d_0_db : float
        Loss at d_0, in dB
    T_pol : int
        Code indicating polarization
        + 0 : POLARIZATION__HORIZONTAL
        + 1 : POLARIZATION__VERTICAL
    params : LineOfSightParams
        Line of sight loss params
    use_reflection : bool
        Set to False if the reflected ray is not accounted for

    
    Returns:
    --------
    params : LineOfSightParams
        Updated line of sight loss params
    R_Tg : float
        Reflection parameter
    """
    
    R_g, phi_g = reflection_coefficients(psi_rad, f_mhz, T_pol)
    
    if np.tan(psi_rad) >= 0.1:
        D_v = 1.0
    else:
        r_1 = params.D_km[0] / np.cos(psi_rad)       # [Eqn 8-3]
        r_2 = params.D_km[1] / np.cos(psi_rad)       # [Eqn 8-3]
        R_r = (r_1 * r_2) / params.r_12_km           # [Eqn 8-4]
        
        term_1 = ((2 * R_r * (1 + (np.sin(psi_rad))**2)) / 
                  (params.a_a_km * np.sin(psi_rad)))
        term_2 = (2 * R_r / params.a_a_km)**2
        D_v = (1.0 + term_1 + term_2)**(-0.5)        # [Eqn 8-5]
    
    # Ray-length factor, [Eqn 8-6]
    F_r = min(params.r_0_km / params.r_12_km, 1)
    
    R_Tg = R_g * D_v * F_r                           # [Eqn 8-7]
    
    params.A_LOS_db = 0
    
    if params.d_km > path.d_0_km:
        # [Eqn 8-1]
        params.A_LOS_db = (((params.d_km - path.d_0_km) * 
                           (A_dML_db - A_d_0_db) / 
                           (path.d_ML_km - path.d_0_km)) + A_d_0_db)
    else:
        if not use_reflection:
            return params, R_Tg
        
        lambda_km = 0.2997925 / f_mhz    # [Eqn 8-2]
        
        if psi_rad > psi_limit:
            # Ignore the phase lag; Step 8-2
            params.A_LOS_db = 0
        else:
            # Total phase lag of the ground reflected ray relative to the direct ray
            
            # [Eqn 8-8]
            phi_Tg = (2 * np.pi * params.delta_r_km / lambda_km) + phi_g
            
            # [Eqn 8-9]
            cplx = R_Tg * np.cos(phi_Tg) - 1j * R_Tg * np.sin(phi_Tg)
            
            # [Eqn 8-10]
            W_RL = min(abs(1.0 + cplx), 1.0)
            
            # [Eqn 8-11]
            W_R0 = W_RL**2
            
            # [Eqn 8-12]
            params.A_LOS_db = 10.0 * np.log10(W_R0)
    
    return params, R_Tg

def reflection_coefficients(psi_rad: float, f_mhz: float, T_pol: int
                            ) -> Tuple[float, float]:
    """
    Compute the reflection coefficients.
    
    This function computes the reflection coefficients as described in Annex 2,
    Section 9 of Recommendation ITU-R P.528-5.
    
    Parameters:
    -----------
    psi_rad : float
        Reflection angle, in rad
    f_mhz : float
        Frequency, in MHz
    T_pol : int
        Code indicating polarization
        + 0 : POLARIZATION__HORIZONTAL
        + 1 : POLARIZATION__VERTICAL

    
    Returns:
    --------
    R_g : float
        Magnitude (real part)
    phi_g : float
        Phase angle (imaginary part)
    """
    
    # Handle edge cases for psi_rad
    if psi_rad <= 0.0:
        psi_rad = 0.0
        sin_psi = 0.0
        cos_psi = 1.0
    elif psi_rad >= np.pi / 2:
        psi_rad = np.pi / 2
        sin_psi = 1.0
        cos_psi = 0.0
    else:
        sin_psi = np.sin(psi_rad)
        cos_psi = np.cos(psi_rad)
    
    X = (18000.0 * Const.sigma) / f_mhz              # [Eqn 9-1]
    Y = Const.epsilon_r - (cos_psi)**2               # [Eqn 9-2]
    T = np.sqrt(Y**2 + X**2) + Y                     # [Eqn 9-3]
    P = np.sqrt(T * 0.5)                             # [Eqn 9-4]
    Q = X / (2.0 * P)                                # [Eqn 9-5]
    
    # [Eqn 9-6]
    if T_pol == Const.POLARIZATION__HORIZONTAL:
        B = 1.0 / (P**2 + Q**2)
    else:
        B = ((Const.epsilon_r)**2 + X**2) / (P**2 + Q**2)
    
    # [Eqn 9-7]
    if T_pol == Const.POLARIZATION__HORIZONTAL:
        A = (2.0 * P) / (P**2 + Q**2)
    else:
        A = (2.0 * (P * Const.epsilon_r + Q * X)) / (P**2 + Q**2)
    
    # [Eqn 9-8]
    R_g = np.sqrt((1.0 + (B * sin_psi**2) - (A * sin_psi)) /
                  (1.0 + (B * sin_psi**2) + (A * sin_psi)))
    
    # [Eqn 9-9]
    if T_pol == Const.POLARIZATION__HORIZONTAL:
        alpha = np.arctan2(-Q, sin_psi - P)
    else:
        alpha = np.arctan2((Const.epsilon_r * sin_psi) - Q,
                          Const.epsilon_r * sin_psi - P)
    
    # [Eqn 9-10]
    if T_pol == Const.POLARIZATION__HORIZONTAL:
        beta = np.arctan2(Q, sin_psi + P)
    else:
        beta = np.arctan2((X * sin_psi) + Q,
                         Const.epsilon_r * sin_psi + P)
    
    # [Eqn 9-11]
    phi_g = alpha - beta
    
    return R_g, phi_g

def long_term_variability(d_r1_km: float, d_r2_km: float, d_km: float,
                         f_mhz: float, p: float, f_theta_h: float,
                         A_T: float) -> Tuple[float, float]:
    """
    Compute long-term variability.
    
    Parameters:
    -----------
    d_r1_km : float
        Horizon distance of terminal 1, in km
    d_r2_km : float
        Horizon distance of terminal 2, in km
    d_km : float
        Path distance, in km
    f_mhz : float
        Frequency, in MHz
    p : float
        Time percentage
    f_theta_h : float
        Elevation angle factor
    A_T : float
        Terrain attenuation

    
    Returns:
    --------
    Y_e_db : float
        Long-term variability, in dB
    A_Y : float
        Correction factor
    """
    
    d_qs_km = 65.0 * (100.0 / f_mhz)**Const.THIRD              # [Eqn 14-1]
    d_Lq_km = d_r1_km + d_r2_km                                # [Eqn 14-2]
    d_q_km = d_Lq_km + d_qs_km                                 # [Eqn 14-3]
    
    # [Eqn 14-4]
    if d_km <= d_q_km:
        d_e_km = (130.0 * d_km) / d_q_km
    else:
        d_e_km = 130.0 + d_km - d_q_km
    
    # [Eqns 14-5 and 14-6]
    if f_mhz > 1600.0:
        g_10 = 1.05
        g_90 = 1.05
    else:
        g_10 = (0.21 * np.sin(5.22 * np.log10(f_mhz / 200.0))) + 1.28
        g_90 = (0.18 * np.sin(5.22 * np.log10(f_mhz / 200.0))) + 1.23
    
    # Data Source for Below Consts: Tech Note 101, Vol 2
    # Column 1: Table III.4, Row A* (Page III-50)
    # Column 2: Table III.3, Row A* (Page III-49)
    # Column 3: Table III.5, Row Continental Temperate (Page III-51)
    c_1 = np.array([2.93e-4, 5.25e-4, 1.59e-5])
    c_2 = np.array([3.78e-8, 1.57e-6, 1.56e-11])
    c_3 = np.array([1.02e-7, 4.70e-7, 2.77e-8])
    
    n_1 = np.array([2.00, 1.97, 2.32])
    n_2 = np.array([2.88, 2.31, 4.08])
    n_3 = np.array([3.15, 2.90, 3.25])
    
    f_inf = np.array([3.2, 5.4, 0.0])
    f_m = np.array([8.2, 10.0, 3.9])
    
    # [Y_0(90) Y_0(10) V(50)]
    Z_db = np.zeros(3)
    
    for i in range(3):
        f_2 = f_inf[i] + ((f_m[i] - f_inf[i]) * 
                          np.exp(-c_2[i] * (d_e_km**n_2[i])))
        
        Z_db[i] = ((c_1[i] * (d_e_km**n_1[i]) - f_2) * 
                   np.exp(-c_3[i] * (d_e_km**n_3[i])) + f_2)
    
    if p == 50:
        Y_p_db = Z_db[2]
    elif p > 50:
        z_90 = inverse_complementary_cumulative_distribution_function(90.0 / 100.0)
        z_p = inverse_complementary_cumulative_distribution_function(p / 100.0)
        c_p = z_p / z_90
        
        Y = c_p * (-Z_db[0] * g_90)
        Y_p_db = Y + Z_db[2]
    else:
        if p >= 10:
            z_10 = inverse_complementary_cumulative_distribution_function(10.0 / 100.0)
            z_p = inverse_complementary_cumulative_distribution_function(p / 100.0)
            c_p = z_p / z_10
        else:
            # Source for values p < 10: [15], Table 10, Page 34, Climate 6
            ps = np.array([1, 2, 5, 10])
            c_ps = np.array([1.9507, 1.7166, 1.3265, 1.0000])
            
            dist = distance_upper(data_p(), p)
            c_p = linear_interpolation(ps[dist - 1], c_ps[dist - 1], 
                                      ps[dist], c_ps[dist], p)
        
        Y = c_p * (Z_db[1] * g_10)
        Y_p_db = Y + Z_db[2]
    
    Y_10_db = (Z_db[1] * g_10) + Z_db[2]       # [Eqn 14-20]
    Y_eI_db = f_theta_h * Y_p_db               # [Eqn 14-21]
    Y_eI_10_db = f_theta_h * Y_10_db           # [Eqn 14-22]
    
    # A_Y "is used to prevent available signal powers from exceeding levels 
    # expected for free-space propagation by an unrealistic amount when the 
    # variability about L_b(50) is large and L_b(50) is near its free-space 
    # level" [ES-83-3, p3-4]
    A_YI = (A_T + Y_eI_10_db) - 3.0            # [Eqn 14-23]
    A_Y = max(A_YI, 0)                         # [Eqn 14-24]
    Y_e_db = Y_eI_db - A_Y                     # [Eqn 14-25]
    
    # For percentages less than 10%, do a correction check to,
    # "prevent available signal powers from exceeding levels expected from 
    # free-space levels by unrealistic amounts" [Gierhart 1970]
    if p < 10:
        c_Y = np.array([-5.0, -4.5, -3.7, 0.0])
        
        get_P = data_p()
        dist = distance_upper(get_P, p)
        
        c_Yi = linear_interpolation(get_P[dist - 1], c_Y[dist - 1],
                                    get_P[dist], c_Y[dist], p)
        
        Y_e_db = Y_e_db + A_T
        
        if Y_e_db > -c_Yi:
            Y_e_db = -c_Yi
        
        Y_e_db = Y_e_db - A_T
    
    return Y_e_db, A_Y

def inverse_complementary_cumulative_distribution_function(q: float) -> float:
    """
    Compute the inverse complementary cumulative distribution function.
    
    This function computes the inverse complementary cumulative distribution
    function approximation as described in Recommendation ITU-R P.1057.
    This approximation is sourced from Formula 26.2.23 in Abramowitz & Stegun.
    This approximation has an error of abs(epsilon(p)) < 4.5e-4
    
    Parameters:
    -----------
    q : float
        Probability, 0.0 < q < 1.0
    
    Returns:
    --------
    Q_q : float
        Q(q)^-1
    """
    C_0 = 2.515516
    C_1 = 0.802853
    C_2 = 0.010328
    D_1 = 1.432788
    D_2 = 0.189269
    D_3 = 0.001308
    
    x = q
    if q > 0.5:
        x = 1.0 - x
    
    T_x = np.sqrt(-2.0 * np.log(x))
    
    zeta_x = (((C_2 * T_x + C_1) * T_x + C_0) / 
              (((D_3 * T_x + D_2) * T_x + D_1) * T_x + 1.0))
    
    Q_q = T_x - zeta_x
    
    if q > 0.5:
        Q_q = -Q_q
    
    return Q_q


def linear_interpolation(x1: float, y1: float, x2: float, y2: float, 
                         x: float) -> float:
    """
    Perform linear interpolation between two points.
    
    Parameters:
    -----------
    x1, y1 : float
        Point 1 coordinates
    x2, y2 : float
        Point 2 coordinates
    x : float
        Value of the independent variable
    
    Returns:
    --------
    y : float
        Linearly interpolated value
    """
    y = (y1 * (x2 - x) + y2 * (x - x1)) / (x2 - x1)
    return y


def distance_upper(vector: np.ndarray, value: float) -> int:
    """
    Return an index of the first element in the vector that is greater than value,
    or of the last element if no such element is found.
    
    Parameters:
    -----------
    vector : np.ndarray
        Array to search
    value : float
        Value to compare against
    
    Returns:
    --------
    dist : int
        Index of first element > value, or of last element 
    """
    k = np.where(vector > value)[0]
    if len(k) == 0:
        dist = len(vector) - 1
    else:
        dist = k[0]      
    return dist

def distance_lower(vector: np.ndarray, value: float) -> int:
    """
    Return an index of the first element in the vector that is greater than 
    or equal to value, or of the last element if no such element is found.
    
    This function mimics the behavior of C++'s lower_bound function.
    
    Parameters:
    -----------
    vector : np.ndarray
        Array to search
    value : float
        Value to compare against
    
    Returns:
    --------
    dist : int
        Index of first element >= value 
        or the last element of vector if no such element exists
    
    """
    k = np.where(vector >= value)[0]
    
    if len(k) == 0:
        dist = len(vector) - 1
    else:
        dist = k[0]
    
    return dist

def data_p() -> np.ndarray:
    """
    Return the standard percentage data array.
    
    Returns:
    --------
    P : np.ndarray
        Array of percentage values
    """
    P = np.array([1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 98, 99])
    return P

def data_nakagami_rice_curves() -> np.ndarray:
    """
    Return the Nakagami-Rice distribution curves data.
    
    This is a 17x17 matrix where:
    - Rows correspond to different K values: 
      [-40, -25, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 20]
    - Columns correspond to different percentages:
      [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 98, 99]
    
    Returns:
    --------
    nkr : np.ndarray
        17x17 array of Nakagami-Rice curve values
    """
    
    nkr = np.array([
        # K = -40 distribution
        [
            -0.1417,   -0.1252,   -0.1004,   -0.0784,   -0.0634,
            -0.0515,   -0.0321,   -0.0155,    0.0000,    0.0156,    0.0323,
             0.0518,    0.0639,    0.0791,    0.1016,    0.1271,    0.1441
        ],
        # K = -25 distribution
        [
            -0.7676,   -0.6811,   -0.5497,   -0.4312,   -0.3504,
            -0.2856,   -0.1790,   -0.0870,    0.0000,    0.0878,    0.1828,
             0.2953,    0.3651,    0.4537,    0.5868,    0.7390,    0.8420
        ],
        # K = -20 distribution
        [
            -1.3183,   -1.1738,   -0.9524,   -0.7508,   -0.6121,
            -0.5003,   -0.3151,   -0.1537,    0.0000,    0.1564,    0.3269,
             0.5308,    0.6585,    0.8218,    1.0696,    1.3572,    1.5544
        ],
        # K = -18 distribution
        [
            -1.6263,   -1.4507,   -1.1805,   -0.9332,   -0.7623,
            -0.6240,   -0.3940,   -0.1926,    0.0000,    0.1969,    0.4127,
             0.6722,    0.8355,    1.0453,    1.3660,    1.7417,    2.0014
        ],
        # K = -16 distribution
        [
            -1.9963,   -1.7847,   -1.4573,   -1.1557,   -0.9462,
            -0.7760,   -0.4916,   -0.2410,    0.0000,    0.2478,    0.5209,
             0.8519,    1.0615,    1.3326,    1.7506,    2.2463,    2.5931
        ],
        # K = -14 distribution
        [
            -2.4355,   -2.1829,   -1.7896,   -1.4247,   -1.1695,
            -0.9613,   -0.6113,   -0.3007,    0.0000,    0.3114,    0.6573,
             1.0802,    1.3505,    1.7028,    2.2526,    2.9156,    3.3872
        ],
        # K = -12 distribution
        [
            -2.9491,   -2.6507,   -2.1831,   -1.7455,   -1.4375,
            -1.1846,   -0.7567,   -0.3737,    0.0000,    0.3903,    0.8281,
             1.3698,    1.7198,    2.1808,    2.9119,    3.8143,    4.4714
        ],
        # K = -10 distribution
        [
            -3.5384,   -3.1902,   -2.6407,   -2.1218,   -1.7535,
            -1.4495,   -0.9307,   -0.4619,    0.0000,    0.4874,    1.0404,
             1.7348,    2.1898,    2.7975,    3.7820,    5.0373,    5.9833
        ],
        # K = -8 distribution
        [
            -4.1980,   -3.7974,   -3.1602,   -2.5528,   -2.1180,
            -1.7565,   -1.1345,   -0.5662,    0.0000,    0.6045,    1.2999,
             2.1887,    2.7814,    3.5868,    4.9288,    6.7171,    8.1319
        ],
        # K = -6 distribution
        [
            -4.9132,   -4.4591,   -3.7313,   -3.0306,   -2.5247,
            -2.1011,   -1.3655,   -0.6855,    0.0000,    0.7415,    1.6078,
             2.7374,    3.5059,    4.5714,    6.4060,    8.9732,   11.0973
        ],
        # K = -4 distribution
        [
            -5.6559,   -5.1494,   -4.3315,   -3.5366,   -2.9578,
            -2.4699,   -1.6150,   -0.8154,    0.0000,    0.8935,    1.9530,
             3.3611,    4.3363,    5.7101,    8.1216,   11.5185,   14.2546
        ],
        # K = -2 distribution
        [
            -6.3810,   -5.8252,   -4.9219,   -4.0366,   -3.3871,
            -2.8364,   -1.8638,   -0.9455,    0.0000,    1.0458,    2.2979,
             3.9771,    5.1450,    6.7874,    9.6276,   13.4690,   16.4251
        ],
        # K = 0 distribution
        [
            -7.0247,   -6.4249,   -5.4449,   -4.4782,   -3.7652,
            -3.1580,   -2.0804,   -1.0574,    0.0000,    1.1723,    2.5755,
             4.4471,    5.7363,    7.5266,   10.5553,   14.5401,   17.5511
        ],
        # K = 2 distribution
        [
            -7.5229,   -6.8862,   -5.8424,   -4.8090,   -4.0446,
            -3.3927,   -2.2344,   -1.1347,    0.0000,    1.2535,    2.7446,
             4.7144,    6.0581,    7.9073,   11.0003,   15.0270,   18.0526
        ],
        # K = 4 distribution
        [
            -7.8532,   -7.1880,   -6.0963,   -5.0145,   -4.2145,
            -3.5325,   -2.3227,   -1.1774,    0.0000,    1.2948,    2.8268,
             4.8377,    6.2021,    8.0724,   11.1869,   15.2265,   18.2566
        ],
        # K = 6 distribution
        [
            -8.0435,   -7.3588,   -6.2354,   -5.1234,   -4.3022,
            -3.6032,   -2.3656,   -1.1975,    0.0000,    1.3130,    2.8619,
             4.8888,    6.2610,    8.1388,   11.2607,   15.3047,   18.3361
        ],
        # K = 20 distribution
        [
            -8.2238,   -7.5154,   -6.3565,   -5.2137,   -4.3726,
            -3.6584,   -2.3979,   -1.2121,    0.0000,    1.3255,    2.8855,
             4.9224,    6.2992,    8.1814,   11.3076,   15.3541,   18.3864
        ]
    ])
    
    return nkr


def data_k() -> np.ndarray:
    """
    Return the K values corresponding to Nakagami-Rice distribution curves.
    
    Returns:
    --------
    K : np.ndarray
        Array of K values in dB
    """
    K = np.array([-40, -25, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 
                  0, 2, 4, 6, 20])
    return K

def find_k_for_ypi_at_99_percent(Y_pi_99_db: float) -> float:
    """
    Return the K-value of the Nakagami-Rice distribution for the given 
    value of Y_pi(99).
    
    This function searches through the Nakagami-Rice distribution curves
    and interpolates to find the K value corresponding to a given Y_pi at
    99th percentile.
    
    Parameters:
    -----------
    Y_pi_99_db : float
        Y_pi(99), in dB
    Const : Const
        Constants class (needs Y_pi_99_INDEX attribute)
    
    Returns:
    --------
    K : float
        K-value in dB
    """
    
    get_nakagami_rice_curves = data_nakagami_rice_curves()
    get_K = data_k()
    
    # Is Y_pi_99_db smaller than the smallest value in the distribution data
    # Note: MATLAB index (1, Const.Y_pi_99_INDEX+1) → Python [0, Const.Y_pi_99_INDEX]
    if Y_pi_99_db < get_nakagami_rice_curves[0, Const.Y_pi_99_INDEX]:
        K = get_K[0]
        return K
    
    # Search the distribution data and interpolate to find K (dependent variable)
    for i in range(len(get_K)):
        if (Y_pi_99_db - get_nakagami_rice_curves[i, Const.Y_pi_99_INDEX]) < 0:
            # Linear interpolation formula
            K = ((get_K[i] * (Y_pi_99_db - get_nakagami_rice_curves[i - 1, Const.Y_pi_99_INDEX]) - 
                  get_K[i - 1] * (Y_pi_99_db - get_nakagami_rice_curves[i, Const.Y_pi_99_INDEX])) / 
                 (get_nakagami_rice_curves[i, Const.Y_pi_99_INDEX] - 
                  get_nakagami_rice_curves[i - 1, Const.Y_pi_99_INDEX]))
            return K
    
    # No match. Y_pi_99_db is greater than the data contains. Return largest K
    K = get_K[-1]
    return K


def nakagami_rice(K: float, p: float) -> float:
    """
    Compute the value of the Nakagami-Rice distribution for K and p%.
    
    This function uses bilinear interpolation on the Nakagami-Rice distribution
    curves to find the variability value for given K and percentage values.
    
    Parameters:
    -----------
    K : float
        K-value in dB
    p : float
        Time percentage
    Const : Const
        Constants class
    
    Returns:
    --------
    Y_pi_db : float
        Variability, in dB
    """
    
    get_nakagami_rice_curves = data_nakagami_rice_curves()
    get_P = data_p()
    get_K = data_k()
    
    # Find lower bound indices
    d_K = distance_lower(get_K, K)
    d_p = distance_lower(get_P, p)
    
    # Debug output (commented out, uncomment if needed)
    # print(f'Python: K = {K:.6f}, d_K = {d_K}')
    # print(f'Python: p = {p:.6f}, d_p = {d_p}')
    
    if d_K == 0:  # K <= -40
        if d_p == 0:
            Y_pi_db = get_nakagami_rice_curves[0, 0]
            return Y_pi_db
        else:
            Y_pi_db = linear_interpolation(
                get_P[d_p], get_nakagami_rice_curves[0, d_p],
                get_P[d_p - 1], get_nakagami_rice_curves[0, d_p - 1],
                p
            )
            return Y_pi_db
    
    elif d_K == len(get_K):  # K > 20
        if d_p == 0:
            Y_pi_db = get_nakagami_rice_curves[d_K - 1, 0]
            return Y_pi_db
        else:
            Y_pi_db = linear_interpolation(
                get_P[d_p], get_nakagami_rice_curves[d_K - 1, d_p],
                get_P[d_p - 1], get_nakagami_rice_curves[d_K - 1, d_p - 1],
                p
            )
            return Y_pi_db
    
    else:  # K is between -40 and 20
        if d_p == 0:
            Y_pi_db = linear_interpolation(
                get_K[d_K], get_nakagami_rice_curves[d_K, 0],
                get_K[d_K - 1], get_nakagami_rice_curves[d_K - 1, 0],
                K
            )
            return Y_pi_db
        else:
            # Interpolate between K's at constant p first
            v1 = linear_interpolation(
                get_K[d_K], get_nakagami_rice_curves[d_K, d_p],
                get_K[d_K - 1], get_nakagami_rice_curves[d_K - 1, d_p ],
                K
            )
            
            v2 = linear_interpolation(
                get_K[d_K], get_nakagami_rice_curves[d_K, d_p - 1],
                get_K[d_K - 1], get_nakagami_rice_curves[d_K - 1, d_p - 1],
                K
            )
            
            Y_pi_db = linear_interpolation(
                get_P[d_p], v1,
                get_P[d_p - 1], v2,
                p
            )
            return Y_pi_db


def transhorizon_search(path, terminal_1, terminal_2, f_mhz: float,
                       A_dML_db: float, M_d: float, A_d0: float
                       ) -> Tuple[int, float, float, float, int]:
    """
    Compute Step 6 in Annex 2, Section 3 of Recommendation ITU-R P.528-5,
    "Propagation curves for aeronautical mobile and radionavigation services
    using the VHF, UHF and SHF bands"
    
    This function searches past the horizon to find the crossover point
    between diffraction and troposcatter propagation modes.
    
    Parameters:
    -----------
    path : Path
        Structure containing parameters dealing with the propagation path
    terminal_1 : Terminal
        Structure containing parameters dealing with the geometry of the low terminal
    terminal_2 : Terminal
        Structure containing parameters dealing with the geometry of the high terminal
    f_mhz : float
        Frequency, in MHz
    A_dML_db : float
        Diffraction loss at d_ML, in dB
    M_d : float
        Slope of the diffraction line
    A_d0 : float
        Intercept of the diffraction line
    Const : Const
        Constants class
    
    Returns:
    --------
    rtn : int
        SUCCESS or warning code
    M_d : float
        Updated slope of the diffraction line
    A_d0 : float
        Updated intercept of the diffraction line
    d_crx_km : float
        Final search distance, in km
    CASE : int
        Case as defined in Step 6.5
    """
    
    CASE = Const.CONST_MODE__SEARCH
    k = 0
    
    tropo = TropoParams()
    
    # Step 6.1. Initialize search parameters
    d_search_km = np.zeros(2)
    d_search_km[0] = path.d_ML_km + 3  # d', [Eqn 3-8]
    d_search_km[1] = path.d_ML_km + 2  # d", [Eqn 3-9]
    
    A_s_db = np.zeros(2)
    M_s = 0
    
    SEARCH_LIMIT = 100  # 100 km beyond starting point
    
    for i_search in range(SEARCH_LIMIT):
        
        A_s_db[1] = A_s_db[0]
        
        # Step 6.2
        tropo = troposcatter(terminal_1, terminal_2, d_search_km[0], f_mhz)
        A_s_db[0] = tropo.A_s_db
        
        # If loss is less than 20 dB, the result is not within valid part of model
        if tropo.A_s_db < 20.0:
            d_search_km[1] = d_search_km[0]
            d_search_km[0] = d_search_km[0] + 1
            continue
        
        k = k + 1
        if k <= 1:  # Need two points to draw a line and we don't have them both yet
            d_search_km[1] = d_search_km[0]
            d_search_km[0] = d_search_km[0] + 1
            continue
        
        # Step 6.3
        M_s = ((A_s_db[0] - A_s_db[1]) / 
               (d_search_km[0] - d_search_km[1]))  # [Eqn 3-10]
        
        if M_s <= M_d:
            
            d_crx_km = d_search_km[0]
            
            # Step 6.6
            A_d_db = M_d * d_search_km[1] + A_d0  # [Eqn 3-11]
            
            if A_s_db[1] >= A_d_db:
                CASE = Const.CASE_1
            else:
                # Adjust the diffraction line to the troposcatter model
                M_d = ((A_s_db[1] - A_dML_db) / 
                       (d_search_km[1] - path.d_ML_km))  # [Eqn 3-12]
                A_d0 = A_s_db[1] - (M_d * d_search_km[1])  # [Eqn 3-13]
                
                CASE = Const.CASE_2
            
            rtn = Const.SUCCESS
            return rtn, M_d, A_d0, d_crx_km, CASE
        
        d_search_km[1] = d_search_km[0]
        d_search_km[0] = d_search_km[0] + 1
    
    # M_s was always greater than M_d. Default to diffraction-only transhorizon model
    CASE = Const.CONST_MODE__DIFFRACTION
    d_crx_km = d_search_km[1]
    
    rtn = Const.WARNING__DFRAC_TROPO_REGION
    return rtn, M_d, A_d0, d_crx_km, CASE


def troposcatter(terminal_1, terminal_2, d_km: float, f_mhz: float
                ) -> TropoParams:
    """
    Compute the Troposcatter loss as described in Annex 2, Section 11 of
    Recommendation ITU-R P.528-5, "Propagation curves for aeronautical mobile
    and radionavigation services using the VHF, UHF and SHF bands"
    
    Parameters:
    -----------
    terminal_1 : Terminal
        Struct containing low terminal parameters
    terminal_2 : Terminal
        Struct containing high terminal parameters
    d_km : float
        Path distance, in km
    f_mhz : float
        Frequency, in MHz

    
    Returns:
    --------
    tropo : TropoParams
        Struct containing resulting parameters
    """
    
    tropo = TropoParams()
    
    tropo.d_s_km = d_km - terminal_1.d_r_km - terminal_2.d_r_km  # [Eqn 11-2]
    
    if tropo.d_s_km <= 0.0:
        tropo.d_z_km = 0.0
        tropo.A_s_db = 0.0
        tropo.d_s_km = 0.0
        tropo.h_v_km = 0.0
        tropo.theta_s = 0.0
        tropo.theta_A = 0.0
    else:
        ###################################
        # Compute the geometric parameters
        #
        
        tropo.d_z_km = 0.5 * tropo.d_s_km  # [Eqn 11-6]
        
        A_m = 1 / Const.a_0_km  # [Eqn 11-7]
        dN = A_m - (1.0 / Const.a_e_km)  # [Eqn 11-8]
        gamma_e_km = (Const.N_s * 1e-6) / dN  # [Eqn 11-9]
        
        z_a_km = (1.0 / (2 * Const.a_e_km) * 
                 ((tropo.d_z_km / 2)**2))  # [Eqn 11-10]
        z_b_km = (1.0 / (2 * Const.a_e_km) * 
                 (tropo.d_z_km**2))  # [Eqn 11-11]
        
        Q_o = A_m - dN  # [Eqn 11-12]
        
        Q_a = A_m - dN / np.exp(min(35.0, z_a_km / gamma_e_km))  # [Eqn 11-13]
        Q_b = A_m - dN / np.exp(min(35.0, z_b_km / gamma_e_km))  # [Eqn 11-13]
        
        Z_a_km = ((7.0 * Q_o + 6.0 * Q_a - Q_b) * 
                 ((tropo.d_z_km**2) / 96.0))  # [Eqn 11-14]
        Z_b_km = ((Q_o + 2.0 * Q_a) * 
                 ((tropo.d_z_km**2) / 6.0))  # [Eqn 11-15]
        
        Q_A = A_m - dN / np.exp(min(35.0, Z_a_km / gamma_e_km))  # [Eqn 11-16]
        Q_B = A_m - dN / np.exp(min(35.0, Z_b_km / gamma_e_km))  # [Eqn 11-16]
        
        tropo.h_v_km = ((Q_o + 2.0 * Q_A) * 
                       ((tropo.d_z_km**2) / 6.0))  # [Eqn 11-17]
        
        tropo.theta_A = ((Q_o + 4.0 * Q_A + Q_B) * 
                        tropo.d_z_km / 6.0)  # [Eqn 11-18]
        
        tropo.theta_s = 2 * tropo.theta_A  # [Eqn 11-19]
        
        #
        # Compute the geometric parameters
        ###################################
        
        #########################################
        # Compute the scattering efficiency term
        #
        epsilon_1 = (5.67e-6 * (Const.N_s**2) - 
                    0.00232 * Const.N_s + 0.031)  # [Eqn 11-20]
        epsilon_2 = (0.0002 * (Const.N_s**2) - 
                    0.06 * Const.N_s + 6.6)  # [Eqn 11-21]
        
        gamma = (0.1424 * (1.0 + epsilon_1 / 
                np.exp(min(35.0, (tropo.h_v_km / 4.0)**6))))  # [Eqn 11-22]
        
        S_e_db = (83.1 - epsilon_2 / (1.0 + 0.07716 * (tropo.h_v_km**2)) + 
                 20 * np.log10((0.1424 / gamma)**2 * 
                              np.exp(gamma * tropo.h_v_km)))  # [Eqn 11-23]
        
        #
        # Compute the scattering efficiency term
        #########################################
        
        #####################################
        # Compute the scattering volume term
        #
        
        X_A1_km2 = ((terminal_1.h_e_km)**2 + 
                   4.0 * (Const.a_e_km + terminal_1.h_e_km) * Const.a_e_km * 
                   (np.sin(terminal_1.d_r_km / (Const.a_e_km * 2))**2))  # [Eqn 11-24]
        X_A2_km2 = ((terminal_2.h_e_km)**2 + 
                   4.0 * (Const.a_e_km + terminal_2.h_e_km) * Const.a_e_km * 
                   (np.sin(terminal_2.d_r_km / (Const.a_e_km * 2))**2))  # [Eqn 11-24]
        
        ell_1_km = np.sqrt(X_A1_km2) + tropo.d_z_km  # [Eqn 11-25]
        ell_2_km = np.sqrt(X_A2_km2) + tropo.d_z_km  # [Eqn 11-25]
        ell_km = ell_1_km + ell_2_km  # [Eqn 11-26]
        
        s = (ell_1_km - ell_2_km) / ell_km  # [Eqn 11-27]
        eta = gamma * tropo.theta_s * ell_km / 2  # [Eqn 11-28]
        
        kappa = f_mhz / 0.0477  # [Eqn 11-29]
        
        rho_1_km = 2.0 * kappa * tropo.theta_s * terminal_1.h_e_km  # [Eqn 11-30]
        rho_2_km = 2.0 * kappa * tropo.theta_s * terminal_2.h_e_km  # [Eqn 11-30]
        
        SQRT2 = np.sqrt(2)
        
        A = (1 - s**2)**2  # [Eqn 11-36]
        
        X_v1 = (1 + s)**2 * eta  # [Eqn 11-32]
        X_v2 = (1 - s)**2 * eta  # [Eqn 11-33]
        
        q_1 = X_v1**2 + rho_1_km**2  # [Eqn 11-34]
        q_2 = X_v2**2 + rho_2_km**2  # [Eqn 11-35]
        
        # [Eqn 11-37]
        B_s = (6 + 8 * s**2 +
               8 * (1.0 - s) * X_v1**2 * rho_1_km**2 / (q_1**2) +
               8 * (1.0 + s) * X_v2**2 * rho_2_km**2 / (q_2**2) +
               2 * (1.0 - s**2) * (1 + 2 * X_v1**2 / q_1) * 
               (1 + 2 * X_v2**2 / q_2))
        
        # [Eqn 11-38]
        C_s = (12 *
               ((rho_1_km + SQRT2) / rho_1_km)**2 *
               ((rho_2_km + SQRT2) / rho_2_km)**2 *
               (rho_1_km + rho_2_km) / (rho_1_km + rho_2_km + 2 * SQRT2))
        
        temp = ((A * eta**2 + B_s * eta) * q_1 * q_2 / 
               ((rho_1_km**2) * (rho_2_km**2)))
        
        S_v_db = 10 * np.log10(temp + C_s)
        
        #
        # Compute the scattering volume term
        #####################################
        
        tropo.A_s_db = (S_e_db + S_v_db + 
                       10.0 * np.log10(kappa * (tropo.theta_s**3) / ell_km))
    
    return tropo

def combine_distributions(A_M: float, A_p: float, B_M: float, B_p: float,
                         p: float) -> float:
    """
    Combine two distributions A and B, returning the resulting percentile.
    
    Parameters:
    -----------
    A_M : float
        Mean of distribution A
    A_p : float
        p% of distribution A
    B_M : float
        Mean of distribution B
    B_p : float
        p% of distribution B
    p : float
        Percentage
    
    Returns:
    --------
    C_p : float
        p% of resulting distribution C
    """
    
    C_M = A_M + B_M
    
    Y_1 = A_p - A_M
    Y_2 = B_p - B_M
    
    Y_3 = np.sqrt((Y_1**2) + (Y_2**2))
    
    if p < 50:
        C_p = C_M + Y_3
    else:
        C_p = C_M - Y_3
    
    return C_p


###############################################################################
# ITU-R P.835 - Reference Standard Atmospheres
###############################################################################

def global_temperature(h_km: float) -> float:
    """
    The mean annual global reference atmospheric temperature, in Kelvin.
    
    Parameters:
    -----------
    h_km : float
        Geometric height, in km

    
    Returns:
    --------
    T_kelvin : float
        Temperature, in Kelvin. Or error code (negative number).
    """
    
    if h_km < 0:
        T_kelvin = Const.ERROR_HEIGHT_TOO_SMALL
        return T_kelvin
    
    if h_km > 100:
        T_kelvin = Const.ERROR_HEIGHT_TOO_LARGE
        return T_kelvin
    
    if h_km < 86:
        h_prime_km = convert_to_geopotential_height(h_km)
        T_kelvin = global_temperature_regime1(h_prime_km)
        return T_kelvin
    else:
        T_kelvin = global_temperature_regime2(h_km)
        return T_kelvin


def global_temperature_regime1(h_prime_km: float) -> float:
    """
    The mean annual global reference atmospheric temperature, in Kelvin,
    for the first height regime. See Equations (2a-g).
    
    Parameters:
    -----------
    h_prime_km : float
        Geopotential height, in km'
    Const : Const
        Constants class
    
    Returns:
    --------
    T_kelvin : float
        Temperature, in Kelvin. Or error code (negative number).
    """
    
    if h_prime_km < 0:
        T_kelvin = Const.ERROR_HEIGHT_TOO_SMALL
        return T_kelvin
    elif h_prime_km <= 11:
        T_kelvin = 288.15 - 6.5 * h_prime_km
        return T_kelvin
    elif h_prime_km <= 20:
        T_kelvin = 216.65
        return T_kelvin
    elif h_prime_km <= 32:
        T_kelvin = 216.65 + (h_prime_km - 20)
        return T_kelvin
    elif h_prime_km <= 47:
        T_kelvin = 228.65 + 2.8 * (h_prime_km - 32)
        return T_kelvin
    elif h_prime_km <= 51:
        T_kelvin = 270.65
        return T_kelvin
    elif h_prime_km <= 71:
        T_kelvin = 270.65 - 2.8 * (h_prime_km - 51)
        return T_kelvin
    elif h_prime_km <= 84.852:
        T_kelvin = 214.65 - 2.0 * (h_prime_km - 71)
        return T_kelvin
    else:
        T_kelvin = Const.ERROR_HEIGHT_TOO_LARGE
        return T_kelvin


def global_temperature_regime2(h_km: float) -> float:
    """
    The mean annual global reference atmospheric temperature, in Kelvin,
    for the second height regime. See Equations (4a-b).
    
    Parameters:
    -----------
    h_km : float
        Geometric height, in km
    Const : Const
        Constants class
    
    Returns:
    --------
    T_kelvin : float
        Temperature, in Kelvin. Or error code (negative number).
    """
    
    if h_km < 86:
        T_kelvin = Const.ERROR_HEIGHT_TOO_SMALL
        return T_kelvin
    elif h_km <= 91:
        T_kelvin = 186.8673
        return T_kelvin
    elif h_km <= 100:
        T_kelvin = (263.1905 - 76.3232 * 
                   np.sqrt(1 - ((h_km - 91) / 19.9429)**2))
        return T_kelvin
    else:
        T_kelvin = Const.ERROR_HEIGHT_TOO_LARGE
        return T_kelvin


def global_pressure(h_km: float) -> float:
    """
    The mean annual global reference atmospheric pressure, in hPa.
    
    Parameters:
    -----------
    h_km : float
        Geometric height, in km
    Const : Const
        Constants class
    
    Returns:
    --------
    p_hPa : float
        Dry air pressure, in hPa. Or error code (negative number).
    """
    
    if h_km < 0:
        p_hPa = Const.ERROR_HEIGHT_TOO_SMALL
        return p_hPa
    
    if h_km > 100:
        p_hPa = Const.ERROR_HEIGHT_TOO_LARGE
        return p_hPa
    
    if h_km < 86:
        h_prime_km = convert_to_geopotential_height(h_km)
        p_hPa = global_pressure_regime1(h_prime_km)
        return p_hPa
    else:
        p_hPa = global_pressure_regime2(h_km)
        return p_hPa


def global_pressure_regime1(h_prime_km: float) -> float:
    """
    The mean annual global reference atmospheric pressure, in hPa,
    for the first height regime. See Equations (3a-g).
    
    Parameters:
    -----------
    h_prime_km : float
        Geopotential height, in km'
    Const : Const
        Constants class
    
    Returns:
    --------
    p_hPa : float
        Dry air pressure, in hPa. Or error code (negative number).
    """
    
    if h_prime_km < 0:
        p_hPa = Const.ERROR_HEIGHT_TOO_SMALL
        return p_hPa
    elif h_prime_km <= 11:
        p_hPa = 1013.25 * (288.15 / (288.15 - 6.5 * h_prime_km))**(-34.1632 / 6.5)
        return p_hPa
    elif h_prime_km <= 20:
        p_hPa = 226.3226 * np.exp(-34.1632 * (h_prime_km - 11) / 216.65)
        return p_hPa
    elif h_prime_km <= 32:
        p_hPa = 54.74980 * (216.65 / (216.65 + (h_prime_km - 20)))**34.1632
        return p_hPa
    elif h_prime_km <= 47:
        p_hPa = (8.680422 * (228.65 / (228.65 + 2.8 * (h_prime_km - 32)))**
                (34.1632 / 2.8))
        return p_hPa
    elif h_prime_km <= 51:
        p_hPa = 1.109106 * np.exp(-34.1632 * (h_prime_km - 47) / 270.65)
        return p_hPa
    elif h_prime_km <= 71:
        p_hPa = (0.6694167 * (270.65 / (270.65 - 2.8 * (h_prime_km - 51)))**
                (-34.1632 / 2.8))
        return p_hPa
    elif h_prime_km <= 84.852:
        p_hPa = (0.03956649 * (214.65 / (214.65 - 2.0 * (h_prime_km - 71)))**
                (-34.1632 / 2.0))
        return p_hPa
    else:
        p_hPa = Const.ERROR_HEIGHT_TOO_LARGE
        return p_hPa


def global_pressure_regime2(h_km: float) -> float:
    """
    The mean annual global reference atmospheric pressure, in hPa,
    for the second height regime. See Equation (5).
    
    Parameters:
    -----------
    h_km : float
        Geometric height, in km
    Const : Const
        Constants class
    
    Returns:
    --------
    p_hPa : float
        Dry air pressure, in hPa. Or error code (negative number).
    """
    
    if h_km < 86:
        p_hPa = Const.ERROR_HEIGHT_TOO_SMALL
        return p_hPa
    
    if h_km > 100:
        p_hPa = Const.ERROR_HEIGHT_TOO_LARGE
        return p_hPa
    
    a_0 = 95.571899
    a_1 = -4.011801
    a_2 = 6.424731e-2
    a_3 = -4.789660e-4
    a_4 = 1.340543e-6
    
    p_hPa = np.exp(a_0 + a_1 * h_km + a_2 * (h_km**2) + 
                   a_3 * (h_km**3) + a_4 * (h_km**4))
    
    return p_hPa


def global_water_vapour_density(h_km: float, rho_0: float) -> float:
    """
    The mean annual global reference atmospheric water vapour density,
    in g/m^3. See Equation (6).
    
    Parameters:
    -----------
    h_km : float
        Geometric height, in km
    rho_0 : float
        Ground-level water vapour density, in g/m^3
    Const : Const
        Constants class
    
    Returns:
    --------
    rho : float
        Water vapour density, in g/m^3. Or error code (negative number).
    """
    
    if h_km < 0:
        rho = Const.ERROR_HEIGHT_TOO_SMALL
        return rho
    
    if h_km > 100:
        rho = Const.ERROR_HEIGHT_TOO_LARGE
        return rho
    
    h_0_km = 2  # scale height
    
    rho = rho_0 * np.exp(-h_km / h_0_km)
    
    return rho


def global_water_vapour_pressure(h_km: float, rho_0: float) -> float:
    """
    The mean annual global reference atmospheric water vapour pressure,
    in hPa.
    
    Parameters:
    -----------
    h_km : float
        Geometric height, in km
    rho_0 : float
        Ground-level water vapour density, in g/m^3
    Const : Const
        Constants class
    
    Returns:
    --------
    e_hPa : float
        Water vapour pressure, e(h), in hPa. Or error code (negative number).
    """
    
    if h_km < 0:
        e_hPa = Const.ERROR_HEIGHT_TOO_SMALL
        return e_hPa
    
    if h_km > 100:
        e_hPa = Const.ERROR_HEIGHT_TOO_LARGE
        return e_hPa
    
    rho = global_water_vapour_density(h_km, rho_0)
    
    if h_km < 86:
        # Convert to geopotential height
        h_prime_km = convert_to_geopotential_height(h_km)
        T_kelvin = global_temperature_regime1(h_prime_km)
    else:
        T_kelvin = global_temperature_regime2(h_km)
    
    e_hPa = water_vapour_density_to_pressure(rho, T_kelvin)
    
    return e_hPa


def convert_to_geopotential_height(h_km: float) -> float:
    """
    Convert from geometric height, in km, to geopotential height, in km'.
    See Equation (1a).
    
    Parameters:
    -----------
    h_km : float
        Geometric height, in km
    
    Returns:
    --------
    k_prime_km : float
        Geopotential height, in km'
    """
    
    k_prime_km = (6356.766 * h_km) / (6356.766 + h_km)
    
    return k_prime_km


def convert_to_geometric_height(h_prime_km: float) -> float:
    """
    Convert from geopotential height, in km', to geometric height, in km.
    See Equation (1b).
    
    Parameters:
    -----------
    h_prime_km : float
        Geopotential height, in km'
    
    Returns:
    --------
    k_km : float
        Geometric height, in km
    """
    
    k_km = (6356.766 * h_prime_km) / (6356.766 - h_prime_km)
    
    return k_km


def water_vapour_density_to_pressure(rho: float, T_kelvin: float) -> float:
    """
    Convert water vapour density, in g/m^3, to water vapour pressure, in hPa.
    See Equation (8).
    
    Parameters:
    -----------
    rho : float
        Water vapour density, rho(h), in g/m^3
    T_kelvin : float
        Temperature, T(h), in Kelvin
    
    Returns:
    --------
    e_hPa : float
        Water vapour pressure, e(h), in hPa
    """
    
    e_hPa = (rho * T_kelvin) / 216.7
    
    return e_hPa

def slant_path_attenuation(f_ghz: float, h_1_km: float, h_2_km: float,
                           beta_1_rad: float) -> SlantPathResult:
    """
    Calculate the slant path attenuation due to atmospheric gases.
    
    This is the main function for ITU-R P.676 slant path calculations.
    It handles both positive and negative elevation angles.
    
    Parameters:
    -----------
    f_ghz : float
        Frequency, in GHz
    h_1_km : float
        Height of the low terminal, in km
    h_2_km : float
        Height of the high terminal, in km
    beta_1_rad : float
        Elevation angle (from zenith), in rad
    Const : Const
        Constants class
    
    Returns:
    --------
    result : SlantPathResult
        Structure containing computed parameters
    """
    
    if beta_1_rad > np.pi / 2:
        # Negative elevation angle
        # Find h_G and then trace in each direction
        # See Section 2.2.2
        
        # Compute refractive index at h_1
        p_hPa = global_pressure(h_1_km)
        T_kelvin = global_temperature(h_1_km)
        e_hPa = global_wet_pressure(h_1_km)
        
        n_1 = refractive_index(p_hPa, T_kelvin, e_hPa)
        
        # Set initial h_G at mid-point between h_1 and surface of the earth
        # Then binary search to converge
        h_G_km = h_1_km
        delta = h_1_km / 2
        diff = 100
        
        while True:
            if diff > 0:
                h_G_km = h_G_km - delta
            else:
                h_G_km = h_G_km + delta
            delta = delta / 2
            
            p_hPa = global_pressure(h_G_km)
            T_kelvin = global_temperature(h_G_km)
            e_hPa = global_wet_pressure(h_G_km)
            
            n_G = refractive_index(p_hPa, T_kelvin, e_hPa)
            
            grazing_term = n_G * (Const.a_0_km + h_G_km)
            start_term = n_1 * (Const.a_0_km + h_1_km) * np.sin(beta_1_rad)
            
            diff = grazing_term - start_term
            if abs(diff) <= 0.001:
                break
        
        # Converged on h_G. Now call RayTrace in both directions with grazing angle
        beta_graze_rad = np.pi / 2
        result_1 = ray_trace(f_ghz, h_G_km, h_1_km, beta_graze_rad)
        result_2 = ray_trace(f_ghz, h_G_km, h_2_km, beta_graze_rad)
        
        result = SlantPathResult()
        result.angle_rad = result_2.angle_rad
        result.A_gas_db = result_1.A_gas_db + result_2.A_gas_db
        result.a_km = result_1.a_km + result_2.a_km
        result.bending_rad = result_1.bending_rad + result_2.bending_rad
        result.delta_L_km = result_1.delta_L_km + result_2.delta_L_km
    else:
        result = ray_trace(f_ghz, h_1_km, h_2_km, beta_1_rad)
    
    return result


def refractive_index(p_hPa: float, T_kelvin: float, e_hPa: float) -> float:
    """
    Compute the refractive index.
    
    Parameters:
    -----------
    p_hPa : float
        Dry pressure, in hPa
    T_kelvin : float
        Temperature, in Kelvin
    e_hPa : float
        Water vapour pressure, in hPa
    
    Returns:
    --------
    n : float
        Refractive index
    """
    
    # Dry term of refractivity
    N_dry = 77.6 * p_hPa / T_kelvin
    
    # Wet term of refractivity
    N_wet = 72 * e_hPa / T_kelvin + 3.75e5 * e_hPa / (T_kelvin**2)
    
    N = N_dry + N_wet
    
    n = 1 + N * (10**(-6))
    
    return n


def ray_trace(f_ghz: float, h_1_km: float, h_2_km: float, beta_1_rad: float
             ) -> SlantPathResult:
    """
    Trace the ray from terminal h_1 to terminal h_2 and compute results
    such as atmospheric absorption loss and ray path length.
    
    Parameters:
    -----------
    f_ghz : float
        Frequency, in GHz
    h_1_km : float
        Height of the low terminal, in km
    h_2_km : float
        Height of the high terminal, in km
    beta_1_rad : float
        Elevation angle (from zenith), in rad
    Const : Const
        Constants class
    
    Returns:
    --------
    result : SlantPathResult
        Ray trace result structure
    """
    
    # Equations 16(a)-(c)
    i_lower = int(np.floor(100 * np.log(1e4 * h_1_km * 
                                        (np.exp(1. / 100.) - 1) + 1) + 1))
    i_upper = int(np.ceil(100 * np.log(1e4 * h_2_km * 
                                       (np.exp(1. / 100.) - 1) + 1) + 1))
    m = (((np.exp(2. / 100.) - np.exp(1. / 100.)) / 
          (np.exp(i_upper / 100.) - np.exp(i_lower / 100.))) * 
         (h_2_km - h_1_km))
    
    alpha_i_rad = beta_1_rad
    beta_ii_rad = beta_1_rad
    
    # Initialize results
    result = SlantPathResult()
    
    # Initialize starting layer
    delta_i_km = layer_thickness(m, i_lower)
    h_i_km = (h_1_km + m * ((np.exp((i_lower - 1) / 100.) - 
                            np.exp((i_lower - 1) / 100.)) / 
                           (np.exp(1 / 100.) - 1)))
    n_i, gamma_i = get_layer_properties(f_ghz, h_i_km + delta_i_km / 2)
    r_i_km = Const.a_0_km + h_i_km
    
    # Record bottom layer properties for alpha and beta calculations
    r_1_km = r_i_km
    n_1 = n_i
    
    # Summation from Equation 13
    for i in range(i_lower, i_upper):
        
        delta_ii_km = layer_thickness(m, i + 1)
        h_ii_km = (h_1_km + m * ((np.exp((i + 1 - 1) / 100.) - 
                                 np.exp((i_lower - 1) / 100.)) / 
                                (np.exp(1 / 100.) - 1)))
        
        n_ii, gamma_ii = get_layer_properties(f_ghz, h_ii_km + delta_ii_km / 2
                                              )
        
        r_ii_km = Const.a_0_km + h_ii_km
        
        delta_i_km = layer_thickness(m, i)
        
        # Equation 19b
        beta_i_rad = np.arcsin(min(1, (n_1 * r_1_km) / (n_i * r_i_km) * 
                                   np.sin(beta_1_rad)))
        
        # Entry angle into the layer interface, Equation 18a
        alpha_i_rad = np.arcsin(min(1, (n_1 * r_1_km) / (n_i * r_ii_km) * 
                                    np.sin(beta_1_rad)))
        
        # Path length through ith layer, Equation 17
        a_i_km = (-r_i_km * np.cos(beta_i_rad) + 
                 np.sqrt((r_i_km**2) * (np.cos(beta_i_rad))**2 + 
                        2 * r_i_km * delta_i_km + (delta_i_km**2)))
        
        result.a_km = result.a_km + a_i_km
        result.A_gas_db = result.A_gas_db + a_i_km * gamma_i
        result.delta_L_km = result.delta_L_km + a_i_km * (n_i - 1)  # Summation, Equation 23
        
        beta_ii_rad = np.arcsin(n_i / n_ii * np.sin(alpha_i_rad))
        
        # Summation of the bending angle, Equation 22a
        # The summation only goes to i_max - 1
        if i != i_upper - 1:
            result.bending_rad = result.bending_rad + beta_ii_rad - alpha_i_rad
        
        # Shift for next loop
        h_i_km = h_ii_km
        n_i = n_ii
        gamma_i = gamma_ii
        r_i_km = r_ii_km
    
    result.angle_rad = alpha_i_rad
    
    return result

def layer_thickness(m: float, i: int) -> float:
    """
    Thickness of the ith layer.
    
    Parameters:
    -----------
    m : float
        Internal parameter
    i : int
        Layer of interest
    
    Returns:
    --------
    delta_i_km : float
        Layer thickness, in km
    """
    # Equation 14
    delta_i_km = m * np.exp((i - 1) / 100.)
    
    return delta_i_km


def get_layer_properties(f_ghz: float, h_i_km: float) -> Tuple[float, float]:
    """
    Determine the parameters for the ith layer.
    
    Parameters:
    -----------
    f_ghz : float
        Frequency, in GHz
    h_i_km : float
        Height of the ith layer, in km
    Const : Const
        Constants class
    
    Returns:
    --------
    n : float
        Refractive index
    gamma : float
        Specific attenuation, in dB/km
    """
    # Use function pointers to get atmospheric parameters
    T_kelvin = global_temperature(h_i_km)
    p_hPa = global_pressure(h_i_km)
    e_hPa = global_wet_pressure(h_i_km)
    
    # Compute the refractive index for the current layer
    n = refractive_index(p_hPa, T_kelvin, e_hPa)
    
    # Specific attenuation of layer
    gamma = specific_attenuation(f_ghz, T_kelvin, e_hPa, p_hPa)
    
    return n, gamma


def global_wet_pressure(h_km: float) -> float:
    """
    Calculate the water vapour pressure at a given height.
    
    Parameters:
    -----------
    h_km : float
        Geometric height, in km
    Const : Const
        Constants class
    
    Returns:
    --------
    e_hPa : float
        Water vapour pressure, in hPa
    """
    T_kelvin = global_temperature(h_km)
    P_hPa = global_pressure(h_km)
    rho_g_m3 = max(global_water_vapour_density(h_km, Const.RHO_0__M_KG),
                   2 * (10**(-6)) * 216.7 * P_hPa / T_kelvin)
    
    e_hPa = water_vapour_density_to_partial_pressure(rho_g_m3, T_kelvin)
    
    return e_hPa


def line_shape_factor(f_ghz: float, f_i_ghz: np.ndarray, delta_f_ghz: np.ndarray,
                      delta: np.ndarray) -> np.ndarray:
    """
    Calculate the line shape factor for spectral lines.
    
    Parameters:
    -----------
    f_ghz : float
        Frequency, in GHz
    f_i_ghz : np.ndarray
        Line center frequencies, in GHz
    delta_f_ghz : np.ndarray
        Line widths, in GHz
    delta : np.ndarray
        Line interference parameter
    
    Returns:
    --------
    F_i : np.ndarray
        Line shape factors
    """
    term1 = f_ghz / f_i_ghz
    term2 = ((delta_f_ghz - delta * (f_i_ghz - f_ghz)) / 
             ((f_i_ghz - f_ghz)**2 + (delta_f_ghz**2)))
    term3 = ((delta_f_ghz - delta * (f_i_ghz + f_ghz)) / 
             ((f_i_ghz + f_ghz)**2 + (delta_f_ghz**2)))
    
    F_i = term1 * (term2 + term3)
    
    return F_i


def nonresonant_debye_attenuation(f_ghz: float, e_hPa: float, p_hPa: float,
                                  theta: float) -> float:
    """
    Calculate non-resonant Debye attenuation.
    
    Parameters:
    -----------
    f_ghz : float
        Frequency, in GHz
    e_hPa : float
        Water vapour pressure, in hPa
    p_hPa : float
        Dry air pressure, in hPa
    theta : float
        Temperature ratio (300/T)
    
    Returns:
    --------
    N_D : float
        Non-resonant Debye contribution
    """
    # Width parameter for the Debye spectrum, Equation 9
    d = 5.6e-4 * (p_hPa + e_hPa) * theta**(0.8)
    
    # Equation 8
    frac_1 = 6.14e-5 / (d * (1 + (f_ghz / d)**2))
    frac_2 = ((1.4e-12 * p_hPa * theta**(1.5)) / 
             (1 + 1.9e-5 * (f_ghz**(1.5))))
    N_D = f_ghz * p_hPa * (theta**2) * (frac_1 + frac_2)
    
    return N_D


@dataclass
class OxygenData:
    """Spectroscopic data for oxygen attenuation (Table 1)"""
    f_0: np.ndarray = field(default_factory=lambda: np.array([]))
    a_1: np.ndarray = field(default_factory=lambda: np.array([]))
    a_2: np.ndarray = field(default_factory=lambda: np.array([]))
    a_3: np.ndarray = field(default_factory=lambda: np.array([]))
    a_4: np.ndarray = field(default_factory=lambda: np.array([]))
    a_5: np.ndarray = field(default_factory=lambda: np.array([]))
    a_6: np.ndarray = field(default_factory=lambda: np.array([]))


def get_oxygen_data() -> OxygenData:
    """
    Get spectroscopic data for oxygen attenuation (Table 1).
    
    Returns:
    --------
    result : OxygenData
        Oxygen spectroscopic parameters
    """
    result = OxygenData()
    
    result.f_0 = np.array([
        50.474214,  50.987745,  51.503360,  52.021429,  52.542418,  53.066934,  53.595775,
        54.130025,  54.671180,  55.221384,  55.783815,  56.264774,  56.363399,  56.968211,
        57.612486,  58.323877,  58.446588,  59.164204,  59.590983,  60.306056,  60.434778,
        61.150562,  61.800158,  62.411220,  62.486253,  62.997984,  63.568526,  64.127775,
        64.678910,  65.224078,  65.764779,  66.302096,  66.836834,  67.369601,  67.900868,
        68.431006,  68.960312, 118.750334, 368.498246, 424.763020, 487.249273,
        715.392902, 773.839490, 834.145546
    ])
    
    result.a_1 = np.array([
        0.975,    2.529,    6.193,   14.320,   31.240,   64.290,  124.600,  227.300,
        389.700,  627.100,  945.300,  543.400, 1331.800, 1746.600, 2120.100, 2363.700,
        1442.100, 2379.900, 2090.700, 2103.400, 2438.000, 2479.500, 2275.900, 1915.400,
        1503.000, 1490.200, 1078.000,  728.700,  461.300,  274.000,  153.000,   80.400,
        39.800,   18.560,    8.172,    3.397,    1.334,  940.300,   67.400,  637.700,
        237.400,   98.100,  572.300,  183.100
    ])
    
    result.a_2 = np.array([
        9.651, 8.653, 7.709, 6.819, 5.983, 5.201, 4.474, 3.800, 3.182, 2.618, 2.109,
        0.014, 1.654, 1.255, 0.910, 0.621, 0.083, 0.387, 0.207, 0.207, 0.386, 0.621,
        0.910, 1.255, 0.083, 1.654, 2.108, 2.617, 3.181, 3.800, 4.473, 5.200, 5.982,
        6.818, 7.708, 8.652, 9.650, 0.010, 0.048, 0.044, 0.049, 0.145, 0.141, 0.145
    ])
    
    result.a_3 = np.array([
        6.690,  7.170,  7.640,  8.110,  8.580,  9.060,  9.550,  9.960, 10.370,
        10.890, 11.340, 17.030, 11.890, 12.230, 12.620, 12.950, 14.910, 13.530,
        14.080, 14.150, 13.390, 12.920, 12.630, 12.170, 15.130, 11.740, 11.340,
        10.880, 10.380,  9.960,  9.550,  9.060,  8.580,  8.110,  7.640,  7.170,
        6.690, 16.640, 16.400, 16.400, 16.000, 16.000, 16.200, 14.700
    ])
    
    result.a_4 = np.zeros(44)
    
    result.a_5 = np.array([
        2.566,  2.246,  1.947,  1.667,  1.388,  1.349,  2.227,  3.170,  3.558,  2.560,
        -1.172,  3.525, -2.378, -3.545, -5.416, -1.932,  6.768, -6.561,  6.957, -6.395,
        6.342,  1.014,  5.014,  3.029, -4.499,  1.856,  0.658, -3.036, -3.968, -3.528,
        -2.548, -1.660, -1.680, -1.956, -2.216, -2.492, -2.773, -0.439,  0.000,  0.000,
        0.000,  0.000,  0.000,  0.000
    ])
    
    result.a_6 = np.array([
        6.850,  6.800,  6.729,  6.640,  6.526,  6.206,  5.085,  3.750,  2.654,  2.952,
        6.135, -0.978,  6.547,  6.451,  6.056,  0.436, -1.273,  2.309, -0.776,  0.699,
        -2.825, -0.584, -6.619, -6.759,  0.844, -6.675, -6.139, -2.895, -2.590, -3.680,
        -5.002, -6.091, -6.393, -6.475, -6.545, -6.600, -6.650,  0.079,  0.000,  0.000,
        0.000,  0.000,  0.000,  0.000
    ])
    
    return result


@dataclass
class WaterVapourData:
    """Spectroscopic data for water vapor attenuation (Table 2)"""
    f_0: np.ndarray = field(default_factory=lambda: np.array([]))
    b_1: np.ndarray = field(default_factory=lambda: np.array([]))
    b_2: np.ndarray = field(default_factory=lambda: np.array([]))
    b_3: np.ndarray = field(default_factory=lambda: np.array([]))
    b_4: np.ndarray = field(default_factory=lambda: np.array([]))
    b_5: np.ndarray = field(default_factory=lambda: np.array([]))
    b_6: np.ndarray = field(default_factory=lambda: np.array([]))


def get_water_vapour_data() -> WaterVapourData:
    """
    Get spectroscopic data for water vapor attenuation (Table 2).
    
    Returns:
    --------
    result : WaterVapourData
        Water vapour spectroscopic parameters
    """
    result = WaterVapourData()
    
    result.f_0 = np.array([
        22.235080,  67.803960, 119.995940, 183.310087, 321.225630, 325.152888,  336.227764,
        380.197353, 390.134508, 437.346667, 439.150807, 443.018343, 448.001085,  470.888999,
        474.689092, 488.490108, 503.568532, 504.482692, 547.676440, 552.020960,  556.935985,
        620.700807, 645.766085, 658.005280, 752.033113, 841.051732, 859.965698,  899.303175,
        902.611085, 906.205957, 916.171582, 923.112692, 970.315022, 987.926764, 1780.000000
    ])
    
    result.b_1 = np.array([
        0.1079, 0.0011,   0.0007,  2.273, 0.0470, 1.514,    0.0010, 11.67,   0.0045,
        0.0632, 0.9098,   0.1920, 10.41,  0.3254, 1.260,    0.2529,  0.0372, 0.0124,
        0.9785, 0.1840, 497.0,     5.015, 0.0067, 0.2732, 243.4,     0.0134, 0.1325,
        0.0547, 0.0386,   0.1836,  8.400, 0.0079, 9.009,  134.6,     17506.0
    ])
    
    result.b_2 = np.array([
        2.144, 8.732, 8.353, 0.668, 6.179, 1.541, 9.825, 1.048, 7.347, 5.048,
        3.595, 5.048, 1.405, 3.597, 2.379, 2.852, 6.731, 6.731, 0.158, 0.158,
        0.159, 2.391, 8.633, 7.816, 0.396, 8.177, 8.055, 7.914, 8.429, 5.110,
        1.441, 10.293, 1.919, 0.257, 0.952
    ])
    
    result.b_3 = np.array([
        26.38, 28.58, 29.48, 29.06, 24.04, 28.23, 26.93, 28.11, 21.52, 18.45, 20.07,
        15.55, 25.64, 21.34, 23.20, 25.86, 16.12, 16.12, 26.00, 26.00, 30.86, 24.38,
        18.00, 32.10, 30.86, 15.90, 30.60, 29.85, 28.65, 24.08, 26.73, 29.00, 25.50,
        29.85, 196.3
    ])
    
    result.b_4 = np.array([
        0.76, 0.69, 0.70, 0.77, 0.67, 0.64, 0.69, 0.54, 0.63, 0.60, 0.63, 0.60, 0.66, 0.66,
        0.65, 0.69, 0.61, 0.61, 0.70, 0.70, 0.69, 0.71, 0.60, 0.69, 0.68, 0.33, 0.68, 0.68,
        0.70, 0.70, 0.70, 0.70, 0.64, 0.68, 2.00
    ])
    
    result.b_5 = np.array([
        5.087, 4.930, 4.780, 5.022, 4.398, 4.893, 4.740, 5.063, 4.810, 4.230, 4.483,
        5.083, 5.028, 4.506, 4.804, 5.201, 3.980, 4.010, 4.500, 4.500, 4.552, 4.856,
        4.000, 4.140, 4.352, 5.760, 4.090, 4.530, 5.100, 4.700, 5.150, 5.000, 4.940,
        4.550, 24.15
    ])
    
    result.b_6 = np.array([
        1.00, 0.82, 0.79, 0.85, 0.54, 0.74, 0.61, 0.89, 0.55, 0.48, 0.52, 0.50, 0.67, 0.65,
        0.64, 0.72, 0.43, 0.45, 1.00, 1.00, 1.00, 0.68, 0.50, 1.00, 0.84, 0.45, 0.84,
        0.90, 0.95, 0.53, 0.78, 0.80, 0.67, 0.90, 5.00
    ])
    
    return result


def oxygen_refractivity(f_ghz: float, T_kelvin: float, e_hPa: float, p_hPa: float,
                       oxygen_data: OxygenData) -> float:
    """
    Calculate oxygen refractivity contribution.
    
    Parameters:
    -----------
    f_ghz : float
        Frequency, in GHz
    T_kelvin : float
        Temperature, in Kelvin
    e_hPa : float
        Water vapour pressure, in hPa
    p_hPa : float
        Dry air pressure, in hPa
    oxygen_data : OxygenData
        Oxygen spectroscopic data
    
    Returns:
    --------
    N_o : float
        Oxygen refractivity
    """
    theta = 300 / T_kelvin
    
    # Vectorized calculations (commented loop above for reference)
    S_i = (oxygen_data.a_1 * 1e-7 * p_hPa * (theta**3) * 
           np.exp(oxygen_data.a_2 * (1 - theta)))
    
    delta_f_ghz = (oxygen_data.a_3 * 1e-4 * 
                   (p_hPa * (theta**(0.8 - oxygen_data.a_4)) + 1.1 * e_hPa * theta))
    
    # Modify line width to account for Zeeman splitting of oxygen lines
    delta_f_ghz = np.sqrt((delta_f_ghz**2) + 2.25e-6)
    
    # Correction factor due to interference effects in oxygen lines
    delta = ((oxygen_data.a_5 + oxygen_data.a_6 * theta) * 1e-4 * 
            (p_hPa + e_hPa) * (theta**0.8))
    
    F_i = line_shape_factor(f_ghz, oxygen_data.f_0, delta_f_ghz, delta)
    
    N_D = nonresonant_debye_attenuation(f_ghz, e_hPa, p_hPa, theta)
    N = np.sum(S_i * F_i)
    
    N_o = N + N_D
    
    return N_o


def water_vapour_refractivity(f_ghz: float, T_kelvin: float, e_hPa: float, P_hPa: float,
                              water_vapour_data: WaterVapourData) -> float:
    """
    Calculate water vapour refractivity contribution.
    
    Parameters:
    -----------
    f_ghz : float
        Frequency, in GHz
    T_kelvin : float
        Temperature, in Kelvin
    e_hPa : float
        Water vapour pressure, in hPa
    P_hPa : float
        Total pressure, in hPa
    water_vapour_data : WaterVapourData
        Water vapour spectroscopic data
    
    Returns:
    --------
    N_w : float
        Water vapour refractivity
    """
    theta = 300 / T_kelvin
    
    # Vectorized calculations (commented loop above for reference)
    S_i = (0.1 * water_vapour_data.b_1 * e_hPa * (theta**3.5) * 
           np.exp(water_vapour_data.b_2 * (1 - theta)))
    
    delta_f_ghz = (1e-4 * water_vapour_data.b_3 * 
                   (P_hPa * (theta**(water_vapour_data.b_4)) + 
                    water_vapour_data.b_5 * e_hPa * (theta**(water_vapour_data.b_6))))
    
    # Modify line width to account for Doppler broadening of water vapour lines
    term1 = (0.217 * (delta_f_ghz**2) + 
            (2.1316e-12 * (water_vapour_data.f_0**2) / theta))
    delta_f_ghz = 0.535 * delta_f_ghz + np.sqrt(term1)
    
    # For water vapour, delta = 0
    delta = 0
    
    F_i = line_shape_factor(f_ghz, water_vapour_data.f_0, delta_f_ghz, delta)
    N_w = np.sum(S_i * F_i)
    
    return N_w


def specific_attenuation(f_ghz: float, T_kelvin: float, e_hPa: float, 
                        p_hPa: float) -> float:
    """
    Calculate the total specific attenuation due to atmospheric gases.
    
    Parameters:
    -----------
    f_ghz : float
        Frequency, in GHz
    T_kelvin : float
        Temperature, in Kelvin
    e_hPa : float
        Water vapour pressure, in hPa
    p_hPa : float
        Dry air pressure, in hPa
    
    Returns:
    --------
    gamma : float
        Specific attenuation, in dB/km
    """
    oxygen_data = get_oxygen_data()
    water_vapour_data = get_water_vapour_data()
    
    gamma_o = oxygen_specific_attenuation(f_ghz, T_kelvin, e_hPa, p_hPa, oxygen_data)
    gamma_w = water_vapour_specific_attenuation(f_ghz, T_kelvin, e_hPa, p_hPa, 
                                                water_vapour_data)
    
    gamma = gamma_o + gamma_w  # [Eqn 1]
    
    return gamma


def oxygen_specific_attenuation(f_ghz: float, T_kelvin: float, e_hPa: float, 
                                p_hPa: float, oxygen_data: OxygenData) -> float:
    """
    Calculate oxygen specific attenuation.
    
    Parameters:
    -----------
    f_ghz : float
        Frequency, in GHz
    T_kelvin : float
        Temperature, in Kelvin
    e_hPa : float
        Water vapour pressure, in hPa
    p_hPa : float
        Dry air pressure, in hPa
    oxygen_data : OxygenData
        Oxygen spectroscopic data
    
    Returns:
    --------
    gamma_o : float
        Oxygen specific attenuation, in dB/km
    """
    # Partial Eqn 1
    N_o = oxygen_refractivity(f_ghz, T_kelvin, e_hPa, p_hPa, oxygen_data)
    gamma_o = 0.1820 * f_ghz * N_o
    
    return gamma_o


def water_vapour_specific_attenuation(f_ghz: float, T_kelvin: float, e_hPa: float, 
                                      p_hPa: float, 
                                      water_vapour_data: WaterVapourData) -> float:
    """
    Calculate water vapour specific attenuation.
    
    Parameters:
    -----------
    f_ghz : float
        Frequency, in GHz
    T_kelvin : float
        Temperature, in Kelvin
    e_hPa : float
        Water vapour pressure, in hPa
    p_hPa : float
        Dry air pressure, in hPa
    water_vapour_data : WaterVapourData
        Water vapour spectroscopic data
    
    Returns:
    --------
    gamma_w : float
        Water vapour specific attenuation, in dB/km
    """
    # Partial Eqn 1
    N_w = water_vapour_refractivity(f_ghz, T_kelvin, e_hPa, p_hPa, water_vapour_data)
    gamma_w = 0.1820 * f_ghz * N_w
    
    return gamma_w


def terrestrial_path(f_ghz: float, T_kelvin: float, e_hPa: float, p_hPa: float,
                    r_0_km: float) -> float:
    """
    Calculate total path attenuation for a terrestrial path.
    
    Parameters:
    -----------
    f_ghz : float
        Frequency, in GHz
    T_kelvin : float
        Temperature, in Kelvin
    e_hPa : float
        Water vapour pressure, in hPa
    p_hPa : float
        Dry air pressure, in hPa
    r_0_km : float
        Path length, in km
    
    Returns:
    --------
    A_db : float
        Total path attenuation, in dB
    """
    gamma = specific_attenuation(f_ghz, T_kelvin, e_hPa, p_hPa)
    
    # Equation 10
    A_db = gamma * r_0_km
    
    return A_db


def water_vapour_density_to_partial_pressure(rho_g_m3: float, T_kelvin: float) -> float:
    """
    Convert water vapour density to partial pressure.
    
    Parameters:
    -----------
    rho_g_m3 : float
        Water vapour density, in g/m³
    T_kelvin : float
        Temperature, in Kelvin
    
    Returns:
    --------
    e_hPa : float
        Water vapour pressure, in hPa
    """
    # Equation 4
    e_hPa = (rho_g_m3 * T_kelvin) / 216.7
    
    return e_hPa



