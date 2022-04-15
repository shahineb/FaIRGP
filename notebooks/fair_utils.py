from __future__ import division

import inspect
import numpy as np
import warnings

from fair.constants import molwt, lifetime, radeff
from fair.constants.general import M_ATMOS, ppm_gtc
from fair.defaults import carbon, thermal
from fair.gas_cycle.fair1 import carbon_cycle
from fair.forcing.ghg import co2_log
from fair.temperature.millar import calculate_q, forcing_to_temperature


# TODO: unified interface to the different carbon cycles


def emis_to_conc(c0, e0, e1, ts, lt, vm):
    """Calculate concentrations of well mixed GHGs from emissions for simple
    one-box model.

    Inputs (all can be scalar or 1D arrays for multiple species):
        c0: concentrations in timestep t-1
        e0: emissions in timestep t-1
        e1: emissions in timestep t
        ts: length of timestep. Use 1 for sensible results in FaIR 1.3.
        lt: atmospheric (e-folding) lifetime of GHG
        vm: conversion from emissions units (e.g. Mt) to concentrations units
            (e.g. ppb)

    Outputs:
        c1: concentrations in timestep t
    """
    c1 = c0 - c0 * (1.0 - np.exp(-ts/lt)) + 0.5 * ts * (e1 + e0) * vm
    return c1











def init_fair(
    emissions=False,
    C=None,
    other_rf=0.0,
    q        = thermal.q,
    tcrecs   = thermal.tcrecs,
    d        = thermal.d,
    F2x      = thermal.f2x,
    tcr_dbl  = thermal.tcr_dbl,
    a        = carbon.a,
    tau      = carbon.tau,
    r0       = carbon.r0,
    rc       = carbon.rc,
    rt       = carbon.rt,
    iirf_max = carbon.iirf_max,
    iirf_h   = carbon.iirf_h,
    C_pi=np.array([278., 722., 273., 34.497] + [0.]*25 + [13.0975, 547.996]),
    scale=None,
    ):
    # is iirf_h < iirf_max? Don't stop the code, but warn user
    if iirf_h < iirf_max:
        warnings.warn('iirf_h=%f, which is less than iirf_max (%f)'
          % (iirf_h, iirf_max), RuntimeWarning)

    # Conversion between ppb/ppt concentrations and Mt/kt emissions
    # in the RCP databases ppb = Mt and ppt = kt so factor always 1e18
    emis2conc = M_ATMOS/1e18*np.asarray(molwt.aslist)/molwt.AIR

    # Funny units for nitrogen emissions - N2O is expressed in N2 equivalent
    n2o_sf = molwt.N2O/molwt.N2
    emis2conc[2] = emis2conc[2] / n2o_sf

    # Convert any list to a numpy array for (a) speed and (b) consistency.
    # Goes through all variables in scope and converts them.
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for arg_to_check in args:
        if type(values[arg_to_check]) is list:
            exec(arg_to_check + '= np.array(' + arg_to_check + ')')

    # Set up the output timeseries variables depending on options and perform basic sense checks
    # initialisations for CO2-only mode
    ngas = 1
    nF   = 1
    if type(emissions) is np.ndarray:
        if emissions.ndim != 1:
            raise ValueError(
              "In CO2-only mode, emissions should be a 1D array")
        nt = emissions.shape[0]
        carbon_boxes_shape = (nt, a.shape[0])
        thermal_boxes_shape = (nt, d.shape[0])
    elif type(other_rf) is np.ndarray:
        if other_rf.ndim != 1:
            raise ValueError(
              "In CO2-only mode, other_rf should be a 1D array")
        nt = other_rf.shape[0]
        carbon_boxes_shape = (nt, a.shape[0])
        thermal_boxes_shape = (nt, d.shape[0])
        emissions = np.zeros(nt)
    else:
        raise ValueError(
          "Neither emissions or other_rf is defined as a timeseries")

    # check scale factor is correct shape - either scalar or 1D
    scale = np.ones(nt)

    # If TCR and ECS are supplied, calculate q coefficients
    if type(tcrecs) is np.ndarray:
        q = calculate_q(tcrecs, d, F2x, tcr_dbl, nt)

    # Check a and tau are same size
    if a.ndim != 1:
        raise ValueError("a should be a 1D array")
    if tau.ndim != 1:
        raise ValueError("tau should be a 1D array")
    if len(a) != len(tau):
        raise ValueError("a and tau should be the same size")
    if not np.isclose(np.sum(a), 1.0):
        raise ValueError("a should sum to one")

    # Allocate intermediate and output arrays
    F = np.zeros((nt, nF))
    C_acc = np.zeros(nt)
    ξ_j = np.zeros(thermal_boxes_shape)
    C_0 = np.copy(C_pi)
    C = np.zeros((nt, ngas))
    R_i = np.zeros(carbon_boxes_shape)

    # Initialise the carbon pools to be correct for first timestep in
    # numerical method
    R_i[0,:] = a * emissions[0,np.newaxis] / ppm_gtc
    C[0,0] = np.sum(R_i[0,:],axis=-1) + C_0[0]

    # Calculate forcings from concentrations
    # Calculate forcing for CO2-only mode, first time step
    if np.isscalar(other_rf):
        F[0,0] = co2_log(C[0,0], C_pi[0], F2x) + other_rf
    else:
        F[0,0] = co2_log(C[0,0], C_pi[0], F2x) + other_rf[0]
    F[0,0] = F[0,0] * scale[0]


    # Calculate temperatures for first time step
    # Update the thermal response boxes
    ξ_j[0, :] = (q[0, :] / d) * (np.sum(F[0, :]))
    return F, C_acc, ξ_j, C_pi, C, R_i, q, scale














def fair_iteration(t, emissions, other_rf, scale, F, C_acc, ξ_j, C_pi, C, R_i, T, q,
                    tcrecs   = thermal.tcrecs,
                    d        = thermal.d,
                    F2x      = thermal.f2x,
                    tcr_dbl  = thermal.tcr_dbl,
                    a        = carbon.a,
                    tau      = carbon.tau,
                    r0       = carbon.r0,
                    rc       = carbon.rc,
                    rt       = carbon.rt,
                    iirf_max = carbon.iirf_max,
                    iirf_h   = carbon.iirf_h,):
    if t == 1:
        time_scale_sf = 0.16
    C[t,0], C_acc[t], R_i[t,:], time_scale_sf = carbon_cycle(
      emissions[t-1],
      C_acc[t-1],
      T[t-1],
      r0,
      rc,
      rt,
      iirf_max,
      time_scale_sf,
      a,
      tau,
      iirf_h,
      R_i[t-1,:],
      C_pi[0],
      C[t-1,0],
      emissions[t]
    )

    if np.isscalar(other_rf):
        F[t,0] = co2_log(C[t,0], C_pi[0], F2x) + other_rf
    else:
        F[t,0] = co2_log(C[t,0], C_pi[0], F2x) + other_rf[t]

    F[t,0] = F[t,0] * scale[t]

    ξ_j[t, :] = q[t, :] * (1. - np.exp(-1. / d)) * F[t, :]
    return F, C_acc, ξ_j, C_pi, C, R_i



















def fair_scm(
    emissions=False,
    C=None,
    other_rf=0.0,
    q        = thermal.q,
    tcrecs   = thermal.tcrecs,
    d        = thermal.d,
    F2x      = thermal.f2x,
    tcr_dbl  = thermal.tcr_dbl,
    a        = carbon.a,
    tau      = carbon.tau,
    r0       = carbon.r0,
    rc       = carbon.rc,
    rt       = carbon.rt,
    iirf_max = carbon.iirf_max,
    iirf_h   = carbon.iirf_h,
    C_pi=np.array([278., 722., 273., 34.497] + [0.]*25 + [13.0975, 547.996]),
    scale=None,
    ):
    # is iirf_h < iirf_max? Don't stop the code, but warn user
    if iirf_h < iirf_max:
        warnings.warn('iirf_h=%f, which is less than iirf_max (%f)'
          % (iirf_h, iirf_max), RuntimeWarning)

    # Conversion between ppb/ppt concentrations and Mt/kt emissions
    # in the RCP databases ppb = Mt and ppt = kt so factor always 1e18
    emis2conc = M_ATMOS/1e18*np.asarray(molwt.aslist)/molwt.AIR

    # Funny units for nitrogen emissions - N2O is expressed in N2 equivalent
    n2o_sf = molwt.N2O/molwt.N2
    emis2conc[2] = emis2conc[2] / n2o_sf

    # Convert any list to a numpy array for (a) speed and (b) consistency.
    # Goes through all variables in scope and converts them.
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for arg_to_check in args:
        if type(values[arg_to_check]) is list:
            exec(arg_to_check + '= np.array(' + arg_to_check + ')')

    # Set up the output timeseries variables depending on options and perform basic sense checks
    # initialisations for CO2-only mode
    ngas = 1
    nF   = 1
    if type(emissions) is np.ndarray:
        if emissions.ndim != 1:
            raise ValueError(
              "In CO2-only mode, emissions should be a 1D array")
        nt = emissions.shape[0]
        carbon_boxes_shape = (nt, a.shape[0])
        thermal_boxes_shape = (nt, d.shape[0])
    elif type(other_rf) is np.ndarray:
        if other_rf.ndim != 1:
            raise ValueError(
              "In CO2-only mode, other_rf should be a 1D array")
        nt = other_rf.shape[0]
        carbon_boxes_shape = (nt, a.shape[0])
        thermal_boxes_shape = (nt, d.shape[0])
        emissions = np.zeros(nt)
    else:
        raise ValueError(
          "Neither emissions or other_rf is defined as a timeseries")

    # check scale factor is correct shape - either scalar or 1D
    scale = np.ones(nt)

    # If TCR and ECS are supplied, calculate q coefficients
    if type(tcrecs) is np.ndarray:
        q = calculate_q(tcrecs, d, F2x, tcr_dbl, nt)

    # Check a and tau are same size
    if a.ndim != 1:
        raise ValueError("a should be a 1D array")
    if tau.ndim != 1:
        raise ValueError("tau should be a 1D array")
    if len(a) != len(tau):
        raise ValueError("a and tau should be the same size")
    if not np.isclose(np.sum(a), 1.0):
        raise ValueError("a should sum to one")

    # Allocate intermediate and output arrays
    F = np.zeros((nt, nF))
    C_acc = np.zeros(nt)
    T_j = np.zeros(thermal_boxes_shape)
    T = np.zeros(nt)
    C_0 = np.copy(C_pi)
    C = np.zeros((nt, ngas))
    R_i = np.zeros(carbon_boxes_shape)

    # Initialise the carbon pools to be correct for first timestep in
    # numerical method
    R_i[0,:] = a * emissions[0,np.newaxis] / ppm_gtc
    C[0,0] = np.sum(R_i[0,:],axis=-1) + C_0[0]

    # Calculate forcings from concentrations
    # Calculate forcing for CO2-only mode, first time step
    if np.isscalar(other_rf):
        F[0,0] = co2_log(C[0,0], C_pi[0], F2x) + other_rf
    else:
        F[0,0] = co2_log(C[0,0], C_pi[0], F2x) + other_rf[0]
    F[0,0] = F[0,0] * scale[0]


    # Calculate temperatures for first time step
    # Update the thermal response boxes
    T_j[0, :] = (q[0, :] / d) * (np.sum(F[0, :]))
    T[0] = np.sum(T_j[0,:])


    for t in range(1, nt):
        if t == 1:
            time_scale_sf = 0.16
        C[t,0], C_acc[t], R_i[t,:], time_scale_sf = carbon_cycle(
          emissions[t-1],
          C_acc[t-1],
          T[t-1],
          r0,
          rc,
          rt,
          iirf_max,
          time_scale_sf,
          a,
          tau,
          iirf_h,
          R_i[t-1,:],
          C_pi[0],
          C[t-1,0],
          emissions[t]
        )

        if np.isscalar(other_rf):
            F[t,0] = co2_log(C[t,0], C_pi[0], F2x) + other_rf
        else:
            F[t,0] = co2_log(C[t,0], C_pi[0], F2x) + other_rf[t]

        F[t,0] = F[t,0] * scale[t]

        T_j[t,:] = forcing_to_temperature(
          T_j[t-1,:], q[t,:], d, F[t,:])
        T[t] = np.sum(T_j[t,:])

    C = np.squeeze(C)
    F = np.squeeze(F)
    return C, F, T
