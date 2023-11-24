import os
import sys
import numpy as np


base_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(base_dir)

import src.fair.tools as tools


def _run(
    inp_ar,
    a1,
    a2,
    a3,
    a4,
    tau1,
    tau2,
    tau3,
    tau4,
    r0,
    rC,
    rT,
    rA,
    PI_conc,
    PI_SO2,
    emis2conc,
    f1,
    f2,
    f3,
    f1aci,
    f2aci,
    d,
    q,
    ext_forcing,
    timestep,
    **kwargs
):
    """
    Run FaIR 2.0 from numpy array
    This function can *only* run one scenario,
    thermal parameter set & gas parameter set at a time
    Parameters
    ----------
    inp_ar : :obj:`np.ndarray`
        Input :obj:`np.ndarray` containing the timeseries to run. No checks
        of the column order are performed here.
        format: [[species],[time]]
    a1, a2, a3, a4, tau1, tau2, tau3, tau4, r0, rC,
    rT, rA, PI_conc, emis2conc, f1, f2, f3 : :obj:`np.ndarray`
        Input :obj:`np.ndarray` containing gas parameters in format:
        [species],
        note: all species contain the same number of gas/thermal pool
        indices (some are simply populated with 0)
    d, q : obj:`np.ndarray`
        Input :obj:`np.ndarray` containing thermal parameters in format:
        [response box]
    ext_forcing : :obj:`np.ndarray`
        Input :obj:`np.ndarray` containing any other prescribed forcing
        in format: [time]
    timestep : :obj:`np.ndarray`
        Input :obj:`np.ndarray`
        specifying the length of each entry in inp_ar in years.
        For example: if inp_ar were an nx4 array,
        representing times 2020-2021, 2021-2023, 2023-2027 and 2027-2028:
        timestep would be: np.array([1,2,4,1])
    Returns
    -------
    dict
        Dictionary containing the results of the run.
        Keys are 'C', 'RF', 'T', and 'alpha'
        (Concentration, Radiative Forcing, Temperature and Alpha)
        Values are in :obj:`np.ndarray` format,
        with the final index representing 'timestep'
    """

    n_species, n_timesteps = inp_ar.shape
    # Concentration, Radiative Forcing and Alpha
    C, RF, alpha = np.zeros((3, n_species, n_timesteps))
    # Temperature
    T = np.zeros(n_timesteps)
    # S represents the results of the calculations from the thermal boxes,
    # an Impulse Response calculation (T = sum(S))
    S = np.zeros((n_timesteps, len(d)))
    # G represents cumulative emissions,
    # while G_A represents emissions accumulated since pre-industrial times,
    # both in the same units as emissions
    # So at any point, G - G_A is equal
    # to the amount of a species that has been absorbed
    G_A, G = np.zeros((2, n_species))
    # R in format [[index],[species]]
    R = np.zeros((4, n_species))
    # a,tau in format [[index], [species]]
    a = np.array([a1, a2, a3, a4]).reshape(4, -1)
    tau = np.array([tau1, tau2, tau3, tau4]).reshape(4, -1)
    # g0, g1 in format [species]
    g0, g1 = tools.calculate_g(a=a, tau=tau)
    for i, tstep in enumerate(timestep):
        alpha[..., i] = tools.calculate_alpha(
            G=G, G_A=G_A, T=np.sum(S[max(i - 1, 0)], axis=0), r0=r0, rC=rC, rT=rT, rA=rA, g0=g0, g1=g1
        )
        C[..., i], R, G_A = tools.step_concentration(
            emissions=inp_ar[np.newaxis, ..., i],
            a=a,
            dt=tstep,
            alpha=alpha[np.newaxis, ..., i],
            tau=tau,
            R_old=R,
            G_A_old=G_A,
            PI_conc=PI_conc,
            emis2conc=emis2conc,
        )
        RF[..., i] = tools.step_forcing(
            C=C[..., i], PI_conc=PI_conc, f1=f1, f2=f2, f3=f3, f1aci=f1aci, f2aci=f2aci, PI_SO2=PI_SO2, emissions=inp_ar[..., i],
        )
        S[i], T[i] = tools.step_temperature(
            S_old=S[max(i - 1, 0)], F=np.sum(RF[..., i], axis=0) + ext_forcing[i], q=q, d=d, dt=tstep
        )
        G += inp_ar[..., i]
    res = {"C": C, "RF": RF, "T": T, "S": S, "alpha": alpha}
    return res
