import numexpr as ne
import numpy as np
import torch
import pandas as pd

from .ancil.gas_parameters import get_gas_params
from .ancil.thermal_parameters import get_thermal_params
from .ancil.units import Units
from .constants import TCR_DBL


def calculate_alpha(G, G_A, T, r0, rC, rT, rA, g0, g1, iirf100_max=False):
    iirf100_val = ne.evaluate("abs(r0 + rC * (G-G_A) + rT * T + rA * G_A)")
    if iirf100_max:
        iirf100_val = ne.evaluate(  # noqa: F841
            "where(iirf100_val>iirf100_max,iirf100_max,iirf100_val)"
        )
    alpha_val = ne.evaluate("g0 * exp(iirf100_val / g1)")
    return alpha_val


def calculate_g(a, tau):
    g1 = ne.evaluate(
        "sum( a * tau * ( 1. - ( 1. + 100/tau ) * exp(-100/tau) ), axis = 0)"
    )
    g0 = np.exp(-1 * np.sum(a * tau * (1.0 - np.exp(-100 / tau)), axis=0) / g1)
    return g0, g1


def step_concentration(
    emissions, a, dt, alpha, tau, R_old, G_A_old, PI_conc, emis2conc
):
    decay_rate = ne.evaluate("1/(alpha*tau)")  # noqa: F841
    decay_factor = ne.evaluate("exp(-dt*decay_rate)")  # noqa: F841
    R = ne.evaluate(
        "emissions * a / decay_rate *\
            ( 1. - decay_factor ) + R_old * decay_factor"
    )
    G_A = ne.evaluate("sum(R,axis=0)")
    C = ne.evaluate("PI_conc + emis2conc * (G_A + G_A_old) / 2")
    return C, R, G_A


def step_forcing(C, PI_conc, f1, f2, f3, f1aci, f2aci, PI_SO2, emissions):
    logforc = ne.evaluate(
        "f1 * where( (C/PI_conc) <= 0, 0, log(C/PI_conc) )",
        {"f1": f1, "C": C, "PI_conc": PI_conc},
    )
    linforc = ne.evaluate(
        "f2 * (C - PI_conc)",
        {"f2": f2, "C": C, "PI_conc": PI_conc},
    )
    sqrtforc = ne.evaluate(
        "f3 * ( (sqrt( where(C<0 ,0 ,C ) ) - sqrt(PI_conc)) )",
        {"f3": f3, "C": C, "PI_conc": PI_conc},
    )
    ari_linforc = f2 * emissions
    linforc[2:] = 0.
    ari_linforc[:2] = 0.
    aci_logforc = f1aci * np.log(1 + emissions / PI_SO2)
    aci_linforc = f2aci * emissions
    RF = logforc + linforc + sqrtforc + aci_logforc + aci_linforc
    return RF


def step_temperature(S_old, F, q, d, dt=1):
    decay_factor = ne.evaluate("exp(-dt/d)")  # noqa: F841
    S_new = ne.evaluate("q * F * (1 - decay_factor) + S_old * decay_factor")
    T = ne.evaluate("sum( (S_old + S_new)/2, axis=0 )")
    return S_new, T


def step_I(I_old, K, q, d, dt=1):
    """Takes next time step to construct recursively the I matrix where

        I_{i,j} = (q/d)∫K(ti, s)exp(-(tj-s)/d)ds from 0 to tj

    Rows fix the timestep inside K(ti, s) and colums determine the exponential term and
    integration bounds.

        I_{i,j} = d * k(ti,tj) * (1 - exp(-dt/d)) + I_{i,j-1} * exp(-dt/d)

    Args:
        I_old (np.ndarray): column for previous time step t_{j-1} (i.e. at fixed ti)
        K (np.ndarray): k(ti, s) for all times s
        d (np.ndarray): (nboxes,)
        dt (float): timestep

    Returns:
        type: np.ndarray column for time step t_j
    """
    decay_factor = torch.exp(-dt / d)
    I_new = q * K * (1 - decay_factor) + I_old * decay_factor
    I_new = (I_new + I_old) / 2
    return I_new


def step_kernel(Kj_old, I_row, q, d, dt=1):
    """Takes next time step to construct recursively kernel matrix Kj

        kj(ti, tj) = kj(t_{i-1},tj) * exp(-dt/d) + (q^2 / d) * I_{i,j} * (1 - exp(-dt/d))

    Args:
        Kj_old (np.ndarray): row for previous time step
        I_row (np.ndarray): Description of parameter `I_new`.
        q (np.ndarray): (nboxes,)
        d (np.ndarray): (nboxes,)
        dt (float): timestep

    Returns:
        type: np.ndarray row for time step t_i

    """
    decay_factor = torch.exp(-dt / d)
    Kj_new = Kj_old * decay_factor + q * I_row * (1 - decay_factor)
    Kj_new = (Kj_new + Kj_old) / 2
    return Kj_new
