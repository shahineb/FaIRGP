import torch
import torch.nn as nn


class FaIR(nn.Module):

    def __init__(self,
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
                 emis2conc,
                 f1,
                 f2,
                 f3,
                 forcing_pattern,
                 requires_grad=True,
                 **kwargs):
        super().__init__()
        self.a1 = nn.Parameter(torch.from_numpy(a1))
        self.a2 = nn.Parameter(torch.from_numpy(a2))
        self.a3 = nn.Parameter(torch.from_numpy(a3))
        self.a4 = nn.Parameter(torch.from_numpy(a4))

        self.tau1 = nn.Parameter(torch.from_numpy(tau1))
        self.tau2 = nn.Parameter(torch.from_numpy(tau2))
        self.tau3 = nn.Parameter(torch.from_numpy(tau3))
        self.tau4 = nn.Parameter(torch.from_numpy(tau4))

        self.r0 = nn.Parameter(torch.from_numpy(r0))
        self.rC = nn.Parameter(torch.from_numpy(rC))
        self.rT = nn.Parameter(torch.from_numpy(rT))
        self.rA = nn.Parameter(torch.from_numpy(rA))

        self.f1 = nn.Parameter(torch.from_numpy(f1).float())
        self.f2 = nn.Parameter(torch.from_numpy(f2).float())
        self.f3 = nn.Parameter(torch.from_numpy(f3).float())

        self.register_buffer('forcing_pattern', torch.from_numpy(forcing_pattern).float())
        self.register_buffer('PI_conc', torch.from_numpy(PI_conc).float())
        self.register_buffer('emis2conc', torch.from_numpy(emis2conc).float())

        self.requires_grad = requires_grad
        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    @property
    def tau(self):
        return torch.stack([self.tau1, self.tau2, self.tau3, self.tau4], dim=0)

    @property
    def a(self):
        return torch.stack([self.a1, self.a2, self.a3, self.a4], dim=0)

    @property
    def r(self):
        return torch.stack([self.r1, self.r2, self.r3, self.r4], dim=0)

    @property
    def f(self):
        return torch.stack([self.f1, self.f2, self.f3, self.f4], dim=0)

    def calculate_g(self):
        g1 = torch.sum(self.a * self.tau * (1 - (1 + 100 / self.tau) * torch.exp(-100 / self.tau)), dim=0)
        g0 = torch.exp(-torch.sum(self.a * self.tau * (1. - torch.exp(-100. / self.tau)), dim=0) / g1)
        return g0, g1

    def calculate_alpha(self, G, G_A, T, g0, g1, iirf100_max=False):
        iirf100_val = torch.abs(self.r0 + self.rC * (G - G_A) + self.rT * T + self.rA * G_A)
        if iirf100_max:
            iirf100_val = iirf100_val.clip(max=iirf100_max)
        alpha_val = g0 * torch.exp(iirf100_val / g1)
        return alpha_val

    def step_concentration(self, emissions, dt, alpha, R_old, G_A_old):
        decay_rate = 1 / (alpha * self.tau)
        decay_factor = torch.exp(-dt * decay_rate)
        R = emissions * self.a / decay_rate * (1 - decay_factor) + R_old * decay_factor
        G_A = torch.sum(R, dim=0)
        C = self.PI_conc + self.emis2conc * (G_A + G_A_old) / 2
        return C, R, G_A

    def step_forcing(self, C):
        logvalid_C = C / self.PI_conc
        logvalid_C[logvalid_C <= 0] = 1.
        logforc = self.f1 * torch.log(logvalid_C)

        linforc = self.f2 * (C - self.PI_conc)

        sqrtvalid_C = torch.clip(C, min=0)
        sqrtforc = self.f3 * (torch.sqrt(sqrtvalid_C) - torch.sqrt(self.PI_conc))

        RF = logforc + linforc + sqrtforc
        return RF

    def step_temperature(self, S_old, F, q, d, dt=1):
        decay_factor = torch.exp(-dt / d)  # (n_boxes, n_lat, n_lon)
        S_new = q * (1 - decay_factor) * F.unsqueeze(0) + S_old * decay_factor
        T = torch.sum((S_old + S_new) / 2, dim=0)
        return S_new, T

    def forward(self, inp_ar, timestep, q, d, weights, S0):
        n_species, n_timesteps = inp_ar.shape
        # Concentration, Radiative Forcing and Alpha
        C_ts, RF_ts, alpha_ts = [], [], []
        # Temperature
        T_ts = []
        glob_T_ts = []
        # S represents the results of the calculations from the thermal boxes,
        # an Impulse Response calculation (T = sum(S))
        S_ts = []
        # G represents cumulative emissions,
        # while G_A represents emissions accumulated since pre-industrial times,
        # both in the same units as emissions
        # So at any point, G - G_A is equal
        # to the amount of a species that has been absorbed
        G_A, G = torch.zeros((2, n_species)).to(inp_ar.device)
        # R in format [[index],[species]]
        R = torch.zeros((4, n_species)).to(inp_ar.device)
        # a,tau in format [[index], [species]]
        # g0, g1 in format [species]
        g0, g1 = self.calculate_g()
        for i, dt in enumerate(timestep):
            S_old = S_ts[-1] if S_ts else torch.zeros_like(S0)
            glob_T = S_old.sum(dim=0).mul(weights.view(-1, 1)).sum().div(weights.sum() * S_old.size(-1))
            alpha = self.calculate_alpha(G=G, G_A=G_A, T=glob_T, g0=g0, g1=g1)
            C, R, G_A = self.step_concentration(emissions=inp_ar[:, i],
                                                dt=dt,
                                                alpha=alpha,
                                                R_old=R,
                                                G_A_old=G_A)
            RF = self.step_forcing(C=C)
            F = self.forcing_pattern * RF.sum()
            S, T = self.step_temperature(S_old=S_old, F=F, q=q, d=d, dt=dt)
            G += inp_ar[:, i]

            alpha_ts.append(alpha)
            C_ts.append(C)
            RF_ts.append(RF)
            S_ts.append(S)
            T_ts.append(T)
            glob_T_ts.append(glob_T)
        res = {"C": torch.stack(C_ts),
               "RF": torch.stack(RF_ts),
               "T": torch.stack(T_ts).add(S0.sum(dim=0)),
               "glob_T": torch.stack(glob_T_ts),
               "S": torch.stack(S_ts).add(S0),
               "alpha": torch.stack(alpha_ts)}
        return res
