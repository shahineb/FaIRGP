import torch
from gpytorch import kernels



class ThermalBoxKernel(kernels.Kernel):

    def __init__(self, emission_kernel, space_kernel, d_map, q_map):
        self.emission_kernel = emission_kernel
        self.space_kernel = space_kernel
        self.register_buffer('d_map', d_map)
        self.register_buffer('q_map', q_map)
