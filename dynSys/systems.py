import numpy as np


class LorenzSystem:
    def __init__(self, sigma=10, beta=8/3, rho=28):
        self.name = 'Lorenz'
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.numStateVars = 3

    def initialize(self):
        return [0., 1., 1.05] + np.random.normal(loc=0, scale=0.1, size=(self.numStateVars,))
        
    def dynamics(self, state):
        dx = self.sigma * (state[1] - state[0])
        dy = self.rho * state[0] - state[1] - state[0] * state[2]
        dz = state[0] * state[1] - self.beta * state[2]
        return np.array([dx, dy, dz])
