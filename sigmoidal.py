# Implements shooting method for Sigmoidal kinetics model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sigmoidal(nn.Module):
    def __init__(self, n_vars):
        super(Sigmoidal, self).__init__()
        self.n_vars = n_vars
        self.lin2 = nn.Linear(self.n_vars, self.n_vars)
        self.lin1 = nn.Linear(self.n_vars, self.n_vars)

    def forward(self, state):
        z = torch.sigmoid(self.lin1(state))
        return self.lin2(z)

    # parameter inference with stochastic gradient descent
    # inputs:
    #     states: np.array(steps, n_vars)
    #     velocities: np.array(steps, n_vars)
    #     steps: int
    # output:
    #     np.array(n_vars, n_vars) of final parameters
    def solve(self, states, velocities, iterations=1000, lr=1.0):
        states = torch.tensor(states, dtype=torch.float)
        velocities = torch.tensor(velocities, dtype=torch.float)
        optim = torch.optim.SGD(self.parameters(), lr=lr)
        for i in range(iterations):
            optim.zero_grad()
            output = self.forward(states)
            loss = F.mse_loss(output, velocities)
            loss.backward()
            optim.step()

    # rolls out a trajectory from optional initial condition
    # inputs:
    #     state: np.array(n_vars)    - initial states
    #     velocity: np.array(n_vars) - initial time derivative
    #     delta: float               - time difference between steps
    #     steps: int                 - number of steps to roll out
    # output:
    #     np.array(steps, n_vars), np.array(steps, n_vars)
    #     states and velocities for each simulated timestep
    def run(self, state=None, velocity=None, delta=0.001, steps=100):
        if state is None:
            state = np.random.random(self.n_vars)
        if velocity is None:
            velocity = np.random.random(self.n_vars)
        states = []
        velocities = []
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            velocity = torch.tensor(velocity, dtype=torch.float)
            for i in range(steps):
                velocity = self.forward(state)
                states.append(state.numpy().copy())
                velocities.append(velocity.numpy().copy())
                state += velocity * delta
        return np.vstack(states), np.vstack(velocities)
