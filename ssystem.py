# Implements shooting method for S-System kinetics model

import numpy as np
import numpy.linalg as la


class SSystem(object):
    def __init__(self, n_vars):
        self.n_vars = n_vars
        # g is matrix of production degrees
        # h is matrix of degradation degrees
        self.g = np.zeros((self.n_vars, self.n_vars))
        self.h = np.zeros((self.n_vars, self.n_vars))
        self.alpha = np.zeros(self.n_vars)
        self.beta = np.zeros(self.n_vars)

    # parameter inference with alternating regression
    # inputs:
    #     states: np.array(steps, n_vars)
    #     velocities: np.array(steps, n_vars)
    #     steps: int
    # output:
    #     np.array(n_vars, n_vars) of final parameters
    def solve(self, states, velocities, iterations=100):

        # number of timesteps
        N = states.shape[0]

        # Compute L, matrix of log regressors, and C=(L^TL)^-1L^T
        L = np.hstack([np.ones((N,1)), np.log(states)])
        C = np.matmul(la.inv(np.matmul(L.T, L)), L.T)

        g_constraint = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        h_constraint = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])

        # continue random generation while NaNs exist
        tries = 0
        while True:
            # increment number of tries
            tries += 1

            # randomly initialize parameters
            self.g = np.random.random((self.n_vars, self.n_vars))
            self.h = np.random.random((self.n_vars, self.n_vars))
            self.g *= g_constraint
            self.h *= h_constraint
            self.alpha = np.random.random(self.n_vars)
            self.beta = np.random.random(self.n_vars) * 10

            for i in range(iterations):
                # Estimate production parameters
                xh = np.product(np.power(states[:,np.newaxis,:], self.h), axis=2)
                yd = np.log(velocities + self.beta * xh)
                
                bp = np.matmul(C, yd)
                self.alpha = np.exp(bp[0])
                self.g = bp[1:].T
                self.g *= g_constraint

                # Estimate degradation parameters
                xg = np.product(np.power(states[:,np.newaxis,:], self.g), axis=2)
                yp = np.log(self.alpha * xg - velocities)
                bd = np.matmul(C, yp)
                self.beta = np.exp(bd[0])
                self.h = bd[1:].T
                self.h *= h_constraint

            if not np.any(np.isnan(self.alpha)) and not np.any(np.isnan(self.beta)) and not np.any(np.isnan(self.g)) and not np.any(np.isnan(self.h)):
                print("Took {} tries".format(tries))
                break


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
        print(state, velocity)
        states = []
        velocities = []
        self.state = state.copy()
        self.velocity = velocity.copy()
        for i in range(steps):
            xg = np.product(np.power(self.state, self.g), axis=1)
            xh = np.product(np.power(self.state, self.h), axis=1)
            self.velocity = self.alpha * xg - self.beta * xh
            states.append(self.state.copy())
            velocities.append(self.velocity.copy())
            self.state += self.velocity * delta
        return np.vstack(states), np.vstack(velocities)
