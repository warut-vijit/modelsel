# Implements shooting method for Michaelis-Menten dynamics

import numpy as np
import matplotlib.pyplot as plt


class DynamicsModel(object):
    def __init__(self, n_vars):
        self.n_vars = n_vars
        self.state = np.zeros(n_vars)
        self.velocity = np.zeros(n_vars)

    def run(self, state, velocity, delta=0.001, steps=100):
        states = [state.copy()]
        velocities = [velocity]
        self.state = state
        self.velocity = velocity
        for i in range(steps):
            self.velocity = self.get_vel()
            self.state += self.velocity * delta
            states.append(self.state.copy())
            velocities.append(self.velocity.copy())
        return np.vstack(states), np.vstack(velocities)


# toy example of oscillator with two units
class Oscillator(DynamicsModel):
    def __init__(self):
        super().__init__(2)

    def get_vel(self):
        v = np.zeros(2)
        v[0] = 2 * self.state[1]
        v[1] = -1 * self.state[0]

        return v


# A Model for Circadian Oscillations in the Drosophila Period Protein (PER)
class Goldbeter_1995(DynamicsModel):
    def __init__(self, Vs=0.76, Vm=0.65, n=4, Kl=1, Km=0.5, ks=0.38, V1=3.2, K1=2, K2=2, V2=1.58, V3=5, K3=2, V4=2.5, K4=2, Vd=0.95, Kd=0.2, k1=1.9, k2=1.3):
        super().__init__(5)
        self.Vs = Vs
        self.Vm = Vm
        self.n = n
        self.Kl = Kl
        self.Km = Km
        self.ks = ks
        self.V1 = V1
        self.K1 = K1
        self.K2 = K2
        self.V2 = V2
        self.V3 = V3
        self.K3 = K3
        self.V4 = V4
        self.K4 = K4
        self.Vd = Vd
        self.Kd = Kd
        self.k1 = k1
        self.k2 = k2

    def get_vel(self):
        v = np.zeros(5)
        v[0] = self.Vs * np.power(self.Kl, self.n) / (np.power(self.Kl, self.n) + np.power(self.state[4], self.n)) - self.Vm * self.state[0] / (self.Km + self.state[0])
        v[1] = self.ks * self.state[0] - self.V1 * self.state[1] / (self.K1 + self.state[1]) + self.V2 * self.state[2] / (self.K2 + self.state[2])
        v[2] = self.V1 * self.state[1] / (self.K1 + self.state[1]) - self.V2 * self.state[2] / (self.K2 + self.state[2]) - self.V3 * self.state[2] / (self.K3 + self.state[2]) + self.V4 * self.state[3] / (self.K4 + self.state[3])
        v[3] = self.V3 * self.state[2] / (self.K3 + self.state[2]) - self.V4 * self.state[3] / (self.K4 + self.state[3]) - self.Vd * self.state[3] / (self.Kd + self.state[3]) - self.k1 * self.state[3] + self.k2 * self.state[4]
        v[4] = self.k1 * self.state[3] - self.k2 * self.state[4]

        return v


if __name__ == "__main__":
    model = Goldbeter_1995()
    states, velocities = model.run(state=np.random.random(5) * 3, velocity=np.random.random(5), delta=0.1, steps=1000)
    for i in range(states.shape[1]):
        plt.plot(states[:,i], label="X {}".format(i+1))
    plt.legend()
    plt.show()
