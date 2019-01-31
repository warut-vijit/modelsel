from mmsystem import Goldbeter_1995
from ssystem import SSystem
from sigmoidal import Sigmoidal
import matplotlib.pyplot as plt
import numpy as np

mm_model = Goldbeter_1995()
steps = 50
delta = 0.01

#states, velocities = mm_model.run(state=initial_state, velocity=initial_velocity, delta=0.1, steps=3)
#for i in range(states.shape[1]):
#    plt.plot(states[:,i], label="MM X {}".format(i+1))
trainer = SSystem(n_vars=4)
trainer.g = np.array([[0, 0, -0.8, 0], [0.5, 0, 0, 0], [0, 0.75, 0, 0], [0.5, 0, 0, 0]])
trainer.h = np.array([[0.5, 0, 0, 0], [0, 0.75, 0, 0], [0, 0, 0.5, 0.2], [0, 0, 0, 0.8]])
trainer.alpha = np.array([12., 8., 3., 2.])
trainer.beta = np.array([10., 3., 5., 6.])

all_states = []
all_velocities = []

while len(all_states) < 1:
    initial_state = np.random.random(4)
    initial_velocity = np.random.random(4)
    states, velocities = trainer.run(state=initial_state, velocity=initial_velocity, delta=delta, steps=steps)
    if not np.any(np.isnan(states)) and not np.any(np.isnan(velocities)):
        all_states.append(states)
        all_velocities.append(velocities)

all_states = np.vstack(all_states)
all_velocities = np.vstack(all_velocities)
for i in range(states.shape[1]):
    plt.plot(states[:,i], label="Trainer X {}".format(i+1))

#ssystem = SSystem(n_vars=4)
#ssystem.solve(all_states, all_velocities, iterations=1)
#states, velocities = ssystem.run(state=initial_state, velocity=initial_velocity, delta=delta, steps=steps)
#for i in range(states.shape[1]):
#    plt.plot(states[:,i], label="S-Sys X {}".format(i+1))

nnsystem = Sigmoidal(n_vars=4)
nnsystem.solve(all_states, all_velocities)
states, velocities = nnsystem.run(state=initial_state, velocity=initial_velocity, delta=delta, steps=steps)
for i in range(states.shape[1]):
    plt.plot(states[:,i], label="S-Sys X {}".format(i+1))

plt.legend()
plt.show()
