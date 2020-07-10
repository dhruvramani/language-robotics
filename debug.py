import numpy as np

vobs = np.ones((2, 2, 3))
dof = np.zeros((8))

obs = np.array([vobs, dof])
action = np.random.random([8])
s_a1 = np.array([obs, action])
s_a1 = np.expand_dims(s_a1, 0)
print(s_a1)
print(s_a1.shape)
print(s_a1[0].shape)

s_a2 = np.array([obs, action])
s_a2 = np.expand_dims(s_a2, 0)

traj = np.concatenate((s_a1, s_a2), 0)
print(traj.shape)