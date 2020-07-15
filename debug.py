# import numpy as np

# vobs = np.ones((2, 2, 3))
# dof = np.zeros((8))

# obs = np.array([vobs, dof])
# action = np.random.random([8])
# s_a1 = np.array([obs, action])
# s_a1 = np.expand_dims(s_a1, 0)
# print(s_a1)
# print(s_a1.shape)
# print(s_a1[0].shape)

# s_a2 = np.array([obs, action])
# s_a2 = np.expand_dims(s_a2, 0)

# traj = np.concatenate((s_a1, s_a2), 0)
# print(traj.shape)

# class Outer():
#     def __init__(self):
#         self.a = 10
#         self.inner = self.Inner(self.a)
#         print(self.inner.b)

#     class Inner():
#         def __init__(self, a):
#             self.ab = a
#             print(self.ab)
#             self.b = 100

# o = Outer()

layers_size = [2048] * 4
for i, (in_size) in enumerate(layers_size[1:]):
    print(in_size)
