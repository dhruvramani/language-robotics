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

# import sys
# import os

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print("hello")


# def foo():
#     print(sys.path)
import numpy as np
import torch

class Mod(torch.nn.Module):
    def __init__(self):
        super(Mod, self).__init__()
        self.a = np.array(10)
        self.b = [1, 2, 4, 5 ]
        self.lin1 = torch.nn.Linear(10, 20)
        self.lin2 = torch.nn.Linear(20, 30)
        self.lin3 = torch.nn.Linear(30, 40)

    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(x)))

a = Mod()
print(list(a.state_dict()))
