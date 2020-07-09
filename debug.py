# import os, sys
import torch
a = torch.zeros((5, 5))
a = torch.Tensor(a)
print(a.cpu())

# sys.path.insert(1, os.path.join(sys.path[0], '../model'))

# import utils

# class ModuleA(torch.nn.Module):
#     def __init__(self):
#         super(ModuleA, self).__init__()
#         self.a = torch.nn.Linear(500, 200)

#     def forward(self, x):
#         return self.a(x)

# class ModuleB(torch.nn.Module):
#     def __init__(self, modA):
#         super(ModuleB, self).__init__()
#         self.ab = torch.nn.Linear(200, 100)
#         self.moda = modA

#     def forward(self, x):
#         x = self.moda(x)
#         return self.ab(x)

# a = ModuleA()
# b = ModuleB(a)

# print(utils.count_vars(a))
# print(utils.count_vars(b))

# from global_config import *
# print(BASE_DIR)
# print(TIME_STAMP)

# parser = get_global_parser()
# parser.add_argument('--foo', type=str2bool, default='True')
# config = parser.parse_args()
# config.foo = str2bool('False')
# print(config.foo)