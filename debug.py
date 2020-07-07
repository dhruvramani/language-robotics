import os, sys
import torch

sys.path.insert(1, os.path.join(sys.path[0], '../model'))

import utils

class ModuleA(torch.nn.Module):
    def __init__(self):
        super(ModuleA, self).__init__()
        self.a = torch.nn.Linear(500, 200)

    def forward(self, x):
        return self.a(x)

class ModuleB(torch.nn.Module):
    def __init__(self, modA):
        super(ModuleB, self).__init__()
        self.ab = torch.nn.Linear(200, 100)
        self.moda = modA

    def forward(self, x):
        x = self.moda(x)
        return self.ab(x)

a = ModuleA()
b = ModuleB(a)

print(utils.count_vars(a))
print(utils.count_vars(b))
