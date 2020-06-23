import torch
import torch.nn.Functional as F
from layers import SpatialSoftmax

class PerceptionModule(torch.nn.Module):
    '''
        Maps raw observation (image & proprioception) O_t to a low dimension embedding s_t. 
        TODO : Normalize proprioception to have zero mean and unit variance.
    '''
    def __init__(self, visual_obv_dim, dof_obv_dim, embedding_dim=64):
        super(self, PerceptionModule).__init__()
        
        self.visual_obv_dim = visual_obv_dim
        self.dof_obv_dim = dof_obv_dim
        self.embedding_dim = embedding_dim

        assert self.visual_obv_dim[0] == self.visual_obv_dim[1]

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(8, 8), stride=4, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)
        self.ss = SpatialSoftmax(22, 22, 64)
        self.lin1 = torch.nn.Linear(22 * 22* 64, 512)
        self.lin2 = torch.nn.Linear(512, self.embedding_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, visual_obv, dof_obv=None):
        output = self.relu(self.conv1(visual_obv))
        output = self.relu(self.conv2(output))
        output = self.conv3(output)
        output = self.ss(output)
        output = torch.flatten(output)
        output = self.relu(self.lin1(output))
        output = self.lin2(output)

        if dof_obv:
            dof_obv = torch.flatten(dof_obv)
            output = torch.cat((output, dof_obv), 0)

        return output

class GoalEncoder(torch.nn.Module):
    '''
        Maps goal s_g to latent embedding z
        + NOTE First pass through the PerceptionModule to get s_g = P(O_{-1}).
    '''
    def __init__(self, embedding_dim=64, latent_dim=2048):
        super(self, GoalEncoder).__init__()

        self.embedding_dim == embedding_dim
        self.latent_dim = latent_dim
        self.lin1 = torch.nn.Linear(embedding_dim, self.latent_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, visual_obv, perception_module, dof_obv=None):
        output = perception_module(visual_obv, dof_obv)
        output = self.lin1(x)
        output = self.relu(output)
        return output

class PlanRecognizerModule(torch.nn.Module):
    ''' TODO
        A SeqVAE which maps the entire trajectory T to latent space z_p ~ q(z_p|T).
    '''
    def __init__(self):
        super(self, PlanRecognizerModule).__init__()
        pass

class PlanProposerModule(torch.nn.Module):
    ''' TODO
        A ConditionalVAE which maps initial state s_0 and goal z to latent space z_p ~ p(z_p|s_0, z)
    '''
    def __init__(self):
        super(self, PlanProposerModule).__init__()
        pass

class ControlModule(torch.nn.Module):
    ''' TODO
        RNN based goal, z_p conditioned policy : a_t ~ \pi(a_t | s_t, s_g, z_p).
    '''
    def __init__(self):
        super(self, ControlModule).__init__()