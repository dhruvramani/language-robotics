import torch
import torch.nn.Functional as F
from layers import SpatialSoftmax, ConditionalVAE, SeqVAE

class PerceptionModule(torch.nn.Module):
    '''
        Maps raw observation (image & proprioception) O_t to a low dimension embedding s_t. 
        TODO : Normalize proprioception to have zero mean and unit variance.
    '''
    def __init__(self, visual_obv_dim=[200, 200], dof_obv_dim=[8], embedding_dim=64):
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
            output = torch.cat((output, dof_obv), -1)

        return output

class GoalEncoder(torch.nn.Module):
    '''
        Maps goal s_g to latent embedding z
        + NOTE First pass through the PerceptionModule to get s_g = P(O_{-1}).
    '''
    def __init__(self, state_dim=64, latent_dim=2048):
        super(self, GoalEncoder).__init__()

        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.lin1 = torch.nn.Linear(state_dim, self.latent_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, visual_obv, perception_module):
        assert visual_obv.size() != 2 # To filter dof-obvs
        if(visual_obv.size([-1]) != self.state_dim):
            visual_obv = perception_module(visual_obv, None)
        output = self.lin1(visual_obv)
        output = self.relu(output)
        return output

class PlanRecognizerModule(torch.nn.Module):
    ''' 
        A SeqVAE which maps the entire trajectory T to latent space z_p ~ q(z_p|T).
        NOTE : Pass a single trajectory [[[Ot/st, at], ... K]] only, where Ot = [Vobv, DOF].
        NOTE : Trajectory shape : (1, K, 2)
    '''
    def __init__(self, sequence_length, state_dim=72, latent_dim=256):
        super(self, PlanRecognizerModule).__init__()
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.seqVae = SeqVAE(self.sequence_length, self.state_dim, latent_size=self.latent_dim)

    def forward(self, trajectory, perception_module):
        assert trajectory.size()[0] == 1 and trajectory.size()[1] == self.sequence_length
        if trajectory[0][0][0].size()[-1] != self.state_dim:
            for i in self.sequence_length:
                visual_obv, dof_obv = trajectory[0][i][0]
                trajectory[0][i][0] = perception_module(visual_obv, dof_obv)

        for i in self.sequence_length:
            trajectory[0][i] = torch.cat((trajectory[0][i][0], trajectory[0][i][1]), -1)

        z, mean, logv = self.seqVae(trajectory)
        return z, mean, logv

class PlanProposerModule(torch.nn.Module):
    ''' 
        A ConditionalVAE which maps initial state s_0 and goal z to latent space z_p ~ p(z_p|s_0, z)
    '''
    def __init__(self, state_dim=72, goal_dim=2048, latent_dim=256):
        super(self, PlanProposerModule).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim
        self.cVae = ConditionalVAE(state_size + goal_dim, latent_dim)

    def forward(self, initial_obv, goal_obv, perception_module, goal_encoder):
        if initial_obv.size()[-1] != self.state_dim:
            visual_obv, dof_obv = initial_obv
            initial_obv = perception_module(visual_obv, dof_obv)

        if goal_obv.size()[-1] != self.goal_dim:
            goal_obv = goal_encoder(goal_obv, perception_module)

        z, mean, logv = self.cVae(initial_obv, goal_obv)
        return z, mean, logv

class ControlModule(torch.nn.Module):
    ''' TODO
        RNN based goal, z_p conditioned policy : a_t ~ \pi(a_t | s_t, s_g, z_p).
    '''
    def __init__(self):
        super(self, ControlModule).__init__()