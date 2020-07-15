from furniture.env import make_env
from deg_base import DataEnvGroup

# TODO
class FurnitureDataEnvGroup(DataEnvGroup):
    def __init__(self, get_episode_type=None):
        super(FurnitureDataEnvGroup, self).__init__(get_episode_type)
        assert self.env_name == 'FURNITURE'
        
        raise NotImplementedError
        self.vis_obv_key = None
        self.dof_obv_key = None
        
        self.obs_space = {key : value.shape for key, value in obs.items()}
        self.action_space = dummy_env.dof
