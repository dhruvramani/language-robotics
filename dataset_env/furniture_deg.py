from furniture.env import make_env
from deg_base import DataEnvGroup

# TODO
class FurnitureDataEnvGroup(DataEnvGroup):
    def __init__(self):
        super(FurnitureDataEnvGroup, self).__init__()
        assert self.env_name == 'FURNITURE'
        
        raise NotImplementedError

        self.obs_space = {key : value.shape for key, value in obs.items()}
        self.action_space = dummy_env.dof
