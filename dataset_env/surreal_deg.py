import robosuite.robosuite as suite
from collections import OrderedDict

from deg_base import DataEnvGroup

class SurrealDataEnvGroup(DataEnvGroup):
    def __init__(self):
        super(SurrealDataEnvGroup, self).__init__()
        assert self.env_name == 'SURREAL'
        
        dummy_env = suite.make(self.config.env_type, **self.config.env_args)
        obs = dummy_env.reset()

        # TODO : DEBUG here maybe
        self.obs_space = {key : value.shape for key, value in obs.items()}
        self.action_space = dummy_env.dof

        del dumm_env

    def get_env(self):
        env = suite.make(self.config.env_type, **self.config.env_args)
        env.reset()
        return env