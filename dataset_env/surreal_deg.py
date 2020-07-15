import robosuite.robosuite as suite
from collections import OrderedDict

from deg_base import DataEnvGroup

class SurrealDataEnvGroup(DataEnvGroup):
    ''' DataEnvGroup for Surreal Robotics Suite environment. 
        
        + The observation space can be modified through `config.env_args`
        + Observation space:
            - 'robot-state': proprioceptive feature - vector of:
                > cos and sin of robot joint positions
                > robot joint velocities 
                > current configuration of the gripper.
            - 'object-state': object-centric feature 
            - 'image': RGB/RGB-D image 
                > (256 x 256 by default)
            - Refer: https://github.com/StanfordVL/robosuite/tree/master/robosuite/environments

        + The action spaces by default are joint velocities and gripper actuations.
            - To use the end-effector action-space use inverse-kinematics using IKWrapper.
            - Refer: https://github.com/StanfordVL/robosuite/tree/master/robosuite/wrappers
    '''
    def __init__(self, get_episode_type=None):
        super(SurrealDataEnvGroup, self).__init__(get_episode_type)
        assert self.env_name == 'SURREAL'
        
        dummy_env = suite.make(self.config.env_type, **self.config.env_args)
        obs = dummy_env.reset()

        self.vis_obv_key = 'image'
        self.dof_obv_key = 'robot-state'
        self.obs_space = {self.vis_obv_key: (256, 256, 3), self.dof_obv_key: (8)}
        self.action_space = dummy_env.dof  # TODO : DEBUG here maybe

        del dumm_env

    def get_env(self):
        env = suite.make(self.config.env_type, **self.config.env_args)
        _ = env.reset()
        return env

    def play_trajectory(self):
        # TODO 
        # Refer https://github.com/StanfordVL/robosuite/blob/master/robosuite/scripts/playback_demonstrations_from_hdf5.py
        raise NotImplementedError