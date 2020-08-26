import os 
import sys
import numpy as np
import torch

from demons_config import get_demons_args

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset_env'))

from file_storage import store_trajectoy
from surreal_deg import SurrealDataEnvGroup

from surreal import robosuite
from surreal.robosuite.wrappers import IKWrapper
import surreal.robosuite.utils.transform_utils as T

def collect_human_demonstrations(config):
    assert config.env == 'SURREAL'
    deg = config.deg()
    env = deg.get_env()
    # Need to use inverse-kinematics controller to set position using device 
    env = IKWrapper(env)
    
    if config.device == "keyboard":
        from surreal.robosuite.devices import Keyboard
        device = Keyboard()
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif config.device == "spacemouse":
        from surreal.robosuite.devices import SpaceMouse
        device = SpaceMouse()
    
    for run in range(config.n_runs):
        obs = env.reset()
        env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708]) 
        # rotate the gripper so we can see it easily - NOTE : REMOVE MAYBE
        env.viewer.set_camera(camera_id=2)
        env.render()
        device.start_control()

        reset = False
        task_completion_hold_count = -1
        step = 0
        tr_vobvs, tr_dof, tr_actions = [], [], []
        
        while not reset:
            if int(step % config.collect_freq) == 0:
                tr_vobvs.append(np.array(obs[deg.vis_obv_key]))
                tr_dof.append(np.array(obs[deg.dof_obv_key].flatten()))
            
            device_state = device.get_controller_state()
            dpos, rotation, grasp, reset = (
                device_state["dpos"],
                device_state["rotation"],
                device_state["grasp"],
                device_state["reset"],
            )

            current = env._right_hand_orn
            drotation = current.T.dot(rotation)  
            dquat = T.mat2quat(drotation)
            grasp = grasp - 1. 
            ik_action = np.concatenate([dpos, dquat, [grasp]])

            obs, _, done, _ = env.step(ik_action)
            env.render()

            joint_velocities = np.array(env.controller.commanded_joint_velocities)
            if env.env.mujoco_robot.name == "sawyer":
                gripper_actuation = np.array(ik_action[7:])
            elif env.env.mujoco_robot.name == "baxter":
                gripper_actuation = np.array(ik_action[14:])

            # NOTE: Action for the normal environment (not inverse kinematic)
            action = np.concatenate([joint_velocities, gripper_actuation], axis=0)
            
            if int(step % config.collect_freq) == 0:
                tr_actions.append(action)

            if (int(step % config.flush_freq) == 0) or (config.break_traj_success and task_completion_hold_count == 0):
                print("Storing Trajectory")
                trajectory = {deg.vis_obv_key : np.array(tr_vobvs), deg.dof_obv_key : np.array(tr_dof), 'action' : np.array(tr_actions)}
                store_trajectoy(trajectory, 'play')
                trajectory, tr_vobvs, tr_dof, tr_actions = {}, [], [], []

            if config.break_traj_success and env._check_success():
                if task_completion_hold_count > 0:
                    task_completion_hold_count -= 1 # latched state, decrement count
                else:
                    task_completion_hold_count = 10 # reset count on first success timestep
            else:
                task_completion_hold_count = -1

            step += 1

        env.close()

def collect_random_demonstration(config):
    assert config.env == 'SURREAL'
    deg = config.deg()
    env = deg.get_env()
    obs = env.reset()
    env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708]) 

    tr_vobvs, tr_dof, tr_actions = [], [], []
    for step in range(10):
        tr_vobvs.append(np.array(obs[deg.vis_obv_key]))
        tr_dof.append(np.array(obs[deg.dof_obv_key].flatten()))
        
        action = np.random.randn(env.dof)
        obs, reward, done, info = env.step(action)

        tr_actions.append(action)
    
    print("Storing Trajectory")
    trajectory = {deg.vis_obv_key : np.array(tr_vobvs), deg.dof_obv_key : np.array(tr_dof), 'action' : np.array(tr_actions)}
    store_trajectoy(trajectory, 'random')
    env.close()

if __name__ == "__main__":
    demon_config = get_demons_args()
    if demon_config.collect_by == 'play':
        collect_human_demonstrations(demon_config)
    elif demon_config.collect_by == 'random':
        collect_random_demonstration(demon_config)
    elif demons_config.collect_by == 'imitation':
        # NOT TESTED
        import imitate_play
        
        if demons_config.train_imitation:
            imitate_play.train_imitation(demons_config)
        imitate_play.imitate_play()