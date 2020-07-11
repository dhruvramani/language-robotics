import numpy
import torch

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
def test_experiment(deg, config):
    env = deg.get_env()
    vobs_dim, dof_dim = deg.obs_space['image'], deg.obs_space['robot-state'] 
    act_dim = deg.action_space # TODO : DEBUG here

    with torch.no_grad():
        perception_module = PerceptionModule(vobs_dim, dof_dim, config.visual_state_dim).to(device)
        visual_goal_encoder = VisualGoalEncoder(config.visual_state_dim, config.goal_dim).to(device)
        plan_proposer = PlanProposerModule(config.combined_state_dim, config.goal_dim, config.latent_dim).to(device)
        control_module = ControlModule(act_dim, config.combined_state_dim, config.goal_dim, config.latent_dim).to(device)

        perception_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'perception.pth')))
        visual_goal_encoder.load_state_dict(torch.load(os.path.join(config.models_save_path, 'visual_goal.pth')))
        plan_proposer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'plan_proposer.pth')))
        control_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'control_module.pth')))

        obvs = env.reset()
        for i in range(config.n_test_evals):
            obvs = env.reset()
            goal_obv = get_goal() # TODO *IMPORTANT*
            t, done = 0, False

            while (not done) and t <= config.max_test_timestep:
                visual_obv, dof_obv = obvs['image'], obvs['robot-state']
                state = perception_module(visual_obv, dof_obv)
                goal = perception_module(goal_obv)
                goal = visual_goal_encoder(goal) 
                z_p, _, _ = plan_proposer(state, goal)
                
                action, _ = control_module.step(state, goal, z_p)
                obvs, _, done, _ = env.step(action)
                env.render()
                t += 1