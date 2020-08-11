import numpy
import torch

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def test_experiment(config):
    deg = config.deg()
    env = deg.get_env()
    vobs_dim, dof_dim = deg.obs_space[deg.vis_obv_key], deg.obs_space[deg.dof_obv_key] 
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
            goal = deg.get_random_goal()
            goal = perception_module(goal_obv)
            goal, _. _ = visual_goal_encoder(goal) 
            t, done = 0, False

            while (not done) and t <= config.max_test_timestep: 
                # TODO : Figure out way to set done tru when goal is reached
                visual_obv, dof_obv = obvs[deg.vis_obv_key], obvs[deg.dof_obv_key]
                state = perception_module(visual_obv, dof_obv)
                z_p, _, _ = plan_proposer(state, goal)
                
                action, _ = control_module.step(state, goal, z_p)
                obvs, _, done, _ = env.step(action)
                env.render()
                t += 1

def test_with_lang(config):
    deg = config.deg()
    env = deg.get_env()
    vobs_dim, dof_dim = deg.obs_space[deg.vis_obv_key], deg.obs_space[deg.dof_obv_key] 
    act_dim = deg.action_space # TODO : DEBUG here

    with torch.no_grad():
        perception_module = PerceptionModule(vobs_dim, dof_dim, config.visual_state_dim).to(device)
        plan_proposer = PlanProposerModule(config.combined_state_dim, config.goal_dim, config.latent_dim).to(device)
        control_module = ControlModule(act_dim, config.combined_state_dim, config.goal_dim, config.latent_dim).to(device)
        if config.use_lang_model:
            instruction_encoder = LanguageModelInstructionEncoder(config.lang_model, config.latent_dim).to(device)
        else:
            instruction_encoder = BasicInstructionEncoder(config.latent_dim).to(device)

        perception_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'perception.pth')))
        plan_proposer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'plan_proposer.pth')))
        control_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'control_module.pth')))
        if config.use_lang_model:
            instruction_encoder.goal_dist.load_state_dict(torch.load(os.path.join(config.models_save_path, 'lang_model_{}.pth'.format(config.lang_model))))
        else:
            instruction_encoder.load_state_dict(torch.load(os.path.join(config.models_save_path, 'basic_instruct_model.pth')))
        
        obvs = env.reset()
        for i in range(config.n_test_evals):
            obvs = env.reset()
            instruction = input("Type instruction: ")
            goal = instruction_encoder(instruction)
            t, done = 0, False

            while (not done) and t <= config.max_test_timestep: 
                # TODO : Figure out way to set done true when goal is reached
                visual_obv, dof_obv = obvs[deg.vis_obv_key], obvs[deg.dof_obv_key]
                state = perception_module(visual_obv, dof_obv)
                z_p, _, _ = plan_proposer(state, goal)
                
                action, _ = control_module.step(state, goal, z_p)
                obvs, _, done, _ = env.step(action)
                env.render()
                t += 1