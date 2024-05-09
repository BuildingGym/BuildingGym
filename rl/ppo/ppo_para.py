from dataclasses import dataclass

@dataclass
class Args:
    exp_name: str = 'buildinggym-ppo'
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "energygym"
    """the wandb's project name"""
    wandb_entity: str = 'buildinggym'
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "EnergyGym-ppo-v1"
    """the id of the environment"""
    total_timesteps: int = 1000
    """total timesteps of the experiments"""
    """the id of the environment"""
    input_dim: int = 5
    """the id of the environment"""
    output_dim: int = 5
    """the id of the environment"""         
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 1000
    """the replay memory buffer size"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    start_e: float = 0.5
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""        
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    work_time_start: str = '6:00'
    """the begining of working time"""
    work_time_end: str = '22:00'
    """the end of working time"""        
    n_time_step: int = 6
    """the number of steps in one hour"""
    outlook_step: int = 6
    """the number of steps to outlook for accumulate rewards"""    
    batch_size: int = 64
    """the batch size (computed in runtime)"""
    learning_starts: int = 1
    """the batch size (computed in runtime)"""    
    train_frequency: int = 1
    """the batch size (computed in runtime)"""        
