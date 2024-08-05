from dataclasses import dataclass
from typing import Type
import torch as th

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
    wandb_project_name: str = "energygym-ppo"
    """the wandb's project name"""
    wandb_entity: str = 'buildinggym'
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "EnergyGym-ppo-v1"
    """the id of the environment"""
    """training time of per epoch"""
    input_dim: int = 5
    """the id of the environment"""
    output_dim: int = 5
    """the id of the environment"""         
    buffer_size: int = 1000
    """the replay memory buffer size"""
    # batch_size: int = 64
    # """the minibatch size"""
    optimizer_class: Type[th.optim.Optimizer] = th.optim.SGD
    """the optimizer"""    
    minibatch_size: int = 64
    """the minibatch size"""    
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
    # gamma: float = 0.99
    # """the discount factor gamma"""
    # gae_lambda: float = 0.95
    # """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 1
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    # ent_coef: float = 0.1
    # """coefficient of the entropy"""
    # vf_coef: float = 1
    # """coefficient of the value function"""
    max_grad_norm: float = 10
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    work_time_start: str = '6:00'
    """the begining of working time"""
    work_time_end: str = '22:00'
    """the end of working time"""        
    n_time_step: int = 6
    """the number of steps in one hour"""
    # outlook_step: int = 6
    # """the number of steps to outlook for accumulate rewards"""    
    # batch_size: int = 64
    # """the batch size (computed in runtime)"""
    # learning_starts: int = 1
    """the batch size (computed in runtime)"""    
    # train_frequency: int = 1
    """the batch size (computed in runtime)"""        

    learning_rate: float = 0.001
    n_epochs: int = 2
    alpha: float = 0.99
    outlook_steps: int = 6
    step_size: int = 1
    batch_size: int = 6
    n_steps: int = 2
    gamma: float = 0.9
    gae_lambda: float = 0.95
    ent_coef: float = 0
    vf_coef: float = .5
    max_grad_norm: float = 50
    use_sde: bool = False
    sde_sample_freq: int = -1
    # train_perEp: int = 1
    pol_coef: float = 1
    total_epoch: int = 200
    # max_train_perEp: int = 1
    # xa_init_gain: float = 1.
    optimizer_class: Type[th.optim.Optimizer] = th.optim.RMSprop
