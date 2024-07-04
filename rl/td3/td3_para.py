from dataclasses import dataclass
from typing import Type, Union
import torch as th

@dataclass
class Args:
    exp_name: str = 'buildinggym-td3'
    """the name of this experiment"""
    seed: int = None
    """seed of the experiment"""
    # torch_deterministic: bool = True
    # """if toggled, `torch.backends.cudnn.deterministic=False`"""
    # cuda: bool = True
    # """if toggled, cuda will be enabled by default"""
    # track: bool = True
    # """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "energygym-td3"
    """the wandb's project name"""
    wandb_entity: str = 'buildinggym'
    """the entity (team) of wandb's project"""
    # capture_video: bool = False
    # """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "EnergyGym-td3-v1"
    """the id of the environment"""
    # total_timesteps: int = 1000
    """total timesteps of the experiments"""
    """the id of the environment"""
    # input_dim: int = 5
    # """the id of the environment"""
    # output_dim: int = 5
    # """the id of the environment"""         
    # learning_rate: float = 2.5e-2
    # """the learning rate of the optimizer"""
    # buffer_size: int = 1000
    # """the replay memory buffer size"""
    # minibatch_size: int = 64
    # """the minibatch size"""    
    # num_steps: int = 128
    # """the number of steps to run in each environment per policy rollout"""
    # start_e: float = 0.5
    # """the starting epsilon for exploration"""
    # end_e: float = 0.05
    # """the ending epsilon for exploration"""
    # exploration_fraction: float = 0.5
    # """the fraction of `total-timesteps` it takes from start-e to go end-e"""        
    # anneal_lr: bool = True
    # """Toggle learning rate annealing for policy and value networks"""
    # gamma: float = 0.99
    # """the discount factor gamma"""
    # gae_lambda: float = 0.95
    # """the lambda for the general advantage estimation"""
    # num_minibatches: int = 4
    # """the number of mini-batches"""
    # update_epochs: int = 1
    # """the K epochs to update the policy"""
    # norm_adv: bool = True
    # """Toggles advantages normalization"""
    # clip_coef: float = 2
    # """the surrogate clipping coefficient"""
    # clip_vloss: bool = False
    # """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    # ent_coef: float = 0.1
    # """coefficient of the entropy"""
    # vf_coef: float = 1
    # """coefficient of the value function"""
    # max_grad_norm: float = 10
    # """the maximum norm for the gradient clipping"""
    # target_kl: float = None
    # """the target KL divergence threshold"""
    work_time_start: str = '8:00'
    """the begining of working time"""
    work_time_end: str = '19:00'
    """the end of working time"""        
    n_time_step: int = 6
    """the number of steps in one hour"""
    # outlook_step: int = 6
    # """the number of steps to outlook for accumulate rewards"""    
    # batch_size: int = 64
    # """the batch size (computed in runtime)"""
    learning_starts: int = -1
    """the batch size (computed in runtime)"""    
    train_frequency: int = 2000
    """the batch size (computed in runtime)"""        

    device: str = 'cuda'
    learning_rate: float = 0.01
    alpha: float = 0.99
    outlook_steps: int = 6
    step_size: int = 1
    batch_size: int = 256
    # n_steps: int = 2
    # n_epochs: int = 10
    # clip_range: int = 50
    gradient_steps: int = 1
    epsilon_start: float = 0.5
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.98
    # clip_range_vf: Union[float, None] = None
    # normalize_advantage: bool = True
    tau: float = 1
    target_policy_noise: float = 0.2
    noise_std: float = 0.2
    target_noise_clip: float = 5
    policy_delay: int = 2
    gamma: float = 0.9
    max_buffer_size: int = 2000
    # gae_lambda: float = 1
    # ent_coef: float = 0
    # vf_coef: float = 0.5
    # max_grad_norm: float = 50
    use_sde: bool = False
    sde_sample_freq: int = -1
    # train_perEp: int = 1
    # pol_coef: float = 1
    total_epoch: int = max(int(100/gradient_steps),200)
    # max_train_perEp: int = 1
    # xa_init_gain: float = 1.
    optimizer_class: Type[th.optim.Optimizer] = th.optim.SGD
