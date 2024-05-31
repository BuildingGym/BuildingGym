from dataclasses import dataclass
from typing import Type
import torch as th

@dataclass
class Args:
    env_id: str = "PG-v1"
    """the id of the environment"""    
    exp_name: str = 'buildinggym-PG'
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "energygym-PG"
    """the wandb's project name"""
    wandb_entity: str = 'buildinggym'
    """the entity (team) of wandb's project"""


    work_time_start: str = '6:00'
    """the begining of working time"""
    work_time_end: str = '22:00'
    """the end of working time"""   
    n_time_step: int = 6
    """the number of steps in one hour"""    

    learning_rate: float = 0.0003
    alpha: float = 0.97
    outlook_steps: int = 3
    step_size: int = 3
    shuffle: bool = True
    batch_size: int = 6
    # n_steps: int = 2
    gamma: float = 0.9
    # gae_lambda: float = 1
    ent_coef: float = 0
    # vf_coef: float = 0.5
    max_grad_norm: float = 50
    use_sde: bool = False
    sde_sample_freq: int = -1
    # train_perEp: int = 1
    pol_coef: float = 1
    total_epoch: int = 3000
    max_train_perEp: int = 1
    xa_init_gain: float = 1.
    optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam

