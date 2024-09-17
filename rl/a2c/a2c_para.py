from dataclasses import dataclass
from typing import Type
import torch as th

@dataclass
class Args:
    env_id: str = "A2C-v1"
    """the id of the environment"""    
    exp_name: str = 'buildinggym-a2c'
    """the name of this experiment"""
    seed: int = 6
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "energygym-a2c-paper-fintune"
    """the wandb's project name"""
    wandb_entity: str = 'buildinggym'
    """the entity (team) of wandb's project"""


    work_time_start: str = '6:00'
    """the begining of working time"""
    work_time_end: str = '22:00'
    """the end of working time"""   
    n_time_step: int = 6
    """the number of steps in one hour"""    

    learning_rate: float = 0.001
    alpha: float = 0.99
    outlook_steps: int = 6
    step_size: int = 1
    batch_size: int = 6
    n_steps: int = 2
    gamma: float = 0.9
    gae_lambda: float = 1
    ent_coef: float = 0
    vf_coef: float = 0.5
    max_grad_norm: float = 50
    use_sde: bool = False
    sde_sample_freq: int = -1
    # train_perEp: int = 1
    pol_coef: float = 1
    total_epoch: int = 200
    # max_train_perEp: int = 1
    # xa_init_gain: float = 1.
    optimizer_class: Type[th.optim.Optimizer] = th.optim.RMSprop

