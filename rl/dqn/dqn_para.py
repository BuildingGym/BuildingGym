from dataclasses import dataclass
import os

@dataclass
class Args:
    cuda: bool = True
    """if toggled, cuda will be enabled by default, use GPU to train model"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "energygym"
    """the wandb's project name"""
    wandb_entity: str = 'buildinggym'
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "EnergyGym-v1"
    """the id of the environment"""
    input_dim: int = 5
    """the id of the environment"""
    output_dim: int = 5
    """the id of the environment"""        
    total_timesteps: int = 1000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-2
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000
    """the replay memory buffer size"""
    gamma: float = 0.9
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 30
    """the timesteps it takes to update the target network"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    start_e: float = 0.5
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 0
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""
    work_time_start: str = '6:00'
    """the begining of working time"""
    work_time_end: str = '22:00'
    """the end of working time"""        
    n_time_step: int = 6
    """the number of steps in one hour"""
    outlook_step: int = 6
    """the number of steps to outlook for accumulate rewards"""