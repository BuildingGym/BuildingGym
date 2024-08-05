import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
import math

class mycallback(BaseCallback):
    def __init__(self, model, verbose: int = 0):
        self.model = model
        super().__init__(verbose)

    def on_rollout_end(self) -> None:
        super().on_rollout_end()
        result = np.mean(self.model.env.sensor_dic['results'].iloc[np.where(self.model.env.sensor_dic['Working time'])[0]])
        reward = np.mean(self.model.env.sensor_dic['rewards'].iloc[np.where(self.model.env.sensor_dic['Working time'])[0]])
        # prob = np.mean(np.exp(self.model.env.sensor_dic['logprobs'].iloc[np.where(self.model.env.sensor_dic['Working time'])[0]]))
        p_loss = np.mean(self.model.env.p_loss_list)
        v_loss =  np.mean(self.model.env.v_loss_list)
        # prob = self.model.env.prob
        lr = self.model.learning_rate
        wandb.log({'reward_curve': reward}, step=self.num_timesteps)        
        wandb.log({'result_curve': result}, step=self.num_timesteps)
        # wandb.log({'action prob': prob}, step=self.num_timesteps)
        wandb.log({'p_loss_curve': float(p_loss)}, step=self.num_timesteps)
        wandb.log({'v_loss_curve': float(v_loss)}, step=self.num_timesteps)

    def on_epoch_end(self):
        result = np.mean(self.model.env.sensor_dic['results'].iloc[np.where(self.model.env.sensor_dic['Working time'])[0]])
        reward = np.mean(self.model.env.sensor_dic['rewards'].iloc[np.where(self.model.env.sensor_dic['Working time'])[0]])        
        wandb.log({'reward_curve': reward}, step=self.num_timesteps)
        wandb.log({'result_curve': result}, step=self.num_timesteps)
        wandb.log({'v_loss_curve': float(self.model.critic_losses_i)}, step=self.num_timesteps)
        if not math.isnan(self.model.actor_losses_i):
            wandb.log({'p_loss_curve': float(self.model.actor_losses_i)}, step=self.num_timesteps)

    def _on_step(self):
        pass
    
    def per_time_step(self, var = None) -> None:
        # super().on_epoch_end()
        if var is not None:
            p_loss = var['loss'].item()
            # wandb.log({'p_loss_curve': float(p_loss)}, step=self.num_timesteps)       