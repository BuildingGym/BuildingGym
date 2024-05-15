from env.env import buildinggym_env
import gymnasium as _gymnasium_
from energyplus.ooep.addons.rl import (
    VariableBox,
    SimulatorEnv,
)
import math
from energyplus.ooep import (
    Actuator,
    OutputVariable,
)
from rl.a2c.a2c_policy import ActorCriticPolicy
from rl.util.schedule import ConstantSchedule
from gymnasium.spaces import (
    Box,
    Discrete
)
import numpy as _numpy_
import numpy as np
from rl.a2c.a2c_para import Args
from rl.a2c.a2c import A2C
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import tyro
import time

observation_space = _gymnasium_.spaces.Dict({
            't_out': VariableBox(
                low=22.8, high=33.3,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Site Outdoor Air Drybulb Temperature',
                key='Environment',
            )),
            't_in': VariableBox(
                low=22, high=27,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Zone Mean Air Temperature',
                key='CORE_BOTTOM ZN',
            )),
            'occ': VariableBox(
                low=0, high=1,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Schedule Value',
                key='Large Office Bldg Occ',
            )),
            'light': VariableBox(
                low=0, high=1,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Schedule Value',
                key='Large Office Bldg Light',
            )),
            'Equip': VariableBox(
                low=0, high=1,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Schedule Value',
                key='Large Office Bldg Equip',
            )),   
            'Chiller Electricity Rate': VariableBox(
                low=0, high=1000,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Chiller Electricity Rate',
                key='DOE REF 1980-2004 WATERCOOLED  CENTRIFUGAL CHILLER 0 1100TONS 0.7KW/TON',
            )),
        })
action_space = _gymnasium_.spaces.Dict({
                    'Thermostat': VariableBox(
                        low=22., high=30.,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(Actuator.Ref(
                        type='Schedule:Compact',
                        control_type='Schedule Value',
                        key='ANN-ctrl',
                    ))
                })
schedule = ConstantSchedule(0.0001)
input_sp = Box(np.array([0] * 5), np.array([1] * 5))
action_sp = Discrete(5)
agent = ActorCriticPolicy(input_sp, action_sp, schedule.value)
env = buildinggym_env('Large office - 1AV232.idf',
                    'USA_FL_Miami.722020_TMY2.epw',
                    observation_space,
                    action_space,
                    agent,
                    Args)

class callback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def on_rollout_end(self) -> None:
        super().on_rollout_end()
        if hasattr(self.model.env, 'p_loss'):
            result = np.mean(self.model.data_wt['results'])
            reward = np.mean(self.model.data_wt['rewards'])
            p_loss = self.model.env.p_loss
            v_loss = self.model.env.v_loss
            prob = self.model.env.prob
            lr = self.model.learning_rate
            wandb.log({'reward_curve': reward}, step=self.num_timesteps)        
            wandb.log({'result_curve': result}, step=self.num_timesteps)        
            wandb.log({'action prob': math.exp(prob)}, step=self.num_timesteps)        
            wandb.log({'p_loss_curve': float(p_loss)}, step=self.num_timesteps)      
            wandb.log({'v_loss_curve': float(v_loss)}, step=self.num_timesteps)      

my_callback = callback()
args = tyro.cli(Args)
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"



a = A2C(ActorCriticPolicy, env, Args, my_callback)

wandb.init(
    project=args.wandb_project_name,
    entity=args.wandb_entity,
    sync_tensorboard=True,
    config=args,
    name=run_name,
    save_code=False,
)
_, performance = a.learn(args.total_epoch, my_callback)



parameters_dict = {
'learning_rate': {
    'values': [1e-2, 1e-4, 1e-6]
    },
'train_perEp': {
        'values': [1, 10, 100]
    },  
'gamma': {
        'values': [0.9, 0.8, 0.5]
    },         
# 'start_e': {
#       'values': [0.8, 0.5, 0.2]
#     },     
# 'train_frequency': {
#       'values': [1, 5, 10]
#     },   
# 'target_network_frequency': {
#       'values': [5, 20, 30]
#     },                                
}
sweep_config = {
'method': 'random'
}
metric = {
'name': 'performance',
'goal': 'maximize'   
}
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

# sweep_id = wandb.sweep(sweep_config, project="a2c-auto")

# wandb.agent(sweep_id, a.train_auto_fine_tune, count=6) 

dxl = 'success'
