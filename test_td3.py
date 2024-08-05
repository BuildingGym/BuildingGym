# from env.env_new_r import buildinggym_env
from env.env_offline import buildinggym_env
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

from rl.td3.network import Agent
from rl.util.schedule import ConstantSchedule
from gymnasium.spaces import (
    Box,
    Discrete
)
import numpy as _numpy_
import numpy as np
from rl.td3.td3_para import Args
from rl.td3.td3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import tyro
import time
import torch as th

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
                key='Perimeter_ZN_1 ZN',
            )),
            'occ': VariableBox(
                low=0, high=1,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Schedule Value',
                key='Small Office Bldg Occ',
            )),
            'light': VariableBox(
                low=0, high=1,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Schedule Value',
                key='Office Bldg Light',
            )),
            'Equip': VariableBox(
                low=0, high=1,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Schedule Value',
                key='Small Office Bldg Equip',
            )),   
            'Energy_1': VariableBox(
                low=0, high=10000000,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Cooling Coil Total Cooling Rate',
                key='CORE_ZN ZN PSZ-AC-1 1SPD DX AC CLG COIL 34KBTU/HR 9.7SEER',
            )),
            'Energy_2': VariableBox(
                low=0, high=10000000,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Cooling Coil Total Cooling Rate',
                key='PERIMETER_ZN_1 ZN PSZ-AC-2 1SPD DX AC CLG COIL 33KBTU/HR 9.7SEER',
            )),
            'Energy_3': VariableBox(
                low=0, high=10000000,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Cooling Coil Total Cooling Rate',
                key='PERIMETER_ZN_2 ZN PSZ-AC-3 1SPD DX AC CLG COIL 23KBTU/HR 9.7SEER',
            )),
            'Energy_4': VariableBox(
                low=0, high=10000000,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Cooling Coil Total Cooling Rate',
                key='PERIMETER_ZN_3 ZN PSZ-AC-4 1SPD DX AC CLG COIL 33KBTU/HR 9.7SEER',
            )),
            'Energy_5': VariableBox(
                low=0, high=10000000,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(OutputVariable.Ref(
                type='Cooling Coil Total Cooling Rate',
                key='PERIMETER_ZN_4 ZN PSZ-AC-5 1SPD DX AC CLG COIL 25KBTU/HR 9.7SEER',
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
                        key='Always 26',
                    ))
                })
# schedule = ConstantSchedule(0.0001)
input_sp = Box(np.array([0] * 5), np.array([1] * 5))
# action_sp = Box(np.array([0, -0.5]), np.array([1, 0.5]))
action_sp = Discrete(3)
if isinstance(action_sp, Discrete):
    action_dim = action_sp.n
elif isinstance(action_sp, Box):
    action_dim = action_sp.shape[0]
# agent = Agent(input_sp, action_sp, schedule.value)
args = tyro.cli(Args)
env = buildinggym_env('Small office-1A-Long.idf',
                    'USA_FL_Miami.722020_TMY2.epw',
                    observation_space,
                    action_space,
                    input_sp.shape[0],
                    action_sp,
                    args)

# class callback(BaseCallback):
#     def __init__(self, verbose: int = 0):
#         super().__init__(verbose)

#     def on_rollout_end(self) -> None:
#         super().on_rollout_end()
#         result = np.mean(self.model.env.sensor_dic['results'].iloc[np.where(env.sensor_dic['Working time'])[0]])
#         reward = np.mean(self.model.env.sensor_dic['rewards'].iloc[np.where(env.sensor_dic['Working time'])[0]])
#         prob = np.mean(np.exp(self.model.env.sensor_dic['logprobs'].iloc[np.where(env.sensor_dic['Working time'])[0]]))
#         p_loss = np.mean(self.model.env.p_loss_list)
#         # v_loss = self.model.env.v_loss
#         # prob = self.model.env.prob
#         lr = self.model.learning_rate
#         wandb.log({'reward_curve': reward}, step=self.num_timesteps)        
#         wandb.log({'result_curve': result}, step=self.num_timesteps)
#         wandb.log({'action prob': prob}, step=self.num_timesteps)
#         wandb.log({'p_loss_curve': float(p_loss)}, step=self.num_timesteps)
#         # wandb.log({'v_loss_curve': float(v_loss)}, step=self.num_timesteps)

#     def on_epoch_end(self):
#         wandb.log({'reward_curve': self.model.policy}, step=self.num_timesteps)    

#     def per_time_step(self, var = None) -> None:
#         # super().on_epoch_end()
#         if var is not None:
#             p_loss = var['loss'].item()
#             # wandb.log({'p_loss_curve': float(p_loss)}, step=self.num_timesteps)       


# my_callback = callback()
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"


a = TD3(Agent,
        env,
        args,
        run_name,
        None,
        policy_kwargs = {'optimizer_class': args.optimizer_class},
        # max_train_perEp = args.max_train_perEp,
        )
env.setup(algo=a)

wandb.init(
    project=args.wandb_project_name,
    entity=args.wandb_entity,
    sync_tensorboard=True,
    config=args,
    name=run_name,
    save_code=False,
)
_, performance = a.learn(args.total_epoch, None)



# parameters_dict = {
# 'learning_rate': {
#     'values': [0.01, 0.001, 0.005, 0.008]
#     },
# 'alpha': {
#         'values': [0.98, 0.95, 0.9, 0.8]
#     },  
# 'outlook_steps': {
#         'values': [12, 2, 6]
#     },  
# 'step_size': {
#         'values': [12, 2, 6]
#     },          
# 'gamma': {
#         'values': [0.5, 0.9, 0.99]
#     },         
# 'batch_size': {
#       'values': [16, 32, 1]
#     },     
# 'ent_coef': {
#       'values': [0.01, 0.05, 0.1, 0.5]
#     },   
# # 'gae_lambda': {
# #       'values': [1, 0.1]
# #     },                                
# }
# sweep_config = {
# 'method': 'random'
# }
# metric = {
# 'name': 'performance',
# 'goal': 'maximize'   
# }
# sweep_config['metric'] = metric
# sweep_config['parameters'] = parameters_dict

# sweep_id = wandb.sweep(sweep_config, project="a2c-auto")

# wandb.agent(sweep_id, a.train_auto_fine_tune, count=50) 

dxl = 'success'
