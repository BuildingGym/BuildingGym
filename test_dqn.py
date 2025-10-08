from env.env_offline import buildinggym_env
import gymnasium as _gymnasium_
# from energyplus.ooep.addons.rl import (
#     VariableBox,
#     SimulatorEnv,
# )
# import math
# from energyplus.ooep import (
#     Actuator,
#     OutputVariable,
# )
import random
from rl.dqn.network import policy_network
from rl.util.schedule import ConstantSchedule
from gymnasium.spaces import (
    Box,
    Discrete
)
import numpy as _numpy_
import numpy as np
from rl.dqn.dqn_para import Args
from rl.dqn.dqn import DQN
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import tyro
import time
import torch as th
from controllables.core.tools.gymnasium import (
    DictSpace,
    BoxSpace,
    Agent,
)
from controllables.energyplus import (
    System,
    #WeatherModel,
    #Report,
    Actuator,
    OutputVariable,
)
idf_file = 'Small office-1A-Long.idf'
epw_file = 'USA_FL_Miami.722020_TMY2.epw'


ep_world = System(
    building=idf_file,
    #world='tmp_timestep 10 min.idf',
    weather=epw_file,

    report='tmp/ooep-report-9e1287d2-8e75-4cf5-bbc5-f76580b56a69',
    repeat=False,
    # design_day=False,
).add('logging:progress')

ep_para = Agent(dict(
        action_space=DictSpace({
            'Thermostat': BoxSpace(
                low=22., high=30.,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(ep_world[Actuator.Ref(
                type='Schedule:Compact',
                control_type='Schedule Value',
                key='Always 26',
            )])
        }),    
        observation_space=DictSpace({
            't_in': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(ep_world[OutputVariable.Ref(
                        type='Zone Mean Air Temperature',
                        key='Perimeter_ZN_1 ZN',
                    )]),
            't_out': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(ep_world[OutputVariable.Ref(
                        type='Site Outdoor Air Drybulb Temperature',
                        key='Environment',
                    )]),
            'occ': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(ep_world[OutputVariable.Ref(
                        type='Schedule Value',
                        key='Small Office Bldg Occ',
                    )]),
            'light': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(ep_world[OutputVariable.Ref(
                        type='Schedule Value',
                        key='Office Bldg Light',
                    )]),
            'Equip': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(ep_world[OutputVariable.Ref(
                        type='Schedule Value',
                        key='Small Office Bldg Equip',
                    )]),
            'Energy_1': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(ep_world[OutputVariable.Ref(
                        type='Cooling Coil Total Cooling Rate',
                        key='CORE_ZN ZN PSZ-AC-1 1SPD DX AC CLG COIL 34KBTU/HR 9.7SEER',
                    )]),
            'Energy_2': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(ep_world[OutputVariable.Ref(
                        type='Cooling Coil Total Cooling Rate',
                        key='PERIMETER_ZN_1 ZN PSZ-AC-2 1SPD DX AC CLG COIL 33KBTU/HR 9.7SEER',
                    )]),
            'Energy_3': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(ep_world[OutputVariable.Ref(
                        type='Cooling Coil Total Cooling Rate',
                        key='PERIMETER_ZN_2 ZN PSZ-AC-3 1SPD DX AC CLG COIL 23KBTU/HR 9.7SEER',
                    )]),
            'Energy_4': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(ep_world[OutputVariable.Ref(
                        type='Cooling Coil Total Cooling Rate',
                        key='PERIMETER_ZN_3 ZN PSZ-AC-4 1SPD DX AC CLG COIL 33KBTU/HR 9.7SEER',
                    )]),
            'Energy_5': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(ep_world[OutputVariable.Ref(
                        type='Cooling Coil Total Cooling Rate',
                        key='PERIMETER_ZN_4 ZN PSZ-AC-5 1SPD DX AC CLG COIL 25KBTU/HR 9.7SEER',
                    )]),                                                                                                                                                                                                                                                                                                                       
        }),
    ))  
action_sp = Discrete(3)

class my_env(buildinggym_env):
    def __init__(self, idf_file, epw_file, ep_world, ep_para, action_type, args, inter_obs_var, ext_obs_var=None, agent=None):
        super().__init__(idf_file, epw_file, ep_world, ep_para, action_type, args, inter_obs_var, ext_obs_var, agent)

    def cal_r_i(self, obs, state, time):
        ##____important: must define the rewards as the reward at each time step
        cooling_energy =  obs['Energy_1'].item() + obs['Energy_2'].item() + obs['Energy_3'].item() + obs['Energy_4'].item() + obs['Energy_5'].item()
        if self.ext_obs_bool:
            signal = state[-1]
        else:
            signal = 0.5               
        # baseline = pd.read_csv('Data\Day_mean.csv')
        hour = time.hour
        min = time.minute
        idx = int(hour*6+int(min/10))
        # baseline_i = self.baseline['Day_mean'].iloc[idx]
        baseline_i = 20000
        # reward_i = max(round(0.3 - abs(data ** 2 - baseline_i ** 2)/baseline_i ** 2,2),-0.4)*5
        # result_i = round(1 - abs(data - baseline_i)/baseline_i,2)
        # return reward_i, result_i, baseline_i
        # baseline_energy = self.baseline['cooling_energy'].iloc[idx]
        actual_reduction = (baseline_i - cooling_energy) / baseline_i
        
        # Target reduction percentage
        target_reduction = 0.3 * signal
        
        # if abs(actual_reduction-target_reduction) < 0.05:
        #     self.reward_i = 5
        # elif abs(actual_reduction-target_reduction) < 0.15:
        #     self.reward_i = 2
        # else:
        #     self.reward_i = -1
        self.reward_i = 10 - abs(actual_reduction - target_reduction) * 10
        # if self.reward_i<-5:
        #     self.reward_i = -5
        return {'rewards':self.reward_i, 'actual_reduction': actual_reduction, 'baseline_i': baseline_i, 'signal': signal, 'cooling_energy': cooling_energy}        
    
    def get_ext_var(self, t=None):
        ext_obs_var = {}
        if t.hour >=11 and t.hour<=13:
            for i in self.ext_obs_var:
                ext_obs_var[i] = random.choice([1])
        elif t.hour >=14 and t.hour<=16:
            for i in self.ext_obs_var:
                ext_obs_var[i] = random.choice([0.5])     
        elif t.hour >=17 and t.hour<=19:
            for i in self.ext_obs_var:
                ext_obs_var[i] = random.choice([1])                    
        else:
            for i in self.ext_obs_var:
                ext_obs_var[i] = random.choice([0])                        
        return ext_obs_var    
    
    def control_fun(self, actions):
        self.com +=  (actions.cpu().item()-1)*0.5
        self.com = max(min(self.com, 27), 23)
        self.ep_para.action.value = {
        'Thermostat': self.com,
        }  
        
    def normalize_input_i(self, state):
        # nor_min = np.array([22.8, 22, 0, 0, 0])
        if self.ext_obs_bool:
            nor_mean = np.array([29.3, 25, 0.78, 0.58, 0.89, 0])
            std = np.array([2, 2, 0.39, 0.26, 0.26, 1])
        else:
            nor_mean = np.array([29.3, 25, 0.78, 0.58, 0.89])
            std = np.array([2, 2, 0.39, 0.26, 0.26])
            
        # nor_mean = np.array([28.7, 26, 0.78, 0.58, 0.89])

        # nor_mean = np.array([28.7, 26, 0.78, 0.58, 0.89])
        # std = np.array([2.17, 0.5, 0.39, 0.26, 0.26, 1])
        # std = np.array([2.17, 0.5, 0.39, 0.26, 0.26])

        # std = np.array([2.17, 0.5, 0.39, 0.26, 0.26])


        # nor_min = np.array([0, 0, 0, 0, 0])
        # nor_max = np.array([33.3, 27, 1, 1, 1])
        # nor_max = np.array([1, 1, 1, 1, 1])
        return (state- nor_mean)/std            

# if isinstance(action_sp, Discrete):
#     action_dim = action_sp.n
# elif isinstance(action_sp, Box):
#     action_dim = action_sp.shape[0]
# agent = Agent(input_sp, action_sp, schedule.value)
env = my_env(idf_file,
                    epw_file,
                    ep_world,
                    ep_para,                    
                    # observation_space,
                    # action_space,
                    # input_sp.shape[0],
                    action_sp,
                    Args,
                    inter_obs_var = ['t_out', 't_in', 'occ', 'light', 'Equip'],
                    ext_obs_var = ['signal']) # ext_obs_var = None if not external signals

# class callback(BaseCallback):
#     def __init__(self, verbose: int = 0):
#         super().__init__(verbose)

#     def on_rollout_end(self) -> None:
#         super().on_rollout_end()
#         result = np.mean(self.model.env.sensor_dic['results'].iloc[np.where(env.sensor_dic['Working time'])[0]])
#         reward = np.mean(self.model.env.sensor_dic['rewards'].iloc[np.where(env.sensor_dic['Working time'])[0]])
#         prob = np.mean(np.exp(self.model.env.sensor_dic['logprobs'].iloc[np.where(env.sensor_dic['Working time'])[0]]))
#         p_loss = np.mean(self.model.env.loss_list)
#         # v_loss = self.model.env.v_loss
#         # prob = self.model.env.prob
#         lr = self.model.learning_rate
#         wandb.log({'reward_curve': reward}, step=self.num_timesteps)        
#         wandb.log({'result_curve': result}, step=self.num_timesteps)
#         wandb.log({'action prob': prob}, step=self.num_timesteps)
#         wandb.log({'p_loss_curve': float(p_loss)}, step=self.num_timesteps)
#         # wandb.log({'v_loss_curve': float(v_loss)}, step=self.num_timesteps)      

#     def per_time_step(self, var = None) -> None:
#         # super().on_epoch_end()
#         if var is not None:
#             p_loss = var['loss'].item()
#             # wandb.log({'p_loss_curve': float(p_loss)}, step=self.num_timesteps)       


# my_callback = callback()
args = tyro.cli(Args)
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"


a = DQN(policy_network,
        env,
        args,
        run_name,
        # my_callback,
        policy_kwargs = {'optimizer_class': args.optimizer_class},
        # max_train_perEp = args.max_train_perEp,
        )
env.setup(algo=a)

if args.log_wandb:
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=args,
        name=run_name,
        save_code=False,
    )
_, performance = a.learn(args.total_epoch, None)

# observation_space = _gymnasium_.spaces.Dict({
#             't_out': VariableBox(
#                 low=22.8, high=33.3,
#                 dtype=_numpy_.float32,
#                 shape=(),
#             ).bind(OutputVariable.Ref(
#                 type='Site Outdoor Air Drybulb Temperature',
#                 key='Environment',
#             )),
#             't_in': VariableBox(
#                 low=22, high=27,
#                 dtype=_numpy_.float32,
#                 shape=(),
#             ).bind(OutputVariable.Ref(
#                 type='Zone Mean Air Temperature',
#                 key='Perimeter_ZN_1 ZN',
#             )),
#             'occ': VariableBox(
#                 low=0, high=1,
#                 dtype=_numpy_.float32,
#                 shape=(),
#             ).bind(OutputVariable.Ref(
#                 type='Schedule Value',
#                 key='Small Office Bldg Occ',
#             )),
#             'light': VariableBox(
#                 low=0, high=1,
#                 dtype=_numpy_.float32,
#                 shape=(),
#             ).bind(OutputVariable.Ref(
#                 type='Schedule Value',
#                 key='Office Bldg Light',
#             )),
#             'Equip': VariableBox(
#                 low=0, high=1,
#                 dtype=_numpy_.float32,
#                 shape=(),
#             ).bind(OutputVariable.Ref(
#                 type='Schedule Value',
#                 key='Small Office Bldg Equip',
#             )),   
#             'Energy_1': VariableBox(
#                 low=0, high=10000000,
#                 dtype=_numpy_.float32,
#                 shape=(),
#             ).bind(OutputVariable.Ref(
#                 type='Cooling Coil Total Cooling Rate',
#                 key='CORE_ZN ZN PSZ-AC-1 1SPD DX AC CLG COIL 34KBTU/HR 9.7SEER',
#             )),
#             'Energy_2': VariableBox(
#                 low=0, high=10000000,
#                 dtype=_numpy_.float32,
#                 shape=(),
#             ).bind(OutputVariable.Ref(
#                 type='Cooling Coil Total Cooling Rate',
#                 key='PERIMETER_ZN_1 ZN PSZ-AC-2 1SPD DX AC CLG COIL 33KBTU/HR 9.7SEER',
#             )),
#             'Energy_3': VariableBox(
#                 low=0, high=10000000,
#                 dtype=_numpy_.float32,
#                 shape=(),
#             ).bind(OutputVariable.Ref(
#                 type='Cooling Coil Total Cooling Rate',
#                 key='PERIMETER_ZN_2 ZN PSZ-AC-3 1SPD DX AC CLG COIL 23KBTU/HR 9.7SEER',
#             )),
#             'Energy_4': VariableBox(
#                 low=0, high=10000000,
#                 dtype=_numpy_.float32,
#                 shape=(),
#             ).bind(OutputVariable.Ref(
#                 type='Cooling Coil Total Cooling Rate',
#                 key='PERIMETER_ZN_3 ZN PSZ-AC-4 1SPD DX AC CLG COIL 33KBTU/HR 9.7SEER',
#             )),
#             'Energy_5': VariableBox(
#                 low=0, high=10000000,
#                 dtype=_numpy_.float32,
#                 shape=(),
#             ).bind(OutputVariable.Ref(
#                 type='Cooling Coil Total Cooling Rate',
#                 key='PERIMETER_ZN_4 ZN PSZ-AC-5 1SPD DX AC CLG COIL 25KBTU/HR 9.7SEER',
#             )),                                                
#         })
# action_space = _gymnasium_.spaces.Dict({
#                     'Thermostat': VariableBox(
#                         low=22., high=30.,
#                         dtype=_numpy_.float32,
#                         shape=(),
#                     ).bind(Actuator.Ref(
#                         type='Schedule:Compact',
#                         control_type='Schedule Value',
#                         key='Always 26',
#                     ))
#                 })
# schedule = ConstantSchedule(0.0001)
# input_sp = Box(np.array([0] * 5), np.array([1] * 5))
# # action_sp = Box(np.array([0, -0.5]), np.array([1, 0.5]))
# action_sp = Discrete(3)
# if isinstance(action_sp, Discrete):
#     action_dim = action_sp.n
# elif isinstance(action_sp, Box):
#     action_dim = action_sp.shape[0]
# # agent = Agent(input_sp, action_sp, schedule.value)
# args = tyro.cli(Args)
# env = buildinggym_env('Small office-1A-Long.idf',
#                     'USA_FL_Miami.722020_TMY2.epw',
#                     # observation_space,
#                     # action_space,
#                     input_sp.shape[0],
#                     action_sp,
#                     args)

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
# run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"


# a = DQN(Agent,
#         env,
#         args,
#         run_name,
#         None,
#         policy_kwargs = {'optimizer_class': args.optimizer_class},
#         # max_train_perEp = args.max_train_perEp,
#         )
# env.setup(algo=a)

# if args.log_wandb:
#     wandb.init(
#         project=args.wandb_project_name,
#         entity=args.wandb_entity,
#         sync_tensorboard=True,
#         config=args,
#         name=run_name,
#         save_code=False,
#     )
# _, performance = a.learn(args.total_epoch, None)



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
