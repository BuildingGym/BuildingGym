from energyplus.ooep.addons.progress import ProgressProvider
import asyncio
import pandas as pd
# from rl.ppo.network import Agent
import random
import numpy as np
import time
from energyplus import ooep
# import rl
import torch.nn.functional as F
import os
import wandb
import math
import tyro
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from gymnasium.spaces import (
    Box,
    Discrete
)
import torch
from energyplus.ooep import (
    Simulator,
    Model,
    Weather,
    Report,
)
import numpy as _numpy_
import gymnasium as _gymnasium_
from energyplus.ooep.components.variables import WallClock
from energyplus.ooep.addons.rl import (
    VariableBox,
    SimulatorEnv,
)
from energyplus.ooep import (
    Actuator,
    OutputVariable,
)
from energyplus.ooep.addons.rl.gymnasium import ThinEnv
# from energyplus.dataset.basic import dataset as _epds_
import torch.nn as nn
import wandb
from rl.util.replaybuffer import ReplayBuffer
from controllables.energyplus import (
    System,
    #WeatherModel,
    #Report,
    Actuator,
    OutputVariable,
)

# from energyplus.dataset.basic import dataset as _epds_
from controllables.energyplus.events import Event



from controllables.core import TemporaryUnavailableError 

from controllables.core.tools.gymnasium import (
    DictSpace,
    BoxSpace,
    Agent,
)

# async def energyplus_running(simulator, idf_file, epw_file):
#     await simulator.awaitable.run(
#         input=Simulator.InputSpecs(
#             model=(
#                 idf_file
#             ),
#             weather=(epw_file),
#         ),
#         output=Simulator.OutputSpecs(
#             #report=('/tmp/ooep-report-9e1287d2-8e75-4cf5-bbc5-f76580b56a69'),
#         ),
#         options=Simulator.RuntimeOptions(
#             #design_day=True,
#         ),
#     ) 

# def energyplus_running(simulator, idf_file, epw_file):
#     simulator.run(
#         input=Simulator.InputSpecs(
#             model=(
#                 idf_file
#             ),
#             weather=(epw_file),
#         ),
#         output=Simulator.OutputSpecs(
#             #report=('/tmp/ooep-report-9e1287d2-8e75-4cf5-bbc5-f76580b56a69'),
#         ),
#         options=Simulator.RuntimeOptions(
#             #design_day=True,
#         ),
#     ) 

class buildinggym_env():
    def __init__(self, idf_file,
                 epw_file,
                 observation_space,
                 action_space,
                 observation_dim,
                 action_type,
                 args=None,
                 ext_obs_bool = False,
                 agent = None) -> None:
        # global thinenv
        # self.simulator = Simulator().add(
        #     ProgressProvider(),
        #     #LogProvider(),
        # )
        self.buffer = ReplayBuffer(
            info=['obs', 'actions', 'rewards', 'nxt_obs'],
            args=args
            )
        self.idf_file = idf_file
        self.epw_file = epw_file
        self.ext_obs_bool = ext_obs_bool
        # self.simulator.add(
        #     thinenv := ThinEnv(
        #         action_space=action_space,    
        #         observation_space=observation_space,
        #     )
        # )
        # To update:
        self.observation_space = Box(np.array([-np.inf] * observation_dim), np.array([np.inf] * observation_dim))
        
        if isinstance(action_type, Box):
            self.action_space = Box(action_type.low, action_type.high)
        if isinstance(action_type, Discrete):
            self.action_space = Discrete(action_type.n)
        self.inter_obs_var = ['t_out', 't_in', 'occ', 'light', 'Equip']
        self.ext_obs_var = ['signal']       
        # self.observation_var = ['t_out', 't_in', 'occ', 'light', 'Equip']
        self.action_var = ['Thermostat']
        self.num_envs = 1
        self.agent = agent
        self.ready_to_train = False
        self.args = args
        self.p_loss_list = []
        self.v_loss_list = []
        self.success_n = 0
        self.batch_n = 0
        if args.device == 'cpu':
            self.obs_batch = torch.zeros(args.batch_size, observation_dim)
            self.action_batch = torch.zeros(args.batch_size, 1)
            self.return_batch = torch.zeros(args.batch_size, 1)           
        else: 
            self.obs_batch = torch.zeros(args.batch_size, observation_dim).to('cuda')
            self.action_batch = torch.zeros(args.batch_size, 1).to('cuda')
            self.return_batch = torch.zeros(args.batch_size, 1).to('cuda')
        # self.simulator.events.on('end_zone_timestep_after_zone_reporting', self.handler)
        self.baseline = pd.read_csv('Data\\Day_mean.csv')
        self.com = 24
        self.best_performance = 0
        # self.baseline['Time'] = pd.to_datetime(self.baseline['Time'], format='%m/%d/%Y %H:%M')

        # self.world = world = World(
        #     input=World.Specs.Input(
        #         world='Small office-1A-Long.idf',
        #         #world='tmp_timestep 10 min.idf',
        #         weather='USA_FL_Miami.722020_TMY2.epw',
        #     ),
        #     output=World.Specs.Output(
        #         report='tmp/ooep-report-9e1287d2-8e75-4cf5-bbc5-f76580b56a69',
        #     ),
        #     runtime=World.Specs.Runtime(
        #         recurring=False,
        #         # design_day=False,
        #     ),
        # ).add('logging:progress')

        self.world = world = System(
            building='Small office-1A-Long.idf',
            #world='tmp_timestep 10 min.idf',
            weather='USA_FL_Miami.722020_TMY2.epw',
        
            report='tmp/ooep-report-9e1287d2-8e75-4cf5-bbc5-f76580b56a69',
            repeat=False,
            # design_day=False,
        ).add('logging:progress')


        self.env = Agent(dict(
                action_space=DictSpace({
                    'Thermostat': BoxSpace(
                        low=22., high=30.,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(world[Actuator.Ref(
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
                            ).bind(world[OutputVariable.Ref(
                                type='Zone Mean Air Temperature',
                                key='Perimeter_ZN_1 ZN',
                            )]),
                    't_out': BoxSpace(
                                low=-_numpy_.inf, high=+_numpy_.inf,
                                dtype=_numpy_.float32,
                                shape=(),
                            ).bind(world[OutputVariable.Ref(
                                type='Site Outdoor Air Drybulb Temperature',
                                key='Environment',
                            )]),
                    'occ': BoxSpace(
                                low=-_numpy_.inf, high=+_numpy_.inf,
                                dtype=_numpy_.float32,
                                shape=(),
                            ).bind(world[OutputVariable.Ref(
                                type='Schedule Value',
                                key='Small Office Bldg Occ',
                            )]),
                    'light': BoxSpace(
                                low=-_numpy_.inf, high=+_numpy_.inf,
                                dtype=_numpy_.float32,
                                shape=(),
                            ).bind(world[OutputVariable.Ref(
                                type='Schedule Value',
                                key='Office Bldg Light',
                            )]),
                    'Equip': BoxSpace(
                                low=-_numpy_.inf, high=+_numpy_.inf,
                                dtype=_numpy_.float32,
                                shape=(),
                            ).bind(world[OutputVariable.Ref(
                                type='Schedule Value',
                                key='Small Office Bldg Equip',
                            )]),
                    'Energy_1': BoxSpace(
                                low=-_numpy_.inf, high=+_numpy_.inf,
                                dtype=_numpy_.float32,
                                shape=(),
                            ).bind(world[OutputVariable.Ref(
                                type='Cooling Coil Total Cooling Rate',
                                key='CORE_ZN ZN PSZ-AC-1 1SPD DX AC CLG COIL 34KBTU/HR 9.7SEER',
                            )]),
                    'Energy_2': BoxSpace(
                                low=-_numpy_.inf, high=+_numpy_.inf,
                                dtype=_numpy_.float32,
                                shape=(),
                            ).bind(world[OutputVariable.Ref(
                                type='Cooling Coil Total Cooling Rate',
                                key='PERIMETER_ZN_1 ZN PSZ-AC-2 1SPD DX AC CLG COIL 33KBTU/HR 9.7SEER',
                            )]),
                    'Energy_3': BoxSpace(
                                low=-_numpy_.inf, high=+_numpy_.inf,
                                dtype=_numpy_.float32,
                                shape=(),
                            ).bind(world[OutputVariable.Ref(
                                type='Cooling Coil Total Cooling Rate',
                                key='PERIMETER_ZN_2 ZN PSZ-AC-3 1SPD DX AC CLG COIL 23KBTU/HR 9.7SEER',
                            )]),
                    'Energy_4': BoxSpace(
                                low=-_numpy_.inf, high=+_numpy_.inf,
                                dtype=_numpy_.float32,
                                shape=(),
                            ).bind(world[OutputVariable.Ref(
                                type='Cooling Coil Total Cooling Rate',
                                key='PERIMETER_ZN_3 ZN PSZ-AC-4 1SPD DX AC CLG COIL 33KBTU/HR 9.7SEER',
                            )]),
                    'Energy_5': BoxSpace(
                                low=-_numpy_.inf, high=+_numpy_.inf,
                                dtype=_numpy_.float32,
                                shape=(),
                            ).bind(world[OutputVariable.Ref(
                                type='Cooling Coil Total Cooling Rate',
                                key='PERIMETER_ZN_4 ZN PSZ-AC-5 1SPD DX AC CLG COIL 25KBTU/HR 9.7SEER',
                            )]),                                                                                                                                                                                                                                                                                                                       
                }),
            ))        


        @self.world.on(Event.Ref('end_zone_timestep_after_zone_reporting', include_warmup=False))
        def _(_):
            global thinenv
            try:
                t = self.world['wallclock:calendar'].value
                obs = self.env.observe()
                # t = self.simulator.variables.getdefault(
                #     ooep.WallClock.Ref()
                # ).value
                warm_up = False
            except:
                warm_up = True

            if not warm_up:
                state = [float(obs[i]) for i in self.inter_obs_var]
                if self.ext_obs_bool:
                    if t.hour == 0 or t.hour>self.t_index:
                        self.ext_obs_var = self.get_ext_var(t)
                        self.t_index = t.hour
                    for _, value in self.ext_obs_var.items():
                        state.append(value)
                cooling_energy =  obs['Energy_1'].item() + obs['Energy_2'].item() + obs['Energy_3'].item() + obs['Energy_4'].item() + obs['Energy_5'].item()
                state = self.normalize_input_i(state)
                if self.ext_obs_bool:
                    signal = state[-1]
                else:
                    signal = 0.5
                state = torch.Tensor(state).cuda() if torch.cuda.is_available() and self.args.device == 'cuda'  else torch.Tensor(state).cpu()
                with torch.no_grad():
                    actions = self.agent(state)
                    # actions = torch.argmax(q_values, dim=0).cpu().item()
                if random.random() < self.epsilon:
                # if random.random() < 1.1:
                    if type(self.algo).__name__ == 'DQN':
                        actions = torch.FloatTensor(actions.shape).random_(0, 3).to(device=self.args.device, dtype=actions.dtype)
                    else:
                        actions = torch.FloatTensor(actions.shape).uniform_(-1, 1).to(device=self.args.device, dtype=actions.dtype)
                    # actions = torch.rand(actions.shape, device=self.args.device, dtype = actions.dtype)
                if type(self.algo).__name__ == 'DQN':
                    self.com +=  (actions.cpu().item() - 1) * 0.5
                else:
                    self.com +=  actions.cpu().item() * 0.5
                self.com = max(min(self.com, 27), 23)
                # self.com = 27
                obs = pd.DataFrame(obs, index = [self.sensor_index])                
                obs.insert(0, 'Time', t)
                obs.insert(0, 'day_of_week', t.weekday())
                obs.insert(1, 'Working time', self.label_working_time_i(t))            
                obs.insert(obs.columns.get_loc("t_in") + 1, 'Thermostat', self.com)
                reward_i, result_i, baseline_i = self.cal_r_i(cooling_energy, t, signal)
                obs['cooling_energy'] = cooling_energy
                obs['results'] = result_i
                obs['rewards'] = reward_i
                obs['baseline'] = baseline_i
                obs['Signal'] = signal
                obs['Target'] = 20000 * (1-0.3*signal)            
                obs.insert(obs.columns.get_loc("t_in") + 1, 'actions', actions.cpu().item())
                # obs.insert(obs.columns.get_loc("t_in") + 1, 'logprobs', logprob.cpu().item())
                # obs.insert(obs.columns.get_loc("t_in") + 1, 'values', value.cpu().item())


                if self.sensor_index == 0:
                    self.sensor_dic = pd.DataFrame({})
                    self.sensor_dic = obs
                    # self.logprobs = [logprob]
                    # self.values = [value]
                    self.actions = [actions]
                    self.states = [state]
                    # self.values = [value]
                    self.rewards = [reward_i]
                else:
                    self.sensor_dic = pd.concat([self.sensor_dic, obs])           
                    # self.logprobs.append(logprob) 
                    # self.values.append(value) 
                    self.actions.append(actions)
                    self.states.append(state)
                    # self.values.append(value)
                    self.rewards.append(reward_i)
                actions = actions.cpu().item()
                # com = 25. + actions * 2
                # act = thinenv.act({'Thermostat': self.com})
                self.env.action.value = {
                'Thermostat': self.com,
                }                
                # act = thinenv.act({'Thermostat': 26.2})

                b  = self.args.outlook_steps + 1
                # self.buffer.add([self.states[i], self.actions[i], self.logprobs[i], r_i, value])   # List['obs', 'action', 'logprb', 'rewards', 'values']

                if self.sensor_index > b:
                    i = self.sensor_index-b
                    if i % self.args.step_size == 0:
                        if np.sum(self.sensor_dic['Working time'].iloc[i:(self.sensor_index)]) == b:
                            ob_i = self.states[i]
                            ob_nxt_i = self.states[i+1]
                            r_i = self.rewards[i+1]
                            # logp_i = self.logprobs[i]
                            action_i = self.actions[i]
                            R_i = self.cal_return(self.rewards[i+1:i+b])
                            # if self.batch_n<self.args.batch_size:
                            self.buffer.add([ob_i, action_i, r_i, ob_nxt_i], max_buffer_size = self.args.max_buffer_size)  # List['obs', 'actions', 'rewards', 'nxt_obs']
                                # self.obs_batch[self.batch_n, :] = ob_i
                                # self.return_batch[self.batch_n, :] = R_i
                                # self.action_batch[self.batch_n, :] = action_i
                            #     self.batch_n+=1
                            # else:
                            #     self.batch_n=0
                            #     self.buffer.cal_R_adv()
                            #     p_loss_i, v_loss_i = self.algo.train(self.buffer)
                            #     self.buffer.reset()  # dxl: can update to be able to store somme history info
                            #     self.p_loss_list.append(p_loss_i)
                            #     self.v_loss_list.append(v_loss_i)


                    if i % self.args.train_frequency == 0 and self.buffer.buffer_size>self.args.batch_size and self.train:
                        if type(self.algo).__name__ == 'DQN':
                            self.actor_losses_i, _ = self.algo.train()
                        else:
                            self.actor_losses_i, self.critic_losses_i = self.algo.train()
                        if not math.isnan(self.actor_losses_i):
                            self.p_loss_list.append(self.actor_losses_i)
                        if not type(self.algo).__name__ == 'DQN':
                            self.v_loss_list.append(self.critic_losses_i)                    

                self.sensor_index+=1        

    def setup(self, algo):
        self.algo = algo
        self.agent = self.algo.policy
        self.ready_to_train = True
        
    def run(self, agent = None, epsilon: float = 0, train: bool = True):
        self.epsilon = epsilon
        self.train = train
        self.sensor_index = 0
        # if agent is not None:
        #     self.agent = agent
        # asyncio.run(energyplus_running(self.simulator, self.idf_file, self.epw_file))
        self.world.start().wait()




    def normalize_input_i(self, state):
        nor_min = np.array([22.8, 22, 0, 0, 0])
        nor_mean = np.array([29.3, 25, 0.78, 0.58, 0.89, 0])
        nor_mean = np.array([29.3, 25, 0.78, 0.58, 0.89])
        std = np.array([2, 2, 0.39, 0.26, 0.26, 1])
        std = np.array([2, 2, 0.39, 0.26, 0.26])
        # nor_min = np.array([0, 0, 0, 0, 0])
        nor_max = np.array([33.3, 27, 1, 1, 1])
        # nor_max = np.array([1, 1, 1, 1, 1])
        return (state- nor_mean)/std

    def label_working_time_i(self, t):
        start = pd.to_datetime(self.args.work_time_start, format='%H:%M')
        end = pd.to_datetime(self.args.work_time_end, format='%H:%M')
        # remove data without enough outlook step
        dt = int(60/self.args.n_time_step)
        dt = pd.to_timedelta(dt, unit='min')
        # end -= dt
        day_of_week = t.weekday()
        h = t.hour
        m = t.minute
        t = pd.to_datetime(str(h)+':'+str(m), format='%H:%M')
        if t >= start and t < end and day_of_week<5:
            wt = True
        else:
            wt = False
        if t >= end - dt:
            terminations = True
        else:
            terminations = False
        return wt
        # self.sensor_dic['Terminations'] = terminations            

    # def cal_r(self):
    #     baseline = pd.read_csv('Data\Day_mean.csv')
    #     reward = []
    #     result = []
    #     # Realtime reward function
    #     for j in range(self.sensor_dic.shape[0]):
    #         energy_i = self.sensor_dic['Chiller Electricity Rate'].iloc[j]
    #         k = j % (24*self.args.n_time_step)
    #         baseline_i = baseline['Day_mean'].iloc[k]
    #         reward_i = max(round(0.3 - abs(energy_i ** 2 - baseline_i ** 2)/baseline_i ** 2,2),-0.4)*5
    #         result_i = round(1 - abs(energy_i - baseline_i)/baseline_i,2)
    #         # reward_i = result_i
    #         # if reward_i<0.8:
    #         #     reward_i = reward_i**2
    #         # else:
    #         #     reward_i+=reward_i*5
    #         reward.append(reward_i)
    #         result.append(result_i)          
        
    #     reward = reward[1:]
    #     result = result[1:]
    #     self.actions = self.actions[0:-1]
    #     self.logprobs = self.logprobs[0:-1]
    #     self.sensor_dic =  self.sensor_dic[0:-1]
    #     self.sensor_dic['rewards'] = reward
    #     self.sensor_dic['results'] = result

    def cal_r_i(self, data, time, signal):
        # baseline = pd.read_csv('Data\Day_mean.csv')
        hour = time.hour
        min = time.minute
        idx = int(hour*6+int(min/10))
        baseline_i = self.baseline['Day_mean'].iloc[idx]
        baseline_i = 20000
        # reward_i = max(round(0.3 - abs(data ** 2 - baseline_i ** 2)/baseline_i ** 2,2),-0.4)*5
        # result_i = round(1 - abs(data - baseline_i)/baseline_i,2)
        # return reward_i, result_i, baseline_i
        # baseline_energy = self.baseline['cooling_energy'].iloc[idx]
        actual_reduction = (baseline_i - data) / baseline_i
        
        # Target reduction percentage
        target_reduction = 0.3 * signal
        
        # if abs(actual_reduction-target_reduction) < 0.05:
        #     energy_reward = 5
        # elif abs(actual_reduction-target_reduction) < 0.15:
        #     energy_reward = 2
        # else:
        #     energy_reward = -1
        energy_reward = 10 - abs(actual_reduction - target_reduction) * 10
        # if energy_reward<-5:
        #     energy_reward = -5
        return energy_reward, actual_reduction, baseline_i
        
    
    def cal_return(self, reward_list):
        R = 0
        for r in reward_list[::-1]:
            R = r + R * self.args.gamma
        return R
    
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

    # def handler(self, __event):
    #     global thinenv
    #     try:
    #         obs = thinenv.observe()
    #         t = self.simulator.variables.getdefault(
    #             ooep.WallClock.Ref()
    #         ).value
    #         warm_up = False
    #     except:
    #         warm_up = True

    #     if not warm_up:
    #         state = [float(obs[i]) for i in self.inter_obs_var]
    #         if self.ext_obs_bool:
    #             if t.hour == 0 or t.hour>self.t_index:
    #                 self.ext_obs_var = self.get_ext_var(t)
    #                 self.t_index = t.hour
    #             for _, value in self.ext_obs_var.items():
    #                 state.append(value)
    #         cooling_energy =  obs['Energy_1'].item() + obs['Energy_2'].item() + obs['Energy_3'].item() + obs['Energy_4'].item() + obs['Energy_5'].item()
    #         state = self.normalize_input_i(state)
    #         if self.ext_obs_bool:
    #             signal = state[-1]
    #         else:
    #             signal = 0.5
    #         state = torch.Tensor(state).cuda() if torch.cuda.is_available() and self.args.device == 'cuda'  else torch.Tensor(state).cpu()
    #         with torch.no_grad():
    #             actions = self.agent(state)
    #             # actions = torch.argmax(q_values, dim=0).cpu().item()
    #         if random.random() < self.epsilon:
    #         # if random.random() < 1.1:
    #             if type(self.algo).__name__ == 'DQN':
    #                 actions = torch.FloatTensor(actions.shape).random_(0, 3).to(device=self.args.device, dtype=actions.dtype)
    #             else:
    #                 actions = torch.FloatTensor(actions.shape).uniform_(-1, 1).to(device=self.args.device, dtype=actions.dtype)
    #             # actions = torch.rand(actions.shape, device=self.args.device, dtype = actions.dtype)
    #         if type(self.algo).__name__ == 'DQN':
    #             self.com +=  (actions.cpu().item() - 1) * 0.5
    #         else:
    #             self.com +=  actions.cpu().item() * 0.5
    #         self.com = max(min(self.com, 27), 23)
    #         # self.com = 27
    #         obs = pd.DataFrame(obs, index = [self.sensor_index])                
    #         obs.insert(0, 'Time', t)
    #         obs.insert(0, 'day_of_week', t.weekday())
    #         obs.insert(1, 'Working time', self.label_working_time_i(t))            
    #         obs.insert(obs.columns.get_loc("t_in") + 1, 'Thermostat', self.com)
    #         reward_i, result_i, baseline_i = self.cal_r_i(cooling_energy, t, signal)
    #         obs['cooling_energy'] = cooling_energy
    #         obs['results'] = result_i
    #         obs['rewards'] = reward_i
    #         obs['baseline'] = baseline_i
    #         obs['Signal'] = signal
    #         obs['Target'] = 20000 * (1-0.3*signal)            
    #         obs.insert(obs.columns.get_loc("t_in") + 1, 'actions', actions.cpu().item())
    #         # obs.insert(obs.columns.get_loc("t_in") + 1, 'logprobs', logprob.cpu().item())
    #         # obs.insert(obs.columns.get_loc("t_in") + 1, 'values', value.cpu().item())


    #         if self.sensor_index == 0:
    #             self.sensor_dic = pd.DataFrame({})
    #             self.sensor_dic = obs
    #             # self.logprobs = [logprob]
    #             # self.values = [value]
    #             self.actions = [actions]
    #             self.states = [state]
    #             # self.values = [value]
    #             self.rewards = [reward_i]
    #         else:
    #             self.sensor_dic = pd.concat([self.sensor_dic, obs])           
    #             # self.logprobs.append(logprob) 
    #             # self.values.append(value) 
    #             self.actions.append(actions)
    #             self.states.append(state)
    #             # self.values.append(value)
    #             self.rewards.append(reward_i)
    #         actions = actions.cpu().item()
    #         # com = 25. + actions * 2
    #         # act = thinenv.act({'Thermostat': self.com})
    #         self.env.action.value = {
    #             'Thermostat': 27,
    #             }            
    #         # act = thinenv.act({'Thermostat': 26.2})

    #         b  = self.args.outlook_steps + 1
    #         # self.buffer.add([self.states[i], self.actions[i], self.logprobs[i], r_i, value])   # List['obs', 'action', 'logprb', 'rewards', 'values']

    #         if self.sensor_index > b:
    #             i = self.sensor_index-b
    #             if i % self.args.step_size == 0:
    #                 if np.sum(self.sensor_dic['Working time'].iloc[i:(self.sensor_index)]) == b:
    #                     ob_i = self.states[i]
    #                     ob_nxt_i = self.states[i+1]
    #                     r_i = self.rewards[i+1]
    #                     # logp_i = self.logprobs[i]
    #                     action_i = self.actions[i]
    #                     R_i = self.cal_return(self.rewards[i+1:i+b])
    #                     # if self.batch_n<self.args.batch_size:
    #                     self.buffer.add([ob_i, action_i, r_i, ob_nxt_i], max_buffer_size = self.args.max_buffer_size)  # List['obs', 'actions', 'rewards', 'nxt_obs']
    #                         # self.obs_batch[self.batch_n, :] = ob_i
    #                         # self.return_batch[self.batch_n, :] = R_i
    #                         # self.action_batch[self.batch_n, :] = action_i
    #                     #     self.batch_n+=1
    #                     # else:
    #                     #     self.batch_n=0
    #                     #     self.buffer.cal_R_adv()
    #                     #     p_loss_i, v_loss_i = self.algo.train(self.buffer)
    #                     #     self.buffer.reset()  # dxl: can update to be able to store somme history info
    #                     #     self.p_loss_list.append(p_loss_i)
    #                     #     self.v_loss_list.append(v_loss_i)


    #             if i % self.args.train_frequency == 0 and self.buffer.buffer_size>self.args.batch_size and self.train:
    #                 if type(self.algo).__name__ == 'DQN':
    #                     self.actor_losses_i, _ = self.algo.train()
    #                 else:
    #                     self.actor_losses_i, self.critic_losses_i = self.algo.train()
    #                 if not math.isnan(self.actor_losses_i):
    #                     self.p_loss_list.append(self.actor_losses_i)
    #                 if not type(self.algo).__name__ == 'DQN':
    #                     self.v_loss_list.append(self.critic_losses_i)                    

    #         self.sensor_index+=1