from energyplus.ooep.addons.progress import ProgressProvider
import asyncio
import pandas as pd
from rl.ppo.network import Agent
import random
import numpy as np
import time
from energyplus import ooep
import torch.nn.functional as F
import os
from stable_baselines3.common.buffers import ReplayBuffer
import wandb
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
from energyplus.dataset.basic import dataset as _epds_
import torch.nn as nn
import wandb

async def energyplus_running(simulator, idf_file, epw_file):
    await simulator.awaitable.run(
        input=Simulator.InputSpecs(
            model=(
                idf_file
            ),
            weather=(epw_file),
        ),
        output=Simulator.OutputSpecs(
            #report=('/tmp/ooep-report-9e1287d2-8e75-4cf5-bbc5-f76580b56a69'),
        ),
        options=Simulator.RuntimeOptions(
            #design_day=True,
        ),
    ) 

class buildinggym_env():
    def __init__(self, idf_file,
                 epw_file,
                 observation_space,
                 action_space,
                 observation_dim,
                 action_type,
                 args,
                 agent = None) -> None:
        global thinenv
        self.simulator = Simulator().add(
            ProgressProvider(),
            #LogProvider(),
        )
        self.idf_file = idf_file
        self.epw_file = epw_file
        self.simulator.add(
            thinenv := ThinEnv(
                action_space=action_space,    
                observation_space=observation_space,
            )
        )
        # To update:
        self.observation_space = Box(np.array([-np.inf] * observation_dim), np.array([np.inf] * observation_dim))
        
        if isinstance(action_type, Box):
            self.action_space = Box(action_type.low, action_type.high)
        if isinstance(action_type, Discrete):
            self.action_space = Discrete(action_type.n)
        
        self.observation_var = ['t_out', 't_in', 'occ', 'light', 'Equip']
        self.action_var = ['Thermostat']
        self.num_envs = 1
        self.agent = agent
        self.ready_to_train = False
        self.args = tyro.cli(args)
        self.loss_list = []
        self.success_n = 0
        self.batch_n = 0
        self.obs_batch = torch.zeros(args.batch_size, observation_dim).to('cuda')
        self.action_batch = torch.zeros(args.batch_size, 1).to('cuda')
        self.return_batch = torch.zeros(args.batch_size, 1).to('cuda')
        self.simulator.events.on('end_zone_timestep_after_zone_reporting', self.handler)
        self.baseline = pd.read_csv('Data\\Day_mean.csv')
        # self.baseline['Time'] = pd.to_datetime(self.baseline['Time'], format='%m/%d/%Y %H:%M')

    def setup(self, algo):
        self.algo = algo
        self.agent = self.algo.policy
        self.ready_to_train = True
        
    def run(self, agent = None):
        self.sensor_index = 0
        # if agent is not None:
        #     self.agent = agent
        asyncio.run(energyplus_running(self.simulator, self.idf_file, self.epw_file))

    def normalize_input(self, data=None):
        nor_min = np.array([22.8, 22, 0, 0, 0])
        nor_mean = np.array([28.7, 26, 0.77, 0.57, 0.9])
        # nor_min = np.array([0, 0, 0, 0, 0])
        nor_max = np.array([33.3, 27, 1, 1, 1])
        std = np.array([2, 0.5, 0.4, 0.26, 0.26])
        # nor_max = np.array([1, 1, 1, 1, 1])
        if data == None:
            data = self.sensor_dic[self.observation_var]
        # nor_input = (data - nor_min)/(nor_max - nor_min)
        nor_input = (data - nor_mean)/std
        # nor_input = (data - np.array([27, 25, 0.5, 0.5, 0.5]))/np.array([3, 1, 0.2, 0.2, 0.2])
        j = 0
        for i in self.observation_var:
            col_i =  i + "_nor"
            self.sensor_dic[col_i] = nor_input.iloc[:, j]
            j+=1

    def normalize_input_i(self, state):
        nor_min = np.array([22.8, 22, 0, 0, 0])
        nor_mean = np.array([28.7, 26, 0.78, 0.58, 0.89])
        std = np.array([2.17, 0.5, 0.39, 0.26, 0.26])
        # nor_min = np.array([0, 0, 0, 0, 0])
        nor_max = np.array([33.3, 27, 1, 1, 1])
        # nor_max = np.array([1, 1, 1, 1, 1])
        return (state- nor_mean)/std
        return (state- nor_min)/(nor_max - nor_min)
        # return (state - np.array([27, 25, 0.5, 0.5, 0.5]))/np.array([3, 1, 0.2, 0.2, 0.2])
    
    def label_working_time(self):
        start = pd.to_datetime(self.args.work_time_start, format='%H:%M')
        end = pd.to_datetime(self.args.work_time_end, format='%H:%M')
        # remove data without enough outlook step
        dt = int(60/self.args.n_time_step)
        dt = pd.to_timedelta(dt, unit='min')
        # end -= dt
        wt = [] # wt: working time label
        terminations = [] # terminations: end of working time
        for i in range(int(self.sensor_dic.shape[0])):
            h = self.sensor_dic['Time'].iloc[i].hour
            m = self.sensor_dic['Time'].iloc[i].minute
            t = pd.to_datetime(str(h)+':'+str(m), format='%H:%M')
            if t >= start and t < end:
                wt.append(True)
            else:
                wt.append(False)
            if t >= end - dt:
                terminations.append(True)
            else:
                terminations.append(False)
        self.sensor_dic['Working_time'] = wt
        self.sensor_dic['Terminations'] = terminations    

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

    def cal_r(self):
        baseline = pd.read_csv('Data\Day_mean.csv')
        reward = []
        result = []
        # Realtime reward function
        for j in range(self.sensor_dic.shape[0]):
            energy_i = self.sensor_dic['Chiller Electricity Rate'].iloc[j]
            k = j % (24*self.args.n_time_step)
            baseline_i = baseline['Day_mean'].iloc[k]
            reward_i = max(round(0.3 - abs(energy_i ** 2 - baseline_i ** 2)/baseline_i ** 2,2),-0.4)*5
            result_i = round(1 - abs(energy_i - baseline_i)/baseline_i,2)
            # reward_i = result_i
            # if reward_i<0.8:
            #     reward_i = reward_i**2
            # else:
            #     reward_i+=reward_i*5
            reward.append(reward_i)
            result.append(result_i)          
        
        reward = reward[1:]
        result = result[1:]
        self.actions = self.actions[0:-1]
        self.logprobs = self.logprobs[0:-1]
        self.sensor_dic =  self.sensor_dic[0:-1]
        self.sensor_dic['rewards'] = reward
        self.sensor_dic['results'] = result

    def cal_r_i(self, data, time):
        # baseline = pd.read_csv('Data\Day_mean.csv')
        hour = time.hour
        min = time.minute
        idx = int(hour*6+int(min/10))
        baseline_i = self.baseline['Day_mean'].iloc[idx]
        # reward_i = max(round(0.3 - abs(data ** 2 - baseline_i ** 2)/baseline_i ** 2,2),-0.4)*5
        # result_i = round(1 - abs(data - baseline_i)/baseline_i,2)
        # return reward_i, result_i, baseline_i
        # baseline_energy = self.baseline['cooling_energy'].iloc[idx]
        actual_reduction = (baseline_i - data) / baseline_i
        
        # Target reduction percentage
        target_reduction = 0.25
        
        energy_reward = 2 - abs(actual_reduction - target_reduction) * 10
        return energy_reward, actual_reduction, baseline_i
        
    
    def cal_return(self, reward_list):
        R = 0
        for r in reward_list[::-1]:
            R = r + R * self.args.gamma
        return R

    def handler(self, __event):
        global thinenv
        try:
            obs = thinenv.observe()
            t = self.simulator.variables.getdefault(
                ooep.WallClock.Ref()
            ).value
            warm_up = False
        except:
            warm_up = True

        if not warm_up:
            state = [float(obs[i]) for i in self.observation_var]
            cooling_energy =  obs['Energy_1'].item() + obs['Energy_2'].item() + obs['Energy_3'].item() + obs['Energy_4'].item() + obs['Energy_5'].item()
            state = self.normalize_input_i(state)
            state = torch.Tensor(state).cuda() if torch.cuda.is_available() and self.args.cuda else torch.Tensor(state).cpu()
            with torch.no_grad():
                actions, logprob = self.agent(state)
                # actions = torch.argmax(q_values, dim=0).cpu().numpy()
            obs = pd.DataFrame(obs, index = [self.sensor_index])                
            obs.insert(0, 'Time', t)
            obs.insert(0, 'day_of_week', t.weekday())
            obs.insert(1, 'Working time', self.label_working_time_i(t))            
            obs.insert(obs.columns.get_loc("t_in") + 1, 'Thermostat', 23+4*actions.cpu().numpy())
            reward_i, result_i, baseline_i = self.cal_r_i(cooling_energy, t)
            obs['cooling_energy'] = cooling_energy
            obs['results'] = result_i
            obs['rewards'] = reward_i
            obs['baseline'] = baseline_i
            obs.insert(obs.columns.get_loc("t_in") + 1, 'actions', actions.cpu().numpy())
            obs.insert(obs.columns.get_loc("t_in") + 1, 'logprobs', logprob.cpu().numpy())


            if self.sensor_index == 0:
                self.sensor_dic = pd.DataFrame({})
                self.sensor_dic = obs
                self.logprobs = [logprob]
                # self.values = [value]
                self.actions = [actions]
                self.states = [state]
                self.rewards = [reward_i]
            else:
                self.sensor_dic = pd.concat([self.sensor_dic, obs])           
                self.logprobs.append(logprob) 
                # self.values.append(value) 
                self.actions.append(actions)
                self.states.append(state)
                self.rewards.append(reward_i)
            actions = actions.cpu().numpy()
            com = 23. + actions * 4
            act = thinenv.act({'Thermostat': max(min(com, 27), 23)})
            # act = thinenv.act({'Thermostat': 27})

            b  = self.args.outlook_steps + 1
            if self.sensor_index > b:
                i = self.sensor_index-b
                if i % self.args.step_size == 0:
                    if np.sum(self.sensor_dic['Working time'].iloc[i:(self.sensor_index)]) == b:
                        ob_i = self.states[i]
                        r_i = self.rewards[i]
                        logp_i = self.logprobs[i]
                        action_i = self.actions[i]
                        R_i = self.cal_return(self.rewards[i+1:i+b])
                        if self.batch_n<self.args.batch_size:
                            self.obs_batch[self.batch_n, :] = ob_i
                            self.return_batch[self.batch_n, :] = R_i
                            self.action_batch[self.batch_n, :] = action_i
                            self.batch_n+=1
                        else:
                            self.batch_n=0
                            loss_i = self.algo.train(self.obs_batch, self.action_batch, self.return_batch)
                            self.loss_list.append(loss_i)
                            
            self.sensor_index+=1