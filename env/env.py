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
                 agent,
                 args) -> None:
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
        self.observation_space = Box(np.array([-np.inf] * 5), np.array([np.inf] * 5))
        self.action_space = Discrete(5)
        self.observation_var = ['t_out', 't_in', 'occ', 'light', 'Equip']
        self.action_var = ['Thermostat']
        self.num_envs = 1
        self.agent = agent
        self.args = tyro.cli(args)
        self.simulator.events.on('end_zone_timestep_after_zone_reporting', self.handler)
        
    def run(self):
        self.sensor_index = 0
        asyncio.run(energyplus_running(self.simulator, self.idf_file, self.epw_file))

    def normalize_input(self, data=None):
        nor_min = np.array([22.8, 22, 0, 0, 0])
        nor_max = np.array([33.3, 27, 1, 1, 1])
        if data == None:
            data = self.sensor_dic[self.observation_var]
        nor_input = (data - nor_min)/(nor_max - nor_min)
        j = 0
        for i in self.observation_var:
            col_i =  i + "_nor"
            self.sensor_dic[col_i] = nor_input.iloc[:, j]
            j+=1

    def normalize_input_i(self, state):
        nor_min = np.array([22.8, 22, 0, 0, 0])
        nor_max = np.array([33.3, 27, 1, 1, 1])
        return (state- nor_min)/(nor_max - nor_min)
    
    def label_working_time(self):
        start = pd.to_datetime(self.args.work_time_start, format='%H:%M')
        end = pd.to_datetime(self.args.work_time_end, format='%H:%M')
        # remove data without enough outlook step
        dt = int(60/self.args.n_time_step)
        dt = pd.to_timedelta(dt, unit='min')
        end -= dt
        wt = [] # wt: working time label
        terminations = [] # terminations: end of working time
        for i in range(int(self.sensor_dic.shape[0])):
            h = self.sensor_dic['Time'].iloc[i].hour
            m = self.sensor_dic['Time'].iloc[i].minute
            t = pd.to_datetime(str(h)+':'+str(m), format='%H:%M')
            if t >= start and t < end:
                wt.append(True)
                terminations.append(False)
                if t == end:
                    wt.append(True)
                    terminations.append(True)
            else:
                wt.append(False)
                terminations.append(False)
        self.sensor_dic['Working_time'] = wt
        self.sensor_dic['Terminations'] = terminations    

    def cal_r(self):
        baseline = pd.read_csv('Data\Day_mean.csv')
        reward = []
        result = []
        # Realtime reward function
        for j in range(self.sensor_dic.shape[0]):
            energy_i = self.sensor_dic['Chiller Electricity Rate'].iloc[j]
            k = j % (24*self.args.n_time_step)
            baseline_i = baseline['Day_mean'].iloc[k]
            reward_i = round(0.3 - abs(energy_i ** 2 - baseline_i ** 2)/baseline_i ** 2,1)
            result_i = round(1 - abs(energy_i - baseline_i)/baseline_i,1)
            reward.append(reward_i)
            result.append(result_i)            
        self.sensor_dic['rewards'] = reward
        self.sensor_dic['results'] = result 

    def cal_return(self):
        advantages = np.zeros(self.sensor_dic.shape[0])
        for t in reversed(range(self.sensor_dic.shape[0]-1)):
            with torch.no_grad():
                lastgaelam = 0
                nextnonterminal = 1.0 - self.sensor_dic['Terminations'].iloc[t + 1]
                nextvalues = self.sensor_dic['values'].iloc[t+1].reshape(1, -1)
                delta = self.sensor_dic['rewards'].iloc[t] + self.args.gamma * nextvalues * nextnonterminal - self.sensor_dic['values'].iloc[t]
                delta = delta[0][0]
                lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + self.sensor_dic['values']
        self.sensor_dic['returns'] = returns
        self.sensor_dic['advantages'] = advantages
        self.sensor_dic = self.sensor_dic[:-1]

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
            state = self.normalize_input_i(state)
            state = torch.Tensor(state).cuda() if torch.cuda.is_available() and self.args.cuda else torch.Tensor(state).cpu()
            with torch.no_grad():
                actions, value, logprob = self.agent(state)
                # actions = torch.argmax(q_values, dim=0).cpu().numpy()
            com = 23. + actions.cpu().numpy()[0]

            act = thinenv.act({'Thermostat': com})
            
            obs = pd.DataFrame(obs, index = [self.sensor_index])
            obs.insert(0, 'Time', t)
            # obs.insert(obs.columns.get_loc("t_in") + 1, 'Thermostat', actions.cpu().numpy())
            # obs.insert(obs.columns.get_loc("t_in") + 1, 'logprobs', logprob.cpu().numpy())
            # obs.insert(obs.columns.get_loc("t_in") + 1, 'values', value.flatten().cpu().numpy())
            # obs.insert(obs.columns.get_loc("t_in") + 1, 'Thermostat', 1)
            # obs.insert(obs.columns.get_loc("t_in") + 1, 'logprobs', 1)
            # obs.insert(obs.columns.get_loc("t_in") + 1, 'values', 1)            
            if self.sensor_index == 0:
                self.sensor_dic = pd.DataFrame({})
                self.sensor_dic = obs
            else:
                self.sensor_dic = pd.concat([self.sensor_dic, obs])            
            self.sensor_index+=1
            
