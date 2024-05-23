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
                 action_dim,
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
        self.observation_space = Box(np.array([-np.inf] * observation_dim), np.array([np.inf] * observation_dim))
        self.action_space = Discrete(action_dim)
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
        # nor_min = np.array([0, 0, 0, 0, 0])
        nor_max = np.array([33.3, 27, 1, 1, 1])
        # nor_max = np.array([1, 1, 1, 1, 1])
        if data == None:
            data = self.sensor_dic[self.observation_var]
        nor_input = (data - nor_min)/(nor_max - nor_min)
        # nor_input = (data - np.array([27, 25, 0.5, 0.5, 0.5]))/np.array([3, 1, 0.2, 0.2, 0.2])
        j = 0
        for i in self.observation_var:
            col_i =  i + "_nor"
            self.sensor_dic[col_i] = nor_input.iloc[:, j]
            j+=1

    def normalize_input_i(self, state):
        nor_min = np.array([22.8, 22, 0, 0, 0])
        # nor_min = np.array([0, 0, 0, 0, 0])
        nor_max = np.array([33.3, 27, 1, 1, 1])
        # nor_max = np.array([1, 1, 1, 1, 1])
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
        h = t.hour
        m = t.minute
        t = pd.to_datetime(str(h)+':'+str(m), format='%H:%M')
        if t >= start and t < end:
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
        baseline = pd.read_csv('Data\Day_mean.csv')
        hour = time.hour
        min = time.minute
        idx = int(hour*6+int(min/10))
        baseline_i = baseline['Day_mean'].iloc[idx]
        reward_i = max(round(0.3 - abs(data ** 2 - baseline_i ** 2)/baseline_i ** 2,2),-0.4)*5
        return reward_i
    
    def cal_return(self, reward_list):
        R = 0
        for r in reward_list[::-1]:
            R = r + R * self.args.gamma
        return R
  

    # def cal_return(self):
    #     advantages = np.zeros(self.sensor_dic.shape[0])
    #     for t in reversed(range(self.sensor_dic.shape[0]-1)):
    #         with torch.no_grad():
    #             lastgaelam = 0
    #             nextnonterminal = 1.0 - self.sensor_dic['Terminations'].iloc[t + 1]
    #             nextvalues = self.sensor_dic['values'].iloc[t+1].reshape(1, -1)
    #             delta = self.sensor_dic['rewards'].iloc[t] + self.args.gamma * nextvalues * nextnonterminal - self.sensor_dic['values'].iloc[t]
    #             delta = delta[0][0]
    #             lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
    #             advantages[t] = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
    #     returns = advantages + self.sensor_dic['values']
    #     self.sensor_dic['returns'] = returns
    #     self.sensor_dic['advantages'] = advantages
    #     self.sensor_dic = self.sensor_dic[:-1]

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
            cooling_rate = obs['Chiller Electricity Rate'].item()
            state = self.normalize_input_i(state)
            state = torch.Tensor(state).cuda() if torch.cuda.is_available() and self.args.cuda else torch.Tensor(state).cpu()
            with torch.no_grad():
                actions, logprob = self.agent(state)
                # actions = torch.argmax(q_values, dim=0).cpu().numpy()
            obs = pd.DataFrame(obs, index = [self.sensor_index])                
            obs.insert(0, 'Time', t)
            obs.insert(1, 'Working time', self.label_working_time_i(t))            
            obs.insert(obs.columns.get_loc("t_in") + 1, 'Thermostat', 23+actions.cpu().numpy())
            reward_i = self.cal_r_i(cooling_rate, t)
            # obs.insert(obs.columns.get_loc("t_in") + 1, 'actions', actions)
            # obs.insert(obs.columns.get_loc("t_in") + 1, 'logprobs', logprob)
            # obs.insert(obs.columns.get_loc("t_in") + 1, 'values', value.flatten())
            # obs.insert(obs.columns.get_loc("t_in") + 1, 'Thermostat', 1)
            # obs.insert(obs.columns.get_loc("t_in") + 1, 'logprobs', 1)
            # obs.insert(obs.columns.get_loc("t_in") + 1, 'values', value)            
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
            com = 23. + actions
            act = thinenv.act({'Thermostat': com})
            if self.sensor_index > self.args.outlook_steps:
                i = self.sensor_index-self.args.outlook_steps
                if np.sum(self.sensor_dic['Working time'].iloc[i:self.sensor_index]) == self.args.outlook_steps:
                    ob_i = self.states[i]
                    r_i = self.rewards[i]
                    logp_i = self.logprobs[i]
                    action_i = self.actions[i]
                    R_i = self.cal_return(self.rewards[i:i+self.args.outlook_steps])

                    bbb = 1

            self.sensor_index+=1

    def train(self, obs, actions, returns, policy) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # self.policy.action_network.train()

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)


        # for rollout_data in self.rollout_buffer.get(batch_size=self.batch_size):

        # for rollout_data in self.rollout_buffer.get(batch_size=None):
            # if n_train >= max_train_perEp:
            #     break
        # rollout_data = self.rollout_buffer.get(batch_size=self.batch_size, shuffle=self.args.shuffle)
        actions = actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = actions.long().flatten()

        log_prob, entropy = self.policy.evaluate_actions(obs.float(), actions)
        # for name, param in self.policy.named_parameters():
        #     print(name, param.shape)            
        # values, log_prob, entropy = rollout_data.old_log_prob
        # values = values.flatten()

        # Normalize advantage (not present in the original implementation)
        advantages = returns
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient loss
        policy_loss = -(advantages * log_prob)

        # Value loss using the TD(gae_lambda) target
        # value_loss = F.mse_loss(rollout_data.returns, values)

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        loss = self.args.pol_coef * policy_loss + self.ent_coef * entropy_loss

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        # Check gradient
        # for name, param in self.policy.mlp_extractor.named_parameters():
        # for name, param in self.policy.features_extractor.policy_fe.named_parameters():
        # for name, param in self.policy.action_network.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.grad}")       
        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()


        # explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        # self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        # return policy_loss.item(), np.mean(self.rollout_buffer.logprobs[106])
        return policy_loss.item()            
            
