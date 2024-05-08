from energyplus.ooep.addons.progress import ProgressProvider
import asyncio
import pandas as pd
from rl.dqn.network import QNetwork
import random
import numpy as np
import time
from energyplus import ooep
import torch.nn.functional as F
import os
from stable_baselines3.common.buffers import ReplayBuffer
from rl.dqn.dqn_para import Args
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

simulator = Simulator().add(
    ProgressProvider(),
    #LogProvider(),
)

async def energyplus_running(idf_file,epw_file):
    await simulator.awaitable.run(
        input=Simulator.InputSpecs(
            model=(
                'Large office - 1AV232.idf'
            ),
            weather=('USA_FL_Miami.722020_TMY2.epw'),
        ),
        output=Simulator.OutputSpecs(
            #report=('/tmp/ooep-report-9e1287d2-8e75-4cf5-bbc5-f76580b56a69'),
        ),
        options=Simulator.RuntimeOptions(
            #design_day=True,
        ),
    ) 

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
simulator.add(
    thinenv := ThinEnv(
        action_space=action_space,    
        observation_space=observation_space,
    )
)



class dqn():
    def __init__(self, observation_var, action_var, auto_fine_tune = False, sweep_config = {}) -> None:
        self.observation_var = observation_var
        self.action_var = action_var
        self.sweep_config = sweep_config
        self.args = tyro.cli(Args)
        self.run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])))
        
        self.q_network, self.target_network = self.set_network(self.args.input_dim, self.args.output_dim)
        self.optimizer = self.set_optimizer()
        simulator.events.on('end_zone_timestep_after_zone_reporting', self.handler)


    def train_auto_fine_tune(self):
        with wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                config=self.sweep_config,
                name=self.run_name,
                save_code=True,
            ):
            self.args = wandb.config
            for k, v in tyro.cli(Args).__dict__.items():
                if k not in self.args:
                    self.args[str(k)] = v
            self.train()

    def train(self):
        # if self.args.track:
        #     self.call_track()
        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
        rb = ReplayBuffer(
            self.args.buffer_size,
            Box(np.array([-1] * self.args.input_dim), np.array([1] * self.args.input_dim)),
            Discrete(self.args.output_dim),
            self.device,
            handle_timeout_termination=False,
        )

        start_time = time.time()
        for global_step in range(self.args.total_timesteps):
            # ALGO LOGIC: put action logic here
            epsilon = self.linear_schedule(self.args.start_e, self.args.end_e, self.args.exploration_fraction * self.args.total_timesteps, global_step)
            # if random.random() < epsilon:
            #     actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            # else:
            #     q_values = q_network(torch.Tensor(obs).to(device))
            #     actions = torch.argmax(q_values, dim=1).cpu().numpy()

            self.epsilon = self.linear_schedule(self.args.start_e, self.args.end_e,
                                      self.args.exploration_fraction * self.args.total_timesteps,
                                      global_step)
            self.run()
            self.normalize_input()
            # self.resample()
            self.label_working_time()
            self.cal_r()
            # self.normalize_input()
            Performance = np.mean(self.sensor_dic['result'][self.sensor_dic['Working_time'] == True])
            if  Performance>0.85:
                path_i = os.path.join('Archive results', str(int(time.time())))
                os.mkdir(path_i)
                self.sensor_dic.to_csv(os.path.join(path_i, 'results.csv'))
                torch.save(self.q_network.state_dict(), os.path.join(path_i, 'model.pth'))
            for i in range(self.sensor_dic.shape[0]-1):
                obs_i = self.sensor_dic[self.observation_var].iloc[i]
                next_obs_i = self.sensor_dic[self.observation_var].iloc[i]
                actions_i = self.sensor_dic[self.action_var].iloc[i]
                rewards_i = self.sensor_dic['reward'].iloc[i]
                if self.sensor_dic['Working_time'].iloc[i]:
                    # To Do: too slow
                    rb.add(np.array(obs_i),
                        np.array(next_obs_i),
                        np.array(actions_i),
                        np.array(rewards_i),
                        np.array([False]),
                            '')
            if global_step >= self.args.learning_starts:
                for k in range(1):
                    if global_step % self.args.train_frequency == 0:
                        data = rb.sample(self.args.batch_size)
                        with torch.no_grad():
                            target_max, _ = self.target_network(data.next_observations).max(dim=1)
                            td_target = data.rewards.flatten() + self.args.gamma * target_max
                        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                        loss = F.mse_loss(td_target, old_val)

                        if global_step % 2 == 0:
                            self.writer.add_scalar("losses/td_loss", loss, global_step)
                            self.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                            if self.args.track:
                                if not self.train_auto_fine_tune:
                                    wandb.init(
                                        project=self.args.wandb_project_name,
                                        entity=self.args.wandb_entity,
                                        sync_tensorboard=True,
                                        config=self.args,
                                        name=self.run_name,
                                        save_code=True,
                                    )
                                wandb.log({'reward_curve': np.mean(self.sensor_dic['reward'][self.sensor_dic['Working_time'] == True])}, step=global_step)        
                                wandb.log({'result_curve': Performance}, step=global_step)        
                                wandb.log({'loss_curve': float(loss.cpu().detach().numpy())}, step=global_step)        
                                # wandb.log({'epsilon_curve': float(epsilon)}, step=global_step)        

                            print("SPS:", int(global_step / (time.time() - start_time)))
                            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                        # optimize the model
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                # update target network
                if global_step % self.args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                        target_network_param.data.copy_(
                            self.args.tau * q_network_param.data + (1.0 - self.args.tau) * target_network_param.data
                        )  

    def run(self):
        self.sensor_index = 0
        asyncio.run(energyplus_running('Large office - 1AV232.idf', 'USA_FL_Miami.722020_TMY2.epw'))

                    
    def resample(self):
        month_start = self.sensor_dic['Time'][0].month
        day_start = self.sensor_dic['Time'][0].day
        year_start = self.sensor_dic['Time'][0].year
        start = pd.to_datetime(str(year_start) + '-' + str(month_start) + '-' + str(day_start), format='%Y-%m-%d')
        month_end = self.sensor_dic['Time'][self.sensor_dic.shape[0]-1].month
        day_end = self.sensor_dic['Time'][self.sensor_dic.shape[0]-1].day
        year_end = self.sensor_dic['Time'][self.sensor_dic.shape[0]-1].year
        end = pd.to_datetime(str(year_end) + '-' + str(month_end) + '-' + str(day_end), format='%Y-%m-%d')
        ts = pd.date_range(start=start, end=end, freq=str(int(60/self.args.n_time_step))+'min')
        self.sensor_dic.index = self.sensor_dic['Time']
        # TO DO: check resample results
        ts_m = self.sensor_dic.resample('1min').interpolate(method='linear')
        ts_m = ts_m.resample(str(int(60/self.args.n_time_step))+'min').asfreq()
        self.sensor_dic = ts_m[1:]

    def call_track(self):
        pass

    def normalize_input(self):
        nor_min = np.array([22.8, 22, 0, 0, 0])
        nor_max = np.array([33.3, 27, 1, 1, 1])
        nor_input = (self.sensor_dic[self.observation_var] - nor_min)/(nor_max - nor_min)
        self.sensor_dic[self.observation_var] = nor_input        
            
    def handler(self, __event):
        global thinenv
        try:
            obs = thinenv.observe()
            t = simulator.variables.getdefault(
                ooep.WallClock.Ref()
            ).value
            warm_up = False
        except:
            warm_up = True

        if not warm_up:
            state = [float(obs[i]) for i in self.observation_var]
            state = torch.Tensor(state).cuda() if torch.cuda.is_available() and self.args.cuda else torch.Tensor(state).cpu()
            if random.random() < self.epsilon:
                actions = random.sample(list(np.arange(0, self.args.output_dim)), 1)[0]
            else:
                q_values = self.q_network(state).to(self.device)  
                actions = torch.argmax(q_values, dim=0).cpu().numpy()
            com = 23. + actions

            act = thinenv.act({'Thermostat': com})
            # act = thinenv.act({'Thermostat': 26})
            # thinenv.act(
            #     thinenv.action_space.sample()
            # )
            # t = simulator.variables.getdefault(
            #     ooep.WallClock.Ref()
            # ).value            
            obs = pd.DataFrame(obs, index = [self.sensor_index])
            obs.insert(0, 'Time', t)
            obs.insert(obs.columns.get_loc("t_in") + 1, 'Thermostat', actions)
            if self.sensor_index == 0:
                self.sensor_dic = pd.DataFrame({})
                self.sensor_dic = obs
            else:
                self.sensor_dic = pd.concat([self.sensor_dic, obs])            
            self.sensor_index+=1

    # def to_tensor(self, obs):
    #     value = []
    #     for i in obs:
    #         value.append(obs[i])
    #     return torch.tensor(np.array(value))
    
    def set_network(self, input_dim, output_dim):
        q_network = QNetwork(self.args.input_dim, self.args.output_dim).to(self.device)
        target_network = QNetwork(self.args.input_dim, self.args.output_dim).to(self.device)
        target_network.load_state_dict(q_network.state_dict())
        return q_network, target_network
    
    def set_optimizer(self):
        optimizer = optim.Adam(self.q_network.parameters(), lr=self.args.learning_rate)
        return optimizer
    
    def control_fun(self, observation):
        if random.random() < self.epsilon:
            actions = random.sample(list(np.arange(0, 5)), 1)[0]
        else:
            q_values = self.agent(torch.Tensor(observation).to(self.device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()
        com = [23 + actions]
        # com = [23]
        return com, [actions]

    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)    

    def label_working_time(self):
        start = pd.to_datetime(self.args.work_time_start, format='%H:%M')
        end = pd.to_datetime(self.args.work_time_end, format='%H:%M')
        # remove data without enough outlook step
        dt = int(60/self.args.n_time_step) * self.args.outlook_step
        dt = pd.to_timedelta(dt, unit='min')
        end -= dt
        wt = [] # wt: working time label
        for i in range(int(self.sensor_dic.shape[0])):
            h = self.sensor_dic['Time'].iloc[i].hour
            m = self.sensor_dic['Time'].iloc[i].minute
            t = pd.to_datetime(str(h)+':'+str(m), format='%H:%M')
            if t >= start and t <= end:
                wt.append(True)
            else:
                wt.append(False)
        self.sensor_dic['Working_time'] = wt

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
        self.sensor_dic['reward'] = reward
        self.sensor_dic['result'] = result
        # Return return function (future accmulated reward)
        R_list = []
        for i in range(self.sensor_dic.shape[0] - self.args.outlook_step):
            reward_list = self.sensor_dic['reward'][i:(i+ self.args.outlook_step)]
            R = 0
            for r in reward_list[::-1]:
                R = r + R * self.args.gamma
            R_list.append(R)
        
        # Remove data without enough outlook steps
        self.sensor_dic = self.sensor_dic[:-self.args.outlook_step]
        # if myidf.control:
        #     myidf.cmd_dic = myidf.cmd_dic[:-self.args.outlook_step]
            # append Return data
        self.sensor_dic['Return'] = R_list        

if __name__ == '__main__':
    default_paras = tyro.cli(Args)
    parameters_dict = {
    'learning_rate': {
        'values': [1e-2, 1e-3, 1e-4]
        },
    'batch_size': {
          'values': [32, 64, 128]
        },
    'tau': {
          'values': [0.9, 0.8]
        },     
    'start_e': {
          'values': [0.8, 0.5, 0.2]
        },     
    'train_frequency': {
          'values': [1, 5, 10]
        },   
    'target_network_frequency': {
          'values': [5, 20, 30]
        },                                
    }
    sweep_config = {
    'method': 'random'
    }
    metric = {
    'name': 'Performance',
    'goal': 'maximize'   
    }
    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="energygym-auto")
    observation_var = ['t_out', 't_in', 'occ', 'light', 'Equip']
    action_var = ['Thermostat']
    a = dqn(observation_var, action_var, True, sweep_config)
    wandb.agent(sweep_id, a.train_auto_fine_tune, count=20) 
    # a.train()
