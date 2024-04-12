from epluspy import idf_editor
from epluspy import idf_simu
import energyplus.ooep as _ooep_
import gymnasium as _gymnasium_
import numpy as _numpy_
import energyplus.ooep.addons
from energyplus.ooep.components.variables import (
    Actuator,
    OutputVariable,
)
import logging
from energyplus.ooep.addons.logging import LogProvider

from energyplus.ooep.addons.rl.gymnasium.spaces import VariableBox
from energyplus.ooep import (
    Simulator,
    Model,
    Weather,
    Report,
)
import asyncio
from energyplus.ooep.addons.rl.gymnasium import ThinEnv

import os
import json
from hyper_para.dqn_para import Args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tyro
import random
import time
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
import wandb
import math
import pandas as pd
import logging

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_dim),
        )

    def forward(self, x):
        return self.network(x)
    
class dqn():
    def __init__(self, idf_env, input_var, q_network, target_network) -> None:
        self.idf_env = idf_env
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.input_var = input_var
        self.obs_index = []
        self.q_network = q_network
        self.target_network = target_network
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
        
    def train(self):
        if args.track:
            self.call_track()
        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        rb = ReplayBuffer(
            args.buffer_size,
            Box(np.array([-1] * args.input_dim), np.array([1] * args.input_dim)),
            Discrete(args.output_dim),
            device,
            handle_timeout_termination=False,
        )
        start_time = time.time()
        
        # TRY NOT TO MODIFY: start the game
        for global_step in range(args.total_timesteps):
            if args.track:
                wandb.log({'random_curve':global_step/100+random.random()},step=global_step)
                wandb.log({'log_curve': math.log(global_step+1)},step=global_step)        
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            # if random.random() < epsilon:
            #     actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            # else:
            #     q_values = q_network(torch.Tensor(obs).to(device))
            #     actions = torch.argmax(q_values, dim=1).cpu().numpy()

            epsilon = linear_schedule(args.start_e, args.end_e,
                                      args.exploration_fraction * args.total_timesteps,
                                      global_step)
            myidf.run(epsilon = epsilon)
            self.label_working_time()
            self.cal_r()
            myidf.save()
            if len(self.obs_index) == 0:
                sensor_name_list = list(myidf.sensor_dic.columns)
                for i in self.input_var:
                    assert i in sensor_name_list, "The input variable is not in the sensor list, please add it"
                    self.obs_index.append(sensor_name_list.index(i))
            for i in range(myidf.sensor_dic.shape[0]-1):
                obs = myidf.sensor_dic.iloc[i,self.obs_index]
                next_obs = myidf.sensor_dic.iloc[i+1,self.obs_index]
                rewards = myidf.sensor_dic.iloc[i, list(myidf.sensor_dic.columns).index('reward' )]
                actions = myidf.action_dic.iloc[i, 0]
                if myidf.sensor_dic['Working_time'][i]:
                    rb.add(np.array(obs),
                        np.array(next_obs),
                        np.array(actions),
                        np.array(rewards),
                        np.array([False]),
                            '')
            if global_step > args.learning_starts:
                if global_step % args.train_frequency == 0:
                    data = rb.sample(args.batch_size)
                    with torch.no_grad():
                        target_max, _ = self.target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max
                    old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 10 == 0:
                        self.writer.add_scalar("losses/td_loss", loss, global_step)
                        self.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # update target network
                if global_step % args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )            

    def call_track(self):
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=self.run_name,
            # monitor_gym=True,
            save_code=True,
        )

    def label_working_time(self):
        start = pd.to_datetime(args.work_time_start, format='%H:%M')
        end = pd.to_datetime(args.work_time_end, format='%H:%M')
        # remove data without enough outlook step
        dt = int(60/myidf.n_time_step) * args.outlook_step
        dt = pd.to_timedelta(dt, unit='min')
        end -= dt
        wt = [] # wt: working time label
        for i in range(int(myidf.sensor_dic.shape[0])):
            h = myidf.sensor_dic['Time'][i].minute
            m = myidf.sensor_dic['Time'][i].hour
            t = pd.to_datetime(str(h)+':'+str(m), format='%M:%S')
            if t >= start and t <= end:
                wt.append(True)
            else:
                wt.append(False)
        myidf.sensor_dic['Working_time'] = wt

    def cal_r(self):
        baseline = pd.read_csv('Data\Day_mean.csv')
        reward = []
        for j in range(myidf.n_days+1):
            for k in range(24*myidf.n_time_step):
                reward_i = abs(myidf.sensor_dic['Chiller Electricity Rate@DOE REF 1980-2004 WATERCOOLED  CENTRIFUGAL CHILLER 0 1100TONS 0.7KW/TON'][j*24*myidf.n_time_step+k] - baseline['Day_mean'][k])
                reward.append(reward_i)
        # Realtime reward function
        myidf.sensor_dic['reward'] = reward
        # Return return function (future accmulated reward)
        R_list = []
        for i in range(myidf.sensor_dic.shape[0] - args.outlook_step):
            reward_list = myidf.sensor_dic['reward'][i:(i+ args.outlook_step)]
            R = 0
            for r in reward_list[::-1]:
                R = r + R * args.gamma
            R_list.append(R)
        
        # Remove data without enough outlook steps
        myidf.sensor_dic = myidf.sensor_dic[:-args.outlook_step]
        if myidf.control:
            myidf.cmd_dic = myidf.cmd_dic[:-args.outlook_step]
            # append Return data
        myidf.sensor_dic['Return'] = R_list
    
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class ep_simu(idf_simu.IDF_simu):
    def control_fun(self, senstor_t):
        value = []
        for i in range(len(self.input_var)):
            value.append(list(senstor_t[self.input_var[i]])[0])
        if random.random() < self.epsilon:
            actions = random.sample(list(np.arange(0, 5)), 1)[0]
        else:
            q_values = self.agent(torch.Tensor(value).to(args.devices))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()
        # com = [19 + actions]
        com = [28]
        return com, [actions]
    
# async def energyplus_running(idf_file,epw_file):
#     await simulator.awaitable.run_forever(
#     input=Simulator.InputSpecs(
#         model=Model().open(
#             idf_file
#         ),
#         weather=Weather().open(epw_file),
#     ),
#     # output=Simulator.OutputSpecs(
#     #     report=Report().open('/tmp/ooep-report-9e1287d2-8e75-4cf5-bbc5-f76580b56a69'),
#     # ),
#     options=Simulator.RuntimeOptions(
#         #design_day=True,
#     ),
#     )
    
#     # TODO
#     #simulator.variables.on(...)[...]

if __name__ == '__main__':
    simulator = Simulator().add(
    LogProvider(),
    )
    logging.basicConfig(level='INFO')
    # simulator.add(
    #     thinenv := ThinEnv(
    #         action_space=_gymnasium_.spaces.Dict({
    #                     'thermostat': VariableBox(
    #                         low=15., high=16.,
    #                         dtype=_numpy_.float32,
    #                         shape=(),
    #                     ).bind(Actuator.Ref(
    #                         type='Zone Temperature Control',
    #                         control_type='Heating Setpoint',
    #                         key='CORE_MID',
    #                     ))
    #                 }),    
    #         observation_space=_gymnasium_.spaces.Dict({
    #             'temperature': VariableBox(
    #                 low=-_numpy_.inf, high=+_numpy_.inf,
    #                 dtype=_numpy_.float32,
    #                 shape=(),
    #             ).bind(OutputVariable.Ref(
    #                 type='People Air Temperature',
    #                 key='CORE_MID',
    #             )),
    #         }),
    #     )
    # )
    # try:
    #     print(thinenv.observe())
    # except _ooep_.TemporaryUnavailableError as e:
    #     pass

    run_baseline = True
    idf_file = 'Large office - 1AV940.idf'
    epw_file = 'USA_FL_Miami.722020_TMY2.epw'
    output_path = './test'
    epjson = './Energy+.schema.epJSON'
    args = tyro.cli(Args)
    q_network = QNetwork(args.input_dim, args.output_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    target_network = QNetwork(args.input_dim, args.output_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    input_var = ['Site Outdoor Air Drybulb Temperature@Environment',
                  'Zone Mean Air Temperature@CORE_BOTTOM ZN',
                  'Zone People Sensible Heating Rate@CORE_BOTTOM ZN']
    # # TO ADD: CHECK input_var
    # with open(epjson, 'r') as f:
    #     data = json.load(f)

    # loop = asyncio.get_running_loop()
    # task = loop.create_task(energyplus_running(idf_file = idf_file,
    #                                            epw_file = epw_file))
    myidf = ep_simu(idf_file, epw_file, output_path, '2018-08-01', '2018-08-31', 6, True, True, 5)
    if run_baseline:
        myidf.edit('Thermostatsetpoint:dualsetpoint', 'All', cooling_setpoint_temperature_schedule_name = 'Large Office ClgSetp')
    else:
        myidf.edit('Thermostatsetpoint:dualsetpoint', 'All', cooling_setpoint_temperature_schedule_name = 'ANN-ctrl')
    myidf.sensor_call(dict(Air_System_Outdoor_Air_Mass_Flow_Rate = ['VAV_1'],
                      Chiller_Electricity_Rate  = ['DOE REF 1980-2004 WATERCOOLED  CENTRIFUGAL CHILLER 0 1100TONS 0.7KW/TON'],
                      Site_Outdoor_Air_Drybulb_Temperature = ['Environment'],
                      Zone_Mean_Air_Temperature=['CORE_BOTTOM ZN'],
                      Cooling_Coil_Total_Cooling_Rate=['VAV_1 CLG COIL'],
                      Lights_Total_Heating_Rate=['CORE_BOTTOM ZN OFFICE WHOLEBUILDING - LG OFFICE LIGHTS'],
                      Zone_People_Sensible_Heating_Rate=['CORE_BOTTOM ZN']))
    # To update: directly update to original files
    myidf.actuator_call(Schedule_Value = [['ANN-ctrl', 'Schedule:Compact']])
    # myidf.delete_class('AvailabilityManager:Scheduled')
    # myidf.delete_class('AvailabilityManager:NightCycle')
    # myidf.delete_class('AvailabilityManagerAssignmentList')
    # to update: check why need to be all capital
    myidf.set_agent(q_network, input_var)
    # myidf.run()
    # myidf.save()
    # TO ADD: CHECK AGENT
    rl_env = dqn(myidf, input_var, q_network, target_network)
    rl_env.train()
    # a = 1