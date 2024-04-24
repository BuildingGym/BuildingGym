from epluspy import idf_editor
from epluspy import idf_simu
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

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
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
            if len(self.obs_index) == 0:
                sensor_name_list = list(myidf.sensor_dic.columns)
                for i in self.input_var:
                    assert i in sensor_name_list, "The input variable is not in the sensor list, please add it"
                    self.obs_index.append(sensor_name_list.index(i))
            self.normalize_input()
            if np.mean(myidf.sensor_dic['result'][myidf.sensor_dic['Working_time'] == True]) > 0.85:
                path_i = os.path.join('Archive results', str(int(time.time())))
                os.mkdir(path_i)
                myidf.save(path_i)
                torch.save(q_network.state_dict(), os.path.join(path_i, 'model.pth'))
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
                for k in range(1):
                    if global_step % args.train_frequency == 0:
                        data = rb.sample(args.batch_size)
                        with torch.no_grad():
                            target_max, _ = self.target_network(data.next_observations).max(dim=1)
                            td_target = data.rewards.flatten() + args.gamma * target_max
                        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                        loss = F.mse_loss(td_target, old_val)

                        if global_step % 2 == 0:
                            self.writer.add_scalar("losses/td_loss", loss, global_step)
                            self.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                            wandb.log({'reward_curve': np.mean(myidf.sensor_dic['reward'][myidf.sensor_dic['Working_time'] == True])}, step=global_step)        
                            wandb.log({'result_curve': np.mean(myidf.sensor_dic['result'][myidf.sensor_dic['Working_time'] == True])}, step=global_step)        
                            wandb.log({'loss_curve': float(loss.cpu().detach().numpy())}, step=global_step)        
                            # wandb.log({'epsilon_curve': float(epsilon)}, step=global_step)        

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

    def normalize_input(self):
        nor_min = np.array([22.8, 22, 0, 0, 0])
        nor_max = np.array([33.3, 27, 1, 1, 1])
        nor_input = (myidf.sensor_dic.iloc[:,self.obs_index] - nor_min)/(nor_max - nor_min)
        myidf.sensor_dic.iloc[:,self.obs_index] = nor_input


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
            h = myidf.sensor_dic['Time'][i].hour
            m = myidf.sensor_dic['Time'][i].minute
            t = pd.to_datetime(str(h)+':'+str(m), format='%H:%M')
            if t >= start and t <= end:
                wt.append(True)
            else:
                wt.append(False)
        myidf.sensor_dic['Working_time'] = wt

    def cal_r(self):
        baseline = pd.read_csv('Data\Day_mean.csv')
        reward = []
        result = []
        for j in range(myidf.n_days+1):
            for k in range(24*myidf.n_time_step):
                energy_i = myidf.sensor_dic['Chiller Electricity Rate@DOE REF 1980-2004 WATERCOOLED  CENTRIFUGAL CHILLER 0 1100TONS 0.7KW/TON'][j*24*myidf.n_time_step+k]
                reward_i = round(0.3 - abs(energy_i ** 2 - baseline['Day_mean'][k] ** 2)/baseline['Day_mean'][k] ** 2,1)
                result_i = round(1 - abs(energy_i - baseline['Day_mean'][k])/baseline['Day_mean'][k],1)
                reward.append(reward_i)
                result.append(result_i)
        # Realtime reward function
        myidf.sensor_dic['reward'] = reward
        myidf.sensor_dic['result'] = result
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
        com = [23 + actions]
        # com = [23]
        return com, [actions]
        
if __name__ == '__main__':
    is_train = True
    idf_file = 'Large office - 1AV940.idf'
    epw_file = 'USA_FL_Miami.722020_TMY2.epw'
    output_path = 'test\\'
    epjson = 'C:\\EnergyPlusV9-4-0\\Energy+.schema.epJSON'
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    q_network = QNetwork(args.input_dim, args.output_dim).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(args.input_dim, args.output_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    input_var = ['Site Outdoor Air Drybulb Temperature@Environment',
                  'Zone Mean Air Temperature@CORE_BOTTOM ZN',
                  'Schedule Value@'+'Large Office Bldg Occ'.upper(),
                  'Schedule Value@'+'Large Office Bldg Light'.upper(),
                  'Schedule Value@'+'Large Office Bldg Equip'.upper(),]
    # TO ADD: CHECK input_var
    with open(epjson, 'r') as f:
        data = json.load(f)
    myidf = ep_simu(idf_file, epw_file, output_path, '2018-08-01', '2018-08-31', 6, True, True, 5)
    if is_train:
        myidf.edit('Thermostatsetpoint:dualsetpoint', 'All', cooling_setpoint_temperature_schedule_name = 'ANN-ctrl')
    else:
        myidf.edit('Thermostatsetpoint:dualsetpoint', 'All', cooling_setpoint_temperature_schedule_name = 'Large Office ClgSetp')
    myidf.sensor_call(Air_System_Outdoor_Air_Mass_Flow_Rate = 'VAV_1',
                      Chiller_Electricity_Rate  = ['DOE REF 1980-2004 WATERCOOLED  CENTRIFUGAL CHILLER 0 1100TONS 0.7KW/TON'],
                      Site_Outdoor_Air_Drybulb_Temperature = ['Environment'],
                      Zone_Mean_Air_Temperature=['CORE_BOTTOM ZN'],
                      Cooling_Coil_Total_Cooling_Rate=['VAV_1 CLG COIL'],
                      Lights_Total_Heating_Rate=['CORE_BOTTOM ZN OFFICE WHOLEBUILDING - LG OFFICE LIGHTS'],
                      Zone_People_Sensible_Heating_Rate=['CORE_BOTTOM ZN'],
                      Schedule_Value = ['Large Office Bldg Occ'.upper(), 'Large Office Bldg Light'.upper(),
                                        'Large Office Bldg Equip'.upper()])
                        # to update: check why need to be all capital
    # myidf.actuator_call(Schedule_Value = [['Large Office Bldg Equip', 'Schedule:Year']])
    myidf.actuator_ctrl(Schedule_Value = [['ANN-ctrl', 'Schedule:Compact']])
    # To update: directly update to original files
    # myidf.delete_class('AvailabilityManager:Scheduled')
    # myidf.delete_class('AvailabilityManager:NightCycle')
    # myidf.delete_class('AvailabilityManagerAssignmentList')
    myidf.set_agent(q_network, input_var)
    # myidf.run()
    # myidf.save()
    # TO ADD: CHECK AGENT
    rl_env = dqn(myidf, input_var, q_network, target_network)
    rl_env.train()
    wandb.finish()
    # a = 1