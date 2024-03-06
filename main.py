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
            myidf.save()
            self.cal_r()
            if len(self.obs_index) == 0:
                sensor_name_list = list(myidf.sensor_dic.columns)
                for i in self.input_var:
                    assert i in sensor_name_list, "The input variable is not in the sensor list, please add it"
                    self.obs_index.append(sensor_name_list.index(i))
            for i in range(myidf.sensor_dic.shape[0]-1):
                obs = myidf.sensor_dic.iloc[i,self.obs_index]
                next_obs = myidf.sensor_dic.iloc[i+1,self.obs_index]
                rewards = myidf.sensor_dic['reward'][i]
                actions = myidf.action_dic.iloc[i, 0]
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

                    if global_step % 100 == 0:
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


    def cal_r(self):
        # Realtime reward function
        myidf.sensor_dic['reward'] = myidf.sensor_dic['Cooling Coil Total Cooling Rate@BCA AHU COOLING COIL']/3000
        # Return return function (future accmulated reward)
        R_list = []
        for i in range(myidf.sensor_dic.shape[0] - args.outlook_step):
            reward_list = myidf.sensor_dic['reward'][i:i+5]
            R = 0
            for r in reward_list[::-1]:
                R = r + R * args.gamma
            R_list.append(R)
        
        # Remove data without enough outlook steps
        myidf.sensor_dic = myidf.sensor_dic[:-args.outlook_step]
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
        com = [24 + actions]
        # com = [22]
        return com, [actions]
        
if __name__ == '__main__':
    idf_file = 'Main-PV-v4_ForTrain.idf'
    epw_file = 'SGP_Singapore.486980_IWEC.epw'
    output_path = 'test\\'
    epjson = 'C:\\EnergyPlusV9-4-0\\Energy+.schema.epJSON'
    args = tyro.cli(Args)
    q_network = QNetwork(args.input_dim, args.output_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    target_network = QNetwork(args.input_dim, args.output_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    input_var = ['Site Outdoor Air Drybulb Temperature@Environment',
                  'Zone Mean Air Temperature@BLOCK3:ZONE1',
                  'Zone Other Equipment Total Heating Rate@BLOCK3:ZONE1']
    with open(epjson, 'r') as f:
        data = json.load(f)
    myidf = ep_simu(idf_file, epw_file, output_path, '2018-02-03', '2018-03-05', 2, True, True, 5)
    myidf.sensor_call(Air_System_Outdoor_Air_Mass_Flow_Rate = 'BCA',
                      Site_Outdoor_Air_Drybulb_Temperature = ['Environment'],
                      Zone_Mean_Air_Temperature=['BLOCK3:ZONE1'],
                      Cooling_Coil_Total_Cooling_Rate=['BCA AHU COOLING COIL'],
                      Zone_Other_Equipment_Total_Heating_Rate=['BLOCK3:ZONE1'],
                      Lights_Total_Heating_Rate=['BLOCK3:ZONE1 GENERAL LIGHTING'],
                      Zone_People_Sensible_Heating_Rate=['BLOCK3:ZONE1'],
                      Other_Equipment_Total_Heating_Rate = ['BLOCK3:ZONE1 EQUIPMENT GAIN 1'])
    myidf.actuator_call(Schedule_Value = [['ALWAYS 24', 'Schedule:Compact']])
    # to update: check why need to be all capital
    myidf.set_agent(q_network, input_var)
    rl_env = dqn(myidf, input_var, q_network, target_network)
    rl_env.train()
    a = 1