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
    def __init__(self, idf_env) -> None:
        self.idf_env = idf_env
    
    def train(self):
        for global_step in range(args.total_timesteps):
            epsilon = linear_schedule(args.start_e, args.end_e,
                                      args.exploration_fraction * args.total_timesteps,
                                      global_step)
            myidf.run(epsilon = epsilon)

    
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
        actions = [24 + actions]
        return actions
        
if __name__ == '__main__':
    idf_file = 'Main-PV-v4_ForTrain.idf'
    epw_file = 'USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw'
    output_path = 'test\\'
    epjson = 'C:\\EnergyPlusV9-4-0\\Energy+.schema.epJSON'
    args = tyro.cli(Args)
    q_network = QNetwork(args.input_dim, args.output_dim)
    input_var = ['Site Outdoor Air Drybulb Temperature@Environment',
                  'Site Outdoor Air Drybulb Temperature@Environment',
                  'Site Outdoor Air Drybulb Temperature@Environment']
    with open(epjson, 'r') as f:
        data = json.load(f)
    myidf = ep_simu(idf_file, epw_file, output_path, '2018-02-03', '2018-03-05', 2, True, True, 5)
    myidf.sensor_call(Air_System_Outdoor_Air_Mass_Flow_Rate = 'BCA',
                      Site_Outdoor_Air_Drybulb_Temperature = ['Environment'],
                      Other_Equipment_Total_Heating_Rate = ['BLOCK1:ZONE1 EQUIPMENT GAIN 1', 'BLOCK2:ZONE1 EQUIPMENT GAIN 1'])
    myidf.actuator_call(Schedule_Value = [['ALWAYS 24', 'Schedule:Compact']])
    # to update: check why need to be all capital
    myidf.set_agent(q_network, input_var)
    rl_env = dqn(myidf)
    rl_env.train()
    myidf.save()
    a = 1