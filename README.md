# BuildingGym
![pypi](https://pypi-camo.freetls.fastly.net/49eddcb8b6dd234c39f1459da9dcc601043be0a7/68747470733a2f2f696d672e736869656c64732e696f2f707970692f762f77616e6462)  ![conda](https://camo.githubusercontent.com/ef8ab0467fad6b69d198eddeae5e64463478a8e901683abfebfbf583ca8ba3b5/68747470733a2f2f696d672e736869656c64732e696f2f636f6e64612f766e2f636f6e64612d666f7267652f77616e6462)

<img src="docs\README_images\pytorch_logo.png" width="20%" >


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
## Introduction
BuildingGym is a project that provides an API to easily train reinforcement learning control algorithm for all EnergyPlus envionment, and includes implementations of common reinforcement learning algorithm: Policy gradient, DQN, A2C, A3C, and more. Below is the structure for ```BuildingGym```
<p align="center">
<img src="docs\README_images\Package structure.png" width="50%" >
</p>
## Features
- 😏 Applied to all user-defined Energyplus model
- ❤️ Easy implement for common RL algorithms
- 💥 Included commone RL algorithm
- 😆 Auto-select the best model
- 😋 Track and visualize all the training process
- 😃 Applied to common control problem, e.g. demand respond, energy saving etc.

## Preparation
Install [Energyplus](https://energyplus.net/), [Pytorch](https://pytorch.org/), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), [Wandb](https://wandb.ai/site)
## Quickstart guide
You can utilize the quickstart guide to train your reinforcement learning control algorithm tailored to address the specific problem you aim to solve. Simply execute the necessary steps for a general implementation.

Or you can run the ```main.py``` to test the demo usage of this package
### Import the package
```
from epluspy import idf_editor
from epluspy import idf_simu
```
### Read idf
Read the Energyplus model (```.idf file```)
```
idf_file = 'Large office - 1AV940.idf'
epw_file = 'USA_FL_Miami.722020_TMY2.epw'
output_path = 'test\\'
myidf = ep_simu(idf_file, epw_file, output_path, '2018-08-01', '2018-08-31', 6, True, True, 5)
```
### Define the control algorithm
Hear we showcase the double deep Q learning algorithm.

```
# Define the RL algorithm as double deep Q learning algorithm
rl_env = dqn(myidf)
```
If you want to custonmize the control agent (optional), see the example here
```
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
```

### Define the sensor and acuator in Energyplus environment
For the avaliable sensor and actuator, please check the list '''test\dry_run\rdd.csv''' and '''test\dry_run\edd.csv'''. See the example below to define them.
Define the sensor:
```
myidf.sensor_call(Air_System_Outdoor_Air_Mass_Flow_Rate = 'VAV_1',
                    Chiller_Electricity_Rate  = ['DOE REF 1980-2004 WATERCOOLED  CENTRIFUGAL CHILLER 0 1100TONS 0.7KW/TON'],
                    Site_Outdoor_Air_Drybulb_Temperature = ['Environment'],
                    Zone_Mean_Air_Temperature=['CORE_BOTTOM ZN'],
                    Cooling_Coil_Total_Cooling_Rate=['VAV_1 CLG COIL'],
                    Lights_Total_Heating_Rate=['CORE_BOTTOM ZN OFFICE WHOLEBUILDING - LG OFFICE LIGHTS'],
                    Zone_People_Sensible_Heating_Rate=['CORE_BOTTOM ZN'])
```
Define the actuator (The control variable in RL):
```
myidf.actuator_call(Schedule_Value = [['ANN-ctrl', 'Schedule:Compact']])
```
### Define the control problem and reward function
Define input variables for control algorithm. The format is "Sensor Type@Sensor Name". Please note that you mush claim the sensor using ```sensor_call()```  before define the input variables. See the example below:
```
input_var = ['Site Outdoor Air Drybulb Temperature@Environment',
                'Zone Mean Air Temperature@CORE_BOTTOM ZN',
                'Zone People Sensible Heating Rate@CORE_BOTTOM ZN']
```

### Define the reward function
The reward function is defined according to the control target. During training process, the RL algorithm tries to increase the rewards. So higher reward means better control performance. This step is critical to achieve good control performance. See the example belwo to define the reward function.
```
def cal_r(self):
    ```
    reward: float, realtime reward value.
    R: float, return reward.
    ```
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
```

### Run the training
After the above steps, you can finally start to train the RL control algorithm!
```
rl_env.train()
```

## Methods in IDF class
