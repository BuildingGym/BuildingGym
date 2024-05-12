from env.env import buildinggym_env
import gymnasium as _gymnasium_
from energyplus.ooep.addons.rl import (
    VariableBox,
    SimulatorEnv,
)
from energyplus.ooep import (
    Actuator,
    OutputVariable,
)
from rl.a2c.a2c_policy import ActorCriticPolicy
from rl.util.schedule import ConstantSchedule
from gymnasium.spaces import (
    Box,
    Discrete
)
import numpy as _numpy_
import numpy as np
from rl.ppo.ppo_para import Args
from rl.a2c.a2c import A2C

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
schedule = ConstantSchedule(0.1)
input_sp = Box(np.array([-1] * 5), np.array([1] * 5))
action_sp = Discrete(5)
agent = ActorCriticPolicy(input_sp, action_sp, schedule.value)
env = buildinggym_env('Large office - 1AV232.idf',
                    'USA_FL_Miami.722020_TMY2.epw',
                    observation_space,
                    action_space,
                    agent,
                    Args)


a = A2C(ActorCriticPolicy, env)
a.learn(100)
dxl = 'success'
