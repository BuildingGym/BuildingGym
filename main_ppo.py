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
from rl.ppo.ppo_para import Args
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



class ppo():
    def __init__(self, observation_var, action_var, auto_fine_tune = False, sweep_config = {}) -> None:
        self.observation_var = observation_var
        self.action_var = action_var
        self.sweep_config = sweep_config
        self.auto_fine_tune = auto_fine_tune
        self.args = tyro.cli(Args)
        self.run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])))
        
        self.agent = self.set_network(self.args.input_dim, self.args.output_dim)
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
            self.epsilon = self.linear_schedule(self.args.start_e, self.args.end_e,
                                      self.args.exploration_fraction * self.args.total_timesteps,
                                      global_step)
            self.run()
            self.normalize_input()
            self.label_working_time()
            self.cal_r()
            self.cal_return()
            # self.normalize_input()
            Performance = np.mean(self.sensor_dic['results'][self.sensor_dic['Working_time'] == True])
            if  Performance>0.88:
                path_i = os.path.join('Archive results', str(int(time.time())))
                os.mkdir(path_i)
                self.sensor_dic.to_csv(os.path.join(path_i, 'results.csv'))
                torch.save(self.agent.state_dict(), os.path.join(path_i, 'model.pth'))
            buffer = {
                'obs': self.sensor_dic[self.observation_var].iloc[np.where(self.sensor_dic['Working_time']==True)[0]],
                'next_obs':self.sensor_dic[self.observation_var].iloc[np.where(self.sensor_dic['Working_time']==True)[0]+1],
                'actions': self.sensor_dic[self.action_var].iloc[np.where(self.sensor_dic['Working_time']==True)[0]],
                'rewards': self.sensor_dic['rewards'].iloc[np.where(self.sensor_dic['Working_time']==True)[0]],
                'advantages':  self.sensor_dic['advantages'].iloc[np.where(self.sensor_dic['Working_time']==True)[0]],
                'logprobs': self.sensor_dic['logprobs'].iloc[np.where(self.sensor_dic['Working_time']==True)[0]],
                'values': self.sensor_dic['values'].iloc[np.where(self.sensor_dic['Working_time']==True)[0]],
                'returns': self.sensor_dic['returns'].iloc[np.where(self.sensor_dic['Working_time']==True)[0]],
                'Working_time': self.sensor_dic['Working_time'].iloc[np.where(self.sensor_dic['Working_time']==True)[0]],
                'Terminations':  self.sensor_dic['Terminations'].iloc[np.where(self.sensor_dic['Working_time']==True)[0]]
                }
            # for i in range(self.sensor_dic.shape[0]-1):
            #     obs_i = self.sensor_dic[self.observation_var].iloc[i]
            #     next_obs_i = self.sensor_dic[self.observation_var].iloc[i+1]
            #     actions_i = self.sensor_dic[self.action_var].iloc[i]
            #     rewards_i = self.sensor_dic['rewards'].iloc[i]
            #     advantages_i = self.sensor_dic['advantages'].iloc[i]
            #     logprob_i = self.sensor_dic['logprobs'].iloc[i]
            #     value_i = self.sensor_dic['values'].iloc[i]
            #     return_i = self.sensor_dic['returns'].iloc[i]
            #     if self.sensor_dic['Working_time'].iloc[i]:
            #         # To Do: too slow
            #         rb.add(np.array(obs_i),
            #             np.array(next_obs_i),
            #             np.array(actions_i),
            #             np.array(rewards_i),
            #             np.array([False]),
            #                 {'advantages': advantages_i, 'logprobs':logprob_i, 'values': value_i, 'returns': return_i})
            batch_size = int(np.sum(self.sensor_dic['Working_time'])/self.args.minibatch_size) * self.args.minibatch_size
            # n_minibatch = int(self.sensor_dic.shape[0]/self.args.minibatch_size)
            b_inds = np.arange(batch_size)
            clipfracs = []
            if global_step >= self.args.learning_starts:
                for k in range(self.args.update_epochs):
                    if global_step % self.args.train_frequency == 0:
                        np.random.shuffle(b_inds)
                        minibatch_size = self.args.minibatch_size
                        # for start in range(0, batch_size, minibatch_size):
                        for start in range(0, 1):
                            end = start + minibatch_size
                            mb_inds = b_inds[start:end]
                            b_obs = torch.tensor(np.array(buffer['obs'])).to(self.device)
                            b_actions = torch.tensor(np.array(buffer['actions'])).to(self.device)
                            b_logprobs = torch.tensor(np.array(buffer['logprobs'])).to(self.device)
                            b_advantages = torch.tensor(np.array(buffer['advantages'])).to(self.device)
                            b_returns = torch.tensor(np.array(buffer['returns'])).to(self.device)
                            b_values = torch.tensor(np.array(buffer['values'])).to(self.device)
                            _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds].float(), b_actions.float()[mb_inds])
                            logratio = newlogprob - b_logprobs[mb_inds]
                            ratio = logratio.exp()

                            with torch.no_grad():
                                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                            mb_advantages = b_advantages[mb_inds]
                            if self.args.norm_adv:
                                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                            # Policy loss
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            # Value loss
                            newvalue = newvalue.view(-1)
                            if self.args.clip_vloss:
                                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                                v_clipped = b_values[mb_inds] + torch.clamp(
                                    newvalue - b_values[mb_inds],
                                    -self.args.clip_coef,
                                    self.args.clip_coef,
                                )
                                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                                v_loss = 0.5 * v_loss_max.mean()
                            else:
                                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                            entropy_loss = entropy.mean()
                            loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                            self.optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                            self.optimizer.step()

                        if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                            break
                    # TRY NOT TO MODIFY: record rewards for plotting purposes
                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y      
                if self.args.track:
                    if not self.auto_fine_tune:
                        wandb.init(
                            project=self.args.wandb_project_name,
                            entity=self.args.wandb_entity,
                            sync_tensorboard=True,
                            config=self.args,
                            name=self.run_name,
                            save_code=True,
                        )
                    wandb.log({'reward_curve': np.mean(self.sensor_dic['rewards'][self.sensor_dic['Working_time'] == True])}, step=global_step)        
                    wandb.log({'result_curve': Performance}, step=global_step)        
                    wandb.log({'loss_curve': float(loss.cpu().detach().numpy())}, step=global_step)                                                     
                self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
                self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        self.writer.close()                         


    def run(self):
        self.sensor_index = 0
        asyncio.run(energyplus_running('Large office - 1AV232.idf', 'USA_FL_Miami.722020_TMY2.epw'))


    def call_track(self):
        pass

    def normalize_input(self):
        nor_min = np.array([22.8, 22, 0, 0, 0])
        nor_max = np.array([33.3, 27, 1, 1, 1])
        nor_input = (self.sensor_dic[self.observation_var] - nor_min)/(nor_max - nor_min)
        self.sensor_dic[self.observation_var] = nor_input     

    def normalize_input_i(self, state):
        nor_min = np.array([22.8, 22, 0, 0, 0])
        nor_max = np.array([33.3, 27, 1, 1, 1])
        return (state- nor_min)/(nor_max - nor_min)
            
            
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
            state = self.normalize_input_i(state)
            state = torch.Tensor(state).cuda() if torch.cuda.is_available() and self.args.cuda else torch.Tensor(state).cpu()
            with torch.no_grad():
                actions, logprob, _, value = self.agent.get_action_and_value(state)
                # actions = torch.argmax(q_values, dim=0).cpu().numpy()
            com = 23. + actions

            act = thinenv.act({'Thermostat': com})

            obs = pd.DataFrame(obs, index = [self.sensor_index])
            obs.insert(0, 'Time', t)
            obs.insert(obs.columns.get_loc("t_in") + 1, 'Thermostat', actions.cpu().numpy())
            obs.insert(obs.columns.get_loc("t_in") + 1, 'logprobs', logprob.cpu().numpy())
            obs.insert(obs.columns.get_loc("t_in") + 1, 'values', value.flatten().cpu().numpy())
            if self.sensor_index == 0:
                self.sensor_dic = pd.DataFrame({})
                self.sensor_dic = obs
            else:
                self.sensor_dic = pd.concat([self.sensor_dic, obs])            
            self.sensor_index+=1

    def set_network(self, input_dim, output_dim):
        agent = Agent(input_dim, output_dim).to(self.device)
        return agent
    
    def set_optimizer(self):
        optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate)
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
        
      

if __name__ == '__main__':
    default_paras = tyro.cli(Args)
    parameters_dict = {
    'learning_rate': {
        'values': [1e-2]
        },
    'minibatch_size': {
          'values': [32, 64, 128]
        },
    'update_epochs': {
          'values': [1]
        },     
    # 'start_e': {
    #       'values': [0.8, 0.5, 0.2]
    #     },     
    # 'train_frequency': {
    #       'values': [1, 5, 10]
    #     },   
    # 'target_network_frequency': {
    #       'values': [5, 20, 30]
    #     },                                
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

    sweep_id = wandb.sweep(sweep_config, project="energygym-ppo-auto")
    observation_var = ['t_out', 't_in', 'occ', 'light', 'Equip']
    action_var = ['Thermostat']
    a = ppo(observation_var, action_var, False, sweep_config)
    # wandb.agent(sweep_id, a.train_auto_fine_tune, count=6) 
    a.train()
