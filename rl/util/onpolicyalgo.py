import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from rl.util.base_class import BaseAlgorithm
# from rl.util.buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer
from rl.util.replaybuffer import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from env.env import buildinggym_env
import torch
import random
from rl.a2c.network import Agent
import os

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: ReplayBuffer
    # policy: Union[str, Type[ActorCriticPolicy], Agent]
    policy: Union[str, Type[Agent]]

    def __init__(
        self,
        policy: Union[str, Type[Agent]],
        # policy: Union[str, Type[ActorCriticPolicy], Agent],
        env: buildinggym_env,
        learning_rate: Union[float, Schedule],
        # n_steps: int,
        batch_size: int,
        gamma: float,
        # gae_lambda: float,
        ent_coef: float,
        # vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        buffer_info: List[str],
        # rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        # rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        args = None,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            batch_size = batch_size,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        # self.n_steps = n_steps
        self.gamma = gamma
        # self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        # self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        # self.rollout_buffer_class = rollout_buffer_class
        # self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        self.use_sde = use_sde
        self.buffer_info = buffer_info
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # if self.rollout_buffer_class is None:
        #     if isinstance(self.observation_space, spaces.Dict):
        #         self.rollout_buffer_class = DictRolloutBuffer
        #     else:
        #         self.rollout_buffer_class = RolloutBuffer

        # self.rollout_buffer = self.rollout_buffer_class(
        #     self.batch_size,
        #     self.observation_space,  # type: ignore[arg-type]
        #     self.action_space,
        #     device=self.device,
        #     gamma=self.gamma,
        #     gae_lambda=self.gae_lambda,
        #     n_envs=self.n_envs,
        #     **self.rollout_buffer_kwargs,
        # )

        self.rollout_buffer = ReplayBuffer(self.buffer_info)
        
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
    
    # def reset_buffer(self, new_batch_size):
    #     self.rollout_buffer = self.rollout_buffer_class(
    #         new_batch_size,
    #         self.observation_space,  # type: ignore[arg-type]
    #         self.action_space,
    #         device=self.device,
    #         gamma=self.gamma,
    #         gae_lambda=self.gae_lambda,
    #         n_envs=self.n_envs,
    #         **self.rollout_buffer_kwargs,
    #     )        

    def collect_rollouts(
        self,
        env: buildinggym_env,
        callback: BaseCallback,
        rollout_buffer: ReplayBuffer,
        n_rollout_steps: Union[int, None] = None,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        # assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        # rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # while n_steps < n_rollout_steps:
        while n_steps < 1:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            env.run()
            self._update_learning_rate(self.policy.optimizer, self.args.alpha)
            # env.normalize_input()
            # env.label_working_time()
            n_steps+=1
            # env.cal_r()
            # # env.cal_return()

            # self.data_wt = env.sensor_dic
            # # self.logprobs_wt = env.logprobs[np.where(env.sensor_dic['Working_time'])[0]]
            # self.logprobs_wt = env.logprobs
            # # self.values_wt = env.values[np.where(env.sensor_dic['Working_time'])[0]]
            # # self.values_wt = [env.values[i] for i in np.where(env.sensor_dic['Working_time'])[0]]
            # self.actions_wt = env.actions
            # self.wt_label = list(env.sensor_dic['Working_time'])
            # # rollout_buffer.reset(self.data_wt.shape[0])

            # assert self.batch_size<env.sensor_dic.shape[0], f'Batch size should samller than {self.data_wt.shape[0]}'

            # obs_nor = [env.observation_var[i] + '_nor' for i in range(len(env.observation_var))]
            # _obs = np.array(self.data_wt[obs_nor])
            # # _terminal_state = np.array(self.data_wt['Terminations'])
            # _rewards = np.array(self.data_wt['rewards'])
            # _actions = self.actions_wt
            # # _values = self.values_wt
            # _log_probs = self.logprobs_wt 
            performance = np.mean(env.sensor_dic['rewards'][np.where(env.sensor_dic['Working time'])[0]])
            if performance > 0.4:
                path_i = os.path.join('Archive results', self.run_name)
                if not os.path.exists(path_i):
                    os.mkdir(path_i) 
                env.sensor_dic.to_csv(os.path.join(path_i, 'results.csv'))
                torch.save(self.policy.state_dict(), os.path.join(path_i, 'model.pth'))
        #     training_data = [_obs, _actions, _rewards, _log_probs]
        #     # index = random.randint(0, _obs.shape[0]-self.batch_size-1)

        #     # _terminal_state = _terminal_state[index:index+self.batch_size]
        #     # _obs = _obs[index:index+self.batch_size,:]
        #     # _rewards = _rewards[index:index+self.batch_size]


        #     # with th.no_grad():
        #     #     # Convert to pytorch tensor or to TensorDict
        #     #     obs_tensor = obs_as_tensor(_obs, self.device)
        #     #     actions, values, log_probs = self.policy(obs_tensor.float())
        #     # actions = actions.cpu().numpy()

        #     # Rescale and perform action
        #     # clipped_actions = actions

        #     if isinstance(self.action_space, spaces.Box):
        #         if self.policy.squash_output:
        #             # Unscale the actions to match env bounds
        #             # if they were previously squashed (scaled in [-1, 1])
        #             clipped_actions = self.policy.unscale_action(clipped_actions)
        #         else:
        #             # Otherwise, clip the actions to avoid out of bound error
        #             # as we are sampling from an unbounded Gaussian distribution
        #             clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        #     # new_obs, rewards, dones, infos = env.step(clipped_actions)

        #     self.num_timesteps += env.num_envs

        #     # Give access to local variables
        #     callback.update_locals(locals())
        #     if not callback.on_step():
        #         return False, performance

        #     # self._update_info_buffer(infos, dones)
        #     n_steps += 1

        #     # if isinstance(self.action_space, spaces.Discrete):
        #     #     # Reshape in case of discrete action
        #     #     actions = actions.reshape(-1, 1)

        #     # Handle timeout by bootstraping with value function
        #     # see GitHub issue #633
        #     # for idx, done in enumerate(_terminal_state):
        #     #     if (
        #     #         done
        #     #         # and infos[idx].get("terminal_observation") is not None
        #     #         # and infos[idx].get("TimeLimit.truncated", False)
        #     #     ):
        #     #         terminal_obs = torch.tensor(_obs[idx]).to('cuda').unsqueeze(0).float()
        #     #         with th.no_grad():
        #     #             terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
        #     #         _rewards[idx] += self.gamma * terminal_value

        #     # for k in range(_obs.shape[0]):
        #     # rollout_buffer.add(
        #     #     _obs,  # type: ignore[arg-type]
        #     #     _actions,
        #     #     _rewards,
        #     #     # _terminal_state],  # type: ignore[arg-type]
        #     #     # # self._last_episode_starts,  # type: ignore[arg-type]
        #     #     # _values],
        #     #     _log_probs,
        #     # )
        #     rollout_buffer.add(training_data, self.wt_label)
        #     # a = rollout_buffer.get(12, 5)
        #     self._last_obs = _obs[-1]  # type: ignore[assignment]
        #     # self._last_episode_starts = _terminal_state[k]

        # # with th.no_grad():
        # #     # Compute value for the last timestep
        # #     last_values = self.policy.predict_values(torch.tensor(self._last_obs).to('cuda').unsqueeze(0).float())

        # # rollout_buffer.remove_tail(n_rollout_steps)
        # rollout_buffer.compute_returns(outlook_steps = 5, gamma = self.gamma)
        # self.data_wt = self.data_wt[0:rollout_buffer.buffer_size]
        # rollout_buffer.remove_redundancy(rollout_buffer.wt_label)
        callback.update_locals(locals())

        callback.on_rollout_end()

        return True, performance

    def train(self, max_train_perEp = np.inf) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        max_train_perEp: int = 10
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            self.args.total_epoch,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training, performance = self.collect_rollouts(self.env, callback, self.rollout_buffer)

            if not continue_training:
                break

            iteration += 1
            self.num_timesteps+=1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            # self.env.p_loss, self.env.prob = self.train(max_train_perEp)

        callback.on_training_end()

        return self, performance

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
