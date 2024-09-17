import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from rl.util.offpolicyalgo import OffPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

from stable_baselines3.common.utils import explained_variance, get_schedule_fn

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class DQN(OffPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        args = None,
        run_name = None,
        n_steps: int = 2048,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,        
        max_train_perEp: int = 1,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.args = args
        super().__init__(
            policy,
            env,
            learning_rate=args.learning_rate,
            # n_steps=n_steps,
            gamma=args.gamma,
            # gae_lambda=gae_lambda,
            # ent_coef=ent_coef,
            # vf_coef=vf_coef,
            # max_grad_norm=max_grad_norm,
            use_sde=args.use_sde,
            sde_sample_freq=args.sde_sample_freq,
            # rollout_buffer_class=rollout_buffer_class,
            # rollout_buffer_kwargs=rollout_buffer_kwargs,
            # stats_window_size=args.stats_window_size,
            # tensorboard_log=tensorboard_log,
            # policy_kwargs=args.policy_kwargs,
            # verbose=args.verbose,
            device=args.device,
            seed=args.seed,
            args = args,
            # _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
            ),
        )
        self.run_name = run_name
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        # if normalize_advantage:
        #     assert (
        #         batch_size > 1
        #     ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        # if self.env is not None:
        #     # Check that `n_steps * n_envs > 1` to avoid NaN
        #     # when doing advantage normalization
        #     buffer_size = self.env.num_envs * self.n_steps
        #     assert buffer_size > 1 or (
        #         not normalize_advantage
        #     ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
        #     # Check that the rollout buffer size is a multiple of the mini-batch size
        #     untruncated_batches = buffer_size // batch_size
        #     if buffer_size % batch_size > 0:
        #         warnings.warn(
        #             f"You have specified a mini-batch size of {batch_size},"
        #             f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
        #             f" after every {untruncated_batches} untruncated mini-batches,"
        #             f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
        #             f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
        #             f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
        #         )
        self.batch_size = args.batch_size
        # self.n_epochs = args.n_epochs
        # self.clip_range = args.clip_range
        # self.clip_range_vf = args.clip_range_vf
        # self.normalize_advantage = args.normalize_advantage
        # self.target_kl = target_kl

        self.policy_delay = args.policy_delay
        self.target_noise_clip = args.target_noise_clip
        self.target_policy_noise = args.target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # # Initialize schedules for policy/value clipping
        # self.clip_range = get_schedule_fn(self.clip_range)
        # if self.clip_range_vf is not None:
        #     if isinstance(self.clip_range_vf, (float, int)):
        #         assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

        #     self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        # self.policy.actor_batch_norm_stats = get_parameters_by_name(self.policy.actor, ["running_"])
        self.policy.q_network_batch_norm_stats = get_parameters_by_name(self.policy.q_network, ["running_"])
        # self.policy.actor_batch_norm_stats_target = get_parameters_by_name(self.policy.actor_target, ["running_"])
        self.policy.q_network_batch_norm_stats_target = get_parameters_by_name(self.policy.q_network_target, ["running_"])            

    def train(self) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        # self._update_learning_rate([self.policy.actor.optimizer, self.policy.critic.optimizer])

        losses = []
        for _ in range(self.args.gradient_steps):
            self._n_updates += 1
            # if self.policy_delay>self.args.gradient_steps:
            #     self.policy_delay = self.args.gradient_steps
            # Sample replay buffer
            # replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            data = self.env.buffer.get_sample(self.args.batch_size, shuffle = True)  # type: ignore[union-attr]
            obs = data[0]
            actions = data[1]
            rewards = data[2]
            nxt_obs = data[3]
            dones = 0

            with th.no_grad():
                target_max, _ = self.policy.q_network_target(nxt_obs).max(dim=1)
                td_target = rewards.flatten() + self.args.gamma * target_max
            old_val = self.policy.q_network(obs).gather(1, actions.to(dtype = th.long)).squeeze()
            old_val = old_val.to(dtype = th.float64)
            loss = F.mse_loss(td_target, old_val)
            self.policy.q_network.optimizer.zero_grad()
            loss.backward()
            self.policy.q_network.optimizer.step()
            losses.append(loss.item())

            # with th.no_grad():
            #     # Select action according to policy and add clipped noise
            #     noise = actions.clone().data.normal_(0, self.target_policy_noise)
            #     noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            #     next_actions = (self.policy.actor_target(nxt_obs) + noise).clamp(-1, 1)

            #     # Compute the next Q-values: min over all critics targets
            #     a = self.policy.critic_target(nxt_obs, next_actions)
            #     next_q_values = th.cat(a, dim=1)
            #     next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            #     target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # # Get current Q-values estimates for each critic network
            # current_q_values = self.policy.critic(obs, actions)

            # # Compute critic loss
            # critic_loss = sum(F.mse_loss(current_q.to(th.float64), target_q_values) for current_q in current_q_values)
            # assert isinstance(critic_loss, th.Tensor)
            # critic_losses.append(critic_loss.item())

            # # Optimize the critics
            # self.policy.critic.optimizer.zero_grad()
            # critic_loss.backward()
            # self.policy.critic.optimizer.step()


            # for name, param in self.policy.actor.named_parameters():
            #     if param.requires_grad:
            #         print(f"{name}: {param.grad}")       
            # Delayed policy updates


            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                # actor_loss = -self.policy.critic.q1_forward(obs, self.policy.actor(obs)).mean()
                # obs.requires_grad = True
                # actions.requires_grad = True
                # actor_loss = -self.policy.critic.q1_forward(obs, self.policy.actor(obs, deterministic = True)).mean()
                # actor_losses.append(actor_loss.item())

                # # Optimize the actor
                # self.policy.actor.optimizer.zero_grad()
                # actor_loss.backward()
                # self.policy.actor.optimizer.step()

                polyak_update(self.policy.q_network.parameters(), self.policy.q_network_target.parameters(), self.tau)
                # polyak_update(self.policy.actor.parameters(), self.policy.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.policy.q_network_batch_norm_stats, self.policy.q_network_batch_norm_stats_target, 1.0)
                # polyak_update(self.policy.actor_batch_norm_stats, self.policy.actor_batch_norm_stats_target, 1.0)

        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # if len(actor_losses) > 0:
        #     actor_losses = np.mean(actor_losses)
        #     self.logger.record("train/actor_loss", actor_losses)
        # else:
        #     actor_losses=float('NaN')
        self.logger.record("train/loss", np.mean(losses))
        return np.mean(losses), _

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        _, performance = super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
        return _, performance
    
