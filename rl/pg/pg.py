from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, List
from rl.pg.pg_para import Args
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import tyro
from stable_baselines3.common.buffers import RolloutBuffer
from rl.util.onpolicyalgo import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
from env.env import buildinggym_env
import numpy as np
import wandb
from rl.pg.network import Agent

# SelfA2C = TypeVar("SelfA2C", bound="A2C")


class pg(OnPolicyAlgorithm):
    """
    Policy gradient

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

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
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param normalize_advantage: Whether to normalize or not the advantage
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
        policy: Union[str, Type[Agent]],
        env: buildinggym_env,
        args: Type[Args],
        run_name: str,
        my_callback = None,
        learning_rate: Union[float, Schedule] = 1e-4,
        n_steps: int = 5,
        batch_size: int = 64,
        gamma: float = 0.9,
        # gae_lambda: float = 1.0,
        ent_coef: float = 0,
        # vf_coef: float = 0.5,
        max_grad_norm: float = 10,
        max_train_perEp: int = 100,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        buffer_info: List[str] = ['observations', 'actions', 'rewards', 'logprobs'],
        # rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        # rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.args = tyro.cli(args)
        self.my_callback = my_callback
        self.sweep_config = self.args
        super().__init__(
            policy,
            env,
            learning_rate=self.args.learning_rate,
            # n_steps=self.args.n_steps,
            batch_size = args.batch_size,
            gamma=self.args.gamma,
            # gae_lambda=self.args.gae_lambda,
            ent_coef=self.args.ent_coef,
            # vf_coef=self.args.vf_coef,
            max_grad_norm=self.args.max_grad_norm,
            use_sde=self.args.use_sde,
            sde_sample_freq=self.args.sde_sample_freq,
            # rollout_buffer_class=rollout_buffer_class,
            # rollout_buffer_kwargs=rollout_buffer_kwargs,
            buffer_info = buffer_info,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage
        self.observation_var = env.observation_var
        self.max_train_perEp = max_train_perEp
        self.run_name = run_name
        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()

    def train(self, obs, actions, returns) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # self.policy.action_network.train()

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)


        # for rollout_data in self.rollout_buffer.get(batch_size=self.batch_size):

        # for rollout_data in self.rollout_buffer.get(batch_size=None):
            # if n_train >= max_train_perEp:
            #     break
        # rollout_data = self.rollout_buffer.get(batch_size=self.batch_size, shuffle=self.args.shuffle)
        actions = actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = actions.long().flatten()

        log_prob, entropy = self.policy.evaluate_actions(obs.float(), actions)
        # for name, param in self.policy.named_parameters():
        #     print(name, param.shape)            
        # values, log_prob, entropy = rollout_data.old_log_prob
        # values = values.flatten()

        # Normalize advantage (not present in the original implementation)
        advantages = returns
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient loss
        policy_loss = -(advantages * log_prob)

        # Value loss using the TD(gae_lambda) target
        # value_loss = F.mse_loss(rollout_data.returns, values)

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        loss = self.args.pol_coef * policy_loss + self.ent_coef * entropy_loss

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        # Check gradient
        # for name, param in self.policy.mlp_extractor.named_parameters():
        # for name, param in self.policy.features_extractor.policy_fe.named_parameters():
        # for name, param in self.policy.action_network.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.grad}")       
        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()


        # explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        # self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        # return policy_loss.item(), np.mean(self.rollout_buffer.logprobs[106])
        # self.my_callback.per_time_step(locals())

        return policy_loss.item()


    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 10,
        tb_log_name: str = "A2C",
        reset_num_timesteps: bool = False,
        progress_bar: bool = False,
        max_train_perEp: int = 100,
    ):
        _, performance =  super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            max_train_perEp=self.max_train_perEp
        )
        return performance
    
    def train_auto_fine_tune(self,
                             ):
        with wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                config=self.sweep_config,
                # name=self.run_name,
                save_code=True,
            ):
            self.args = wandb.config
            for k, v in tyro.cli(Args).__dict__.items():
                if k not in self.args:
                    self.args[str(k)] = v
            self.learn(self.args.total_epoch, self.my_callback, max_train_perEp = self.args.max_train_perEp)
