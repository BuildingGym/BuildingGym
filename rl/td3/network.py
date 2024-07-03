from typing import Any, Dict, List, Optional, Type, Union, Tuple

import torch as th
from gymnasium import spaces
from torch import nn

# from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.policies import BasePolicy, BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from rl.td3.td3_para import Args


class Actor(nn.Module):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        # features_dim: int,
        activation_fn: Type[nn.Module],
        features_extractor: List[int] =None,
        args: Args = None,
    ):
        super(Actor, self).__init__()
        # super().__init__(
        #     observation_space,
        #     action_space,
        #     features_extractor=features_extractor,
        #     normalize_images=normalize_images,
        #     squash_output=True,
        # )
        self.args = args
        self.net_arch = net_arch
        if features_extractor == None:
            self.features_dim = last_layer_dim = observation_space.shape[0]
            self.fe_net = None
        else:
            self.features_dim = features_extractor[-1]
            last_layer_dim = observation_space.shape[0]
            fe = []
            for current_layer_dim in features_extractor:
                fe.append(nn.Linear(last_layer_dim, current_layer_dim))
                # if need add activation layers
                last_layer_dim = current_layer_dim
            self.fe_net = nn.Sequential(*fe)
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        # action_dim = action_space.n
        # actor_net = []
        # last_layer_dim_pi = self.features_dim
        # for curr_layer_dim in net_arch:
        #     actor_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
        #     actor_net.append(activation_fn())
        #     last_layer_dim_pi = curr_layer_dim     
        # actor_net.append(nn.Linear(last_layer_dim_pi, 1))
        # actor_net.append(nn.Tanh())
        actor_net = create_mlp(self.features_dim, 1, net_arch, activation_fn, squash_output=True)
        # Deterministic action
        self.mu = nn.Sequential(*actor_net)
        a = 1

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        # features = self.extract_features(obs, self.features_extractor)
        if self.fe_net is not None:
            feature = self.fe_net(obs.to(th.float32))
        else:
            feature = obs.to(th.float32)
        action = self.mu(feature)
        dis = th.distributions.Normal(action, self.args.noise_std)
        if deterministic:
            return action
        else:
            action += dis.sample()
            action = action.clamp(-1,1)
            return action

    # def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
    #     # Note: the deterministic deterministic parameter is ignored in the case of TD3.
    #     #   Predictions are always deterministic.
    #     return self(observation)

    def set_training_mode(self, mode):
        self.train(mode)

class ContinuousCritic(nn.Module):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: Union[List[int], nn.Module] = None,
        # features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        args: Args = None,
    ):
        super(ContinuousCritic, self).__init__()
        # super().__init__(
        #     observation_space,
        #     action_space,
        #     features_extractor=features_extractor,
        #     normalize_images=normalize_images,
        # )
        if isinstance(features_extractor, List):
            features_dim = features_extractor[-1]
            last_layer_dim = observation_space.shape[0]
            fe = []
            for current_layer_dim in features_extractor:
                fe.append(nn.Linear(last_layer_dim, current_layer_dim))
                # if need add activation layers
                last_layer_dim = current_layer_dim
            self.fe_net = nn.Sequential(*fe)            
        elif isinstance(features_extractor, nn.Module):
            features_dim = features_extractor[-1].out_features
            self.fe_net = features_extractor
        else:
            features_dim = observation_space.shape[0]
            self.fe_net = None

        action_dim = get_action_dim(action_space)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net_list)
            q_net.float()
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor

        # dxl: remove feature extractor
        if self.fe_net is not None:
            with th.set_grad_enabled(not self.share_features_extractor):
                feature = self.fe_net(obs.to(th.float32))
        else:
            feature = obs.to(th.float32)

        qvalue_input = th.cat([feature, actions], dim=1).to(th.float32)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """

        # dxl: remove feature extractor
        if self.fe_net is not None:
            with th.no_grad():
                feature = self.fe_net(obs.to(th.float32))
        else:
            feature = obs.to(th.float32)
        return self.q_networks[0](th.cat([feature, actions.to(th.float32)], dim=1))
    
    def set_training_mode(self, mode):
        self.train(mode)    


class Agent(nn.Module):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Sigmoid,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = {},
        n_critics: int = 1,
        share_features_extractor: bool = True,
        args = None,
    ):
        super().__init__()
        # super().__init__(
        #     observation_space,
        #     action_space,
        #     features_extractor_class,
        #     features_extractor_kwargs,
        #     optimizer_class=optimizer_class,
        #     optimizer_kwargs=optimizer_kwargs,
        #     squash_output=True,
        #     normalize_images=normalize_images,
        # )

        # Default network architecture, from the original paper
        self.args = args
        self.optimizer_kwargs = optimizer_kwargs
        if net_arch is None:
            net_arch = [128, 32]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": observation_space,
            "action_space": action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "features_extractor": [64],
            "args": self.args,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)
        a = 1

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = Actor(**self.actor_kwargs)
        self.init_weight(self.actor)
        self.actor_target = Actor(**self.actor_kwargs)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.args.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic_kwargs.update(
                {'features_extractor': self.actor.fe_net}
            )
            self.critic = ContinuousCritic(**self.critic_kwargs)
            # Critic target should not share the features extractor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_kwargs.update(
                {'features_extractor': self.actor_target.fe_net}
            )            
            self.critic_target = ContinuousCritic(**self.critic_kwargs)
        else:
            # Create new features extractor for each network
            self.critic = ContinuousCritic(**self.critic_kwargs)
            self.critic_target = ContinuousCritic(**self.critic_kwargs)
        self.init_weight(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.args.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)
        # self.critic.float()
        # self.critic_target.float()

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def forward(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

    def init_weight(self, network):
        for m in network.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, mean=0, std = 0.1)
                # nn.init.xavier_normal_(m.weight, gain=1)
                # nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)            