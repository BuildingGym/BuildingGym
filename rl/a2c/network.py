import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from stable_baselines3.common.policies import BasePolicy
import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from gymnasium.spaces import (
    Box,
    Discrete
)
import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from rl.util.build_network import MlpBuild
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from rl.util.schedule import ConstantSchedule
import torch

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, 
                observation_space: spaces.Space,
                action_space: spaces.Space,
                lr_schedule: Schedule,
                net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                activation_fn: Type[nn.Module] = nn.Tanh,
                extract_features_bool: bool = False,
                share_features_extractor: bool = False,
                device: Union[str, torch.device] = 'cuda',
                optimizer_kwargs: Optional[Dict[str, Any]] = None,
                use_sde: bool = False,
                ortho_init: bool = False
                ):
        super().__init__()
        self.activation_fn = activation_fn
        self.observation_space = observation_space
        self.action_space = action_space
        self.extract_features_bool = extract_features_bool
        self.share_features_extractor = share_features_extractor
        self.device = device
        self.ortho_init = ortho_init
        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]
        self.action_dist = CategoricalDistribution(self.action_space.n)
        
        
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        if optimizer_kwargs is None:
            self.optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                self.optimizer_kwargs["eps"] = 1e-5
        self.lr_schedule = lr_schedule
        # Default network architecture, from stable-baselines
        if net_arch is None:
                net_arch = dict(pi=[64, 64], vf=[64, 64])
        self.net_arch = net_arch

        self.features_extractor = None
        self.mlp_extractor = MlpBuild(
            self.observation_space.shape[0],
            self.action_space.n,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

        self.action_network = nn.Sequential(nn.Linear(self.mlp_extractor.latent_dim_pi, self.action_space.n),
                                             nn.Softmax(dim=-1),
                                             ).to(self.device)
        self.value_network = nn.Linear(self.mlp_extractor.latent_dim_vf, 1, device=self.device)
        self._build(lr_schedule)


    def set_training_mode(self, mode = True):
        if mode:
            self.train()
        else:
            self.eval()

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        if self.extract_features_bool:
            features = self.extract_features(obs)
        else:
            features = obs
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features=vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_network(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        # actions = actions.reshape((-1, *self.action_space.n)) 
        return actions, values, log_prob
    

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_network(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        if self.extract_features_bool:
            features = self.extract_features(obs)
        else:
            features = obs
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features=vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_network(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        if self.extract_features_bool:
            features = super().extract_features(obs, self.pi_features_extractor)
        else:
            features = obs
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        if self.extract_features_bool:
            features = super().extract_features(obs, self.vf_features_extractor)
        else:
            features = obs
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_network(latent_vf)           
    
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        # latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # if isinstance(self.action_dist, DiagGaussianDistribution):
        #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        #         latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        #     )
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        #         latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
        #     )
        # elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
        #     self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        # else:
        #     raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        # self.action_net.to(self.policydevice)
        # self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1, device=self.policydevice)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_network: 0.1,
                self.value_network: 10,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                if self.extract_features_bool:
                    del module_gains[self.features_extractor]
                    module_gains[self.pi_features_extractor] = np.sqrt(2)
                    module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]    
