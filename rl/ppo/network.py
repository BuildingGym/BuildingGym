import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
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
import os
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
from rl.util.FeatureExt_network import FEBuild
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
                args:None,
                net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                Fe_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                optimizer_class: Type[th.optim.Optimizer] = th.optim.SGD,
                activation_fn: Type[nn.Module] = nn.Tanh,
                extract_features_bool: bool = True,
                share_features_extractor: bool = True,
                # device: Union[str, torch.device] = 'cuda',
                optimizer_kwargs: Optional[Dict[str, Any]] = None,
                use_sde: bool = False,
                ortho_init: bool = False,
                xa_init_gain: float = 0.5,
                ):
        super(Agent, self).__init__()
        self.activation_fn = activation_fn
        self.observation_space = observation_space
        self.action_space = action_space
        self.extract_features_bool = extract_features_bool
        self.args = args
        self.device = self.args.device
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
                net_arch = dict(pi=[32], vf=[32])
        self.net_arch = net_arch

        if Fe_arch is None:
                Fe_arch = [64, 32]
        self.features_extractor = FEBuild(
            self.observation_space.shape[0],
            Fe_arch = Fe_arch,
            share_features_extractor = share_features_extractor,
            # activation_fn=self.activation_fn,
            device=self.device,
        )
        self.mlp_extractor = MlpBuild(
            self.features_extractor.latent_dim_pi,
            self.features_extractor.latent_dim_vf,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

        if type(self.action_space) == Discrete:
            self.dist_type = 'categorical'
            self.action_network = nn.Sequential(
                                        nn.Linear(self.mlp_extractor.latent_dim_pi, self.action_space.n),
                                        nn.Softmax(dim =-1),
                                        ).to(self.device)            
        elif type(self.action_space) == Box:
            self.dist_type = 'normal'
            self.action_network_mu = nn.Sequential(
                                        nn.Linear(self.mlp_extractor.latent_dim_pi, 1),
                                        nn.Sigmoid(),
                                        ).to(self.device)       
            self.action_network_logstd = nn.Sequential(
                                        nn.Linear(self.mlp_extractor.latent_dim_pi, 1),
                                        nn.Sigmoid(),
                                        ).to(self.device)                           
        # self.action_dist = CategoricalDistribution(self.action_space.n)
        self.value_network = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
      
        # self.xa_init_gain = xa_init_gain
        if self.dist_type == 'categorical':
            self.init_weight(self.action_network)
        if self.dist_type == 'normal':
            self.init_weight(self.action_network_mu)
            self.init_weight(self.action_network_logstd)
        self.init_weight(self.mlp_extractor.policy_net)
        self.init_weight(self.mlp_extractor.value_net)
        self.init_weight(self.value_network)
        
        self._build(lr_schedule)
        self._load_pre_train(None)

    def _load_pre_train(self, pre_model: os.PathLike = None):
        if pre_model is not None:
            self.load_state_dict(torch.load(pre_model))


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
            # pi_features = self.features_extractor.extract_features(obs)
            # if self.share_features_extractor:
            pi_features, vf_features = self.features_extractor.extract_features(obs)
        else:
            pi_features=vf_features = obs            
        # else:
        #     pi_features = obs
        # if self.share_features_extractor:
        #     latent_pi, latent_vf = self.features_extractor(features)
        # else:
        #     pi_features=vf_features = features
        latent_pi = self.mlp_extractor.policy_net(pi_features)
        # Evaluate the values for the given observations
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions)
        # actions = actions.reshape((-1, *self.action_space.n)) 

        latent_vf = self.mlp_extractor.value_net(vf_features)
        value = self.value_network(latent_vf)

        return actions, value, log_prob
    

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """


        if self.dist_type == 'categorical':
            mean_actions = self.action_network(latent_pi)
            return Categorical(mean_actions)
        if self.dist_type == 'normal':
            mu = self.action_network_mu(latent_pi)
            std = self.action_network_logstd(latent_pi)      
            return Normal(mu.squeeze(), std.squeeze()*0.3)
        # if isinstance(self.action_dist, DiagGaussianDistribution):
        #     return self.action_dist.proba_distribution(mean_actions, self.log_std)
        # elif isinstance(self.action_dist, CategoricalDistribution):
        #     # Here mean_actions are the logits before the softmax
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, MultiCategoricalDistribution):
        #     # Here mean_actions are the flattened logits
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, BernoulliDistribution):
        #     # Here mean_actions are the logits (before rounding to get the binary actions)
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        # else:
        #     raise ValueError("Invalid action distribution")

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
            # pi_features = self.features_extractor.extract_features(obs)
            # if self.share_features_extractor:
            pi_features, vf_features = self.features_extractor.extract_features(obs)
        else:
            pi_features=vf_features = obs            
        latent_pi = self.mlp_extractor.policy_net(pi_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        # actions = distribution.sample()
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        latent_vf = self.mlp_extractor.value_net(vf_features)
        value = self.value_network(latent_vf)

        return value, log_prob, entropy
    
    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        if self.extract_features_bool:
            pi_features, _ = self.features_extractor.extract_features(obs)
        else:
            pi_features, _ = obs
        latent_pi = self.mlp_extractor(pi_features)
        return self._get_action_dist_from_latent(latent_pi)
    
    def predict_values(self, obs: PyTorchObs) -> th.Tensor:

        if self.extract_features_bool:
            # pi_features = self.features_extractor.extract_features(obs)
            # if self.share_features_extractor:
            _, vf_features = self.features_extractor.extract_features(obs)
        else:
            vf_features = obs            

        latent_vf = self.mlp_extractor.value_net(vf_features)
        value = self.value_network(latent_vf)
        return value  

      
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
            }
            
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]    
        # self.optimizer =torch.optim.SGD(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]    

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=self.xa_init_gain, std=0.5)
            # torch.nn.init.zero_(m.bias)

    # define init method inside your model class
    def init_weight(self, network):
        for m in network.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                # nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)    
