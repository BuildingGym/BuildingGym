from typing import Dict, List, Tuple, Type, Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device




class FEBuild(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        feature_dim: int,
        Fe_arch: Union[List[int], Dict[str, List[int]]],
        share_features_extractor: bool = True,
        activation_fn: Union[Type[nn.Module], None] = None,
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_fe: List[nn.Module] = []
        value_fe: List[nn.Module] = []
        last_layer_dim_vf = last_layer_dim_pi = feature_dim
        self.share_features_extractor = share_features_extractor

        # save dimensions of layers in policy and value nets
        if isinstance(Fe_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = Fe_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = Fe_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = Fe_arch

        if not share_features_extractor:
            # Iterate through the policy layers and build the policy net
            for curr_layer_dim in pi_layers_dims:
                policy_fe.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
                if activation_fn is not None:
                    policy_fe.append(activation_fn())
                last_layer_dim_pi = curr_layer_dim

            # Iterate through the value layers and build the value net
            for curr_layer_dim in vf_layers_dims:
                value_fe.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
                if activation_fn is not None:
                    value_fe.append(activation_fn())
                last_layer_dim_vf = curr_layer_dim     
            # Create networks
            # If the list of layers is empty, the network will just act as an Identity module
            self.policy_fe = nn.Sequential(*policy_fe).to(device)
            self.value_fe = nn.Sequential(*value_fe).to(device)
        else:
            for curr_layer_dim in pi_layers_dims:
                policy_fe.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
                if activation_fn is not None:
                    policy_fe.append(activation_fn())
                last_layer_dim_pi = curr_layer_dim                        
                last_layer_dim_vf = curr_layer_dim                        
            self.value_fe = self.policy_fe = nn.Sequential(*policy_fe).to(device)

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

    def extract_features(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:

        return self.policy_fe(features), self.value_fe(features)

    

    # def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    #     """
    #     :return: latent_policy, latent_value of the specified network.
    #         If all layers are shared, then ``latent_policy == latent_value``
    #     """
    #     return self.policy_fe(features), self.value_fe(features)

    # def forward_actor(self, features: th.Tensor) -> th.Tensor:
    #     return self.policy_fe(features)

    # def forward_critic(self, features: th.Tensor) -> th.Tensor:
    #     return self.value_fe(features)