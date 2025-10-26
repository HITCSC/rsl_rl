from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict import TensorDict 
from rsl_rl.networks import MLP, EmpiricalNormalization, AttentionMapEncoder 


class EncActorCritic(nn.Module):
    is_recurrent = False
    LOAD_POLICY_WEIGHTS = 1
    LOAD_CRITIC_WEIGHTS = 2
    LOAD_ENCODER_WEIGHTS = 4 
    LOAD_NORMALIZER_WEIGHTS = 8

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        embedding_dim=64,
        load_mask:int=LOAD_POLICY_WEIGHTS|LOAD_CRITIC_WEIGHTS|LOAD_ENCODER_WEIGHTS|LOAD_NORMALIZER_WEIGHTS,
        output_attention:bool=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "EncActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0  # obervation dimensions in 1 stamp for the actor
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) > 2, "The EncActorCritic module only supports obs shape [B,H,d,...]. "
            "for IsaacLab, you need to make sure that flatten_history_dim is False."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0 # obervation dimensions in 1 stamp for the critic 
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) > 2, "The EncActorCritic module only supports obs shape [B,H,d,...]. "
            "for IsaacLab, you need to make sure that flatten_history_dim is False."
            num_critic_obs += obs[obs_group].shape[-1]
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        # Encoder :
        # num_perception_obs = 0
        scan_height_shape = []
        for obs_group in obs_groups["perception"]:
            # TODO : 这里需要修改为支持多obs的输入
            # num_perception_obs += obs[obs_group].shape[-1]
            if (obs_group == "perception"):
                scan_height_shape = obs[obs_group].shape # 
        self.embedding_dim = embedding_dim
        self.encoder = AttentionMapEncoder(self.num_actor_obs,embedding_dim=embedding_dim)
        print(f"Encoder : {self.encoder}")
        self.horizon = scan_height_shape[1] 
        self.high_dim_obs_shape = scan_height_shape # [B,H,L,W,C]
        self.load_mask = load_mask  # 加载参数的mask
        self.output_attention = output_attention  # 是否输出attention 
        embedding_actor_dim = self.horizon*(self.embedding_dim + num_actor_obs) # [H*(d_obs+d)]
        embedding_critic_dim = self.horizon*(self.embedding_dim + num_critic_obs) # [H*(d_c+d)]
        self.embedding_actor_dim = embedding_actor_dim
        self.embedding_critic_dim = embedding_critic_dim 
        # 这里需要构造一个从critic到actor obs的mask, 但是当前仍然只支持1d的输入
        critic_to_actor_mask = torch.zeros((num_critic_obs,), dtype=torch.bool)
        tensor_idx = 0
        for i, obs_group in enumerate(obs_groups["critic"]):
            if obs_group in obs_groups["policy"]:
                critic_to_actor_mask[tensor_idx: tensor_idx + obs[obs_group].shape[-1]] = True
            tensor_idx += obs[obs_group].shape[-1]
        self.critic_to_actor_mask = critic_to_actor_mask
        # convert to [B,H,d] style (if obs shape = (B,H*d)) 
        # if (obs_style=='lab'):
        #     self.critic_to_actor_mask = self._lab_to_gym(critic_to_actor_mask,self.horizon,keep_dim=False)
        # else:
        #     self.critic_to_actor_mask = critic_to_actor_mask.reshape(self.horizon,-1)
        # actor
        self.actor = MLP(embedding_actor_dim, num_actions, actor_hidden_dims, activation)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization((self.horizon,num_actor_obs))  # 这里是支持输入[B,H,d]的(self.horizon,num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic
        self.critic = MLP(embedding_critic_dim, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization((self.horizon,num_critic_obs))  # 是不是(self.horizon,num_critic_obs)会更好?
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, prop_obs:torch.Tensor ,perception_obs: torch.Tensor):
        """
        :param prop_obs: [B, H, d_prop]
        :param perception_obs: [B, H, d_obs]
        """
        # compute embedding 
        embedding,_ = self.encoder(perception_obs,prop_obs,embedding_only=False)
        embedding_vec = embedding.view(embedding.shape[0], -1)  # [B,H*(d+d_obs)]
        # compute mean
        mean = self.actor(embedding_vec)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs:TensorDict, **kwargs):
        low_dim_obs,high_dim_obs = self.get_actor_obs(obs)
        low_dim_obs = self.actor_obs_normalizer(low_dim_obs)  # [B,H,d]
        self.update_distribution(low_dim_obs,high_dim_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        low_dim_obs,high_dim_obs = self.get_actor_obs(obs)  # [B,H,d]
        low_dim_obs = self.actor_obs_normalizer(low_dim_obs) # [B,H,d]
        # compute embedding 
        embedding,attention = self.encoder(high_dim_obs,low_dim_obs,embedding_only=False)
        embedding_vec = embedding.view(embedding.shape[0], -1)  # [B,H*(d+d_obs)], gym style 
        # compute mean
        action = self.actor(embedding_vec)
        if (self.output_attention):
            return action,attention
        else:
            return action

    def evaluate(self, obs, **kwargs):
        low_dim_obs,high_dim_obs = self.get_critic_obs(obs)  # [B,H,d]
        low_dim_obs = self.critic_obs_normalizer(low_dim_obs)
        # TODO : 这里需要针对(B,H*d)的情况进行处理
        low_dim_query = low_dim_obs # [B,H,d] 假设是一样的，只不过不带噪声
        # low_dim_query = low_dim_obs[:,:,self.critic_to_actor_mask]  # [B,H,d] for attention query
        embedding,_ = self.encoder(high_dim_obs,low_dim_query,embedding_only=True)
        critic_obs = torch.cat([embedding, low_dim_obs], dim=-1)  # [B,H,d+d_obs]
        critic_obs = critic_obs.view(critic_obs.shape[0], -1)  # [B,H*(d+d_obs)], gym style
        values = self.critic(critic_obs)
        return values
    
    def _gym_to_lab(self,obs:torch.Tensor,horizon:int,keep_dim=False)->torch.Tensor:
        """
        Brief:
            from gym style obs [O_1^1,...,O_1^d,...,O_H^1,...,O_H^d] to 
            lab style obs [O_{1:H}^1,...,O_{1:H}^d]
        Args:
            obs: shape [B, H*d]
            horizon: 时间步数 H
            keep_dim: 是否保持输入维度, False会返回[B,d,H]
        Returns:
            lab_style_obs :  shape [B, d*H]
        """
        B, total_dim = obs.shape
        d = total_dim // horizon
        
        # 检查维度是否可整除
        if total_dim % horizon != 0:
            raise ValueError(f"Total dimension {total_dim} must be divisible by horizon {horizon}")
        # 一步完成转换
        if (keep_dim):
            return obs.view(B, horizon, d).permute(0, 2, 1).reshape(B, -1)
        else:
            return obs.view(B, horizon, d).permute(0, 2, 1)  # [B,d,H]
    
    def _lab_to_gym(self,obs:torch.Tensor,horizon:int,keep_dim=False)->torch.Tensor:
        """
        Brief:
            from lab style obs [O_{1:H}^1,...,O_{1:H}^d] to 
            gym style obs [O_1^1,...,O_1^d,...,O_H^1,...,O_H^d]
        Args:
            obs: shape [B, d*H]
            horizon: 时间步数 H
            keep_dim: 是否保持输入的维度, False会返回[B,H,d]
        Returns:
            gym_style_obs :  shape [B, H*d]
        """
        B, total_dim = obs.shape
        d = total_dim // horizon

        # 检查维度是否可整除
        if total_dim % horizon != 0:
            raise ValueError(f"Total dimension {total_dim} must be divisible by horizon {horizon}")
        if keep_dim:
            return obs.view(B, d, horizon).permute(0, 2, 1).reshape(B, -1)
        else:
            return obs.view(B, d, horizon).permute(0, 2, 1)  # [B,H,d]

    def get_actor_obs(self, obs:TensorDict,style:str='lab')->tuple:
        """
        :param obs: TensorDict, each element shape maybe [B,H*d] or [B,H,d,...]
        :param style : 'lab' or 'gym', for lab style obs the permutation is 
            [O_{1:H}^1,O_{1:H}^2,...,O_{1:H}^n] where n is the index of part/group;
            for gym style obs , the permutation is [O_1^1,...,O_1^n,O_2^1,...,O_2^n,...,O_H^n]
        :return : tuple of TensorDict, each element shape is [B,H,d,...]
        """
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            # 这里假设每个group的历史堆叠形式是gym style的
            # if style == 'lab':
            #     gym_obs = self._lab_to_gym(obs[obs_group], self.horizon,keep_dim=False)  # [B,H,d]
            #     obs_list.append(gym_obs)
            # else:
            #     B = obs[obs_group].shape[0]
            #     obs_list.append(obs[obs_group].reshape(B,self.horizon,-1))  # [B,H,d_i]
            obs_list.append(obs[obs_group]) # [B,H,d_i]
        low_dim_obs = torch.cat(obs_list, dim=-1)  # [B,H,d]
        high_dim_obs_list = []
        for obs_group in self.obs_groups["perception"]:
            high_dim_obs_list.append(obs[obs_group])
        high_dim_obs = torch.cat(high_dim_obs_list, dim=-1)  
        return low_dim_obs,high_dim_obs

    def get_critic_obs(self, obs:TensorDict,style:str='lab')->tuple:
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            # 这里假设每个group的历史堆叠形式是gym style的
            # if style == 'lab':
            #     gym_obs = self._lab_to_gym(obs[obs_group], self.horizon,keep_dim=False)  # [B,H,d]
            #     obs_list.append(gym_obs)
            # else:
            #     B = obs[obs_group].shape[0]
            #     obs_list.append(obs[obs_group].reshape(B,self.horizon,-1))  # [B,H,d_i]
            obs_list.append(obs[obs_group]) # [B,H,d_i]
        low_dim_obs = torch.cat(obs_list, dim=-1)  # [B,H,d]
        high_dim_obs_list = []
        for obs_group in self.obs_groups["perception"]:
            high_dim_obs_list.append(obs[obs_group])
        high_dim_obs = torch.cat(high_dim_obs_list, dim=-1) 
        return low_dim_obs,high_dim_obs

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs,_ = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs,_ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    # for state_dict :
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        module_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        # 不知道为什么, 不能加下面的
        # if self.actor_obs_normalization:
        #     module_dict[prefix + "actor_obs_normalizer"] = self.actor_obs_normalizer.state_dict()
        # if self.critic_obs_normalization:
        #     module_dict[prefix + "critic_obs_normalizer"] = self.critic_obs_normalizer.state_dict()
        return module_dict
    

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        TODO : 
            because of the encoder is independent of the actor and critic, so we need to load the encoder parameters separately. Besides, 
            in the different training process, the critic's observation space is same, hence we can load the critic's parameters directly.
        """
        # 存在的keys : {'std','encoder.*','actor.*','critic.*'}
        if self.load_mask & self.LOAD_POLICY_WEIGHTS:
            actor_state_dict = {k.replace('actor.', '',1): v for k, v in state_dict.items() if k.startswith('actor.')}
            self.actor.load_state_dict(actor_state_dict, strict=strict)
            print("=== EncActorCritic : Load Actor Weights ===")
            # TODO : 这里需要确认一下, 是否需要加载std的参数
        if self.load_mask & self.LOAD_CRITIC_WEIGHTS:
            critic_state_dict = {k.replace('critic.', '',1): v for k, v in state_dict.items() if k.startswith('critic.')}
            self.critic.load_state_dict(critic_state_dict, strict=strict)
            print("=== EncActorCritic : Load Critic Weights ===")
        if self.load_mask & self.LOAD_ENCODER_WEIGHTS:
            enc_state_dict = {k.replace('encoder.', '',1): v for k, v in state_dict.items() if k.startswith('encoder.')}
            self.encoder.load_state_dict(enc_state_dict, strict=strict)
            print("=== EncActorCritic : Load Encoder Weights ===")
        # 这里还需要load normalization的参数
        if (self.load_mask & self.LOAD_NORMALIZER_WEIGHTS):
            # if (self.actor_obs_normalization) and ('actor_obs_normalizer' in state_dict):
            if (self.actor_obs_normalization):
                act_obs_norm_state_dict = {k.replace('actor_obs_normalizer.', '',1): v for k, v in state_dict.items() if k.startswith('actor_obs_normalizer.')}
                self.actor_obs_normalizer.load_state_dict(act_obs_norm_state_dict)
                print("=== EncActorCritic : Load actor normalizer weights ===")
            # if (self.critic_obs_normalization) and ('critic_obs_normalizer' in state_dict):
            if (self.critic_obs_normalization):
                critic_obs_norm_state_dict = {k.replace('critic_obs_normalizer.', '',1): v for k, v in state_dict.items() if k.startswith('critic_obs_normalizer.')}
                self.critic_obs_normalizer.load_state_dict(critic_obs_norm_state_dict)
                print("=== EncActorCritic : Load critic normalizer weights ===")
        # super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes
