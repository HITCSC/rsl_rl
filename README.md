# RSL RL

A fast and simple implementation of RL algorithms, designed to run fully on GPU.
This code is an evolution of `rl-pytorch` provided with NVIDIA's Isaac Gym.

Environment repositories using the framework:

* **`Isaac Lab`** (built on top of NVIDIA Isaac Sim): https://github.com/isaac-sim/IsaacLab
* **`Legged-Gym`** (built on top of NVIDIA Isaac Gym): https://leggedrobotics.github.io/legged_gym/

The main branch supports **PPO** and **Student-Teacher Distillation** with additional features from our research. These include:

* [Random Network Distillation (RND)](https://proceedings.mlr.press/v229/schwarke23a.html) - Encourages exploration by adding
  a curiosity driven intrinsic reward.
* [Symmetry-based Augmentation](https://arxiv.org/abs/2403.04359) - Makes the learned behaviors more symmetrical.

We welcome contributions from the community. Please check our contribution guidelines for more
information.

**Maintainer**: Mayank Mittal and Clemens Schwarke <br/>
**Affiliation**: Robotic Systems Lab, ETH Zurich & NVIDIA <br/>
**Contact**: cschwarke@ethz.ch

> **Note:** The `algorithms` branch supports additional algorithms (SAC, DDPG, DSAC, and more). However, it isn't currently actively maintained.


## Setup

The package can be installed via PyPI with:

```bash
pip install rsl-rl-lib
```

or by cloning this repository and installing it with:

```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip install -e .
```

The package supports the following logging frameworks which can be configured through `logger`:

* Tensorboard: https://www.tensorflow.org/tensorboard/
* Weights & Biases: https://wandb.ai/site
* Neptune: https://docs.neptune.ai/

For a demo configuration of PPO, please check the [example_config.yaml](config/example_config.yaml) file.


## Contribution Guidelines

For documentation, we adopt the [Google Style Guide](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings. Please make sure that your code is well-documented and follows the guidelines.

We use the following tools for maintaining code quality:

- [pre-commit](https://pre-commit.com/): Runs a list of formatters and linters over the codebase.
- [black](https://black.readthedocs.io/en/stable/): The uncompromising code formatter.
- [flake8](https://flake8.pycqa.org/en/latest/): A wrapper around PyFlakes, pycodestyle, and McCabe complexity checker.

Please check [here](https://pre-commit.com/#install) for instructions to set these up. To run over the entire repository, please execute the following command in the terminal:

```bash
# for installation (only once)
pre-commit install
# for running
pre-commit run --all-files
```

## Citing

**We are working on writing a white paper for this library.** Until then, please cite the following work
if you use this library for your research:

```text
@InProceedings{rudin2022learning,
  title = 	 {Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning},
  author =       {Rudin, Nikita and Hoeller, David and Reist, Philipp and Hutter, Marco},
  booktitle = 	 {Proceedings of the 5th Conference on Robot Learning},
  pages = 	 {91--100},
  year = 	 {2022},
  volume = 	 {164},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v164/rudin22a.html},
}
```

If you use the library with curiosity-driven exploration (random network distillation), please cite:

```text
@InProceedings{schwarke2023curiosity,
  title = 	 {Curiosity-Driven Learning of Joint Locomotion and Manipulation Tasks},
  author =       {Schwarke, Clemens and Klemm, Victor and Boon, Matthijs van der and Bjelonic, Marko and Hutter, Marco},
  booktitle = 	 {Proceedings of The 7th Conference on Robot Learning},
  pages = 	 {2594--2610},
  year = 	 {2023},
  volume = 	 {229},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v229/schwarke23a.html},
}
```

If you use the library with symmetry augmentation, please cite:

```text
@InProceedings{mittal2024symmetry,
  author={Mittal, Mayank and Rudin, Nikita and Klemm, Victor and Allshire, Arthur and Hutter, Marco},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  title={Symmetry Considerations for Learning Task Symmetric Robot Policies},
  year={2024},
  pages={7433-7439},
  doi={10.1109/ICRA57147.2024.10611493}
}
```

## Record:
1. get_observation()函数由manager_base中自定义，在attention_env_cfg中配置了obsgroup，并写入键值  ,具体结构如：
```text
{
    "command": torch.Tensor(num_envs, h, D_command),  # 拼接后的指令张量
    "policy": torch.Tensor(num_envs, h, D_policy),    # 拼接后的策略观测张量
    "privileged": torch.Tensor(num_envs, h, D_privileged),  # 拼接后的特权观测张量
    "perception": Tensordict({  # 子Tensordict，包含感知观测项
        "map_scan": torch.Tensor(num_envs, h, D_map_scan)
    }, batch_size=(num_envs, 1))
}
```
2. 速度估计训练：在encoder中加入了两层线性层，接受本体感受（prop),速度估计输出在combination[B,-1,-3:]  
```text
self.proprio_linear = nn.Sequential(
                nn.Linear(d_obs, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, embedding_dim),
            )
```
3. obs维度：
ppo obs_batch -> AC.act(obs) -> AC.get_actor_obs

## 11.18解决：
修复了low_dim_obs数据输入的bug；base_lin_vel提取的错误
## 11.18问题：
1. 如何合理地把网络估计速度加入policy_obs有待商榷
2. 如何去叠观测，目前方案仅支持所有观测history都相同——get_actor_obs中的问题
3. 加入history还是有问题—— 解决：actor MLP的输入维度应该是 （single_obs + 3）* history
各部分输入记录：  
1. PPO中obs_batch 由runner中的get_observation获取的tensordict["key"][envs,H,d_obs]，经过minibatch获得->[B,H,d_obs]  key包含env_cfg中定义的
2. AC网络中 actor_obs ->["command"] + ["policy"]拼接 (B,H,4 + num_policy_obs)   critic_obs -> ["command"] + ["privileged"]
3. low_dim_obs: prop[b,h,d_obs] , high_dim_obs: prep[b,h,L*W*3] 

## 10.20 22-32-32 版训练：
1.网络结构： encoding传入相同数量history的map_scan与prop，critic不通过encoder，仅有actor走encoder。  
2.现象： 给速度指令不走，课程上不去   在play的时候给policy_obs传入真实速度，反而导致robo直接倒。
现在直接不给policy_obs base_lin_vel重训练试试（之前给的 0 0 0） 或者加大vel_reward?   确实可能需要不给policy_obs加入base_lin_vel    不知道是map_scan也叠了history的原因还是速度观测没给对的原因。   目前先测试直接取消policy_obs速度观测  
Q: 开始时候的loss_velocity下降得太快？ 走都不能走如何能估计速度？ 是否需要设计一种前期不能走的时候不去估计，等到能走的时候再估计，如果这样网络结构如何设计？ 分段训练，冻结参数？

A：目前的课程上不去，尝试分离估计器。使用单帧map_scan与多帧prop，多帧prop不输入actor_MLP  
分离速度估计器，先用两层linear_layers(input_dim = h*d_obs(h*88)) 估计速度输出，拼接到low_dim_obs :5,6,7位。 estimator——单帧map_scan

目前 enc_vel_actor_critic 很屎，直接在get_obs里面 估计速度再处理拼接，然后输出单帧的obs----目前encoder只接受一帧数据

## 10.23训练
1. 实际速度不跟踪指令速度而且实际很大是否与传入了错误的policy_obs相关？（前期因为policy_obs传入的速度本身就不准，因为站不住）——如何修改？ 根据episode_length来判断是否使用估计速度？代码实现or分阶段训练？，stage1：前期训练不用速度估计，待epl上去之后才使用速度估计；stage2：Loss_vel下去之后用vel_est代替policy_vel。
Q:而且目前Loss_velocity的曲线也很奇怪，最大才0.03？ epl为什么能上去？800 但是play的时候站不住？ 跟command给了history有关系吗？
速度估计有个错误：传出的[B,H,3] 是否这里没用上历史的观测——或者bp的时候没有成功？ 现在改成输出[B,3]尝试  —————— 同样的现象，小command但大实际速度，根本站不住？
11.24晚：换思路，用两阶段训练，先使用真实速度输入encoder，除去速度的policy_obs输入vel_est。  此时，回复attention_env_cfg中的policy_vel
训练中途报错：
```text
Error executing job with overrides: []
Traceback (most recent call last):
  File "/home/hitcsc/isaac_lab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 101, in hydra_main
    func(env_cfg, agent_cfg, *args, **kwargs)
  File "/home/hitcsc/isaac_lab/Leju-IsaacLab/scripts/rsl_rl/train.py", line 147, in main
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
  File "/home/hitcsc/isaac_lab/rsl_rl/rsl_rl/runners/on_policy_runner.py", line 150, in learn
    loss_dict = self.alg.update()
                ^^^^^^^^^^^^^^^^^
  File "/home/hitcsc/isaac_lab/rsl_rl/rsl_rl/algorithms/ppo.py", line 265, in update
    self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
  File "/home/hitcsc/isaac_lab/rsl_rl/rsl_rl/modules/enc_vel_actor_critic.py", line 203, in act
    return self.distribution.sample()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hitcsc/miniconda3/envs/isaac_lab/lib/python3.11/site-packages/torch/distributions/normal.py", line 74, in sample
    return torch.normal(self.loc.expand(shape), self.scale.expand(shape))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: normal expects all elements of std >= 0.0
noise_std_type='log' 按理说已经当做参数传入进去了？为什么还是会出现std<0的情况？
```
考虑网络结构？ stage1的训练为什么会出现这个错误（使用policy原始91个input，只是加了一个estimator但没有使用其估计数据）

排查是否为网络结构不匹配问题： 去除复杂地形学习速度，resume 一阶段
1125 速度估计效果还行，开始时有些不准，走动后还行0.05-0.8？ 加个滤波？

1126 用速度估计器代替policy中的速度，估计效果还行，但是注意力还得训练。能在小速度时抵抗不良诱惑，梅花桩走不了，加大梅花桩地形比例，减小速度reward再次训练