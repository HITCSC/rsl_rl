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

## 12.24
1. CENet训练Loss降不下去，后续考虑使用一个好的baseline去训练CENet再导出,不再让CENet与PPO一起训练
2. 在enc_vel_actor_critic中增加estimator的输入正则化，训练看效果。需要还原直接解注释。