# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO, DPPO, DMultiAMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    ActorCriticDepthCNN,
    PolicyDepthCNN,
    PolicyHeightMapCNN,
    RMAEncoder,
    MultiDiscriminator,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.modules.discriminator import Normalizer
from rsl_rl.utils import (
    AMPLoader,
    load_optimizer_policy_only,
    resolve_amp_obs_layout,
    resolve_obs_groups,
    slice_amp_obs,
    store_code_state,
)
from rsl_rl.utils.amp_obs_layout import PROPRIO_ONLY_AMP_ACTIVE_TERMS
from rsl_rl.runners import EMPOnPolicyRunner, OnPolicyRunner, NEWEMPOnPolicyRunner

import copy






class OnPolicyRunner_Distillation:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = copy.deepcopy(train_cfg)
    
        self.alg_cfg = copy.deepcopy(train_cfg["algorithm_student"])
        self.policy_cfg = copy.deepcopy(train_cfg["policy_student"])


        
        self.policy_cfg["class_name"] = "ActorCriticDepthCNN"
        self.policy_cfg["noise_std_type"] = "log"
        self.policy_cfg["actor_obs_normalization"] = True
        self.policy_cfg["critic_obs_normalization"] = True
        # Feature switches: enable RMA and/or multi-critic independently via config
        self.policy_cfg.setdefault("use_rma", False)
        self.policy_cfg.setdefault("use_multi_critic", False)
        self.policy_cfg.setdefault("num_critics", 1)
        self.policy_cfg.setdefault("style_id_key", "style_id")
        self.use_multi_amp = "amp_styles" in self.cfg
        if self.use_multi_amp:
            self.alg_cfg["class_name"] = "DMultiAMPPPO"
        else:
            self.alg_cfg["class_name"] = "DPPO"
        self.alg_cfg.setdefault("use_rma", self.policy_cfg["use_rma"])
        self.amp_style_obs_key = self.cfg.get("amp_style_obs_key", "style")
        # self.alg_cfg.setdefault("guide_pure_iters", 5000)
        # self.alg_cfg.setdefault("guide_blend_iters", 1000)
        self.device = device
        self.env = env
       

        ppo_runner = NEWEMPOnPolicyRunner(env, train_cfg, log_dir=None, device=device)
        # teacher_resume_path = "/home/hitcsc/navigation/model_29999.pt"
        # teacher_resume_path = "/home/hitcsc/hhx/Leju-IsaacLab-privileged-policy/logs/rsl_rl/Kuavo/s42/privileged/2026-06-08_21-24-55/model_19999.pt"
        teacher_resume_path = "/home/guojunhong/Downloads/model_48350.pt"
        ppo_runner.load(teacher_resume_path)

    # obtain the trained policy for inference
        self.teacher_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    

        # 修改 student_obs 的键名为 policy
    

        # check if multi-gpu is enabled
        
        
        self._configure_multi_gpu()

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")

        obs_groups = self.cfg.setdefault("obs_groups", {})
        obs_groups.setdefault("policy", ["policy"])
        obs_groups.setdefault("critic", ["policy", "privileged"])
        obs_groups.setdefault("student_perception", ["perception"])
        obs_groups.setdefault("student_precise_perception", ["precise_perception"])
        if self.policy_cfg.get("use_rma", False):
            obs_groups.setdefault("rma", ["policy"])
            obs_groups.setdefault("guide", ["command"])

        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # create the algorithm
        self.alg = self._construct_algorithm(obs)
        if self.use_multi_amp:
            self.amp_obs_layout = self.alg.amp_obs_layout

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations().to(self.device)
        amp_obs = self._get_active_amp_obs(obs) if self.use_multi_amp else None
        style_ids = self._get_style_ids(obs) if self.use_multi_amp else None
        self.train_mode()  # switch to train mode (for dropout for example)
        batch_size = obs.batch_size[0]  # 4096
        half_size = batch_size  # 2048
        teacher_obs = obs
        student_obs = obs
        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        student_flag = True
        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
       
        for it in range(start_iter, tot_iter):
            if self.alg.policy.use_rma:
                guide_mix_ratio = ActorCriticDepthCNN.compute_guide_mix_ratio(
                    it,
                    pure_guide_iters=2000,
                    blend_iters=1000,
                )
                self.alg.policy.set_guide_mix_ratio(guide_mix_ratio)

            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    # actions = self.alg.act(obs)
                    # Get first half (teacher_obs) and second half (student_obs)
                
                    privileged_actions = self.teacher_policy(teacher_obs)
                    # self.alg.transition.observations = obs.detach()
                    self.alg.transition.privileged_actions = privileged_actions.detach()
                    self.alg.transition.teacher_obs = teacher_obs

                    # if self.current_learning_iteration >1000:
                    if self.use_multi_amp:
                        student_actions = self.alg.act(student_obs, amp_obs, style_ids)
                    else:
                        student_actions = self.alg.act(student_obs)
                    # student_actions = self.alg.policy.act_inference(student_obs)
                    # actions = torch.cat([privileged_actions, student_actions], dim=0)
                
                    # print("self.current_learning_iteration", self.current_learning_iteration)
                    if self.current_learning_iteration > -1:
                        actions = student_actions
                        student_flag = True
                    else:
                        # actions = student_actions
                        actions = privileged_actions
                        student_flag = False

                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    next_amp_obs = self._get_active_amp_obs(obs) if self.use_multi_amp else None
                    # rewards = rewards[half_size:]
                    # dones = dones[half_size:]
                    teacher_obs = obs
                    student_obs = obs
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    teacher_obs, student_obs = teacher_obs.to(self.device), student_obs.to(self.device)
                    if self.use_multi_amp:
                        next_amp_obs = next_amp_obs.to(self.device)
                        style_ids = self._get_style_ids(obs)
                        next_amp_obs_with_term = torch.clone(next_amp_obs)
                        if hasattr(self.env, "reset_env_ids"):
                            reset_env_ids = self.env.unwrapped.reset_env_ids
                            if len(reset_env_ids) > 0:
                                terminal_amp_states = self._get_active_amp_obs(obs, env_ids=reset_env_ids)
                                next_amp_obs_with_term[reset_env_ids] = terminal_amp_states.to(self.device)
                        rewards = self.alg.multi_discriminator.predict_amp_reward(
                            amp_obs,
                            next_amp_obs_with_term,
                            rewards,
                            style_ids,
                            self.alg.amp_normalizers,
                        )
                        amp_obs = torch.clone(next_amp_obs)
                   
                    # process the step
                    if self.use_multi_amp:
                        self.alg.process_env_step_distillation(
                            student_obs, rewards, dones, extras, student_flag, half_size, next_amp_obs_with_term, style_ids
                        )
                    else:
                        self.alg.process_env_step_distillation(student_obs, rewards, dones, extras, student_flag, half_size)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

                        # print("!!!!!!!!!!!!!!!!!!!!!!!!", len(locals()["rewbuffer"]), new_ids)
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if student_flag:
                    self.alg.compute_returns(student_obs)

            # update policy
            loss_dict = self.alg.update_rl_distillation(student_flag)
            

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # (log) to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        

        # -- Policy
        # self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                # f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                # f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.use_multi_amp:
            saved_dict["multi_discriminator_state_dict"] = self.alg.multi_discriminator.state_dict()
            saved_dict["amp_normalizers"] = self.alg.amp_normalizers
        # -- Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def _should_use_actor_only_load(self, model_state_dict: dict) -> bool:
        """Use partial actor loading when checkpoint architecture differs (e.g. missing RMA)."""
        policy = self.alg.policy
        if not hasattr(policy, "load_actor_state_dict"):
            return False
        has_rma_in_ckpt = any(key.startswith("rma_encoder.") for key in model_state_dict)
        if getattr(policy, "use_rma", False) and not has_rma_in_ckpt:
            return True
        return False

    @staticmethod
    def _optimizer_state_compatible(optimizer, optimizer_state_dict: dict) -> bool:
        if optimizer_state_dict is None or "param_groups" not in optimizer_state_dict:
            return False
        current_groups = optimizer.param_groups
        loaded_groups = optimizer_state_dict["param_groups"]
        if len(current_groups) != len(loaded_groups):
            return False
        for current_group, loaded_group in zip(current_groups, loaded_groups):
            if len(current_group["params"]) != len(loaded_group["params"]):
                return False
        return True

    def _try_load_optimizer(self, loaded_dict: dict) -> bool:
        if "optimizer_state_dict" not in loaded_dict:
            print("[加载] checkpoint 中无 optimizer_state_dict，跳过")
            return False
        optimizer_state = loaded_dict["optimizer_state_dict"]
        if not self._optimizer_state_compatible(self.alg.optimizer, optimizer_state):
            print(
                "[加载] optimizer 参数组与当前模型不匹配（可能新增了 RMA 等模块），"
                "跳过 optimizer，使用新初始化的优化器"
            )
            return False
        try:
            self.alg.optimizer.load_state_dict(optimizer_state)
        except ValueError as exc:
            print(f"[加载] optimizer 加载失败，跳过: {exc}")
            return False
        if hasattr(self.alg, "rnd") and self.alg.rnd and "rnd_optimizer_state_dict" in loaded_dict:
            try:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
            except ValueError as exc:
                print(f"[加载] RND optimizer 加载失败，跳过: {exc}")
        return True

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        load_actor_only: bool = False,
        optimizer_load_mode: str = "policy_only",
        map_location: str | None = None,
    ):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        model_state_dict = loaded_dict["model_state_dict"]
        use_actor_only = load_actor_only or self._should_use_actor_only_load(model_state_dict)

        if use_actor_only and hasattr(self.alg.policy, "load_actor_state_dict"):
            if not load_actor_only:
                print("[Actor加载] 检测到 checkpoint 无 RMA 模块，自动仅加载 CNN / actor MLP / actor_obs_normalizer / std")
            load_info = self.alg.policy.load_actor_state_dict(model_state_dict)
            self._print_actor_load_summary(load_info)
            resumed_training = False
        else:
            resumed_training = self.alg.policy.load_state_dict(model_state_dict, strict=False)

        if self.use_multi_amp and "multi_discriminator_state_dict" in loaded_dict:
            self.alg.multi_discriminator.load_state_dict(loaded_dict["multi_discriminator_state_dict"])
        elif self.use_multi_amp:
            print("[INFO] Checkpoint has no discriminator weights. Discriminators keep random initialization.")

        if self.use_multi_amp and "amp_normalizers" in loaded_dict:
            self.alg.amp_normalizers = loaded_dict["amp_normalizers"]

        # -- Load RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- load optimizer if used (never when load_actor_only / actor-only auto path)
        if load_optimizer and resumed_training and not use_actor_only:
            if self.use_multi_amp and "optimizer_state_dict" in loaded_dict:
                self._load_optimizer_state(loaded_dict["optimizer_state_dict"], mode=optimizer_load_mode)
            else:
                self._try_load_optimizer(loaded_dict)
        elif use_actor_only:
            print("[Actor加载] 跳过 optimizer 参数，使用新初始化的优化器状态")
        # -- load current learning iteration
        if resumed_training and not use_actor_only:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def _load_optimizer_state(self, checkpoint_optimizer_state: dict, mode: str = "auto"):
        current_groups = len(self.alg.optimizer.param_groups)
        ckpt_groups = len(checkpoint_optimizer_state.get("param_groups", []))

        if mode == "policy_only" or (mode == "auto" and ckpt_groups < current_groups):
            load_optimizer_policy_only(self.alg.optimizer, checkpoint_optimizer_state, self.alg.policy)
            return

        try:
            self.alg.optimizer.load_state_dict(checkpoint_optimizer_state)
            print(f"[INFO] Loaded full optimizer state ({ckpt_groups} param groups).")
        except (ValueError, KeyError) as error:
            if mode == "full":
                raise
            print(f"[WARNING] Full optimizer load failed ({error}). Falling back to policy-only load.")
            load_optimizer_policy_only(self.alg.optimizer, checkpoint_optimizer_state, self.alg.policy)

    def load_actor(self, path: str, map_location: str | None = None):
        """Load CNN, actor MLP, actor_obs_normalizer, and std without optimizer state."""
        return self.load(path, load_optimizer=False, load_actor_only=True, map_location=map_location)

    @staticmethod
    def _print_actor_load_summary(load_info: dict) -> None:
        print(f"[Actor加载] 成功加载 {len(load_info['loaded_keys'])} 个参数（CNN / actor MLP / actor_obs_normalizer / std）")
        if load_info["partial_keys"]:
            print(f"[Actor加载] 部分加载 {len(load_info['partial_keys'])} 个参数（actor 输入维增加，如 RMA）:")
            for key, old_shape, new_shape in load_info["partial_keys"]:
                print(f"       {key}: old {old_shape} -> new {new_shape}")
        if load_info["skipped_shape_keys"]:
            print(f"[Actor加载] 跳过 {len(load_info['skipped_shape_keys'])} 个形状不匹配的 actor 参数:")
            for key, old_shape, new_shape in load_info["skipped_shape_keys"][:10]:
                print(f"       {key}: old {old_shape} vs new {new_shape}")
        if load_info["missing_actor_keys"]:
            print(f"[Actor加载] 新模型中缺失的 actor 参数: {load_info['missing_actor_keys']}")
        if load_info["ignored_keys"]:
            print(
                f"[Actor加载] 忽略 checkpoint 中的非 actor 参数 {len(load_info['ignored_keys'])} 个"
                "（如 rma_encoder / critic）"
            )

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        if getattr(self.alg.policy, "use_rma", False):
            self.alg.policy.set_guide_mix_ratio(0.0)
        return self.alg.policy.act_inference

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        if self.use_multi_amp:
            self.alg.multi_discriminator.train()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        if self.use_multi_amp:
            self.alg.multi_discriminator.eval()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs) -> PPO:
        """Construct the actor-critic algorithm."""
        # resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # initialize the actor-critic
        # actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        # actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
        #     obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        # ).to(self.device)
        # policy_cnn = PolicyDepthCNN()
        policy_cnn = PolicyHeightMapCNN(9, 7, use_output_layernorm=False)

        use_rma = self.policy_cfg.get("use_rma", False)
        rma_encoder = None
        if use_rma:
            rma_input_dim = 0
            for obs_group in self.cfg["obs_groups"]["rma"]:
                rma_input_dim += obs[obs_group].shape[-1]
            rma_encoder = RMAEncoder(input_dim=rma_input_dim)

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))

        actor_critic: ActorCriticDepthCNN = actor_critic_class(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            policy_cnn,
            rma_encoder,
            **self.policy_cfg,
        ).to(self.device)

        alg_class = eval(self.alg_cfg.pop("class_name"))

        if self.use_multi_amp:
            amp_styles = self._resolve_amp_styles()
            style_ids = sorted(int(style_id) for style_id in amp_styles.keys())
            amp_obs_layout = self._resolve_amp_obs_layout()

            amp_loaders = {}
            amp_normalizers = {}
            amp_obs_dim = amp_obs_layout["active_obs_dim"]
            for style_id in style_ids:
                style_cfg = amp_styles[style_id]
                loader = AMPLoader(
                    self.device,
                    time_between_frames=self.env.unwrapped.step_dt,
                    preload_transitions=True,
                    num_preload_transitions=self.cfg["amp_num_preload_transitions"],
                    motion_files=style_cfg["motion_files"],
                    amp_obs_layout=amp_obs_layout,
                )
                amp_loaders[style_id] = loader
                amp_normalizers[style_id] = Normalizer(loader.observation_dim)
                if loader.observation_dim != amp_obs_dim:
                    raise ValueError(
                        f"All AMP motion files must share observation dim {amp_obs_dim}. "
                        f"Style {style_id} loader reports {loader.observation_dim}."
                    )

            use_height_map_cnn = self.cfg.get("amp_use_height_map_cnn")
            if use_height_map_cnn is None:
                use_height_map_cnn = "height_scan" in amp_obs_layout["amp_active_terms"]
            if use_height_map_cnn and "height_scan" not in amp_obs_layout["amp_active_terms"]:
                raise ValueError(
                    "amp_use_height_map_cnn=True requires 'height_scan' in amp_active_terms."
                )

            multi_discriminator = MultiDiscriminator(
                amp_obs_dim,
                amp_styles,
                self.cfg["amp_discr_hidden_dims"],
                self.device,
                self.cfg.get("amp_task_reward_lerp", 0.0),
                amp_obs_layout=amp_obs_layout,
                use_height_map_cnn=use_height_map_cnn,
                cnn_channels=self.cfg.get("amp_cnn_channels", [32, 64]),
                cnn_output_dim=self.cfg.get("amp_cnn_output_dim", 64),
            ).to(self.device)

            alg: DMultiAMPPPO = alg_class(
                actor_critic,
                multi_discriminator,
                amp_loaders,
                amp_normalizers,
                style_ids,
                amp_obs_layout=amp_obs_layout,
                device=self.device,
                amp_replay_buffer_size=self.cfg.get("amp_replay_buffer_size", 100000),
                amploss_coef=self.cfg.get("amploss_coef", 1.0),
                grad_pen_lambda=self.cfg.get("amp_grad_pen_lambda", 10.0),
                **self.alg_cfg,
                multi_gpu_cfg=self.multi_gpu_cfg,
            )
        else:
            alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # initialize the storage
        alg.init_storage(
            "rl+distillation",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def _get_style_ids(self, obs) -> torch.Tensor:
        return obs[self.amp_style_obs_key].to(self.device)

    def _resolve_amp_obs_layout(self) -> dict:
        amp_obs_layout_cfg = self.cfg.get("amp_obs_layout")
        if hasattr(amp_obs_layout_cfg, "to_dict"):
            amp_obs_layout_cfg = amp_obs_layout_cfg.to_dict()
        elif amp_obs_layout_cfg is None:
            amp_obs_layout_cfg = {}

        if self.cfg.get("amp_use_proprio_only", False):
            amp_obs_layout_cfg = dict(amp_obs_layout_cfg)
            amp_obs_layout_cfg["amp_active_terms"] = list(PROPRIO_ONLY_AMP_ACTIVE_TERMS)
        elif "amp_active_terms" in self.cfg and self.cfg["amp_active_terms"] is not None:
            amp_obs_layout_cfg = dict(amp_obs_layout_cfg)
            amp_obs_layout_cfg["amp_active_terms"] = list(self.cfg["amp_active_terms"])

        amp_obs_layout = resolve_amp_obs_layout(amp_obs_layout_cfg)
        print(
            f"AMP obs: file dim={amp_obs_layout['full_obs_dim']}, "
            f"active dim={amp_obs_layout['active_obs_dim']}, "
            f"active terms={amp_obs_layout['amp_active_terms']}"
        )
        return amp_obs_layout

    def _get_active_amp_obs(self, obs, env_ids=None) -> torch.Tensor:
        amp_obs_key = self.cfg.get("amp_obs_key", "AmpObs")
        if amp_obs_key not in obs.keys():
            raise KeyError(
                f"AMP observation key '{amp_obs_key}' not found. Available keys: {list(obs.keys())}"
            )
        amp_obs = obs[amp_obs_key]
        if env_ids is not None:
            amp_obs = amp_obs[env_ids]
        amp_obs = amp_obs.to(self.device)
        return slice_amp_obs(amp_obs, self.amp_obs_layout)

    def _resolve_amp_styles(self) -> dict[int, dict]:
        if "amp_styles" not in self.cfg:
            raise ValueError(
                "Multi-AMP training requires `amp_styles` in the runner config. "
                "Example: amp_styles: {0: {motion_files: [...], amp_reward_coef: 0.3}, ...}"
            )

        amp_styles = {}
        for style_id, style_cfg in self.cfg["amp_styles"].items():
            style_id = int(style_id)
            if "motion_files" not in style_cfg:
                raise ValueError(f"amp_styles[{style_id}] must contain 'motion_files'.")
            if "amp_reward_coef" not in style_cfg:
                raise ValueError(f"amp_styles[{style_id}] must contain 'amp_reward_coef'.")
            amp_styles[style_id] = style_cfg
        return amp_styles

    def _prepare_logging_writer(self):
        """Prepares the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")
