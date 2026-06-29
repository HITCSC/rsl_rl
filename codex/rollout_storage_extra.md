# RolloutStorage Extra 说明

本文档说明 `RolloutStorage.extra` 的用途、当前支持的 key、缓存内容，以及后续扩展时应遵守的约定。

## 基本语义

`RolloutStorage.extra` 是与 rollout transition 对齐的可选 `TensorDict` 缓冲区，用于保存不属于 PPO 核心字段、但 update 阶段需要复用的张量。

核心字段仍然独立保存：

- `observations`
- `actions`
- `rewards`
- `dones`
- `values`
- `actions_log_prob`
- `distribution_params`
- recurrent hidden states

`extra` 只承载附加训练数据。它不会参与环境交互，也不会进入 inference policy 导出。

## 存储规则

`Transition.extra` 可以为 `None`。如果某个 rollout 使用 extra，则必须从第一个 transition 开始提供，并且整个 rollout 的 extra schema 必须固定。

`RolloutStorage._save_extra()` 会在第一次看到 extra 时按 schema 懒分配缓冲区：

```text
[num_transitions_per_env, num_envs, ...extra_leaf_shape]
```

之后每一步会校验：

- nested leaf keys 必须完全一致。
- 每个 leaf 的 shape 必须一致。
- 每个 leaf 的 dtype 必须一致。
- transition extra 的 batch size 必须是 `[num_envs]`。

这能防止 update 阶段 batch 数据错位。

## 当前支持的 Extra Key

### `actor_features`

来源：`PPO.act()` 中的 frozen visual encoder feature cache。

启用条件：

- actor 和 critic 都有 `supports_feature_cache=True`。
- 当前通常来自 `CachedEncoderModelMixin`，例如冻结 encoder 的 DeFM / Depth-Anything-V2 类模型。

内容：

```text
extra["actor_features"][<obs_group>] -> Tensor
```

其中 `<obs_group>` 是 actor 使用的视觉 observation group，例如：

- `actor_depth`
- `actor_rgb`

用途：

- rollout 时只跑一次冻结视觉 encoder。
- PPO update 时通过 `actor.forward_from_features()` 复用缓存特征，避免每个 epoch / mini-batch 重复编码大图像或深度输入。

### `critic_features`

来源和启用条件同 `actor_features`。

内容：

```text
extra["critic_features"][<obs_group>] -> Tensor
```

其中 `<obs_group>` 是 critic 使用的视觉 observation group，例如：

- `critic_depth`
- `critic_rgb`

用途：

- update 阶段通过 `critic.forward_from_features()` 复用 critic 视觉 encoder 输出。

### `next_obs_prediction`

来源：`RolloutStorage.mini_batch_generator()` 的 batch 阶段懒生成。

注意：这个 key 不写入 `Transition.extra`，也不写入 rollout storage 的长期 extra 缓冲区。

内容：

```text
batch.extra["next_obs_prediction", "target"] -> Tensor
```

用途：

- 给可选的 DreamWAQ 风格 next-observation prediction 辅助 loss 提供 target。
- target 由 `storage.observations[t + 1]` 和 rollout 末尾 `final_obs` 临时拼出。
- 这样不需要在 storage 中额外保存一整份 `next_observations`，显存只在当前 mini-batch materialize target。

启用方式：

- `PPO(next_obs_prediction_cfg=...)`
- 默认不启用。

当前限制：

- 只支持 feedforward PPO。
- target observation group 必须是 1D observation。
- recurrent PPO 启用该功能会直接报错。

## Mini-Batch 行为

Feedforward generator 会 flatten `[time, env]` 为 `[batch]`，并用同一组 shuffled index 取：

- `observations`
- `extra`
- `actions`
- `values`
- `advantages`
- `returns`
- `old_actions_log_prob`
- `old_distribution_params`
- `dones`

如果启用了 next-observation prediction：

- 对非最后 rollout step，target 来自同一个 env 的下一步 observation。
- 对最后 rollout step，target 来自 `PPO.compute_returns(final_obs)` 保存的 `final_obs`。
- 生成的 target 会合并进当前 batch 的 `extra`，不会覆盖已有 `actor_features` / `critic_features`。

Recurrent generator 会对 `observations` 和 `extra` 同步做 trajectory split + padding，保证 padded trajectory 中 extra 与 observation 对齐。当前 next-observation prediction 不支持 recurrent generator。

## 扩展建议

新增 extra key 时，优先选择以下两种方式之一：

1. rollout 阶段缓存昂贵但 transition-aligned 的张量：
   - 在 algorithm 的 `act()` 或 `process_env_step()` 中写入 `transition.extra`。
   - 使用 nested key，避免和现有 key 冲突，例如 `extra["my_feature_cache", "encoder_a"]`。
   - 确保第一步就提供完整 schema，后续每步 shape/dtype 不变。

2. mini-batch 阶段懒生成只在 update 使用的 target：
   - 不写入 `Transition.extra`。
   - 在 generator 或 algorithm update 前按 batch index 临时生成。
   - 合并到 `batch.extra["my_auxiliary", ...]`。
   - 适合 next-step target、对比学习 target、只依赖已存 observations/actions 的派生数据。

扩展时应避免：

- 把 inference 不需要的大张量长期塞进 storage。
- 在 rollout 中保存可由 `observations` 和 batch index 便宜恢复的数据。
- 改变已有 extra key 的 shape 或 dtype。
- 在不同 transition 中有时提供 extra、有时不提供 extra。

新增功能时建议同时补测试：

- extra schema 改变时会报错。
- mini-batch 后 extra 与 action/value/observation 使用同一批 index。
- recurrent padding 场景下 extra 与 observation 对齐。
- 如果是 batch-only extra，确认不会污染 storage-level extra schema。
