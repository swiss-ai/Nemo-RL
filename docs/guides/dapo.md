# An in-depth Walkthrough of DAPO in NeMo RL

This guide covers the [Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)](https://arxiv.org/pdf/2503.14476) implementation in NeMo RL.

DAPO introduces four key improvements over Group Relative Policy Optimization (GRPO):
1. **Clip-Higher**, which promotes the diversity of the system and avoids entropy collapse
2. **Dynamic Sampling**, which improves training efficiency and stability
3. **Token-Level Policy Gradient Loss**, which is critical in long-CoT RL scenarios
4. **Overlong Reward Shaping**, which reduces reward noise and stabilizes training

This document focuses on DAPO-specific features: Dynamic Sampling and Overlong Reward Shaping. For foundational concepts on GRPO including data handling, policy training, generation, and loss functions, see the [NeMo RL GRPO Guide](grpo.md).


## Quickstart: Launch a DAPO Run

To get started quickly, use the example configuration [examples/configs/recipes/llm/dapo-qwen2.5-7b.v2.yaml](../../examples/configs/recipes/llm/dapo-qwen2.5-7b.v2.yaml). You can launch this using the same script as GRPO:

```bash
uv run examples/run_grpo.py --config examples/configs/recipes/llm/dapo-qwen2.5-7b.v2.yaml {overrides}
```

**Reminder**: Don't forget to set your HF_HOME, WANDB_API_KEY, and HF_DATASETS_CACHE (if needed). You'll need to do a `huggingface-cli login` as well for LLaMA models.

## Dynamic Sampling

Standard GRPO trains on all generated responses, even when they have identical rewards (zero gradient signal) within a prompt group of generations. Dynamic sampling filters to keep only groups with diverse rewards (`std > 0`), and accumulates them across batches until reaching the target batch size. Dynamic sampling can be enabled by setting `use_dynamic_sampling=True` in your configuration. For implementation details, see the [`dynamic_sampling`](../../nemo_rl/algorithms/grpo.py) function. 

**Algorithm**: For each training step:

1. Sample `batch_multiplier × num_prompts_per_step` prompts from the dataset. The default value of `batch_multiplier` is 1.
2. Generate `num_generations_per_prompt` responses per prompt and compute rewards.
3. Compute the baseline and standard deviation for each prompt group.
4. Filter prompt groups where `std > 0`.
5. Store these prompts in a cache until reaching the target training batch size of `num_prompts_per_step × num_generations_per_prompt` samples.
6. Samples are accumulated until the maximum number of allowed batches (`dynamic_sampling_max_gen_batches`) is reached. If the cache still does not meet the target rollout batch size at that point, an error is raised. To resolve this, consider adjusting parameters such as `num_prompts_per_step` or `num_generations_per_prompt` to increase sample diversity, or revisit the complexity of your data.
7. Perform training on the collected samples with nonzero standard deviation

### Async GRPO

Dynamic sampling is also supported by the legacy async GRPO loop
(`grpo.async_grpo.enabled: true`). Set `grpo.use_dynamic_sampling: true`
and use at least two generations per prompt. The trajectory collector discards
completed prompt groups with identical unshaped rewards (after optional scaling)
before replay insertion. Each candidate dispatch requests
`int(num_prompts_per_step × batch_multiplier)` prompts, including refills, subject
to the remaining candidate budget and available dataset rows. Groups enter replay
as they finish; the collector does not shrink a candidate batch to the number of
missing useful groups. Each target admits at most `num_prompts_per_step` groups.
Useful surplus is discarded rather than carried to another target or blocked on
replay capacity. Discarded groups are not retried as failed rollouts. Incomplete
current targets remain eligible after weight updates.

`dynamic_sampling_max_gen_batches` limits candidates per target to
`dynamic_sampling_max_gen_batches × int(num_prompts_per_step × batch_multiplier)`.
Each candidate group with valid rewards counts once against its dispatch target,
whether retained or discarded.
Training starts as soon as enough useful groups are ready; it does not wait for a
full candidate batch. An incomplete target at the limit stops collection with an
error. Counters reset after target consumption and on checkpoint resume.

Optional reward scaling and shaping run after replay sampling, in the same order
as sync GRPO: `scale_rewards`, then `apply_reward_shaping`, then advantage computation.
Enable scaling with `grpo.reward_scaling.enabled: true` and set the source and target
ranges. Scaling clamps rewards to the source range before mapping them to the target
range. It also processes named GDPO reward components. Enable shaping separately
with `grpo.reward_shaping.enabled: true` and configure the overlong penalty (or
`stop_properly_penalty_coef`).

For dynamic sampling, the collector scales a reward copy before testing equality.
This also rejects groups whose rewards become equal after clamping, as in sync GRPO.
The replay buffer keeps original rewards, so the trainer does not scale them twice.
Collector reward metrics use scaled, unshaped rewards; advantages and training reward
logs use rewards after scaling and shaping. Length penalties cannot make an
otherwise equal-reward group pass filtering.

Dynamic sampling is still unsupported by the single-controller path.

The collector uses the sync DAPO metric names under `collection/dynamic_sampling/`:
- `dynamic_sampling_num_gen_batches`: candidates evaluated in the interval divided
  by `int(num_prompts_per_step × batch_multiplier)`, including fractional batches.
  Internal stream retries
  do not consume another candidate batch.
- `dynamic_sampling_num_discarded_valid_samples`: useful response samples discarded
  because they exceed the target's remaining useful-group quota.
- `reward` and `filtered_reward`: sample-weighted mean rewards before and after
  filtering (after optional scaling, before shaping).

These are drain-to-drain **collection intervals**, not metrics for the trainer's
sampled batch. Passing filtering does not guarantee a group has been buffered or
trained yet. Empty intervals emit no metrics, and reward means with no samples are
omitted. Counters reset after draining and on checkpoint resume.
Equal-reward discards are also printed in collector logs.

### About batch_multiplier

`batch_multiplier` (a float ≥ 1.0) controls the initial prompt pool size by sampling `batch_multiplier × num_prompts_per_step` prompts before dynamic sampling. Higher values increase memory and compute requirements, while very low values (e.g., 1.0) may slow the cache accumulation of prompt groups with nonzero standard deviation. The optimal value depends on the dataset, model capacity, and overall training setup.  When **dynamic sampling** is enabled, we also log two additional metrics:

 * `dynamic_sampling_num_gen_batches`: The number of generation rounds required to produce `num_prompts_per_step * num_generations_per_prompt` samples with a nonzero standard deviation. If this number remains consistently high across iterations, try increasing the `batch_multiplier`. The maximum allowed value for this parameter is determined by `dynamic_sampling_max_gen_batches`.
 * `dynamic_sampling_num_discarded_valid_samples`: The number of samples with a nonzero standard deviation that are discarded because the total exceeds `num_prompts_per_step * num_generations_per_prompt`. If this value is frequently high (e.g., above `0.5 * num_prompts_per_step * num_generations_per_prompt`) and `dynamic_sampling_num_gen_batches` is consistently 1, it suggests that a large fraction of the dataset is being discarded unnecessarily. To improve data efficiency, consider decreasing the `batch_multiplier`.

## Reward Shaping
DAPO introduces an overlong reward shaping mechanism to reduce reward noise and stabilize training. This approach penalizes responses that exceed a specified length threshold, helping to prevent the model from generating excessively long outputs while maintaining solution quality.

For a detailed explanation of the overlong reward shaping mechanism, please refer to Section 3.4 of the [DAPO paper](https://arxiv.org/pdf/2503.14476). For implementation details, see the [`apply_reward_shaping`](../../nemo_rl/algorithms/reward_functions.py) function.

## Configuration

```yaml
grpo:
  use_dynamic_sampling: true  # Enable DAPO dynamic sampling
  num_prompts_per_step: 512   # Target number of prompts per training step
  num_generations_per_prompt: 16  # Generations per prompt
  batch_multiplier: 3    # Dataloader batch size = batch_multiplier × num_prompts_per_step
  dynamic_sampling_max_gen_batches: 10     # Maximum number of batches to be used for accumulating non-zero std prompts
  reward_scaling:
    enabled: true
    source_min: 0.0
    source_max: 1.0
    target_min: -1.0
    target_max: 1.0
  
  reward_shaping:
    enabled: true
    overlong_buffer_length: 4096     # Threshold before penalties apply (paper uses 4096)
    overlong_buffer_penalty: 1.0     # Penalty per excess token
    max_response_length: 20480       # Hard maximum generation length
```

**Key Parameters:**
- **`use_dynamic_sampling`**: When enabled, activates DAPO's dynamic sampling algorithm to filter and accumulate prompt groups with nonzero standard deviation
- **`batch_multiplier`**: Factor that scales the initial prompt pool size for sampling.
- **`dynamic_sampling_max_gen_batches`**: Maximum number of batches to be used for accumulating nonzero standard deviation prompts.
- **`reward_scaling`**: When enabled, clamps each reward in the batch to [source_min, source_max] and linearly rescales it to [target_min, target_max]. Defaults: source_min=0.0, source_max=1.0, target_min=0.0, target_max=1.0.
- **`reward_shaping`**: When enabled, applies the overlong penalty mechanism described in the Reward Shaping section above. Responses exceeding `max_response_length - overlong_buffer_length` receive penalties proportional to their excess length, helping to reduce reward noise and stabilize training.

> [!NOTE]
> When dynamic sampling is enabled, monitor the `filtered_reward` metric to track the average reward of the prompts with std > 0.

> [!NOTE]
> **Clip-Higher** and **Token-Level Policy Gradient Loss** are already supported in NeMo RL and can be configured through the `loss_fn` section of your experiment config:
> - Set `ratio_clip_max` to enable Clip-Higher (e.g., `ratio_clip_max: 0.28`)
> - Set `token_level_loss: true` to enable Token-Level Policy Gradient Loss (see [Loss Normalization](grpo.md#loss-normalization-token_level_loss) for what this changes and the tradeoff against sequence-level normalization)
> 
> See the full [DAPO example config](../../examples/configs/recipes/llm/dapo-qwen2.5-7b.v2.yaml) for reference.

## Example Training Results
Using the [DAPO example config](../../examples/configs/recipes/llm/dapo-qwen2.5-7b.v2.yaml), you can expect to see intermediate plots such as the training reward curve and validation accuracy on AIME24 for Qwen/Qwen2.5-Math-7B. These plots serve as reference outputs to help verify reproducibility. They are not intended to reflect the best accuracy that can be achieved using DAPO for this model.

![DAPO Qwen2.5-7B Training Reward](../assets/dapo_train_reward.png)
![DAPO Qwen2.5-7B Validation Accuracy](../assets/dapo_val_acc.png)

## References

- **DAPO Paper**: [Decoupled Clip and Dynamic Sampling Policy Optimization](https://arxiv.org/pdf/2503.14476)
- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **[NeMo RL GRPO Guide](grpo.md)**