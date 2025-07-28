# RLLM Configuration Parameters Explained

This document provides a comprehensive explanation of RLLM framework configuration parameters in simple terms for software engineers without ML background.

## Table of Contents
1. [Environment Configuration](#environment-configuration)
2. [Agent Configuration](#agent-configuration)
3. [Actor-Rollout-Reference Configuration](#actor-rollout-reference-configuration)
4. [Algorithm Configuration](#algorithm-configuration)
5. [Critic Configuration](#critic-configuration)
6. [Trainer Configuration](#trainer-configuration)
7. [Framework Architecture](#framework-architecture)
8. [Agent and Environment Max Steps](#agent-and-environment-max-steps)

---

## Environment Configuration

### env.env_args.container_config.no_rebuild=False

**What it does:** Controls whether Docker containers are rebuilt from scratch for each training episode.

**In simple terms:** Think of this like deciding whether to create a fresh virtual machine for each test run vs reusing an existing one. When `no_rebuild=False` (the default), the system creates a brand new Docker container for each training episode. This ensures a clean, consistent environment but takes more time. When `no_rebuild=True`, it reuses existing containers, which is faster but might carry over state from previous runs.

**Why it matters:** 
- **False (rebuild)**: Ensures complete isolation between training episodes, preventing any contamination from previous runs
- **True (no rebuild)**: Speeds up training by skipping the container build process, useful when iterating quickly

**Found in:** `src/tbench_rllm/docker_env.py:39`

---

## Agent Configuration

### agent.async_engine=True

**What it does:** Controls whether the agent uses asynchronous (concurrent) or synchronous (sequential) execution when generating training trajectories.

**In simple terms:** Imagine you need to run 8 simulations. With `async_engine=False`, you run them one after another (like washing dishes one at a time). With `async_engine=True`, you run multiple simulations concurrently (like having multiple dishwashers running at once). This is especially useful when each simulation involves waiting for external operations like Docker containers or API calls.

**Why it matters:**
- **True (async)**: Significantly faster when environments have I/O operations (like terminal commands or Docker operations). Multiple trajectories are collected in parallel
- **False (sync)**: Simpler, sequential execution. Easier to debug but slower for I/O-heavy environments

**Technical details:** When async is enabled, the system uses `AsyncAgentExecutionEngine` which runs trajectory generation in separate threads and communicates results via queues.

**Found in:** `external/rllm/rllm/trainer/verl/agent_ppo_trainer.py:519`

---

## Actor-Rollout-Reference Configuration

### actor_rollout_ref.model.use_shm=True

**What it does:** Controls whether model files are copied to shared memory (RAM-based filesystem) for faster access.

**In simple terms:** Think of this like the difference between reading a book from a library shelf (disk) versus having it on your desk (RAM). When `use_shm=True`, the system copies model files to `/dev/shm`, which is a special folder that exists in RAM rather than on your hard drive. This makes loading the model much faster, especially when multiple processes need to access it.

**Why it matters:**
- **True**: Dramatically speeds up model loading, especially for large models. Essential when multiple worker processes need to load the same model
- **False**: Models are loaded from disk each time, which is slower but uses less RAM

**Best practice:** Use `use_shm=True` combined with `load_format: safetensors` for large models to optimize loading performance.

**Technical details:** Files are copied to `/dev/shm/verl-cache/` which is a tmpfs (temporary filesystem in RAM) on Linux systems.

**Found in:** `external/rllm/verl/verl/utils/fs.py`

### actor_rollout_ref.actor.optim.lr=$ACTOR_LR

**What it does:** Sets the learning rate for the actor (policy) network optimizer.

**In simple terms:** Think of the learning rate like the size of steps you take when learning to ride a bike. A smaller learning rate (like 1e-6) means taking tiny, careful steps - you learn slowly but steadily. A larger learning rate means taking bigger steps - you might learn faster but could also overshoot and fall. The actor network is what decides which actions (tokens) to generate, so we typically use small, careful steps to avoid breaking what already works.

**Why it matters:**
- **Lower values (e.g., 1e-6)**: More stable training, less likely to break existing behavior
- **Higher values (e.g., 1e-4)**: Faster learning but risk of instability
- **Too high**: The model might "forget" good behaviors and produce garbage
- **Too low**: Training becomes extremely slow or stops improving

**Technical details:** 
- Uses AdamW optimizer with configurable betas and weight decay
- Default actor LR (1e-6) is typically lower than critic LR (1e-5)
- Supports learning rate scheduling with warmup and cosine decay

**Found in:** `external/rllm/verl/verl/workers/fsdp_workers.py:369`

### actor_rollout_ref.model.use_remove_padding=True

**What it does:** Enables sequence packing by removing padding tokens before model computation.

**In simple terms:** Imagine you're packing a suitcase with clothes of different sizes. Without padding removal, you'd use uniform boxes for each item (lots of wasted space). With padding removal enabled, you pack clothes directly, fitting them together efficiently. In language models, sequences have different lengths but are typically padded to the same length with special "padding" tokens. This feature removes those padding tokens and packs sequences together more efficiently.

**Why it matters:**
- **True**: Significantly improves GPU utilization by not wasting computation on meaningless padding tokens
- **False**: Simpler but less efficient, processes padding tokens unnecessarily
- **Performance gain**: Can improve training throughput by 20-40% on variable-length datasets

**Technical details:**
- Uses Flash Attention's variable-length support
- Currently supported for Llama, Mistral, Gemma1, and Qwen models
- Applies monkey patching to transformer attention implementation
- Maintains correct position IDs for rotary embeddings

**Found in:** `external/rllm/verl/verl/models/transformers/monkey_patch.py`

### actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE and ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU

**What it does:** Controls the batch sizes used during PPO training optimization.

**In simple terms:** Think of this like serving food at a restaurant:
- **Mini-batch size**: The total number of plates you need to serve (e.g., 256 plates)
- **Micro-batch size per GPU**: How many plates each waiter (GPU) can carry at once (e.g., 8 plates)
- If you have fewer waiters or they can only carry a few plates, they'll need multiple trips (gradient accumulation)

**The hierarchy:**
1. **Train batch size** (e.g., 1024): Total customers in the restaurant
2. **PPO mini-batch size** (e.g., 256): Customers served in one round of optimization
3. **PPO micro-batch size per GPU** (e.g., 8): Customers each waiter handles at once

**Why it matters:**
- **Mini-batch size**: Affects training stability and convergence. Larger = more stable but slower updates
- **Micro-batch size**: Affects GPU memory usage. Larger = more memory but faster processing
- **Gradient accumulation** happens when micro < mini (multiple forward passes before updating weights)

**Best practices:**
- Increase micro-batch size as much as GPU memory allows
- Keep mini-batch size reasonable for stable PPO updates (typically 64-512)
- Ideal case: micro-batch = mini-batch (no gradient accumulation needed)

**Technical details:** The framework automatically normalizes these values based on the number of GPUs and rollout workers.

**Found in:** `external/rllm/verl/verl/workers/fsdp_workers.py:162`

### actor_rollout_ref.actor.kl_loss_type=low_var_kl

**What it does:** Specifies the method for calculating KL (Kullback-Leibler) divergence penalty between the current policy and reference policy.

**In simple terms:** KL divergence measures how different your current model is from the original model. Think of it like a safety rope when rock climbing - it prevents your model from changing too drastically and "falling off a cliff". The "low_var_kl" method is like using a more reliable, less shaky rope that gives more consistent measurements.

**Available options:**
- **"kl"**: Standard difference between log probabilities
- **"abs"**: Absolute value of the difference
- **"mse"**: Squared difference (like measuring error in predictions)
- **"low_var_kl"**: Low-variance estimator based on probability ratios
- **"full"**: Not implemented (would need full vocabulary probabilities)

**Why "low_var_kl" is preferred:**
- **More stable**: Provides consistent estimates even with small batches
- **Unbiased**: Gives accurate measurements of policy change
- **Numerically safe**: Clamped between -10 and 10 to prevent extreme values
- **Better for GRPO**: Specifically recommended for Group Relative Policy Optimization

**Technical formula:** `(exp(ref_logprob - logprob) - (ref_logprob - logprob) - 1)`

**Found in:** `external/rllm/verl/verl/trainer/ppo/core_algos.py:592`

### actor_rollout_ref.actor.entropy_coeff=0

**What it does:** Controls the weight of entropy regularization in the policy loss function.

**In simple terms:** Entropy measures how "random" or "uncertain" the model's predictions are. Think of it like this:
- **High entropy**: The model spreads its predictions across many options (like a person who can't decide what to order at a restaurant)
- **Low entropy**: The model is very confident about specific choices (like always ordering the same dish)
- **Entropy coefficient**: How much we encourage the model to keep exploring different options

**Why set it to 0?**
- **Disables exploration bonus**: The model focuses purely on what it thinks is best (exploitation)
- **More deterministic outputs**: The model becomes more consistent and predictable
- **Suitable for language models**: With huge vocabularies (50k+ tokens), there's already natural randomness
- **Alternative regularization**: The config uses KL divergence instead to prevent drastic changes

**When you might want entropy > 0:**
- Early in training when exploring different strategies
- In environments with small action spaces
- When the model gets stuck in local optima

**Technical details:** 
- Entropy is calculated as: `sum(p * log(p))` where p is the probability distribution
- When entropy_coeff = 0, entropy is still logged but not used in the loss

**Found in:** `external/rllm/verl/verl/workers/actor/dp_actor.py`

### actor_rollout_ref.model.enable_gradient_checkpointing=True

**What it does:** Trades computation time for memory savings by not storing intermediate activations during forward pass.

**In simple terms:** Imagine you're solving a complex math problem step by step:
- **Without checkpointing**: You write down every intermediate calculation (uses lots of paper/memory)
- **With checkpointing**: You only write down key checkpoints and recalculate the intermediate steps when needed (uses less paper but takes more time)

**How it works:**
1. During forward pass: Only stores essential "checkpoints" instead of all intermediate values
2. During backward pass: Recomputes the intermediate values from checkpoints when calculating gradients
3. Result: ~50-70% memory savings at the cost of ~20-30% slower training

**Why it matters:**
- **Enabled (True)**: Allows training larger models or bigger batch sizes on limited GPU memory
- **Disabled (False)**: Faster training when you have sufficient GPU memory

**When to use:**
- **Enable**: When training large models that don't fit in GPU memory
- **Enable**: When you want to increase batch size but are memory-limited
- **Disable**: When training speed is critical and memory is abundant
- **Disable**: For small models that fit comfortably in memory

**Technical details:**
- Uses `use_reentrant: False` mode for better compatibility
- Can be combined with activation offloading for even more memory savings
- Applied to both actor and critic models independently

**Found in:** `external/rllm/verl/verl/workers/fsdp_workers.py:273`

### actor_rollout_ref.actor.fsdp_config.param_offload=False and optimizer_offload=False

**What it does:** Controls whether model parameters and optimizer states are offloaded to CPU memory to save GPU memory.

**In simple terms:** Think of this like a workspace with limited desk space (GPU) and unlimited storage shelves (CPU):
- **param_offload**: Moves model weights to storage shelves when not actively using them
- **optimizer_offload**: Moves training history (momentum, variance) to storage shelves
- When needed, items are brought back to the desk for work, then returned to storage

**The two settings:**
1. **param_offload**: Offloads model parameters (weights) to CPU
2. **optimizer_offload**: Offloads optimizer states (Adam momentum, variance, etc.) to CPU

**Why they matter:**
- **False (no offload)**: Keeps everything on GPU for fastest access but uses more GPU memory
- **True (offload)**: Saves GPU memory but adds CPU-GPU transfer overhead, slowing training

**Memory impact:**
- Model parameters: Can save 50% of model memory
- Optimizer states: Can save another 2x model size (Adam uses 2 states per parameter)
- Combined: Can enable training models 3-4x larger than GPU memory

**When to enable:**
- Training very large models that don't fit in GPU memory
- Need larger batch sizes but GPU memory-limited
- Can tolerate 20-50% slower training for memory savings

**When to disable:**
- Have sufficient GPU memory for model + optimizer
- Training speed is critical
- Using smaller models that fit comfortably

**Technical details:** The framework automatically loads/offloads around computation steps to minimize transfer overhead.

**Found in:** `external/rllm/verl/verl/workers/fsdp_workers.py:147`

### actor_rollout_ref.rollout configurations

**tensor_model_parallel_size=$TP_SIZE**
- **What it does:** Splits the model across multiple GPUs for parallel inference
- **In simple terms:** Like having multiple workers assemble different parts of a car simultaneously. With TP=2, the model is split across 2 GPUs, each handling part of the computation
- **Why it matters:** Enables inference of models too large for a single GPU and speeds up generation
- **Best practice:** Use power of 2 (1, 2, 4, 8) based on your model size and GPU count

**gpu_memory_utilization=0.6**
- **What it does:** Controls how much GPU memory vLLM reserves for caching during inference
- **In simple terms:** Like reserving parking spaces - 0.6 means use 60% of GPU memory for the "KV cache" (memory of previous tokens)
- **Why it matters:** Higher values allow longer sequences and bigger batches but leave less room for other operations
- **Trade-off:** Too high (>0.9) risks out-of-memory errors; too low (<0.5) underutilizes GPU

**max_model_len=4096**
- **What it does:** Sets the maximum total sequence length (input + output tokens)
- **In simple terms:** Like setting a word limit for an essay - no sequence can exceed this length
- **Why it matters:** Protects against memory overflow and ensures consistent behavior
- **Default:** If not set, uses prompt_length + response_length from data config

**mode=async**
- **What it does:** Determines whether inference runs synchronously or asynchronously
- **In simple terms:** 
  - **sync**: Like a restaurant where waiters deliver one order at a time
  - **async**: Like a buffet where multiple orders are processed simultaneously
- **Why it matters:** 
  - **async**: Better throughput, non-blocking generation, ideal for I/O-heavy environments
  - **sync**: Simpler, easier to debug, lower latency for individual requests

**Technical details:** These settings configure the vLLM inference engine used during the rollout phase of training.

**Found in:** `external/rllm/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`

### actor_rollout_ref.actor.ulysses_sequence_parallel_size

**What it does:** Splits long sequences across multiple GPUs for parallel processing, enabling training with sequences that wouldn't fit on a single GPU.

**In simple terms:** Imagine reading a very long book:
- **Without sequence parallelism**: One person reads the entire book (one GPU processes full sequence)
- **With sequence parallelism = 4**: Four people each read 1/4 of the book and share notes (4 GPUs each process part of the sequence)

The name "Ulysses" comes from a DeepSpeed paper comparing it to the epic journey in Homer's Odyssey - processing long sequences is like a long journey that needs to be split into manageable parts.

**How it works:**
1. Sequence of 32K tokens with `ulysses_sequence_parallel_size=8`
2. Each GPU processes 32K/8 = 4K tokens
3. GPUs communicate via all-to-all operations to share attention information
4. Results are gathered to produce the final output

**Why it matters:**
- **Enables long context**: Process 32K, 64K, or even 128K token sequences
- **Memory efficiency**: Each GPU only stores 1/N of the sequence
- **Attention memory**: Reduces quadratic attention memory by factor of N
- **Essential for**: Training with sequences longer than ~8K tokens

**Configuration example:**
```yaml
# For 32K token sequences on 8 GPUs
actor_rollout_ref:
  actor:
    ulysses_sequence_parallel_size: 8  # Split across 8 GPUs
```

**When to use:**
- **Set to 1** (default): For sequences up to 8K tokens
- **Set to 2-4**: For sequences 16K-32K tokens
- **Set to 8+**: For very long sequences (64K+)

**Trade-offs:**
- **Benefits**: Enables long sequences, reduces memory per GPU
- **Costs**: All-to-all communication overhead (10-20% performance impact)
- **Constraint**: Number of attention heads must be divisible by SP size

**Compatibility:**
- Works with FSDP (model parallelism)
- Independent of tensor parallelism
- Supported models: LLaMA, Qwen2, and others with compatible attention

**Technical details:**
- Creates 2D device mesh: [data_parallel, sequence_parallel]
- Uses `gather_seq_scatter_heads` operations for attention
- Requires `num_heads % ulysses_sequence_parallel_size == 0`

**Found in:** `external/rllm/verl/verl/utils/ulysses_utils.py`

### actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu and fsdp_config.param_offload=True

**What is the reference model?**
The reference (ref) model is a frozen copy of your initial model used as an anchor point during training. It prevents your trained model from changing too drastically.

**In simple terms:** Think of the reference model like a photograph of yourself before a makeover. As you change your appearance (train the model), you regularly check the photo to ensure you haven't changed beyond recognition. The KL divergence measures how different you've become from the original photo.

**ref.log_prob_micro_batch_size_per_gpu**
- **What it does:** Controls batch size when computing probabilities with the reference model
- **Why separate:** Reference model only does inference (no training), so can use different batch sizes
- **Typical value:** Often larger than actor batch size since no gradients are computed

**ref.fsdp_config.param_offload=True**
- **What it does:** Keeps reference model weights in CPU memory, loading to GPU only when needed
- **Why different from actor:** 
  - Reference model is read-only (never updated)
  - Only used occasionally to compute KL divergence
  - Actor model needs to stay on GPU for frequent updates
- **Memory benefit:** Saves significant GPU memory since reference model = full model size

**How it's used in training:**
1. Generate text with current policy
2. Compute probabilities with reference model
3. Calculate KL divergence: `current_log_prob - reference_log_prob`
4. Use KL to penalize large deviations from original model

**Technical formula:** `final_reward = task_reward - kl_coefficient * kl_divergence`

**Found in:** `external/rllm/verl/verl/workers/fsdp_workers.py:147`

---

## Algorithm Configuration

### algorithm.gamma=0.99

**What it does:** Controls how much future rewards are discounted when calculating returns.

**In simple terms:** Imagine you have a choice between $100 today or $100 in a year. Most people prefer money now because future money is less certain. Gamma works the same way:
- **gamma = 1.0**: Future rewards are valued equally to immediate rewards (no discounting)
- **gamma = 0.99**: Each step into the future multiplies the reward by 0.99 (1% discount per step)
- **gamma = 0.9**: Heavy discounting - rewards 10 steps away are worth only 35% of immediate rewards

**Why it matters:**
- **Higher gamma (0.99-1.0)**: Model considers long-term consequences, good for tasks requiring planning
- **Lower gamma (0.9-0.95)**: Model focuses on immediate rewards, good for tasks with uncertain futures
- **Too high**: Training can be unstable if rewards are sparse
- **Too low**: Model becomes myopic, ignoring valuable future outcomes

**For language models:** Usually set high (0.99-1.0) because generating coherent text requires considering the full sequence.

### algorithm.lam=0.95

**What it does:** Controls the bias-variance tradeoff in Generalized Advantage Estimation (GAE).

**In simple terms:** Think of lam (lambda) like a slider between two methods of estimating how good an action was:
- **lam = 0**: Like judging a chef by just their last dish (low variance, high bias)
- **lam = 1**: Like judging a chef by their entire career (high variance, low bias)  
- **lam = 0.95**: A balanced compromise - considering recent history but not everything

**Technical details:**
- Used in GAE to compute advantages (how much better an action was than expected)
- Interpolates between TD (Temporal Difference) and Monte Carlo methods
- Formula involves exponentially weighted sum of TD residuals

**Why these values:**
- **0.95-0.97**: Standard range that balances stability and accuracy
- **Higher**: More accurate but noisier training
- **Lower**: More stable but potentially biased estimates

**Found in:** `external/rllm/verl/verl/trainer/ppo/core_algos.py:67`

---

## Critic Configuration

### critic.optim.lr=$CRITIC_LR

**What it does:** Sets the learning rate for the critic (value function) network optimizer.

**In simple terms:** The critic is like a judge that estimates how good each state is. While the actor (policy) decides what actions to take, the critic evaluates whether those actions led to good or bad outcomes. The critic needs to learn faster to provide accurate feedback to the actor.

**Why typically higher than actor LR (1e-5 vs 1e-6):**
- **Faster convergence needed**: The critic must quickly learn accurate value estimates for the actor to use
- **Lower risk**: Updating value estimates wrong is less catastrophic than bad policy updates
- **No action consequences**: The critic doesn't directly affect actions, just provides evaluations

### critic.model.path=$MODEL_PATH

**What it does:** Specifies the base model to use for the critic network.

**In simple terms:** Both actor and critic start from the same pre-trained language model but add different "heads":
- **Actor**: Adds a head that predicts next tokens (actions)
- **Critic**: Adds a head that outputs a single number (value estimate)

Think of it like two students who attended the same school (base model) but specialized in different subjects (policy vs value estimation).

### critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU

**What it does:** Controls batch size for critic updates, similar to actor batch sizing.

**Why it matters:**
- **Can be different from actor**: Critic often handles larger batches since it's just computing values, not generating text
- **Memory efficiency**: Larger micro-batches = faster training but more memory use
- **Gradient accumulation**: Works the same as actor - accumulates gradients across micro-batches

**The critic's role in training:**
1. **Estimates state values**: "How good is this partial response?"
2. **Computes advantages**: "Was this action better or worse than expected?"
3. **Provides baselines**: Reduces variance in policy gradient estimates
4. **Enables stable learning**: Critical for PPO's stability guarantees

**Technical details:** The critic minimizes MSE between its predictions and actual returns, using value clipping to prevent large updates.

**Found in:** `external/rllm/verl/verl/workers/critic/dp_critic.py`

---

## Trainer Configuration

### trainer.save_freq=10

**What it does:** Controls how often model checkpoints are saved during training (every N steps).

**In simple terms:** Like auto-saving your work in a document editor. With `save_freq=10`, the model is saved every 10 training steps, allowing you to resume from checkpoints if training crashes or to use intermediate versions.

**Why it matters:**
- **Frequent saves (low value)**: More checkpoints but uses more disk space
- **Infrequent saves (high value)**: Less disk usage but risk losing more progress
- **-1**: Disables automatic saving

### trainer.test_freq=5

**What it does:** Controls how often validation/testing is performed during training (every N steps).

**In simple terms:** Like taking practice exams while studying. Every 5 steps, the model is tested on validation data to see how well it's learning without "cheating" by looking at training data.

**Why it matters:**
- **Frequent testing**: Better monitoring but slows training
- **Infrequent testing**: Faster training but less visibility into progress
- **-1**: Disables periodic testing

### trainer.val_before_train=True

**What it does:** Whether to run validation before starting any training.

**In simple terms:** Like taking a diagnostic test before a course starts to establish a baseline. This helps you see how much the model improves from its starting point.

**Why use it:**
- Establishes baseline metrics
- Ensures validation pipeline works before expensive training
- Helps detect if pre-trained model already performs well

### trainer.rejection_sample=True and rejection_sample_multiplier=2

**What it does:** Filters out training samples that don't provide useful learning signals.

**In simple terms:** Imagine teaching someone to play darts:
- If they miss the board entirely every time (all failures), you can't teach them what works
- If they hit bullseye every time (all successes), they don't need to learn
- You want mixed results to show what works and what doesn't

**How it works:**
1. Groups responses by prompt (e.g., 8 attempts at the same task)
2. Rejects groups where all attempts failed or all succeeded
3. Keeps only groups with mixed outcomes for learning

**rejection_sample_multiplier=2:**
- Doubles the batch size to compensate for rejected samples
- If you want 100 samples but expect 50% rejection, sample 200

**Why critical for GRPO:** Group Relative Policy Optimization needs variance within groups to compute meaningful advantages. Without mixed outcomes, there's no learning signal.

**Found in:** `external/rllm/rllm/trainer/verl/agent_ppo_trainer.py:213`

---

## Framework Architecture

### How agent.name and env.name are mapped and injected

**What it does:** Maps string names to actual Python classes for agents and environments.

**In simple terms:** Think of it like a restaurant menu - instead of describing the whole recipe (importing and instantiating classes), you just say "I'll have the pasta" (use name "terminal_bench").

**The mapping system:**
1. **Central registry**: `external/rllm/rllm/trainer/env_agent_mappings.py` contains two dictionaries
2. **ENV_CLASS_MAPPING**: Maps environment names → environment classes
3. **AGENT_CLASS_MAPPING**: Maps agent names → agent classes

**How it works:**
```python
# In train_agent_ppo.py
env_class = ENV_CLASS_MAPPING[config.env.name]  # "terminal_bench" → DockerIsolatedEnv
agent_class = AGENT_CLASS_MAPPING[config.agent.name]  # "terminal_bench_agent" → TerminalBenchAgent
```

**Two ways to add custom agents/environments:**

**Option 1: Modify the mapping file** (what the comment suggests)
```python
# Add to env_agent_mappings.py
ENV_CLASSES = {
    "terminal_bench": safe_import("src.tbench_rllm.docker_env", "DockerIsolatedEnv"),
    # ... other environments
}
```

**Option 2: Use AgentTrainer directly** (cleaner for external projects)
```python
from rllm.trainer.agent_trainer import AgentTrainer
trainer = AgentTrainer(
    agent_class=YourCustomAgent,  # Pass class directly
    env_class=YourCustomEnv,      # No string mapping needed
    config=config,
)
```

**Why this design:**
- **Simplicity**: Config files can use simple strings instead of Python imports
- **Flexibility**: Easy to switch agents/environments by changing config
- **Limitation**: No plugin system - must modify core files or use AgentTrainer

**Naming conventions:**
- Environments: lowercase with underscores (`terminal_bench`, `browsergym`)
- Agents: lowercase with underscores, often ending in `_agent`

**Found in:** `external/rllm/rllm/trainer/env_agent_mappings.py`

---

## Agent and Environment Max Steps

### Why both agent and env have max_steps?

**What it does:** Provides two-level control over episode length - a hard limit from the execution engine and optional environment-specific limits.

**In simple terms:** Think of it like a timed exam with multiple sections:
- **Agent max_steps**: The overall exam time limit (e.g., 2 hours total)
- **Environment max_steps**: Time limit for a specific section (e.g., 30 minutes for essay)

You must finish within the overall limit, but sections might end early based on their own rules.

**The two parameters:**

**agent.max_steps** (Primary control)
- Set via `config.agent.max_steps` (default: 5)
- Controls the main execution loop in `AgentExecutionEngine`
- Hard upper limit - episode always ends when this is reached
- Passed to agent via info dict so it knows how many steps remain

**env.env_args.container_config.max_steps** (Environment-specific)
- Used by specific environments for their internal logic
- Can cause early termination by returning `done=True`
- Cannot extend beyond agent.max_steps
- Examples:
  - FrozenLake: Ensures generated mazes are solvable within limit
  - ToolEnvironment: Terminates when reaching its step limit
  - Terminal/Docker: Limits commands executed in container

**How they interact:**
1. Engine starts with `agent.max_steps` as the hard limit
2. At each step, engine checks if step count reached `agent.max_steps`
3. Environment can return `done=True` to end early
4. Termination reason is tracked:
   - `"MAX_STEPS"`: Engine limit reached
   - `"ENV_DONE"`: Environment decided to terminate

**Why this design:**
- **Separation of concerns**: Engine controls trajectories, environments control domain logic
- **Safety**: Prevents runaway episodes even if environment misbehaves
- **Flexibility**: Different environments can have different step semantics
- **Information**: Agents can adapt behavior knowing steps remaining

**Best practice:** Set `agent.max_steps` as your safety limit and use environment max_steps for domain-specific constraints.

**Found in:** `external/rllm/rllm/engine/agent_execution_engine.py:272`

---

## Summary

This document explained RLLM configuration parameters for engineers without ML background. Key takeaways:

1. **Performance optimizations** (use_shm, use_remove_padding, gradient_checkpointing) trade memory for speed
2. **Batch sizes** control the hierarchical batching system for distributed training
3. **Learning rates** are carefully tuned with critic learning faster than actor
4. **Algorithm parameters** (gamma, lam) control credit assignment and variance reduction
5. **Framework design** uses string mappings for flexibility but requires core file modification
6. **Two-level control** patterns (like max_steps) provide both safety and flexibility