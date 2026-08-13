# PPO Hyperparameters — Meaning and Influence on the Training Loop

## 1. Big Picture: Where the Hyperparameters Act

A PPO training loop can be simplified as:

```text
┌─────────────────────┐
│  Environment        │
│  state              │
└──────────┬──────────┘
           ↓
      State vector
           ↓
┌─────────────────────┐
│  Actor-Critic       │
│  neural network     │
└──────────┬──────────┘
           ↓
     Action / Value
           ↓
┌─────────────────────┐
│  Environment        │
│  reward + next state│
└──────────┬──────────┘
           ↓
     Collect rollout
           ↓
┌─────────────────────┐
│  GAE / Advantages   │ ← gamma, lambda
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  PPO optimization   │ ← clip_epsilon, cv, ce
│  over n_epochs      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Adam optimizer     │ ← learning rate
└──────────┬──────────┘
           ↓
       New policy
           │
           └──────────────→ repeat
```

Therefore, different hyperparameters affect **different parts of the loop**.

---

# 2. Neural Network Hyperparameters

## `hidden_dim`

### Meaning

Number of neurons in each hidden layer.

```python
hidden_dim=64
```

### Influence on training

It controls the **capacity of the neural network**.

```text
hidden_dim ↑
    ↓
more parameters
    ↓
more representational capacity
    ↓
more computation
    ↓
potentially harder/slower optimization
```

### Practical range

| Range     | Meaning                       |
| --------- | ----------------------------- |
| 8–16      | Very small                    |
| 16–32     | Small                         |
| **32–64** | **Good for your environment** |
| 64–128    | Large                         |
| 128+      | Probably unnecessary          |

### Your problem

Your state vector has only 6 features and there are only 4 actions.

Recommended:

```python
hidden_dim=32
```

or

```python
hidden_dim=64
```

---

# `n_layers`

### Meaning

Number of hidden layers in the Actor-Critic network.

### Influence on training

Controls the **depth/complexity** of the network.

```text
n_layers ↑
    ↓
more complex function
    ↓
more parameters
    ↓
potentially harder optimization
```

Too many layers can make learning unnecessarily difficult for a simple environment.

### Practical range

| Range   | Meaning             |
| ------- | ------------------- |
| 1       | Very simple         |
| **2–3** | **Recommended**     |
| 4–5     | More complex        |
| 5–10    | Usually unnecessary |
| 10+     | Very deep           |

Your:

```python
n_layers=10
```

is probably excessive for this environment.

Recommended:

```python
n_layers=2
```

---

# 3. Optimizer

## `lr` — Learning Rate

### Meaning

Controls the size of the neural-network parameter updates.

### Where it acts

At the very end of the PPO update:

```text
loss
 ↓
backpropagation
 ↓
gradients
 ↓
Adam
 ↓
parameter update ← lr
```

### Influence on training

```text
lr ↑
 ↓
larger updates
 ↓
faster learning
 ↓
higher risk of instability
```

```text
lr ↓
 ↓
smaller updates
 ↓
slower learning
 ↓
usually more stable
```

### Practical range

| Range         | Meaning        |
| ------------- | -------------- |
| < 1e-5        | Very slow      |
| 1e-5–1e-4     | Slow/stable    |
| **1e-4–5e-4** | **Good range** |
| 5e-4–1e-3     | Aggressive     |
| > 1e-3        | Often unstable |

Your:

```python
lr=3e-4
```

is a good starting point.

---

# 4. PPO Parameters

## `clip_epsilon`

### Meaning

Limits how much the new policy can differ from the old policy during a PPO update.

### Where it acts

Inside the PPO policy-loss calculation.

Conceptually:

```text
old policy
     ↓
collect actions
     ↓
new policy
     ↓
compare new vs old
     ↓
PPO clipping ← clip_epsilon
```

### Influence on training

```text
clip_epsilon ↓
    ↓
smaller policy changes
    ↓
more conservative/stable training
```

```text
clip_epsilon ↑
    ↓
larger policy changes
    ↓
potentially faster but less stable learning
```

### Practical range

| Range       | Meaning           |
| ----------- | ----------------- |
| 0.05–0.1    | Very conservative |
| 0.1–0.2     | Conservative      |
| **0.2–0.3** | **Common**        |
| 0.3–0.4     | Aggressive        |
| > 0.4       | Very aggressive   |

Your:

```python
clip_epsilon=0.2
```

is a good default.

---

# `cv` — Value-Loss Coefficient

### Meaning

Controls how strongly the critic/value-function loss contributes to the total PPO loss.

The total loss is conceptually similar to:

```text
total loss =
    policy loss
    + cv × value loss
    - ce × entropy
```

### Where it acts

During PPO optimization.

```text
rollout
   ↓
calculate returns
   ↓
critic predicts values
   ↓
calculate value loss
   ↓
cv controls its importance
```

### Influence on training

```text
cv ↑
 ↓
critic/value function receives more importance
 ↓
better emphasis on value estimation
```

But if it becomes too large, the critic can dominate the optimization.

### Practical range

| Range       | Meaning           |
| ----------- | ----------------- |
| 0           | No value loss     |
| 0.1–0.25    | Weak              |
| 0.25–0.5    | Moderate          |
| **0.5–1.0** | **Strong/common** |
| > 1.0       | Very strong       |

Your:

```python
cv=0.5
```

is reasonable.

---

# `ce` — Entropy Coefficient

### Meaning

Controls how strongly PPO encourages exploration.

Entropy measures how random/uncertain the policy is.

### Where it acts

During the PPO loss calculation.

```text
policy
   ↓
action probabilities
   ↓
entropy
   ↓
entropy bonus ← ce
```

### Influence on training

```text
ce ↑
 ↓
more exploration
 ↓
less premature convergence
```

```text
ce ↓
 ↓
less exploration
 ↓
more exploitation of learned behavior
```

### Practical range

| Range          | Meaning                         |
| -------------- | ------------------------------- |
| 0              | No explicit entropy exploration |
| 0.001–0.005    | Very weak                       |
| **0.005–0.02** | **Moderate**                    |
| 0.02–0.05      | Strong                          |
| > 0.05         | Very strong                     |

Your:

```python
ce=0.01
```

is a good starting point.

---

# `gamma` — Discount Factor

### Meaning

Controls how much the agent values future rewards.

```text
current reward ← important
future reward  ← gamma controls importance
```

### Where it acts

During return/value and advantage calculation.

For example:

```text
reward now = 0
reward next step = 0
reward later = +10
```

`gamma` determines how much that future `+10` influences the current state.

### Influence on training

```text
gamma ↓
 ↓
focus on immediate rewards
 ↓
short-term behavior
```

```text
gamma ↑
 ↓
future rewards matter more
 ↓
long-term planning
```

### Range

`0 ≤ gamma ≤ 1`

| Range         | Behavior               |
| ------------- | ---------------------- |
| 0–0.5         | Immediate rewards      |
| 0.5–0.8       | Short-term planning    |
| 0.8–0.95      | Medium-term            |
| **0.95–0.99** | **Long-term planning** |
| 0.99–1.0      | Very long-term         |

Your:

```python
gamma=0.99
```

is appropriate for navigation.

---

# `lam` — GAE Lambda

### Meaning

Controls how advantages are estimated using **Generalized Advantage Estimation (GAE)**.

### Where it acts

After collecting a rollout:

```text
rollout
   ↓
rewards + values
   ↓
GAE
   ↓
advantages ← gamma + lambda
   ↓
PPO update
```

### Influence on training

`lambda` controls the **bias/variance trade-off**.

```text
lambda ↓
 ↓
less variance
more bias
```

```text
lambda ↑
 ↓
less bias
more variance
```

### Practical range

| Range         | Meaning                      |
| ------------- | ---------------------------- |
| 0–0.5         | Strong smoothing / more bias |
| 0.5–0.8       | Conservative                 |
| 0.8–0.95      | Good general range           |
| **0.95–0.99** | **Common for PPO**           |
| 0.99–1.0      | High variance potential      |

Your:

```python
lam=0.95
```

is a standard choice.

---

# `n_epochs`

### Meaning

Number of times PPO reuses the collected rollout to update the network.

### Where it acts

```text
collect rollout
      ↓
PPO update
      ↓
epoch 1
epoch 2
epoch 3
...
epoch n
```

### Influence on training

```text
n_epochs ↑
 ↓
more optimization from each rollout
 ↓
more efficient use of data
 ↓
risk of over-updating / instability
```

```text
n_epochs ↓
 ↓
less reuse
 ↓
more conservative updates
```

### Practical range

| Range    | Meaning           |
| -------- | ----------------- |
| 1–2      | Very little reuse |
| 3–5      | Moderate          |
| **5–10** | **Common**        |
| 10–20    | Aggressive        |
| >20      | Usually excessive |

Your:

```python
n_epochs=8
```

is reasonable.

For initial experiments, `4–8` is a good range.

---

# 5. Rollout / Data Collection

## `rollout_length`

### Meaning

Number of environment transitions collected before PPO performs an update.

### Where it acts

```text
Environment
    ↓
step 1
step 2
step 3
...
step 64
    ↓
PPO update
```

### Influence on training

```text
rollout_length ↓
 ↓
more frequent updates
 ↓
less data per update
 ↓
noisier estimates
```

```text
rollout_length ↑
 ↓
less frequent updates
 ↓
more data per update
 ↓
potentially more stable estimates
```

### Practical range

| Range     | Meaning                         |
| --------- | ------------------------------- |
| 16–32     | Frequent/noisy                  |
| **32–64** | **Good for small environments** |
| 64–128    | More stable                     |
| 128–512   | Large                           |
| >512      | Very large                      |

Your:

```python
rollout_length=64
```

is a good starting point.

---

# 6. Episode Parameters

## `episodes`

### Meaning

Number of complete episodes used for training.

### Where it acts

It determines how long the outer training loop runs.

```text
Episode 1
   ↓
Episode 2
   ↓
Episode 3
   ↓
...
Episode 5000
```

### Influence on training

```text
episodes ↑
 ↓
more experience
 ↓
more training
 ↓
potentially better policy
```

But eventually additional episodes may produce little improvement.

### Practical range

| Range            | Meaning                   |
| ---------------- | ------------------------- |
| 100–500          | Quick test                |
| 500–2,000        | Small training            |
| **2,000–10,000** | **Good experiment range** |
| 10,000+          | Extensive training        |

Your:

```python
episodes=5000
```

is reasonable.

---

# `steps`

### Meaning

Maximum number of environment steps allowed in one episode.

### Where it acts

```text
Episode
 ↓
step 1
step 2
step 3
...
maximum steps
 ↓
episode terminates
```

### Influence on training

If too small:

```text
steps ↓
 ↓
episode terminates before reaching goal
 ↓
less useful experience
```

If too large:

```text
steps ↑
 ↓
agent can wander for a long time
 ↓
training becomes inefficient
```

### Practical range for your grid

| Range      | Meaning                |
| ---------- | ---------------------- |
| 10–50      | Short                  |
| **50–200** | **Likely appropriate** |
| 200–500    | Long                   |
| 500–5,000  | Very long              |
| 5,000+     | Usually excessive      |

Your current:

```python
steps=5000
```

is probably much larger than necessary for a 6×9 environment.

Try:

```python
steps=100
```

or:

```python
steps=200
```

---

# `random_start`

### Meaning

Determines whether the agent starts from randomly selected states.

### Influence on training

With:

```python
random_start=False
```

the agent can specialize in one route:

```text
START → A → B → C → GOAL
```

With:

```python
random_start=True
```

it experiences many different situations:

```text
state A → ...
state B → ...
state C → ...
state D → ...
```

This generally improves **generalization**.

For your environment:

```python
random_start=True
```

is recommended.

---

# 7. Complete Training Loop

Putting everything together:

```text
             ┌───────────────────────┐
             │      Start episode    │
             └───────────┬───────────┘
                         ↓
                  Random start?
                    ↑          ↑
                  yes          no
                    ↓          ↓
             ┌───────────────────────┐
             │       State          │
             └───────────┬───────────┘
                         ↓
                 State → Network
                         ↓
                  Action + Value
                         ↓
                 Environment step
                         ↓
                  Reward + State
                         ↓
             Collect rollout_length
                    transitions
                         ↓
             ┌───────────────────────┐
             │   Calculate returns   │
             │   and advantages      │
             │                       │
             │ gamma + lambda        │
             └───────────┬───────────┘
                         ↓
             ┌───────────────────────┐
             │      PPO update       │
             │                       │
             │ clip_epsilon          │
             │ cv                    │
             │ ce                    │
             └───────────┬───────────┘
                         ↓
                  Repeat n_epochs
                         ↓
             ┌───────────────────────┐
             │      Adam update      │
             │                       │
             │ learning rate         │
             └───────────┬───────────┘
                         ↓
                  New neural network
                         ↓
                  Continue training
                         ↓
                 Next rollout/episode
```

# 8. Recommended Configuration

For your current 6×9 grid-world:

```python
net1 = ActorCritic(
    n_features1,
    n_actions,
    hidden_dim=64,
    n_layers=2
)

optimizer = torch.optim.Adam(
    net1.parameters(),
    lr=3e-4
)

ppo1 = PPO(
    optimizer=optimizer,
    clip_epsilon=0.2,
    cv=0.5,
    ce=0.01,
    gamma=0.99,
    lam=0.95,
    n_epochs=4
)

training(
    env=env1,
    agent=agent_carter,
    episodes=5000,
    steps=100,
    rollout_length=64,
    random_start=True
)
```

# 9. Quick Reference — Parameter → Training Effect

| Parameter        | Main role                | Increase →                   | Decrease →                   |
| ---------------- | ------------------------ | ---------------------------- | ---------------------------- |
| `hidden_dim`     | Network capacity         | More capacity                | Simpler/faster               |
| `n_layers`       | Network depth            | More complexity              | Simpler optimization         |
| `lr`             | Update size              | Faster/aggressive            | Slower/stable                |
| `clip_epsilon`   | Policy change            | Larger updates               | Conservative updates         |
| `cv`             | Critic importance        | Stronger value learning      | Less critic influence        |
| `ce`             | Exploration              | More exploration             | More exploitation            |
| `gamma`          | Future rewards           | More long-term planning      | More short-term              |
| `lam`            | Advantage estimation     | Lower bias / higher variance | Higher bias / lower variance |
| `n_epochs`       | Rollout reuse            | More optimization            | Less optimization            |
| `rollout_length` | Data per update          | Larger/stabler batches       | Frequent/noisier updates     |
| `episodes`       | Total training           | More experience              | Less training                |
| `steps`          | Episode length           | Longer episodes              | Shorter episodes             |
| `random_start`   | Starting-state diversity | More generalization          | More specialization          |

# 10. What to Tune First

For this environment, tune in roughly this order:

1. **`lr`** — learning speed/stability
2. **`gamma`** — ability to plan toward distant goals
3. **`ce`** — exploration vs exploitation
4. **`n_epochs`** — amount of learning per rollout
5. **`clip_epsilon`** — policy-update stability
6. **`rollout_length`** — quality/frequency of updates
7. **Network size** — only if the network is clearly too small/large

Change **one parameter at a time** when comparing experiments. Otherwise it becomes difficult to determine which parameter caused the change in performance.

