---
title: "DQN, Rollout, Greedy, Random: Four Ways to Make Decisions in Reinforcement Learning"
date: 2026-04-20
draft: false
tags: ["reinforcement-learning", "dqn", "rollout", "gymnasium", "python"]
categories: ["Machine Learning"]
---

When people hear **reinforcement learning (RL)**, they often think about a neural network playing Atari, a robot learning to walk, or an agent landing a lunar lander.

But in real research code, especially in scientific experiment design, we often mix several decision-making methods together:

- `RANDOM`: try something randomly.
- `GREEDY`: choose the best-looking action right now.
- `ROLLOUT`: simulate several possible futures, then choose.
- `DQN`: learn a value function from experience.

These four methods can appear in the same project, but they are not equally “RL”. In my current code, **DQN is the most standard RL algorithm**. Rollout is more like planning / simulation-based decision making. Greedy and Random are useful baselines.

This post explains them with one simple example: **LunarLander**.

We will use `gymnasium`, the maintained successor of OpenAI Gym. The environment is `LunarLander-v3`: the agent sees an 8-dimensional state and chooses one of 4 discrete actions.

---

## The mental model: agent, state, action, reward

Almost all RL problems can be described with the same loop:

```text
state -> choose action -> environment changes -> reward -> next state
```

For LunarLander, imagine a small lander trying to land safely between two flags.

At every step:

- **State**: where the lander is, how fast it is moving, its angle, and whether the legs touch the ground.
- **Action**: do nothing, fire left engine, fire main engine, or fire right engine.
- **Reward**: higher if the lander moves toward a safe landing; lower if it crashes or wastes fuel.
- **Goal**: maximize total reward over the whole episode.

The key difficulty is that the best action is not always the action that looks best immediately.

For example, firing the main engine now may waste fuel, but it may prevent a crash later. This is why RL cares about **long-term return**, not only immediate reward.

---

## 1. Random: the baseline we should not skip

`RANDOM` is the simplest method:

```text
Pick any valid action randomly.
```

It does not learn. It does not think. It does not look ahead.

So why use it?

Because it gives us a baseline. If a fancy RL method is not better than random, something is probably wrong.

In LunarLander, Random means:

```text
Randomly fire engines.
```

It is usually bad, but it is honest.

---

## 2. Greedy: choose what looks best right now

`GREEDY` means:

```text
At the current state, evaluate the possible actions.
Choose the action that looks best immediately.
```

This is simple and often surprisingly useful.

For example, in LunarLander:

- if the lander is falling too fast, fire the main engine;
- if it is tilted, fire a side engine;
- otherwise do nothing.

This is not really RL because it does not learn from experience and does not optimize long-term return. It is a hand-written or model-based rule.

The weakness is clear: Greedy sees only one step ahead.

Sometimes the best first move is not the best-looking immediate move.

---

## 3. Rollout: try several futures before choosing

`ROLLOUT` improves on Greedy by asking:

```text
What may happen after this action?
```

Instead of only checking the next state, Rollout simulates short future trajectories.

A simple rollout procedure is:

```text
For each candidate action:
    simulate N possible futures
    estimate the average total reward
Choose the action with the best average future reward
```

This is still not the same as DQN.

Rollout usually does not train a neural network. It uses simulation directly at decision time. That makes it intuitive but expensive.

In LunarLander, Rollout means:

```text
Before firing an engine, simulate a few short future landings.
Pick the engine action whose simulated futures look best.
```

Rollout is useful when simulation is cheap. It becomes hard when each simulation is expensive.

---

## 4. DQN: learn a Q-function

`DQN` stands for **Deep Q-Network**.

The idea is to train a neural network:

```text
Q(state, action) -> expected future return
```

So instead of simulating many futures every time, DQN learns a reusable function.

At decision time:

```text
Compute Q(state, all actions)
Choose the action with the largest Q-value
```

The important part is that Q-value is not the immediate reward. It is the estimated long-term value.

A DQN usually has a few core components:

1. **Q network**: predicts Q-values.
2. **Replay buffer**: stores old experience `(state, action, reward, next_state, done)`.
3. **Target network**: a slower-moving copy of the Q network for stable training.
4. **Epsilon-greedy exploration**: sometimes act randomly to explore.

In my code, this is why DQN is the real RL algorithm among the four methods.

It interacts with the environment, stores transitions, samples batches, and updates a neural network using the Bellman target:

```text
target = reward + gamma * max_a Q_target(next_state, a)
```

The intuition:

```text
Good actions are actions that lead to good future states.
```

---

## Same environment, different decision styles

Here is the high-level comparison:

| Method  |          Learns? |           Looks ahead? | Fast at inference? | Main idea                      |
| ------- | ---------------: | ---------------------: | -----------------: | ------------------------------ |
| Random  |               No |                     No |                Yes | Try anything                   |
| Greedy  |               No |               One step |                Yes | Choose best-looking action now |
| Rollout | No / not usually |                    Yes |                 No | Simulate possible futures      |
| DQN     |              Yes | Learns long-term value | Yes after training | Learn `Q(state, action)`       |

A useful way to remember it:

```text
Random:  no brain
Greedy:  short-term brain
Rollout: imagination by simulation
DQN:     learned long-term intuition
```

---

## Runnable example: LunarLander with Random, Greedy, Rollout, and DQN

Install dependencies:

```bash
pip install "gymnasium[box2d]" torch numpy
```

Then save the following file as:

```bash
lunar_rl_compare.py
```

```python
"""
Compare four decision styles on Gymnasium LunarLander-v3:

1. Random baseline
2. Greedy heuristic
3. Rollout planning with short simulated futures
4. DQN learning from replay buffer

Install:
    pip install "gymnasium[box2d]" torch numpy

Examples:
    python lunar_rl_compare.py --mode random --episodes 5
    python lunar_rl_compare.py --mode greedy --episodes 5
    python lunar_rl_compare.py --mode rollout --episodes 2 --rollout-horizon 40 --rollouts-per-action 3
    python lunar_rl_compare.py --mode dqn_train --episodes 300
    python lunar_rl_compare.py --mode dqn_eval --model-path dqn_lunarlander.pt --episodes 5
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ENV_ID = "LunarLander-v3"


# ============================================================
# 1. Random and Greedy policies
# ============================================================

def greedy_heuristic_action(obs: np.ndarray) -> int:
    """
    A tiny human-written heuristic for LunarLander.

    Observation roughly contains:
      x, y, x_velocity, y_velocity, angle, angular_velocity,
      left_leg_contact, right_leg_contact

    Actions:
      0 = do nothing
      1 = fire left orientation engine
      2 = fire main engine
      3 = fire right orientation engine

    This is not an optimal controller. It is intentionally simple so that
    GREEDY is easy to understand: choose what looks good right now.
    """
    x, y, vx, vy, angle, angular_v, left_contact, right_contact = obs

    # If we are falling too quickly or are close to the ground, fire main engine.
    if vy < -0.35 or y < 0.45:
        return 2

    # Try to point the lander toward the center and keep it upright.
    target_angle = np.clip(0.5 * x + 1.0 * vx, -0.4, 0.4)
    angle_error = target_angle - angle

    if angle_error > 0.08:
        return 3
    if angle_error < -0.08:
        return 1

    return 0


# ============================================================
# 2. Rollout planning
# ============================================================

class RolloutPlanner:
    """
    A simple rollout planner.

    Gymnasium environments usually do not expose a clean public "clone current
    simulator state" API. To keep this file runnable, we reconstruct the current
    state by resetting with the same seed and replaying the action history.

    This is slow, but it shows the key idea of rollout:
      for each candidate action,
      simulate several short futures,
      choose the action with the best average return.
    """

    def __init__(
        self,
        env_id: str = ENV_ID,
        horizon: int = 30,
        rollouts_per_action: int = 3,
        gamma: float = 0.99,
        seed: int = 0,
    ):
        self.env_id = env_id
        self.horizon = horizon
        self.rollouts_per_action = rollouts_per_action
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.episode_seed = seed
        self.history: list[int] = []

    def reset(self, episode_seed: int) -> None:
        self.episode_seed = int(episode_seed)
        self.history = []

    def observe(self, action: int) -> None:
        self.history.append(int(action))

    def _env_at_current_state(self):
        sim_env = gym.make(self.env_id)
        obs, _ = sim_env.reset(seed=self.episode_seed)
        done = False

        for action in self.history:
            obs, _, terminated, truncated, _ = sim_env.step(action)
            done = terminated or truncated
            if done:
                break

        return sim_env, obs, done

    def act(self, obs: np.ndarray) -> int:
        action_values = []
        n_actions = 4

        for first_action in range(n_actions):
            returns = []

            for _ in range(self.rollouts_per_action):
                sim_env, sim_obs, done = self._env_at_current_state()
                total_return = 0.0
                discount = 1.0

                if not done:
                    sim_obs, reward, terminated, truncated, _ = sim_env.step(first_action)
                    done = terminated or truncated
                    total_return += reward

                    for _ in range(self.horizon - 1):
                        if done:
                            break

                        # Use mostly the greedy heuristic, with a bit of randomness
                        # so different rollouts explore different futures.
                        if self.rng.random() < 0.75:
                            next_action = greedy_heuristic_action(sim_obs)
                        else:
                            next_action = int(self.rng.integers(0, n_actions))

                        sim_obs, reward, terminated, truncated, _ = sim_env.step(next_action)
                        done = terminated or truncated
                        discount *= self.gamma
                        total_return += discount * reward

                sim_env.close()
                returns.append(total_return)

            action_values.append(float(np.mean(returns)))

        return int(np.argmax(action_values))


# ============================================================
# 3. DQN components
# ============================================================

class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(s2, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


def choose_dqn_action(
    qnet: QNet,
    obs: np.ndarray,
    eps: float,
    n_actions: int,
    device: torch.device,
) -> int:
    if random.random() < eps:
        return random.randrange(n_actions)

    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = qnet(x).squeeze(0)
        return int(torch.argmax(q_values).item())


def dqn_update(
    qnet: QNet,
    target_qnet: QNet,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    s, a, r, s2, done = replay.sample(batch_size)

    s_t = torch.tensor(s, dtype=torch.float32, device=device)
    a_t = torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1)
    r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
    s2_t = torch.tensor(s2, dtype=torch.float32, device=device)
    done_t = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(1)

    q_sa = qnet(s_t).gather(1, a_t)

    with torch.no_grad():
        max_q_next = target_qnet(s2_t).max(dim=1, keepdim=True)[0]
        target = r_t + gamma * (1.0 - done_t) * max_q_next

    loss = nn.functional.smooth_l1_loss(q_sa, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


@dataclass
class DQNConfig:
    episodes: int = 300
    max_steps: int = 1000
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 100_000
    learning_starts: int = 2_000
    target_update_every: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    seed: int = 0
    model_path: str = "dqn_lunarlander.pt"


def train_dqn(cfg: DQNConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(ENV_ID)
    env.action_space.seed(cfg.seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    qnet = QNet(obs_dim, n_actions).to(device)
    target_qnet = QNet(obs_dim, n_actions).to(device)
    target_qnet.load_state_dict(qnet.state_dict())

    optimizer = optim.Adam(qnet.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)

    global_step = 0
    recent_returns: Deque[float] = deque(maxlen=20)

    for ep in range(cfg.episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        ep_return = 0.0

        for _ in range(cfg.max_steps):
            frac = min(1.0, global_step / cfg.eps_decay_steps)
            eps = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

            action = choose_dqn_action(qnet, obs, eps, n_actions, device)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_return += reward
            global_step += 1

            if len(replay) >= cfg.learning_starts:
                loss = dqn_update(
                    qnet=qnet,
                    target_qnet=target_qnet,
                    optimizer=optimizer,
                    replay=replay,
                    batch_size=cfg.batch_size,
                    gamma=cfg.gamma,
                    device=device,
                )

                if global_step % cfg.target_update_every == 0:
                    target_qnet.load_state_dict(qnet.state_dict())

            if done:
                break

        recent_returns.append(ep_return)

        if (ep + 1) % 10 == 0:
            avg_return = np.mean(recent_returns)
            print(f"episode={ep+1:4d}  return={ep_return:8.2f}  avg20={avg_return:8.2f}  eps={eps:.3f}")

    torch.save(qnet.state_dict(), cfg.model_path)
    env.close()
    print(f"Saved DQN model to {cfg.model_path}")


def evaluate_policy(args: argparse.Namespace) -> None:
    render_mode = "human" if args.render else None
    env = gym.make(ENV_ID, render_mode=render_mode)
    env.action_space.seed(args.seed)

    qnet = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "dqn_eval":
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        qnet = QNet(obs_dim, n_actions).to(device)
        qnet.load_state_dict(torch.load(args.model_path, map_location=device))
        qnet.eval()

    planner = RolloutPlanner(
        horizon=args.rollout_horizon,
        rollouts_per_action=args.rollouts_per_action,
        seed=args.seed,
    )

    returns = []

    for ep in range(args.episodes):
        episode_seed = args.seed + ep
        obs, _ = env.reset(seed=episode_seed)
        planner.reset(episode_seed)
        ep_return = 0.0

        for _ in range(args.max_steps):
            if args.mode == "random":
                action = env.action_space.sample()
            elif args.mode == "greedy":
                action = greedy_heuristic_action(obs)
            elif args.mode == "rollout":
                action = planner.act(obs)
            elif args.mode == "dqn_eval":
                action = choose_dqn_action(qnet, obs, eps=0.0, n_actions=env.action_space.n, device=device)
            else:
                raise ValueError(f"Unknown eval mode: {args.mode}")

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

            if args.mode == "rollout":
                planner.observe(action)

            obs = next_obs
            if done:
                break

        returns.append(ep_return)
        print(f"episode={ep+1:3d}  return={ep_return:8.2f}")

    env.close()
    print(f"mean_return={np.mean(returns):.2f}  std={np.std(returns):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "greedy", "rollout", "dqn_train", "dqn_eval"], required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")

    # Rollout options
    parser.add_argument("--rollout-horizon", type=int, default=30)
    parser.add_argument("--rollouts-per-action", type=int, default=3)

    # DQN options
    parser.add_argument("--model-path", type=str, default="dqn_lunarlander.pt")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=2000)
    parser.add_argument("--target-update-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "dqn_train":
        cfg = DQNConfig(
            episodes=args.episodes,
            max_steps=args.max_steps,
            gamma=args.gamma,
            lr=args.lr,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            target_update_every=args.target_update_every,
            seed=args.seed,
            model_path=args.model_path,
        )
        train_dqn(cfg)
    else:
        evaluate_policy(args)


if __name__ == "__main__":
    main()

```

---

## How to run it

### Random baseline

```bash
python lunar_rl_compare.py --mode random --episodes 5
```

This should usually perform poorly. That is expected.

---

### Greedy heuristic

```bash
python lunar_rl_compare.py --mode greedy --episodes 5
```

This uses a simple hand-written rule. It may do better than random, but it is not trained.

---

### Rollout planning

```bash
python lunar_rl_compare.py --mode rollout --episodes 2 --rollout-horizon 40 --rollouts-per-action 3
```

This is slower because it simulates short futures before every action.

If you increase the rollout horizon and the number of rollouts per action, the decision quality may improve, but runtime increases quickly.

---

### Train DQN

```bash
python lunar_rl_compare.py --mode dqn_train --episodes 300
```

This trains a small DQN from scratch and saves:

```bash
dqn_lunarlander.pt
```

For a quick test, you can run fewer episodes. For better performance, train longer.

---

### Evaluate DQN

```bash
python lunar_rl_compare.py --mode dqn_eval --model-path dqn_lunarlander.pt --episodes 5
```

With rendering:

```bash
python lunar_rl_compare.py --mode dqn_eval --model-path dqn_lunarlander.pt --episodes 3 --render
```

---

## Practical advice

If I were debugging an RL-style search system, I would not start with DQN immediately.

I would use this order:

1. **Random**: make sure the environment and scoring function work.
2. **Greedy**: check whether the ML model gives useful local guidance.
3. **Rollout**: test whether looking ahead helps.
4. **DQN**: train a reusable policy/value function after the above makes sense.

This order is boring, but it saves time.

Many RL bugs are not neural network bugs. They are reward bugs, environment bugs, masking bugs, or termination-condition bugs.

Similarly, if the reward is designed in the wrong direction, the model can learn the opposite of what we want.

---

## Final takeaway

These four methods are related, but they answer different questions:

```text
Random:  What happens if we just try?
Greedy:  What looks best right now?
Rollout: What looks best after simulating the future?
DQN:     Can we learn a reusable long-term value function?
```

In strict terms, **DQN is the real reinforcement learning algorithm** here.

`Random`, `Greedy`, and `Rollout` are still important because they make the system easier to test, explain, and debug.

In practice, that matters a lot.

A good RL project is not only about using a powerful algorithm. It is about building a reliable decision loop: good states, legal actions, meaningful rewards, and careful evaluation.

---

## References

- [Gymnasium documentation](https://gymnasium.farama.org/)
- [Gymnasium LunarLander-v3 documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
