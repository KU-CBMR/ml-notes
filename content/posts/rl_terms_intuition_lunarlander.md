---
title: "Reinforcement Learning Terms from Intuition: A LunarLander Walkthrough (1)"
date: 2026-02-21
draft: false
math: true
tags:
  [
    "reinforcement-learning",
    "lunarlander",
    "dqn",
    "bellman-equation",
    "value-function",
  ]
categories: ["Machine Learning"]
summary: "A beginner-friendly map of reinforcement learning terms: state, action, environment, reward, return, policy, value function, Q-value, Bellman equation, rollout, and DQN."
---

# Reinforcement Learning Terms from Intuition: A LunarLander Walkthrough

When I first learned reinforcement learning, the hardest part was not one specific algorithm.

The hardest part was the vocabulary.

People say:

```text
state
action
reward
return
policy
value function
Q-value
Bellman equation
transition
trajectory
episode
replay buffer
target network
```

Each word sounds simple.

But if we do not know the order in which these ideas are built, reinforcement learning feels like a bag of formulas.

This post is my attempt to explain the core RL terms from intuition.

We will use **LunarLander** as the running example.

The goal is not to memorize definitions.

The goal is to understand:

```text
Why do these terms exist?
What problem does each term solve?
How does one term lead to the next?
```

---

## 0. The big picture first

Reinforcement learning is about sequential decision making.

The agent does not make one isolated prediction.

It repeatedly acts, observes what happens, and improves its future behavior.

The core loop is:

```text
state
  -> action
  -> environment
  -> reward and next state
  -> repeat
```

In mathematical notation:

$$
s_t, a_t \longrightarrow r_{t+1}, s_{t+1}
$$

For LunarLander:

```text
state:
    Where is the lander?
    How fast is it moving?
    Is it tilted?
    Are its legs touching the ground?

action:
    Do nothing
    Fire left engine
    Fire main engine
    Fire right engine

environment:
    The physics simulator

reward:
    A score telling us whether the last step was good or bad

next state:
    The new position, velocity, angle, and contact information
```

Everything else in RL is built on top of this loop.

---

## 1. Agent

The **agent** is the decision maker.

In LunarLander, the agent is the controller of the lander.

It receives information about the lander and chooses an engine action.

```text
agent = the thing that chooses actions
```

Why do we need this term?

Because RL separates two things:

```text
agent:
    decides what to do

environment:
    decides what happens after the action
```

This separation is important.

The agent does not control physics.

The agent only controls its actions.

---

## 2. Environment

The **environment** is the world the agent interacts with.

In LunarLander, the environment is the simulator.

It knows the physics, gravity, collision, landing pad, engine effects, and reward rule.

When the agent chooses an action, the environment returns:

```text
next state
reward
done flag
```

In code, this usually looks like:

```python
next_state, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

Why do we need this term?

Because RL is not just about prediction.

It is about interaction.

The environment is what makes actions have consequences.

---

## 3. State

The **state** is the information used to make a decision.

At time $t$, we write it as:

$$
s_t
$$

In LunarLander, a state may contain:

```text
x position
y position
x velocity
y velocity
angle
angular velocity
left leg contact
right leg contact
```

The state answers:

```text
Where am I now?
```

Why do we need this term?

Because an action is only meaningful inside a situation.

For example:

```text
fire main engine
```

may be good if the lander is falling too fast.

But it may be bad if the lander is already moving upward too fast.

So RL does not ask:

```text
Is this action good in general?
```

It asks:

```text
Is this action good in this state?
```

---

## 4. Observation

Sometimes people say **observation** instead of state.

They are related but not always identical.

The true state is everything about the world that matters.

The observation is what the agent can actually see.

In many simple environments, the observation is treated as the state.

For LunarLander, the vector returned by the simulator is usually used as the state.

```text
observation = what the agent receives
state       = the information that describes the situation
```

Why do we need this distinction?

Because in real-world problems, the agent may not see everything.

A robot may not know the exact friction of the floor.

A biological experiment agent may not know the true hidden mechanism of a system.

It only sees measurements.

---

## 5. Action

The **action** is what the agent chooses.

At time $t$, we write it as:

$$
a_t
$$

In LunarLander, the discrete actions are:

```text
0: do nothing
1: fire left engine
2: fire main engine
3: fire right engine
```

The action answers:

```text
What do I do now?
```

Why do we need this term?

Because RL is about choosing actions that shape the future.

A supervised learning model usually predicts a label.

An RL agent chooses an action, and that action changes what happens next.

---

## 6. Action space

The **action space** is the set of all actions the agent is allowed to take.

For LunarLander:

$$
\mathcal{A} = \{0,1,2,3\}
$$

In a different problem, actions may be continuous.

For example, a robot arm may choose a motor torque:

$$
a \in \mathbb{R}^n
$$

Why do we need this term?

Because before we can choose a good action, we need to know what actions are available.

DQN is usually used for discrete action spaces because it can output one Q-value per action.

---

## 7. Transition

A **transition** is one step of experience.

It usually has the form:

$$
(s_t, a_t, r_{t+1}, s_{t+1}, d_{t+1})
$$

where:

```text
s_t:
    current state

a_t:
    action taken

r_{t+1}:
    reward received after the action

s_{t+1}:
    next state

d_{t+1}:
    whether the episode ended
```

Why do we need this term?

Because RL algorithms learn from transitions.

A DQN replay buffer is basically a memory full of transitions.

In code, a transition is one row of experience.

---

## 8. Transition dynamics

The **transition dynamics** describe how the environment moves from one state to another.

Mathematically:

$$
P(s_{t+1} \mid s_t, a_t)
$$

This means:

```text
Given current state s_t
and action a_t,
how likely is each possible next state?
```

For LunarLander, this is controlled by physics.

If the lander fires the main engine, the simulator updates velocity and position.

Why do we need this term?

Because actions are not just labels.

Actions cause state changes.

## 9. Reward function

The **reward function** is the rule used by the environment to assign reward.

A common notation is:

$$
R(s_t, a_t, s_{t+1})
$$

It answers:

```text
After this transition, how much score should the agent receive?
```

For LunarLander, the reward function encourages behavior such as:

```text
move toward the landing pad
slow down
keep the lander upright
touch the ground with legs
avoid crashing
avoid wasting too much fuel
```

The exact implementation belongs to the environment.

Why do we need this term?

Because RL needs an objective signal.

The reward function tells the agent what the task cares about.

A bad reward function can teach the wrong behavior.

A good reward function makes the desired behavior learnable.

Important distinction:

```text
reward function = the rule
reward          = the actual number received at one step
```

---

## 10. Reward

The **reward** is the actual number the agent receives after one action.

At time $t+1$, we write it as:

$$
r_{t+1}
$$

For example:

```text
The lander fires the main engine.
The environment updates the physics.
The environment returns reward = -0.3.
```

That number is the reward for that step.

Why do we need this term?

Because it is the immediate feedback from the environment.

But this is also where beginners often get confused.

The reward is not the same as long-term success.

A single step can have a negative reward but still be useful.

For example:

```text
firing the main engine may cost fuel now
but prevent a crash later
```

So reward is local.

RL cares about the future.

This leads to return.

---

## 11. Return

The **return** is the total future reward starting from time $t$.

We usually write it as:

$$
G_t
$$

The standard discounted return is:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + \cdots
$$

or:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

Why do we need this term?

Because one-step reward is too short-sighted.

The agent should not only ask:

```text
Was this action good immediately?
```

It should ask:

```text
Did this action lead to a good future?
```

In LunarLander, a safe landing is the result of many actions.

The return is how we turn a whole future trajectory into one number.

---

## 12. Discount factor

The **discount factor** is:

$$
\gamma
$$

Usually:

$$
0 \leq \gamma \leq 1
$$

It controls how much the agent cares about future rewards.

If:

$$
\gamma = 0
$$

then:

$$
G_t = r_{t+1}
$$

The agent only cares about the next reward.

If:

$$
\gamma \approx 1
$$

the agent cares a lot about long-term consequences.

Why do we use powers of $\gamma$?

Because each step into the future is discounted by the same ratio.

```text
one step later:    gamma
two steps later:   gamma * gamma
three steps later: gamma * gamma * gamma
```

So the weights become:

$$
1,\gamma,\gamma^2,\gamma^3,\ldots
$$

This gives us a powerful recursive identity:

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

This identity is the seed of the Bellman equation.

Why design it this way?

Because it lets us learn long-term value from one-step experience.

Without this recursive structure, DQN and Q-learning would not have such a simple training target.

---

## 13. Episode

An **episode** is one complete run from start to finish.

In LunarLander, one episode begins when the lander starts falling and ends when:

```text
the lander lands successfully
or
the lander crashes
or
the time limit is reached
```

Why do we need this term?

Because many RL tasks naturally have attempts.

One landing attempt is one episode.

One game is one episode.

One search trajectory is one episode.

The agent can learn from many episodes.

---

## 14. Done flag / terminal state

The **done flag** tells us whether the episode ended.

It is often written as:

$$
d
$$

If the episode ended:

$$
d = 1
$$

If the episode continues:

$$
d = 0
$$

In LunarLander:

```text
crash              -> done = 1
successful landing -> done = 1
still flying       -> done = 0
```

Why do we need this term?

Because if the episode is over, there is no future reward after that state.

This is why the DQN target uses:

$$
y = r + \gamma(1-d)\max_{a'}Q(s',a')
$$

If $d=1$, then:

$$
1-d = 0
$$

so:

$$
y = r
$$

The future term disappears.

That is exactly what we want.

---

## 15. Trajectory

A **trajectory** is a sequence of states, actions, and rewards.

For example:

$$
s_0, a_0, r_1, s_1, a_1, r_2, s_2, \ldots
$$

In LunarLander, a trajectory is the full path of one landing attempt.

Why do we need this term?

Because return is computed along a trajectory.

Rollout methods also simulate trajectories to estimate which action is good.

---

## 16. Policy

A **policy** is the agent's behavior rule.

We usually write it as:

$$
\pi(a \mid s)
$$

This means:

```text
Given state s,
what is the probability of choosing action a?
```

A policy can be:

```text
random
rule-based
greedy
epsilon-greedy
a neural network
```

For LunarLander, a policy may say:

```text
If the lander is falling too fast, fire the main engine.
If the lander tilts left, fire the right engine.
```

Or it may be learned from data.

Why do we need this term?

Because values depend on behavior.

The question:

```text
How good is this state?
```

is incomplete unless we know:

```text
What will the agent do from this state onward?
```

That behavior is the policy.

---

## 17. State-value function

The **state-value function** tells us how good a state is under a policy.

For policy $\pi$:

$$
V^\pi(s) = \mathbb{E}_{\pi}[G_t \mid s_t=s]
$$

In words:

```text
If I start from state s
and then follow policy pi,
how much future return do I expect?
```

For LunarLander:

```text
V(s) asks:
    How good is this lander situation,
    assuming I keep using my current landing strategy?
```

Why do we need this term?

Because we often want to evaluate situations.

A lander near the pad, slow, upright, and close to the ground should have a high value.

A lander falling fast, tilted, and far away should have a low value.

But $V(s)$ does not directly tell us which action to take.

It only scores the state.

That leads to Q-value.

---

## 18. Action-value function / Q-value

The **action-value function**, or **Q-function**, tells us how good an action is in a state.

For policy $\pi$:

$$
Q^\pi(s,a) = \mathbb{E}_{\pi}[G_t \mid s_t=s, a_t=a]
$$

In words:

```text
If I am in state s,
take action a first,
and then follow policy pi,
how much future return do I expect?
```

For LunarLander:

```text
Q(s, do nothing):
    If I do nothing now, how good is the future?

Q(s, fire main engine):
    If I fire the main engine now, how good is the future?

Q(s, fire left engine):
    If I fire the left engine now, how good is the future?

Q(s, fire right engine):
    If I fire the right engine now, how good is the future?
```

Why do we need this term?

Because action selection needs action values.

If we only know:

$$
V(s) = 50
$$

we know the state is okay, but we do not know what to do.

If we know:

```text
Q(s, do nothing)        = -20
Q(s, fire left engine)  = 10
Q(s, fire main engine)  = 80
Q(s, fire right engine) = 5
```

then the best action is clear:

$$
a^* = \arg\max_a Q(s,a)
$$

This is why DQN learns Q-values.

---

## 19. Relationship between V and Q

$V$ and $Q$ are closely connected.

$V^\pi(s)$ is the average Q-value under the policy:

$$
V^\pi(s) = \sum_a \pi(a \mid s)Q^\pi(s,a)
$$

This means:

```text
The value of a state
is the average value of the actions the policy may take there.
```

If the policy always chooses the best action, then:

<!-- $$ V^_(s) = \max_a Q^_(s,a) $$
 -->

$V^{\ast}(s) = \max_{a} Q^{\ast}(s,a)$

Why do we need this relationship?

Because it shows that $V$ and $Q$ are not unrelated ideas.

They are two views of the same future return.

```text
V:
    value of a situation

Q:
    value of an action in that situation
```

---

## 20. Bellman equation

The **Bellman equation** is the recursive form of value.

The simplest intuition is:

```text
value now
=
reward from the next step
+
discounted value of the future
```

For Q-values, the optimal Bellman equation is:

$$
Q^{\ast}(s,a) = \mathbb{E} \left[ r + \gamma \max_{a'}Q^{\ast}(s',a') \right]
$$

At first, this formula may look abstract.

But the idea is actually very practical.

In reinforcement learning, the agent usually does not see the whole future immediately.

At one step, the agent only observes one transition:

$$
(s,a,r,s')
$$

This means:

```text
I was in state s.
I took action a.
I received reward r.
The environment moved me to next state s'.
```

But the agent does not only care about this one reward $r$.

The agent wants to know:

```text
If I take action a in state s,
how good will the whole future become?
```

That long-term future is the return:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + \cdots
$$

The problem is:

```text
At time t, the agent only sees r_{t+1} and s_{t+1}.
It does not yet know r_{t+2}, r_{t+3}, r_{t+4}, ...
```

So how can the agent learn long-term value from only one-step experience?

This is exactly why the Bellman equation is useful.

The Bellman idea says:

```text
Do not try to see the entire future at once.

Instead, split the future into two parts:

1. the reward I get immediately
2. the value of the next state
```

So instead of writing the full future as:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

we rewrite it as:

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

This is the key step.

The term $G_{t+1}$ already contains the future after the next state:

$$
G_{t+1} = r_{t+2} + \gamma r_{t+3} + \gamma^2 r_{t+4} + \cdots
$$

So Bellman equation compresses the long future into one number:

```text
future value from the next state
```

For Q-learning and DQN, we do not know the true future return.

So we estimate it using the Q-function.

After reaching the next state $s'$, the best future value is estimated as:

$$
\max_{a'} Q^*(s',a')
$$

This means:

```text
Look at the next state s'.
Try all possible next actions a'.
Take the action with the highest estimated long-term value.
```

Therefore, the value of the current action becomes:

$$
Q^{\ast}(s,a) = \mathbb{E} \left[ r + \gamma \max_{a'} Q^{\ast}(s',a') \right]
$$

In plain English:

```text
The value of taking action a in state s
is the reward I get now
plus
the discounted value of the best thing I can do next.
```

This is the bridge from short-term experience to long-term learning.

---

### LunarLander example

Suppose the lander is falling too fast.

The current state is $s$.

The agent chooses action $a$:

```text
fire main engine
```

The environment gives an immediate reward $r$.

This reward may be slightly negative because firing the engine uses fuel.

So if the agent only looked at immediate reward, it might think:

```text
Firing the main engine is bad.
It costs fuel now.
```

But after firing the engine, the lander may slow down and move into a much safer next state $s'$.

From that next state, the agent may be able to land successfully.

So the action can be good in the long run, even if the immediate reward is not good.

Bellman equation captures this:

$$
Q^{\ast}(s,\text{fire main engine}) = \mathbb{E} \left[ r + \gamma \max_{a'}Q^{\ast}(s',a') \right]
$$

The first term $r$ says:

```text
What happened immediately after firing the engine?
```

The second term says:

```text
After firing the engine, how good is the best future from the next state?
```

This is why Bellman equation matters.

It allows the agent to understand that:

```text
An action can be valuable not because it gives high reward now,
but because it leads to a better future.
```

---

### Why this matters for DQN

DQN uses the Bellman equation to create a training target.

For one transition:

$$
(s,a,r,s',d)
$$

DQN builds the target:

$$
y = r + \gamma(1-d) \max_{a'}Q_{\theta^-}(s',a')
$$

This target says:

```text
The Q-value of the action I just took
should move closer to:

immediate reward
+
discounted best future value
```

So DQN does not need to wait until the entire episode is finished every time.

It can learn from one transition at a time.

That is the power of the Bellman equation.

It turns a long-term learning problem into many local learning problems.

---

### One-sentence summary

The Bellman equation is important because it lets the agent learn long-term value from one-step experience.

It says:

```text
To understand the value of an action now,
look at the reward now
and add the estimated value of the future it leads to.
```

This is the core idea behind Q-learning and DQN.

<!-- ## 20. Bellman equation

The **Bellman equation** is the recursive form of value.

The simplest intuition is:

```text
value now
=
reward now
+
discounted value later
```

For Q-values:

$$
Q^*(s,a)
=
\mathbb{E}
\left[
r + \gamma \max_{a'}Q^*(s',a')
\right]
$$

Why do we need this term?

Because the agent usually sees only one transition:

$$
(s,a,r,s')
$$

But it wants to learn long-term return:

$$
r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\cdots
$$

Bellman equation is the bridge.

It lets us use local information to learn long-term consequences.

This is one of the most important ideas in RL.

--- -->

## 21. Temporal-difference error

The **temporal-difference error**, or **TD error**, measures how different our current estimate is from the Bellman target.

For Q-learning:

$$
\delta = r + \gamma \max_{a'}Q(s',a') - Q(s,a)
$$

Why do we need this term?

Because it tells us how much to update.

If:

$$
\delta > 0
$$

then the action was better than expected.

If:

$$
\delta < 0
$$

then the action was worse than expected.

In LunarLander, if firing the main engine unexpectedly prevents a crash, the TD error may be positive, and the agent will increase the value of that action in similar states.

---

## 22. Q-learning

**Q-learning** is an algorithm that learns the optimal Q-function.

In tabular form, it updates:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'}Q(s',a') - Q(s,a) \right]
$$

Why do we need it?

Because if we can learn $Q^*(s,a)$, then action selection is simple:

$$
a^* = \arg\max_a Q^*(s,a)
$$

For small environments, we can store $Q(s,a)$ in a table.

For large environments, a table is not enough.

That leads to DQN.

---

## 23. DQN

**DQN** stands for Deep Q-Network.

It replaces the Q-table with a neural network:

$$
Q_\theta(s,a) \approx Q^*(s,a)
$$

For LunarLander, the network takes the state vector and outputs four numbers:

```text
Q(s, do nothing)
Q(s, fire left engine)
Q(s, fire main engine)
Q(s, fire right engine)
```

Then it chooses:

$$
a = \arg\max_a Q_\theta(s,a)
$$

Why do we need DQN?

Because many environments have too many possible states for a table.

A neural network can generalize.

If it learns that firing the main engine is useful when the lander is falling fast, it can apply that idea to many similar states.

---

## 24. DQN target

DQN does not know the true label $Q^*(s,a)$.

So it builds a temporary training target:

$$
y = r + \gamma(1-d)\max_{a'}Q_{\theta^-}(s',a')
$$

Then it trains the online network so that:

$$
Q_\theta(s,a)
$$

moves closer to:

$$
y
$$

The loss is often written as:

$$
L(\theta) = \left(Q_\theta(s,a)-y\right)^2
$$

Why do we need this target?

Because RL does not come with supervised labels.

Nobody tells the lander:

```text
The true value of firing the main engine here is 42.7.
```

DQN creates a label from the Bellman equation.

This is how it turns RL into something that looks like supervised learning.

---

## 25. Target network

DQN usually uses two networks:

```text
online network:
    Q_theta
    updated every training step

target network:
    Q_theta_minus
    updated more slowly
```

The target network is used in:

$$
y = r + \gamma(1-d)\max_{a'}Q_{\theta^-}(s',a')
$$

Why do we need it?

Because if the same network creates the label and learns from the label, the target moves too fast.

It is like trying to shoot arrows at a target that jumps every time you move your arm.

The target network makes the label more stable.

It is not a perfect solution.

It is a stabilization trick.

But it is one of the key reasons DQN works in practice.

---

## 26. Replay buffer

The **replay buffer** is a memory of past transitions:

$$
\mathcal{D} = \{(s,a,r,s',d)\}
$$

Why do we need it?

Because consecutive experiences are highly correlated.

In LunarLander, nearby states are almost identical:

```text
state at time t
state at time t+1
state at time t+2
```

If the network only learns from the latest transition, training can become unstable and biased.

The replay buffer helps by:

```text
storing many experiences
mixing old and new transitions
randomly sampling from memory
reusing data multiple times
```

It turns sequential experience into something more like a dataset.

---

## 27. Mini-batch

A **mini-batch** is a small random subset of transitions used for one gradient update.

For example:

<!-- $$(s_i,a_i,r_i,s'_i,d_i)_{i=1}^{64}$$ -->

$(s_i,a_i,r_i,s_i',d_i)_{i=1}^{64}$
This means the batch size is:

$$
B = 64
$$

DQN computes a target for each transition:

$$
y_i = r_i + \gamma(1-d_i)\max_{a'}Q_{\theta^-}(s'_i,a')
$$

Then it minimizes average loss:

$$
L(\theta) = \frac{1}{B} \sum_{i=1}^{B} \left(Q_\theta(s_i,a_i)-y_i\right)^2
$$

Why do we need mini-batches?

Because:

```text
one sample:
    too noisy

all samples:
    too expensive

mini-batch:
    efficient and reasonably stable
```

This is the same practical idea used in deep learning.

---

## 28. Exploration and exploitation

**Exploration** means trying actions to learn more.

**Exploitation** means choosing the best-known action.

In DQN, a common strategy is epsilon-greedy:

$$
a_t =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q_\theta(s_t,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

Why do we need exploration?

Because early Q-values are unreliable.

If the lander always trusts a random untrained network, it may repeat bad actions and never discover better ones.

Exploration lets the agent collect diverse experience.

Over time, $\epsilon$ usually decreases:

```text
early training:
    explore more

later training:
    exploit more
```

---

## 29. Rollout

**Rollout** means simulating possible futures before choosing an action.

For each candidate action $a$, rollout estimates:

<!-- $$
\hat{Q}_{\text{rollout}}(s,a) = \frac{1}{N} \sum_{i=1}^{N} G^{(i)}(s,a)
$$ -->

$$
\hat{Q}_{\mathrm{rollout}}(s,a)
=
\frac{1}{N}
\sum_{i=1}^{N}
G^{(i)}(s,a)
$$

This means:

```text
try action a
simulate the future N times
average the returns
```

Then choose:

$$
a^* = \arg\max_a \hat{Q}_{\text{rollout}}(s,a)
$$

Why do we need rollout?

Because it is a direct way to estimate long-term consequences.

The trade-off is computation.

```text
Rollout:
    expensive during action selection
    no neural network required

DQN:
    expensive during training
    fast during action selection
```

In LunarLander, rollout is like asking:

```text
If I fire the main engine now,
what might happen over the next few seconds?
```

DQN is like asking:

```text
Based on everything I learned before,
how valuable is firing the main engine now?
```

---

## 30. Model-based vs model-free

A **model** in RL usually means a model of the environment:

```text
transition model:
    predicts next state

reward model:
    predicts reward
```

A **model-based** method uses a model or simulator to look ahead.

Rollout is model-based if it uses a simulator.

A **model-free** method does not explicitly predict next states.

DQN is usually model-free.

It does not learn physics directly.

It learns:

$$
Q(s,a)
$$

which predicts long-term action value.

Why do we need this distinction?

Because there are two broad ways to solve sequential decision problems:

```text
simulate the future
or
learn a value function from experience
```

Rollout is closer to the first.

DQN is closer to the second.

---

## 31. A simple LunarLander story using all terms

Imagine the lander is falling fast and tilted slightly to the right.

That situation is the **state** $s_t$.

The agent must choose an **action** $a_t$:

```text
do nothing
fire left engine
fire main engine
fire right engine
```

Suppose it fires the main engine.

The **environment** applies physics.

The lander slows down, fuel is consumed, and the simulator returns:

```text
next state s_{t+1}
reward r_{t+1}
done flag d
```

This one step is a **transition**:

$$
(s_t,a_t,r_{t+1},s_{t+1},d)
$$

The immediate reward may be slightly negative because fuel was used.

But the action may still be good because it prevents a crash later.

So we define **return**:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

The agent follows a **policy** $\pi$ to choose actions.

The **state-value function** asks:

```text
How good is this lander state if I keep following my policy?
```

The **Q-value** asks:

```text
How good is firing the main engine in this exact state?
```

The **Bellman equation** says:

```text
The value of firing the main engine now
should equal:
reward now + discounted best future value
```

DQN learns this with a neural network.

The **replay buffer** stores many transitions.

A **mini-batch** samples some of them.

The **target network** builds a more stable training target.

The online network learns to predict better Q-values.

Eventually, the agent can look at a state and choose the action with the highest predicted long-term value.

That is the whole RL story.

---

## 32. The full construction order

Here is the order I recommend keeping in your head:

```text
1. Environment exists.

2. Agent observes a state.

3. Agent chooses an action.

4. Environment returns next state and reward.

5. One such step is a transition.

6. Many transitions form a trajectory.

7. A complete trajectory is an episode.

8. Reward is one-step feedback.

9. Return is total discounted future reward.

10. Policy is the rule for choosing actions.

11. Value function predicts expected return from a state.

12. Q-function predicts expected return from a state-action pair.

13. Bellman equation turns long-term return into:
        immediate reward + future value.

14. Q-learning uses Bellman equation to update a Q-table.

15. DQN replaces the Q-table with a neural network.

16. Replay buffer stores transitions.

17. Mini-batches train the network efficiently.

18. Target network stabilizes the moving Bellman target.

19. Exploration helps discover better actions.

20. Rollout estimates action values by simulating futures.
```

This order matters.

We do not invent Q-value first.

We first define what the agent experiences.

Then we define what the agent wants.

Then we define functions that predict that objective.

Then we design algorithms to learn those functions.

---

## 33. The one-table summary

| Term             | Intuition                  | LunarLander example          | Why it exists                       |
| ---------------- | -------------------------- | ---------------------------- | ----------------------------------- |
| Agent            | Decision maker             | Lander controller            | Someone must choose actions         |
| Environment      | World                      | Physics simulator            | Actions need consequences           |
| State            | Current situation          | Position, velocity, angle    | Actions depend on context           |
| Observation      | What agent sees            | State vector                 | Agent may not see everything        |
| Action           | Choice                     | Fire engine or not           | Agent must affect the future        |
| Transition       | One experience step        | $(s,a,r,s',d)$               | Basic unit of learning              |
| Reward function  | Scoring rule               | Landing good, crashing bad   | Defines the task objective          |
| Reward           | One-step score             | Reward after firing engine   | Immediate feedback                  |
| Return           | Future total score         | Whole landing outcome        | RL cares about long-term success    |
| Discount factor  | Future weight              | $\gamma=0.99$                | Makes future recursive and stable   |
| Episode          | One attempt                | One landing trial            | Natural training unit               |
| Policy           | Behavior rule              | Which engine to fire         | Defines how the agent acts          |
| Value$V(s)$      | State quality              | How good this situation is   | Evaluates situations                |
| Q-value$Q(s,a)$  | Action quality             | How good firing engine is    | Chooses actions                     |
| Bellman equation | One-step recursion         | Reward now + future value    | Learns long-term value locally      |
| TD error         | Prediction mistake         | Target minus current Q       | Tells update direction              |
| Q-learning       | Table-based value learning | Small discrete world         | Learns optimal action values        |
| DQN              | Neural Q-learning          | Lander state to 4 Q-values   | Handles large state spaces          |
| Replay buffer    | Memory                     | Past flying experiences      | Breaks correlation and reuses data  |
| Mini-batch       | Random training subset     | 64 past transitions          | Stable efficient gradient updates   |
| Target network   | Slow teacher               | Stable Q target              | Reduces moving-label instability    |
| Exploration      | Try uncertain actions      | Random engine choices early  | Discover better behavior            |
| Rollout          | Simulated futures          | Try possible future landings | Estimates action values by planning |

---

## 34. Final intuition

The shortest version is this:

```text
Reward tells us what happened now.

Return tells us whether the future was good.

Value estimates return from a state.

Q-value estimates return from taking an action in a state.

Bellman equation lets us estimate long-term value one step at a time.

DQN learns Q-values with a neural network.

Rollout estimates Q-values by simulating futures.
```

In LunarLander, the agent should not only ask:

```text
Does firing the engine give reward right now?
```

It should ask:

```text
Does firing the engine now create a better landing later?
```

That is the heart of reinforcement learning.

RL is not about immediate reward.

RL is about learning which actions create good futures.

---

## 35. Does changing gamma break learning? Why not directly use a DNN?

There is one subtle question that often appears after learning discounted return and the Bellman equation:

```text
If the whole point is to estimate future return,
why do we need this special recursive form?

Why not just let a neural network directly estimate the return?

And if gamma changes over time,
does reinforcement learning stop working?
```

This is a very good question.

The short answer is:

```text
Changing gamma does not make learning impossible.
Directly training a DNN to estimate return is also possible.

But the standard discounted return gives us a simple recursive structure.
That recursive structure gives DQN and Q-learning a simple one-step training target.
```

Let us unpack this carefully.

---

### 35.1 The standard discounted return

Earlier we defined the discounted return as:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + \cdots
$$

Because the same discount factor $\gamma$ is applied again and again, we get:

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

This identity is powerful because it says:

```text
Long-term return from now
=
one-step reward
+
discounted long-term return from the next step.
```

This is the seed of the Bellman equation.

For Q-learning and DQN, this leads to the familiar one-step target:

$$
y = r + \gamma (1-d) \max_{a'} Q_{\theta^-}(s',a')
$$

This target is simple because it only needs one transition:

$$
(s,a,r,s',d)
$$

That is the main computational advantage.

---

### 35.2 What if gamma changes over time?

Suppose the discount factor is not constant.

For example, imagine:

```text
time t:     gamma_t     = 0.98
time t + 1: gamma_{t+1} = 0.96
time t + 2: gamma_{t+2} = 0.93
```

Can we still define return?

Yes.

One possible definition is:

$$
G_t = r_{t+1} + \gamma_t r_{t+2} + \gamma_t \gamma_{t+1} r_{t+3} + \gamma_t \gamma_{t+1} \gamma_{t+2} r_{t+4} + \cdots
$$

This still has a recursive form:

$$
G_t = r_{t+1} + \gamma_t G_{t+1}
$$

So changing gamma does not automatically destroy learning.

A DQN-style target could become:

$$
y = r + \gamma_t (1-d) \max_{a'} Q(s',a')
$$

So the problem is not that changing gamma is impossible.

The real problem is that the meaning of the Q-function may become less stationary.

---

### 35.3 Why does changing gamma make Q harder to define?

In standard DQN, we want one network to learn:

$$
Q(s,a)
$$

This means:

```text
Given state s and action a,
what long-term return should I expect?
```

But if the discount factor changes with time, then the same state-action pair may have different values at different times.

For example, in LunarLander, imagine the lander is close to the landing pad, falling slowly, and slightly tilted.

That state may appear:

```text
early in the episode
or
near the end of the episode
```

If the discount factor depends on time, then the future is weighted differently in those two cases.

So the value may no longer be fully described by:

$$
Q(s,a)
$$

We may need:

$$
Q(s,a,t)
$$

which means:

```text
The value of action a in state s at time t.
```

This is not impossible.

But it makes the learning problem bigger.

The network may need to know time explicitly.

The training target may need to include $\gamma_t$.

The value function becomes less reusable across different time steps.

This is why the constant $\gamma$ version is so popular.

It gives the same state-action pair a stable meaning across the whole task.

---

### 35.4 Why not directly train a DNN to estimate return?

We can.

This is a real idea.

Instead of using the Bellman target, we could run a full episode, compute the actual return, and train a neural network like this:

$$
Q_\theta(s_t,a_t) \approx G_t
$$

For one trajectory:

```text
s_0, a_0, r_1
s_1, a_1, r_2
s_2, a_2, r_3
...
terminal
```

After the episode ends, we can compute:

$$
G_0, G_1, G_2, \ldots
$$

Then we use these as labels:

```text
input:  (s_t, a_t)
label:  G_t
```

This is often called a Monte Carlo style approach.

So yes, a DNN can directly estimate return.

DQN is not the only possible way.

---

### 35.5 Then why does DQN not simply do that?

Directly training on full returns has several practical issues.

The first issue is that we may need to wait until the episode ends.

In LunarLander, to know the full return of an action, we may need to wait until the lander finally lands or crashes.

But DQN wants to learn from every single transition:

$$
(s,a,r,s')
$$

Bellman learning lets us update immediately.

Instead of waiting for the whole future, DQN uses:

$$
y = r + \gamma \max_{a'} Q(s',a')
$$

This says:

```text
I do not know the full future yet.
But I know the reward right now.
And I have an estimate of the future from the next state.
So I can already make a useful update.
```

This is called bootstrapping.

Bootstrapping means:

```text
Use the current estimate of the future
to improve the current estimate of the present.
```

---

### 35.6 Monte Carlo targets can be noisy

Another issue is variance.

The full return $G_t$ depends on the whole future trajectory.

Even if the current state and action are similar, the future may unfold differently.

In LunarLander, after the same action:

```text
one rollout may land successfully,
one rollout may drift away,
one rollout may crash,
one rollout may waste too much fuel.
```

So the full return label can be very noisy.

A DNN trained directly on full returns may need many complete episodes to average out the noise.

Bellman learning uses a shorter target:

$$
r + \gamma Q(s',a')
$$

This target is still an estimate, but it breaks a long future into smaller local pieces.

That is why DQN can often be more data-efficient than pure Monte Carlo learning.

---

### 35.7 DQN is already using a DNN

The question "Why not directly use a DNN?" can be slightly misleading.

DQN is using a DNN.

The DQN network is:

$$
Q_\theta(s,a)
$$

The neural network is the function approximator.

But the neural network still needs a training signal.

In ordinary supervised learning, labels are given by the dataset.

For example:

```text
input: image
label: cat
```

In DQN, nobody gives us the true label:

$$
Q^*(s,a)
$$

So DQN uses the Bellman equation to construct a temporary label:

$$
y = r + \gamma(1-d) \max_{a'} Q_{\theta^-}(s',a')
$$

Then it trains the DNN by minimizing:

$$
\left(Q_\theta(s,a)-y\right)^2
$$

So the relationship is not:

```text
DNN versus Bellman equation
```

The relationship is:

```text
DNN estimates the Q-function.
Bellman equation provides the training target.
```

---

### 35.8 Three ways to estimate long-term value

Now we can compare three different ideas.

#### Monte Carlo DNN

```text
Run a full episode.
Compute the true realized return G_t.
Train a DNN to map (s,a) to G_t.
```

This is direct, but it may require complete episodes and can have high variance.

#### DQN / Q-learning

```text
Use one transition at a time.
Build a Bellman target.
Train a DNN to match that target.
```

This is more local and data-efficient, but the target is bootstrapped and can be unstable.

That is why DQN needs a target network and replay buffer.

#### Rollout

```text
When choosing an action,
simulate many possible futures.
Average their returns.
Choose the action with the best estimated return.
```

This can work well when simulation is cheap, but it can be slow at decision time.

---

### 35.9 The real reason we like the Bellman form

The Bellman form is not popular because other formulas are impossible.

Other return definitions are possible.

Changing discount factors is possible.

Direct DNN estimation of Monte Carlo returns is possible.

The reason we like the Bellman form is computational.

It turns:

```text
a long future trajectory
```

into:

```text
one immediate reward
+
one estimate of the next state's future
```

That is what allows Q-learning and DQN to learn long-term value from local experience.

In one sentence:

```text
DNN is the function approximator.
Bellman equation is the source of the training signal.
gamma^k is useful because it makes the long-term objective recursively decomposable.
```

That is the key intuition.

## 36. Where do the old estimate and the new target come from?

In the previous section, we talked about the **temporal-difference error**, or **TD error**.

For Q-learning, the TD error is:

$$
\delta = r + \gamma \max_{a'} Q(s',a') - Q(s,a)
$$

At first, this formula can feel confusing.

It compares two quantities:

```text
new target
-
old estimate
```

More explicitly:

$$
\delta = \underbrace{ r+\gamma \max_{a'}Q(s',a') }_{\text{new target}} - \underbrace{ Q(s,a) }_{\text{old estimate}}
$$

The natural question is:

```text
Where do these two estimates come from?
```

This is the key to understanding TD learning.

---

### 36.1 The old estimate: $Q(s,a)$

The old estimate is:

$$
Q(s,a)
$$

This is what the agent believed **before** using the new experience.

Suppose the LunarLander is in state $s$.

The lander may be:

```text
falling too fast
slightly tilted
near the landing pad
```

The agent chooses action $a$:

```text
fire main engine
```

Before the agent actually takes this action, it already has some estimate of how good this action is.

That estimate is:

$$
Q(s,\text{fire main engine})
$$

This number comes from the agent's current Q-function.

If we are using tabular Q-learning, it comes from a Q-table:

```text
Q[state, action]
```

If we are using DQN, it comes from the neural network:

$$
Q_{\theta}(s,a)
$$

For example, the DQN may output four Q-values:

```text
Q(s, do nothing)        = -5
Q(s, fire left engine)  = 3
Q(s, fire main engine)  = 10
Q(s, fire right engine) = 1
```

If the chosen action is `fire main engine`, then the old estimate is:

$$
Q_{\theta}(s,\text{fire main engine}) = 10
$$

So the old estimate means:

```text
Before seeing the new result,
how valuable did the agent think this action was?
```

---

### 36.2 The new target: $r+\gamma \max_{a'}Q(s',a')$

Now the agent actually takes the action.

It sends the action to the environment:

```text
fire main engine
```

The environment responds with:

```text
reward r
next state s'
```

For example:

```text
r = -1
s' = the lander is now slower and safer
```

The reward $r$ is real.

It comes directly from the environment.

But the future after $s'$ has not fully happened yet.

The agent does not know the full future return:

$$
r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

So Bellman idea says:

```text
Do not wait for the whole future.

Use the value of the next state
as a summary of the future.
```

After reaching the next state $s'$, the agent asks:

```text
From this next state,
what is the best action I could take?
```

Mathematically, this is:

$$
\max_{a'} Q(s',a')
$$

This value also comes from the agent's current Q-function.

In DQN, it usually comes from the target network:

$$
Q_{\theta^-}(s',a')
$$

For example, suppose the target network estimates:

```text
Q_target(s', do nothing)        = 5
Q_target(s', fire left engine)  = 20
Q_target(s', fire main engine)  = 30
Q_target(s', fire right engine) = 15
```

Then:

$$
\max_{a'}Q_{\theta^-}(s',a') = 30
$$

If:

$$
r=-1
$$

and:

$$
\gamma=0.99
$$

then the new target is:

$$
\begin{aligned}
r+\gamma \max_{a'}Q_{\theta^-}(s',a')
&= -1 + 0.99 \times 30 \\
&= 28.7
\end{aligned}
$$

This means:

```text
After seeing this transition,
the agent now thinks the value of the action should be closer to 28.7.
```

---

### 36.3 TD error compares old belief with new evidence

Now we have two numbers.

The old estimate was:

$$
Q(s,a)=10
$$

The new target is:

$$ r+\gamma \max\_{a'}Q(s',a')=28.7 $$

So the TD error is:

$$ \delta = 28.7 - 10 = 18.7 $$

The TD error is positive.

This means:

```text
The action was better than the agent expected.
```

The agent originally thought the action was worth about $10$.

But after seeing the reward and the next state, the Bellman target says it should be closer to $28.7$.

So the agent should increase its estimate of:

$$
Q(s,a)
$$

In tabular Q-learning, the update is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \delta
$$

where $\alpha$ is the learning rate.

For example, if:

$$
\alpha = 0.1
$$

then:

$$
Q(s,a) \leftarrow 10 + 0.1 \times 18.7 = 11.87
$$

So the Q-value does not jump all the way to $28.7$.

It moves a little bit toward the new target.

This is learning.

---

### 36.4 What if the TD error is negative?

If:

$$
\delta < 0
$$

then:

```text
new target < old estimate
```

This means:

```text
The action was worse than the agent expected.
```

For example, suppose:

$$
Q(s,a)=40
$$

but the new Bellman target is:

$$
y=15
$$

Then:

$$
\delta = 15 - 40 = -25
$$

The agent overestimated the action.

So Q-learning will reduce $Q(s,a)$.

In plain English:

```text
I thought this action was very good.
But this new experience tells me it was not that good.
So I should lower my estimate.
```

---

### 36.5 What if the TD error is zero?

If:

$$
\delta = 0
$$

then:

```text
new target = old estimate
```

This means the current estimate is already consistent with the Bellman target.

The agent is not surprised.

There is little or no need to update.

In plain English:

```text
What happened matches what I expected.
```

---

### 36.6 Is the new target a true label?

Not exactly.

This is very important.

In supervised learning, we usually have true labels.

For example:

```text
input: image
label: cat
```

But in reinforcement learning, nobody gives us the true value:

```text
Q(s,a) = exactly 28.7
```

The Bellman target is not a perfect label.

It is a temporary learning target built from:

```text
1. the real reward r from the environment
2. the next state s' from the environment
3. the agent's current estimate of future value
```

So TD learning has a special flavor:

```text
Use an estimate of the future
to improve the estimate of the present.
```

This is called **bootstrapping**.

---

### 36.7 Why bootstrapping can work

At first, bootstrapping sounds circular.

The agent uses Q-values to update Q-values.

So why does it work?

The answer is that real rewards keep entering the learning process.

For example, in LunarLander, when the lander successfully lands, the environment may give a large positive reward.

That real reward first updates the Q-value of the action right before landing.

Then that improved Q-value affects the state before that.

Then the value propagates further backward.

Imagine a simple chain:

```text
s0 -> s1 -> s2 -> successful landing
```

Suppose the only large reward happens at the end:

```text
successful landing: +100
```

At the beginning, the agent may know nothing:

```text
Q(s0, a0) = 0
Q(s1, a1) = 0
Q(s2, a2) = 0
```

After seeing the final transition:

```text
s2 -> successful landing
```

the agent learns:

$$
Q(s2,a2)
$$

should be high.

Then, when it sees:

```text
s1 -> s2
```

it learns:

$$
Q(s1,a1) \approx 0 + \gamma Q(s2,a2)
$$

So $Q(s1,a1)$ becomes high.

Then, when it sees:

```text
s0 -> s1
```

it learns:

$$
Q(s0,a0) \approx 0 + \gamma Q(s1,a1)
$$

So $Q(s0,a0)$ also becomes high.

This is how future reward propagates backward.

The agent does not need to see the whole future every time.

It learns long-term value one step at a time.

---

### 36.8 How this looks in DQN

In DQN, the same idea is implemented with neural networks.

The old estimate comes from the online network:

$$
Q_{\theta}(s,a)
$$

The new target is:

$$
y = r + \gamma(1-d) \max_{a'}Q_{\theta^-}(s',a')
$$

where:

```text
r:
    real reward from the environment

s':
    real next state from the environment

d:
    whether the episode ended

Q_theta_minus:
    target network used to estimate future value
```

So the DQN TD error is:

$$
\delta = y - Q_{\theta}(s,a)
$$

DQN then trains the online network so that:

$$
Q_{\theta}(s,a)
$$

moves closer to:

$$
y
$$

This is usually done by minimizing a loss such as:

$$
L(\theta) = \left( Q_{\theta}(s,a)-y \right)^2
$$

or the Huber loss.

So in DQN:

```text
old estimate:
    online network's current prediction

new target:
    reward now + target network's estimate of best future value

TD error:
    how much the old estimate differs from the new target
```

---

### 36.9 One-sentence summary

The TD error compares what the agent previously believed with what the latest transition suggests.

```text
old estimate:
    what I thought before

new target:
    what this new experience suggests

TD error:
    how much I should correct my belief
```

This is the core mechanism of Q-learning and DQN.
