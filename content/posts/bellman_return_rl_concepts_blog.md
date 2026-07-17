---
title: "Why Bellman Equation Exists: From Reward to Return to Q-Value (2)"
date: 2026-02-20
draft: false
# math: true
tags: ["reinforcement-learning", "bellman-equation", "dqn", "machine-learning"]
categories: ["Machine Learning"]
# summary: "A high-level but intuitive explanation of how RL concepts are built: state, environment, reward, return, value function, Q-value, Bellman equation, and DQN."
---

# Why Bellman Equation Exists: From Reward to Return to Q-Value

When I first learned reinforcement learning, one thing confused me a lot:

> Why do we define return as a discounted sum of rewards?
> Why does the Bellman equation suddenly appear?
> Is the Bellman equation the definition of return, or is it derived from return?

This post is my attempt to explain the logic from top to bottom.

The goal is not to memorize formulas.

The goal is to understand the order in which the concepts appear.

In reinforcement learning, the ideas are built like a ladder:

```text
state
  -> action
  -> environment transition
  -> reward
  -> return
  -> value function
  -> Q-value
  -> Bellman equation
  -> Q-learning / DQN
```

Once this order is clear, the Bellman equation feels much less mysterious.

---

## 1. The agent starts with a state

At time step $t$, the agent observes a state:

$$
s_t
$$

A state is just the information the agent uses to make a decision.

For example, in LunarLander, the state may include the lander's position, velocity, angle, angular velocity, and whether each leg touches the ground.

The state answers:

> Where am I now?

---

## 2. The agent chooses an action

Given the state $s_t$, the agent chooses an action:

$$
a_t
$$

In LunarLander, the action could be:

```text
do nothing
fire left engine
fire main engine
fire right engine
```

The action answers:

> What do I do now?

---

## 3. The environment decides what happens next

After the agent chooses $a_t$, the environment responds.

It gives two things:

$$
s_{t+1}, r_{t+1}
$$

That means:

```text
next state:     s_{t+1}
immediate reward: r_{t+1}
```

This is the basic interaction loop of reinforcement learning:

$$
s_t, a_t \longrightarrow s_{t+1}, r_{t+1}
$$

More formally, the environment has two important parts.

First, the transition dynamics:

$$
P(s_{t+1} \mid s_t, a_t)
$$

This tells us how likely the next state is.

Second, the reward function:

$$
R(s_t, a_t, s_{t+1})
$$

This tells us how much reward the agent receives after this transition.

So the environment answers:

> If you do this action in this state, what happens next, and how much reward do you get?

---

## 4. Reward function vs reward

This is an important distinction.

The reward function is the rule.

The reward is the actual number received at one step.

For example, a reward function may say:

```python
if landed_successfully:
    reward = +100
elif crashed:
    reward = -100
else:
    reward = small_shaping_reward
```

The function is the rule.

The actual number returned by the environment at time $t+1$ is:

$$
r_{t+1}
$$

So:

```text
reward function = how the environment gives scores
reward          = the actual score received at one step
```

In standard RL theory, the reward function is part of the environment.

But in engineering practice, if we build our own environment, then we often design the reward function ourselves.

Reward function is just an environment info that we created.

---

## 5. One-step reward is not enough

Now comes the real reason reinforcement learning is hard.

The agent should not only ask:

> Is this action good right now?

It should ask:

> Where will this action lead me in the future?

For example, in LunarLander, firing the main engine may cost fuel right now. So the immediate reward may be slightly worse.

But firing the engine may slow the lander down and prevent a crash later.

So a locally bad action may be globally good.

This is why RL needs something bigger than one-step reward.

That bigger object is called return.

---

## 6. Return is the total future reward

The return from time $t$ is usually written as:

$$
G_t
$$

It represents the total future reward starting from time $t$.

The simplest idea is:

$$
G_t = r_{t+1} + r_{t+2} + r_{t+3} + \cdots
$$

This says:

> Add up all future rewards.

For short finite tasks, this definition can work.

But for long or infinite tasks, there is a problem.

If the agent receives reward $+1$ forever, then:

$$
G_t = 1 + 1 + 1 + 1 + \cdots = \infty
$$

Then many policies may all look infinitely good.

That is not useful.

So we introduce a discount factor.

---

## 7. Discounted return

The standard discounted return is:

$$
G_t
=
r_{t+1}
+
\gamma r_{t+2}
+
\gamma^2 r_{t+3}
+
\gamma^3 r_{t+4}
+
\cdots
$$

or more compactly:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

Here $\gamma$ is the discount factor:

$$
0 \leq \gamma \leq 1
$$

If $\gamma = 0$, the agent only cares about the immediate reward.

If $\gamma$ is close to $1$, the agent cares a lot about long-term consequences.

For example, if $\gamma = 0.99$, then future rewards are still very important.

---

## 8. Why use powers of gamma?

This is the key question.

Why do we use:

$$
1, \gamma, \gamma^2, \gamma^3, \ldots
$$

Why not use something like:

$$
1, 0.99, 0.98, 0.93, \ldots
$$

The short answer is:

> We can use other weighting schemes, but powers of $\gamma$ give us time consistency and recursive structure.

Let us unpack that.

Using powers of $\gamma$ means:

> Every time we move one step into the future, we multiply importance by the same factor.

For example, if $\gamma = 0.99$:

```text
reward 1 step in the future:   weight = 0.99
reward 2 steps in the future:  weight = 0.99 × 0.99 = 0.99²
reward 3 steps in the future:  weight = 0.99 × 0.99 × 0.99 = 0.99³
```

So the $k$-step future reward naturally receives weight $\gamma^k$.

This is not the only possible choice.

But it gives us a beautiful property:

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

This one-line recursive structure is the beginning of the Bellman equation.

---

## 9. Bellman equation is not the definition of return

This is the most important conceptual correction.

The Bellman equation does not come first.

The return comes first.

The Bellman equation is derived from the discounted return.

Start from:

$$
G_t
=
r_{t+1}
+
\gamma r_{t+2}
+
\gamma^2 r_{t+3}
+
\gamma^3 r_{t+4}
+
\cdots
$$

Now write the return from the next time step:

$$
G_{t+1}
=
r_{t+2}
+
\gamma r_{t+3}
+
\gamma^2 r_{t+4}
+
\cdots
$$

Multiply $G_{t+1}$ by $\gamma$:

$$
\gamma G_{t+1}
=
\gamma r_{t+2}
+
\gamma^2 r_{t+3}
+
\gamma^3 r_{t+4}
+
\cdots
$$

Then add $r_{t+1}$:

$$
r_{t+1} + \gamma G_{t+1}
=
r_{t+1}
+
\gamma r_{t+2}
+
\gamma^2 r_{t+3}
+
\gamma^3 r_{t+4}
+
\cdots
$$

This is exactly $G_t$.

Therefore:

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

So the Bellman structure is not arbitrary.

It is a consequence of discounted return.

---

## 10. Why is this recursive structure so powerful?

Because the agent usually does not see the entire future.

At one interaction step, the agent only observes:

$$
(s_t, a_t, r_{t+1}, s_{t+1})
$$

This is just one transition.

But the agent wants to learn long-term return:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

So the question is:

> How can we learn long-term value from short local experience?

The Bellman equation gives the answer:

> The value of now equals immediate reward plus the value of next.

This allows local learning of long-term consequences.

That is the central trick of reinforcement learning.

---

## 11. From return to value function

The return $G_t$ is a random variable.

Why random?

Because the future may be uncertain.

The same action in the same state may lead to different future states if the environment is stochastic.

So instead of only looking at one realized return, we often care about expected return.

The state-value function is:

$$
V^{\pi}(s) = \mathbb{E}_{\pi}[G_t \mid s_t = s]
$$

This means:

> If I start in state $s$ and follow policy $\pi$, how much return do I expect to get?

Here $\pi$ is the policy.

A policy is the agent's behavior rule:

$$
\pi(a \mid s)
$$

It tells us the probability of choosing action $a$ in state $s$.

So the value function is not the reward itself.

It is a prediction of future return.

---

## 12. From value function to Q-value

The value function $V(s)$ tells us how good a state is.

But when choosing an action, we need to know how good each action is.

So we define the action-value function, also called the Q-function:

$$
Q^{\pi}(s,a)
=
\mathbb{E}_{\pi}[G_t \mid s_t=s, a_t=a]
$$

This means:

> If I am in state $s$, take action $a$, and then follow policy $\pi$, how much return do I expect?

This is the number DQN tries to learn.

In plain language:

```text
reward:  how good was this one step?
return:  how good is the whole future?
Q-value: how good is this action considering the whole future?
```

---

## 13. Bellman equation for V

Because return satisfies:

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

the value function also has a recursive form.

For a fixed policy $\pi$:

$$
V^{\pi}(s)
=
\mathbb{E}_{\pi}
\left[
    r_{t+1} + \gamma V^{\pi}(s_{t+1})
    \mid s_t=s
\right]
$$

This says:

> The value of a state is the expected immediate reward plus the discounted value of the next state.

This is a Bellman equation.

Again, it is not magic.

It is return recursion plus expectation.

---

## 14. Bellman equation for Q

Similarly, for the Q-function:

$$
Q^{\pi}(s,a)
=
\mathbb{E}_{\pi}
\left[
    r_{t+1}
    +
    \gamma Q^{\pi}(s_{t+1}, a_{t+1})
    \mid s_t=s, a_t=a
\right]
$$

This says:

> The value of taking action $a$ in state $s$ equals the immediate reward plus the discounted value of the next action.

If we are trying to find the best possible policy, we use the optimal Q-function:

$$
Q^*(s,a)
=
\mathbb{E}
\left[
    r_{t+1}
    +
    \gamma \max_{a'} Q^*(s_{t+1}, a')
    \mid s_t=s, a_t=a
\right]
$$

The $\max_{a'}$ means:

> After arriving at the next state, choose the best next action.

This is the key equation behind Q-learning.

---

## 15. From Bellman equation to Q-learning

The optimal Bellman equation says:

$$
Q^*(s_t,a_t)
\approx
r_{t+1}
+
\gamma \max_{a'} Q^*(s_{t+1}, a')
$$

Q-learning turns this idea into an update rule.

The target is:

$$
y_t
=
r_{t+1}
+
\gamma \max_{a'} Q(s_{t+1}, a')
$$

Then we update our estimate of $Q(s_t,a_t)$ toward $y_t$.

In tabular Q-learning, this looks like:

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\left[
    y_t - Q(s_t,a_t)
\right]
$$

where $\alpha$ is the learning rate.

The term:

$$
y_t - Q(s_t,a_t)
$$

is called the temporal-difference error, or TD error.

It measures:

> How wrong was my current Q-value estimate compared with the Bellman target?

---

## 16. From Q-learning to DQN

Q-learning works well when the number of states is small.

But in many real problems, the state space is too large.

For example, an image input may contain thousands or millions of possible states.

So instead of storing a table:

$$
Q(s,a)
$$

DQN uses a neural network:

$$
Q_{\theta}(s,a)
$$

where $\theta$ represents the network parameters.

The DQN target is:

$$
y_t
=
r_{t+1}
+
\gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a')
$$

Here $Q_{\theta^-}$ is the target network.

The DQN loss is:

$$
L(\theta)
=
\left(
    Q_{\theta}(s_t,a_t) - y_t
\right)^2
$$

or sometimes the Huber loss is used instead of squared error.

So DQN is basically:

```text
Bellman equation
    + neural network function approximation
    + replay buffer
    + target network
```

The neural network is not predicting the immediate reward.

It is predicting expected discounted return.

That is the Q-value.

---

## 17. Why not use a different return formula?

We can.

The discounted return is not the only possible objective.

For finite-horizon tasks, we can use undiscounted return:

$$
G_t = r_{t+1} + r_{t+2} + \cdots + r_T
$$

For continuing tasks, we can use average reward:

$$
\lim_{T \to \infty}
\frac{1}{T}
\sum_{t=1}^{T} r_t
$$

For risk-sensitive RL, we may care not only about the mean return, but also about variance or worst-case outcomes.

So other objectives are possible.

But discounted return is extremely popular because it gives us:

```text
1. finite values for infinite-horizon tasks
2. time consistency
3. recursive Bellman structure
4. efficient local updates
5. compatibility with Q-learning and DQN
```

The most important one is recursive structure.

Without it, we cannot easily turn long-term optimization into one-step learning.

---

## 18. The big picture

Here is the full conceptual chain:

```text
1. The agent observes a state s_t.

2. The agent chooses an action a_t.

3. The environment responds with:
       reward r_{t+1}
       next state s_{t+1}

4. The reward function is the environment's rule for producing reward.

5. The realized reward is the actual number received at one step.

6. The return G_t is the discounted sum of future rewards.

7. The value function V(s) predicts expected return from a state.

8. The Q-function Q(s,a) predicts expected return after taking an action.

9. Because discounted return is recursive, we get the Bellman equation.

10. Because of the Bellman equation, we can learn long-term value from one-step transitions.

11. Q-learning uses this idea in a table.

12. DQN uses a neural network to approximate the Q-table.
```

This is the cleanest way to understand reinforcement learning.

Bellman equation is not just another formula.

It is the bridge between short-term experience and long-term decision making.

---

## 19. A simple mental model

If I had to summarize the whole idea in one sentence, I would say:

> RL is about learning how much future a current action creates.

The reward tells us what happened now.

The return tells us how good the future is.

The Q-value estimates that future.

The Bellman equation lets us estimate that future one step at a time.

And DQN uses a neural network to learn this estimate when the state space is too large for a table.

That is the high-level story.

---

## 20. Final summary

The Bellman equation is not chosen because other formulas are impossible.

Other return definitions are possible.

But discounted return has a special recursive property:

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

This property allows us to write:

$$
Q(s_t,a_t)
\approx
r_{t+1}
+
\gamma \max_{a'} Q(s_{t+1},a')
$$

And this is exactly the idea behind Q-learning and DQN.

So the real reason we like the Bellman form is not just mathematical beauty.

It is computational power.

It lets us use local data to learn long-term consequences.

That is the heart of reinforcement learning.
