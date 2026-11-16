"""Bounded training and greedy evaluation on the same environment contract."""

import numpy as np

from .preprocessing import FrameStack


def run(env, agent, steps, seed=0, training=False):
    if steps <= 0:
        raise ValueError("steps must be positive")
    stack = FrameStack()
    state = stack.reset(env.reset(seed=seed)[0])
    reward_sum = 0.0
    episodes = 0
    losses = []
    for _ in range(steps):
        action = agent.act(state, epsilon=0.1 if training else 0.0)
        image, reward, terminated, truncated, _ = env.step(action)
        next_state = stack.append(image)
        if training:
            losses.append(agent.learn(state, action, reward, next_state, terminated))
        reward_sum += reward
        state = next_state
        if terminated or truncated:
            episodes += 1
            state = stack.reset(env.reset(seed=seed + episodes)[0])
    return {
        "steps": steps,
        "episodes": episodes,
        "reward": float(reward_sum),
        "loss": float(np.mean(losses)) if losses else None,
    }
