"""Checks for the short-assignment PPO implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


CODE_DIR = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(CODE_DIR))

from rl_utils import ReplayBuffer, compute_advantage  # noqa: E402


def load_ppo_class():
    try:
        import gym  # noqa: F401
    except ImportError:
        import gymnasium

        sys.modules["gym"] = gymnasium

    import ppo

    return ppo.PPO


def check_buffer_alignment() -> None:
    buffer = ReplayBuffer(capacity=5)
    for index in range(5):
        buffer.add(index, 10 + index, 100 + index, 1000 + index, False)

    states, actions, rewards, next_states, dones = buffer.sample(3)
    assert len(states) == len(actions) == len(rewards) == 3
    assert len(next_states) == len(dones) == 3
    for state, action, reward, next_state in zip(
        states, actions, rewards, next_states
    ):
        assert action == state + 10
        assert reward == state + 100
        assert next_state == state + 1000

    try:
        buffer.sample(6)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized sample should fail")


def check_buffer_dictionary_data() -> None:
    buffer = ReplayBuffer(capacity=2)
    agents = ("speaker", "listener")
    first = {agent: index for index, agent in enumerate(agents)}
    second = {agent: index + 2 for index, agent in enumerate(agents)}
    buffer.add(first, first, first, second, {a: False for a in agents})
    buffer.add(second, second, second, first, {a: True for a in agents})

    states, actions, rewards, next_states, dones = buffer.sample(2)
    for index in range(2):
        assert states[index] is actions[index]
        assert actions[index] is rewards[index]
        assert states[index][agents[0]] in (0, 2)
        assert set(next_states[index]) == set(agents)
        assert set(dones[index]) == set(agents)


def check_gae_shape_and_values() -> None:
    td_delta = torch.tensor([1.0, 2.0, 3.0])
    advantage = compute_advantage(0.5, 0.5, td_delta)
    expected = torch.tensor([1.6875, 2.75, 3.0])
    assert torch.allclose(advantage, expected)


def check_clipped_gradient_directions() -> None:
    def gradient(ratio_value: float, advantage_value: float) -> float:
        ratio = torch.tensor(ratio_value, requires_grad=True)
        advantage_value = torch.tensor(advantage_value)
        clipped = torch.clamp(ratio, 0.8, 1.2) * advantage_value
        loss = -torch.min(ratio * advantage_value, clipped)
        loss.backward()
        return float(ratio.grad)

    positive_grad = gradient(1.5, 1.0)
    negative_grad = gradient(0.6, -1.0)
    assert positive_grad == 0.0
    assert negative_grad == 0.0


def check_ppo_update() -> None:
    PPO = load_ppo_class()
    torch.manual_seed(0)
    agent = PPO(
        state_dim=4,
        hidden_dim=16,
        action_dim=2,
        actor_lr=1e-3,
        critic_lr=1e-2,
        lmbda=0.95,
        epochs=1,
        eps=0.2,
        gamma=0.98,
        device=torch.device("cpu"),
    )
    states = [
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.1, 0.0],
        [0.2, 0.0, -0.1, 0.0],
    ]
    transition = {
        "states": states,
        "actions": [0, 1, 0],
        "next_states": states[1:] + [states[-1]],
        "rewards": [1.0, 1.0, 1.0],
        "dones": [False, False, True],
    }
    before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    agent.update(transition)
    assert any(
        not torch.equal(parameter, original)
        for parameter, original in zip(agent.actor.parameters(), before)
    )
    action = agent.take_action(states[0])
    assert isinstance(action, int)
    assert action in (0, 1)


def main() -> int:
    check_buffer_alignment()
    check_buffer_dictionary_data()
    check_gae_shape_and_values()
    check_clipped_gradient_directions()
    check_ppo_update()
    print("PPO utility checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
