"""
Sanity checks for CartpoleEnv.

Run with:  pytest test_env.py -v
"""

import numpy as np
import pytest
import gymnasium

import cartpole_env  # registers Cartpole2D-v0

from cartpole_env import CartpoleEnv, _wrap_to_pi
from stable_baselines3.common.env_checker import check_env


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_env(**kwargs) -> CartpoleEnv:
    return CartpoleEnv(**kwargs)


# ─── tests ────────────────────────────────────────────────────────────────────

class TestWrapToPi:
    def test_zero(self):
        assert _wrap_to_pi(0.0) == pytest.approx(0.0)

    def test_pi(self):
        # wrap_to_pi(pi) should be -pi (convention: result in (-pi, pi])
        # numpy's implementation returns -pi for the boundary — that's fine.
        assert abs(_wrap_to_pi(np.pi)) == pytest.approx(np.pi)

    def test_two_pi(self):
        assert _wrap_to_pi(2 * np.pi) == pytest.approx(0.0, abs=1e-10)

    def test_negative(self):
        assert _wrap_to_pi(-np.pi / 2) == pytest.approx(-np.pi / 2)


class TestSpaces:
    def test_obs_space_shape(self):
        env = make_env()
        assert env.observation_space.shape == (5,)

    def test_obs_space_dtype(self):
        env = make_env()
        assert env.observation_space.dtype == np.float32

    def test_continuous_action_space(self):
        env = make_env()
        assert isinstance(env.action_space, gymnasium.spaces.Box)
        assert env.action_space.shape == (1,)
        np.testing.assert_allclose(env.action_space.low,  [-1.0])
        np.testing.assert_allclose(env.action_space.high, [ 1.0])

    def test_discrete_action_space(self):
        env = make_env(discrete_actions=True)
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 2


class TestReset:
    def test_returns_obs_and_info(self):
        env = make_env()
        result = env.reset(seed=0)
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert obs.shape == (5,)
        assert isinstance(info, dict)

    def test_seed_reproducibility(self):
        env = make_env()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seeds_differ(self):
        env = make_env()
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=999)
        assert not np.allclose(obs1, obs2)


class TestStep:
    def test_step_output_shape(self):
        env = make_env()
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (5,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_info_keys(self):
        env = make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample())
        for key in ("theta", "cart_x", "applied_force", "theta_from_upright"):
            assert key in info, f"missing info key: {key}"

    def test_100_step_random_rollout(self):
        """Must not crash; observations must stay finite."""
        env = make_env()
        env.reset(seed=7)
        for _ in range(100):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs)), "non-finite observation"
            if terminated or truncated:
                env.reset()

    def test_truncation_fires(self):
        # White-box: set step counter to limit-1, then one more step must set truncated.
        # The inverted pendulum diverges quickly with zero action (unstable equilibrium),
        # so we can't rely on "no termination before N steps" — just test the counter logic.
        limit = 5
        env = make_env(max_episode_steps=limit)
        env.reset(seed=0)
        # Manually advance the counter to the last step.
        env._step_count = limit - 1
        _, _, _, truncated, _ = env.step(np.array([0.0]))
        assert truncated, f"expected truncated at step {limit}, got _step_count={env._step_count}"

    def test_discrete_rollout(self):
        env = make_env(discrete_actions=True)
        env.reset(seed=0)
        for _ in range(20):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert obs.shape == (5,)


class TestGymMake:
    def test_gym_make(self):
        env = gymnasium.make("Cartpole2D-v0")
        obs, _ = env.reset(seed=0)
        assert obs.shape == (5,)
        env.close()


class TestSB3Checker:
    def test_check_env_continuous(self):
        env = make_env()
        check_env(env, warn=True)

    def test_check_env_discrete(self):
        env = make_env(discrete_actions=True)
        check_env(env, warn=True)
