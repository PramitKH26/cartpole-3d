"""
CartpoleEnv: MuJoCo 2-D cartpole with Gymnasium API.

Pole-upright goal.  qpos = [cart_x, theta], where theta=pi means straight up.
theta_from_upright = wrap_to_pi(theta - pi)  →  0 at upright, ±pi at downward.

Design choices documented inline.  See class docstring for reward rationale.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding

# ── Module-level registration so `gym.make("Cartpole2D-v0")` works when the
#    module is imported either directly or via a package __init__.
gymnasium.register(
    id="Cartpole2D-v0",
    entry_point="cartpole_env:CartpoleEnv",
    max_episode_steps=500,
)

_XML_PATH = Path(__file__).parent / "cartpole.xml"


def _wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class CartpoleEnv(gymnasium.Env):
    """
    MuJoCo 2-D cartpole — pole-upright balance task.

    Observation (5-D):
        [cart_x, cart_xdot, sin(theta_from_upright), cos(theta_from_upright), theta_dot]
        sin/cos encoding avoids the ±pi discontinuity that breaks policy gradients.

    Action:
        Continuous Box(-1, 1) by default.  Pass discrete_actions=True for
        Discrete(2) → {-1, +1} to match classic CartPole-v1 call patterns.

    Reward shaping (per sim step, summed over frame_skip):
        r = cos(theta_from_upright) - 0.01*u² - 0.001*cart_x²

        Rationale: cos(theta) gives +1 at upright and −1 fully inverted — a
        smooth, informative dense reward.  The action-penalty (0.01*u²) discourages
        chattering and keeps the pole centered via torque economy.  The position
        penalty (0.001*x²) softly discourages drift toward the rails without
        dominating the balance signal.  All coefficients are small relative to the
        alive bonus encoded in cos(0)=1, so the dominant gradient still points
        toward upright.

    Termination (hard):
        |cart_x| > CART_X_LIMIT  or  |theta_from_upright| > THETA_LIMIT_RAD
    Truncation (soft):
        step ≥ max_episode_steps
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    # ── Class constants (no magic numbers in step/reset) ──────────────────────
    CART_X_LIMIT: float = 2.4          # metres — matches slider range in XML
    THETA_LIMIT_RAD: float = 0.4       # ~23° from upright
    CART_INIT_NOISE: float = 0.05      # uniform half-width for reset perturbations
    VEL_INIT_NOISE: float = 0.05
    REWARD_ACTION_COEF: float = 0.01
    REWARD_POS_COEF: float = 0.001
    RENDER_WIDTH: int = 480
    RENDER_HEIGHT: int = 480

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        frame_skip: int = 5,
        discrete_actions: bool = False,
    ):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.discrete_actions = discrete_actions

        # Load model once; share between all internal data objects.
        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.data = mujoco.MjData(self.model)

        # Joint/actuator indices (resolved by name so XML changes don't silently break us)
        self._slider_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slider")
        self._hinge_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
        # qpos addresses
        self._qpos_slider = self.model.jnt_qposadr[self._slider_id]
        self._qpos_hinge  = self.model.jnt_qposadr[self._hinge_id]
        # qvel addresses
        self._qvel_slider = self.model.jnt_dofadr[self._slider_id]
        self._qvel_hinge  = self.model.jnt_dofadr[self._hinge_id]

        # Spaces
        obs_high = np.full(5, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        if discrete_actions:
            self.action_space = spaces.Discrete(2)  # 0→-1, 1→+1
        else:
            self.action_space = spaces.Box(
                low=np.float32(-1.0), high=np.float32(1.0), shape=(1,), dtype=np.float32
            )

        # Rendering state (lazy-init)
        self._viewer = None
        self._renderer = None

        self._step_count: int = 0
        self.np_random: np.random.Generator = np.random.default_rng()

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        cart_x   = self.data.qpos[self._qpos_slider]
        theta    = self.data.qpos[self._qpos_hinge]
        cart_xdot = self.data.qvel[self._qvel_slider]
        theta_dot = self.data.qvel[self._qvel_hinge]
        tfu = _wrap_to_pi(theta - np.pi)  # theta_from_upright
        # sin/cos encoding avoids ±pi discontinuity
        return np.array(
            [cart_x, cart_xdot, np.sin(tfu), np.cos(tfu), theta_dot],
            dtype=np.float32,
        )

    def _action_to_ctrl(self, action) -> float:
        if self.discrete_actions:
            # Discrete(2): 0 → -1.0, 1 → +1.0
            return float(action) * 2.0 - 1.0
        return float(np.clip(np.asarray(action, dtype=np.float64).flat[0], -1.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Use parent-seeded np_random (gymnasium sets self.np_random in super().reset)
        rng = self.np_random

        mujoco.mj_resetData(self.model, self.data)

        # qpos[0] = cart_x near 0, qpos[1] = theta near pi (upright)
        self.data.qpos[self._qpos_slider] = rng.uniform(
            -self.CART_INIT_NOISE, self.CART_INIT_NOISE
        )
        self.data.qpos[self._qpos_hinge] = np.pi + rng.uniform(
            -self.CART_INIT_NOISE, self.CART_INIT_NOISE
        )
        self.data.qvel[self._qvel_slider] = rng.uniform(
            -self.VEL_INIT_NOISE, self.VEL_INIT_NOISE
        )
        self.data.qvel[self._qvel_hinge] = rng.uniform(
            -self.VEL_INIT_NOISE, self.VEL_INIT_NOISE
        )

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        ctrl = self._action_to_ctrl(action)
        self.data.ctrl[0] = ctrl

        reward = 0.0
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            cart_x = self.data.qpos[self._qpos_slider]
            theta  = self.data.qpos[self._qpos_hinge]
            tfu    = _wrap_to_pi(theta - np.pi)
            # Accumulate reward every substep so frame_skip doesn't change scale.
            # Divide by frame_skip to keep reward magnitude consistent regardless
            # of how many substeps are taken.
            reward += (
                np.cos(tfu)
                - self.REWARD_ACTION_COEF * ctrl ** 2
                - self.REWARD_POS_COEF * cart_x ** 2
            ) / self.frame_skip

        self._step_count += 1

        cart_x = float(self.data.qpos[self._qpos_slider])
        theta  = float(self.data.qpos[self._qpos_hinge])
        tfu    = float(_wrap_to_pi(theta - np.pi))

        terminated = bool(
            abs(cart_x) > self.CART_X_LIMIT
            or abs(tfu) > self.THETA_LIMIT_RAD
        )
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "theta": theta,
            "theta_from_upright": tfu,
            "cart_x": cart_x,
            "applied_force": ctrl * 10.0,  # gear=10
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model, height=self.RENDER_HEIGHT, width=self.RENDER_WIDTH
                )
            self._renderer.update_scene(self.data)
            return self._renderer.render()  # uint8 HxWx3

        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
