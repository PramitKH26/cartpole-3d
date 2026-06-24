"""
Cartpole3DEnv — MuJoCo cartpole on a 2D plane (cart moves in x AND y).

The cart slides on the floor in two directions. The pole sits on a universal
joint (two perpendicular hinges) so it can fall in any direction. The agent
gets 2 actions (force_x, force_y) and must keep the pole vertical.

Compared with the 2D version this is genuinely harder:
  - 4 degrees of freedom (cart_x, cart_y, tilt_x, tilt_y) instead of 2
  - Coupled dynamics — pushing the cart in x doesn't help if the pole's tilting in y
  - The agent must compose two 1D balance policies simultaneously
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import gymnasium
from gymnasium import spaces

_XML_PATH = Path(__file__).parent / "cartpole3d.xml"

gymnasium.register(
    id="Cartpole3D-v0",
    entry_point="cartpole3d_env:Cartpole3DEnv",
    max_episode_steps=500,
)


class Cartpole3DEnv(gymnasium.Env):
    """
    Observation (10-D):
        [cart_x, cart_y, cart_xdot, cart_ydot,
         sin(tilt_x), cos(tilt_x), sin(tilt_y), cos(tilt_y),
         tilt_x_dot, tilt_y_dot]

    Action (2-D continuous):
        [force_x, force_y] ∈ [-1, 1]² — applied via the cart's two motors
        (gear=10 in XML, so each component maps to ±10 N).

    Reward (per step):
        r = pole_tip_z - 0.01*||u||² - 0.001*(cart_x² + cart_y²)

        pole_tip_z is the z-coordinate of the pole tip, in [-1, +1].
        When the pole is fully upright the tip sits at +1 (relative to the
        pivot's z). When horizontal it's 0. When upside-down it's -1.
        This is the smooth multi-axis generalization of cos(θ) from the 2D env.

    Termination:
        - cart out of bounds: |cart_x| > 2.4 or |cart_y| > 2.4
        - pole tipped too far: tilt magnitude > 0.4 rad (~23°), measured as
          arccos(pole_tip_z / 1.0) — angle between pole and world +z axis.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    CART_LIMIT: float       = 2.4
    TILT_LIMIT_RAD: float   = 0.4
    POSE_INIT_NOISE: float  = 0.03
    VEL_INIT_NOISE: float   = 0.03
    REWARD_ACTION_COEF: float = 0.01
    REWARD_POS_COEF: float    = 0.001
    RENDER_WIDTH: int  = 480
    RENDER_HEIGHT: int = 480

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        frame_skip: int = 5,
    ):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip

        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.data = mujoco.MjData(self.model)

        # qpos addresses for each joint (resolved by name)
        joints = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                  for n in ("slide_x", "slide_y", "hinge_x", "hinge_y")}
        self._qx  = self.model.jnt_qposadr[joints["slide_x"]]
        self._qy  = self.model.jnt_qposadr[joints["slide_y"]]
        self._qtx = self.model.jnt_qposadr[joints["hinge_x"]]
        self._qty = self.model.jnt_qposadr[joints["hinge_y"]]
        self._vx  = self.model.jnt_dofadr[joints["slide_x"]]
        self._vy  = self.model.jnt_dofadr[joints["slide_y"]]
        self._vtx = self.model.jnt_dofadr[joints["hinge_x"]]
        self._vty = self.model.jnt_dofadr[joints["hinge_y"]]

        self._tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pole_tip")
        self._pivot_z_offset = 0.05   # cart height + pivot pos (matches XML)

        obs_high = np.full(10, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.float32(-1.0), high=np.float32(1.0), shape=(2,), dtype=np.float32,
        )

        self._viewer = None
        self._renderer = None
        self._step_count = 0
        self.np_random: np.random.Generator = np.random.default_rng()

    # ─── helpers ──────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        tx, ty = self.data.qpos[self._qtx], self.data.qpos[self._qty]
        return np.array([
            self.data.qpos[self._qx], self.data.qpos[self._qy],
            self.data.qvel[self._vx], self.data.qvel[self._vy],
            np.sin(tx), np.cos(tx), np.sin(ty), np.cos(ty),
            self.data.qvel[self._vtx], self.data.qvel[self._vty],
        ], dtype=np.float32)

    def _pole_tip_height(self) -> float:
        """z-coordinate of the pole tip relative to the pivot, in [-1, +1]."""
        return float(self.data.geom_xpos[self._tip_id][2]) - self._pivot_z_offset

    def _tilt_from_upright(self) -> float:
        """Angle between pole axis and world +z, in radians ∈ [0, π]."""
        h = self._pole_tip_height()
        return float(np.arccos(np.clip(h, -1.0, 1.0)))

    # ─── gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = self.np_random
        mujoco.mj_resetData(self.model, self.data)

        # Small random perturbations on all four DoFs
        self.data.qpos[self._qx]  = rng.uniform(-self.POSE_INIT_NOISE, self.POSE_INIT_NOISE)
        self.data.qpos[self._qy]  = rng.uniform(-self.POSE_INIT_NOISE, self.POSE_INIT_NOISE)
        self.data.qpos[self._qtx] = rng.uniform(-self.POSE_INIT_NOISE, self.POSE_INIT_NOISE)
        self.data.qpos[self._qty] = rng.uniform(-self.POSE_INIT_NOISE, self.POSE_INIT_NOISE)
        self.data.qvel[self._vx]  = rng.uniform(-self.VEL_INIT_NOISE, self.VEL_INIT_NOISE)
        self.data.qvel[self._vy]  = rng.uniform(-self.VEL_INIT_NOISE, self.VEL_INIT_NOISE)
        self.data.qvel[self._vtx] = rng.uniform(-self.VEL_INIT_NOISE, self.VEL_INIT_NOISE)
        self.data.qvel[self._vty] = rng.uniform(-self.VEL_INIT_NOISE, self.VEL_INIT_NOISE)

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        ctrl = np.clip(np.asarray(action, dtype=np.float64).flatten()[:2], -1.0, 1.0)
        self.data.ctrl[0] = float(ctrl[0])
        self.data.ctrl[1] = float(ctrl[1])

        reward = 0.0
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            tip_z  = self._pole_tip_height()
            cart_x = self.data.qpos[self._qx]
            cart_y = self.data.qpos[self._qy]
            reward += (
                tip_z
                - self.REWARD_ACTION_COEF * float(ctrl @ ctrl)
                - self.REWARD_POS_COEF * (cart_x**2 + cart_y**2)
            ) / self.frame_skip

        self._step_count += 1

        cart_x = float(self.data.qpos[self._qx])
        cart_y = float(self.data.qpos[self._qy])
        tilt   = self._tilt_from_upright()

        terminated = bool(
            abs(cart_x) > self.CART_LIMIT
            or abs(cart_y) > self.CART_LIMIT
            or tilt > self.TILT_LIMIT_RAD
        )
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "cart_x": cart_x, "cart_y": cart_y,
            "tilt": tilt,
            "tip_z": self._pole_tip_height(),
            "force_x": float(ctrl[0]) * 10.0,
            "force_y": float(ctrl[1]) * 10.0,
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
            return self._renderer.render()

        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
