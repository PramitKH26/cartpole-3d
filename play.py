"""
Interactive cartpole: steer with arrow keys, watch reward/punishment live.

Run with:
    cd cartpole_rl && mjpython play.py

Controls:
    LEFT  arrow  — push cart left  (-10 N)
    RIGHT arrow  — push cart right (+10 N)
    Hold nothing — zero force (pole will fall)
    Q / ESC      — quit
"""

import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from cartpole_env import CartpoleEnv, _wrap_to_pi

# ── ANSI helpers ──────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"
CLEAR  = "\033[2J\033[H"      # clear screen + move cursor to top

def bar(value, lo, hi, width=20, fill="█", empty="░"):
    """ASCII progress bar, value clamped to [lo, hi]."""
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    n = round(frac * width)
    return fill * n + empty * (width - n)

def angle_color(tfu_abs):
    if tfu_abs < 0.15:  return GREEN
    if tfu_abs < 0.30:  return YELLOW
    return RED

def reward_color(r):
    if r > 0.95: return GREEN
    if r > 0.80: return YELLOW
    return RED

# ── Shared state (written by key callback, read by main loop) ─────────────────
_action   = 0.0      # current action in [-1, 1]
_quit     = False

GLFW_LEFT  = 263
GLFW_RIGHT = 262
GLFW_Q     = 81
GLFW_ESC   = 256

def key_callback(keycode):
    global _action, _quit
    if   keycode == GLFW_RIGHT: _action =  1.0
    elif keycode == GLFW_LEFT:  _action = -1.0
    elif keycode in (GLFW_Q, GLFW_ESC): _quit = True

# Key-release: MuJoCo viewer fires the callback on press only (no release event),
# so we decay the action to zero after a short hold using a timer.
_release_timer = None
HOLD_DURATION  = 0.15   # seconds before action resets to zero

def key_callback_with_decay(keycode):
    global _release_timer
    key_callback(keycode)
    if keycode in (GLFW_LEFT, GLFW_RIGHT):
        if _release_timer: _release_timer.cancel()
        _release_timer = threading.Timer(HOLD_DURATION, _reset_action)
        _release_timer.start()

def _reset_action():
    global _action
    _action = 0.0

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _quit

    env = CartpoleEnv(render_mode=None, frame_skip=5)

    # Launch passive viewer with key callback
    viewer = mujoco.viewer.launch_passive(
        env.model, env.data,
        key_callback=key_callback_with_decay,
    )
    env._viewer = viewer   # so env.close() can shut it down

    obs, _ = env.reset(seed=0)

    ep          = 1
    ep_reward   = 0.0
    ep_steps    = 0
    total_steps = 0
    history      = []    # last N rewards for sparkline
    MAX_HIST     = 40

    print(CLEAR, end="", flush=True)

    try:
        while viewer.is_running() and not _quit:
            t_start = time.perf_counter()

            act = np.array([_action])
            obs, reward, terminated, truncated, info = env.step(act)

            ep_reward += reward
            ep_steps  += 1
            total_steps += 1

            tfu      = info["theta_from_upright"]
            cart_x   = info["cart_x"]
            force    = info["applied_force"]

            history.append(reward)
            if len(history) > MAX_HIST:
                history.pop(0)

            # ── Terminal display ──────────────────────────────────────────────
            acol  = angle_color(abs(tfu))
            rcol  = reward_color(reward)

            # Sparkline: map reward 0.5–1.0 to chars
            sparks = " ▁▂▃▄▅▆▇█"
            def spark(r):
                idx = int(max(0.0, min(0.999, (r - 0.5) / 0.5)) * (len(sparks) - 1))
                return sparks[idx]
            sparkline = "".join(spark(r) for r in history)

            direction = ""
            if   force >  0.5: direction = f"{GREEN}>>> pushing RIGHT{RESET}"
            elif force < -0.5: direction = f"{BLUE}<<< pushing LEFT{RESET}"
            else:               direction = f"{DIM}    (no force)   {RESET}"

            done_msg = ""
            if terminated:
                done_msg = (f"\n  {RED}{BOLD}★ TERMINATED{RESET}  "
                            f"pole={tfu:+.3f} rad  cart_x={cart_x:+.3f} m")
            elif truncated:
                done_msg = (f"\n  {GREEN}{BOLD}★ SURVIVED 500 steps!{RESET}")

            lines = [
                f"{BOLD}MuJoCo Cartpole  —  episode {ep}{RESET}",
                "",
                f"  {BOLD}Pole angle{RESET}  θ̃ = {acol}{tfu:+.4f} rad{RESET}",
                f"  {acol}  |{'█' * round(abs(tfu)/0.4*20):.<20}|{RESET}  limit ±0.4 rad",
                "",
                f"  {BOLD}Cart position{RESET}  x = {cart_x:+.4f} m",
                f"  {'█' * round((cart_x+2.4)/4.8*20):.<20}  limit ±2.4 m",
                "",
                f"  {BOLD}Control{RESET}  {direction}  force = {force:+.1f} N",
                "",
                f"  {BOLD}Reward{RESET}  {rcol}{reward:+.4f}{RESET}  (this step)",
                f"  {rcol}{bar(reward, 0.5, 1.0)}{RESET}  range [0.5 → 1.0]",
                "",
                f"  {BOLD}Episode reward{RESET}  {ep_reward:.2f}  over {ep_steps} steps",
                f"  Recent: {DIM}{sparkline}{RESET}",
                "",
                f"  {DIM}Total steps: {total_steps}{RESET}",
                done_msg,
                "",
                f"  {DIM}← → arrow keys to push  |  Q or ESC to quit{RESET}",
            ]

            sys.stdout.write(CLEAR + "\n".join(lines) + "\n")
            sys.stdout.flush()

            viewer.sync()

            if terminated or truncated:
                time.sleep(1.2)   # pause so the player sees the outcome
                ep += 1
                ep_reward = 0.0
                ep_steps  = 0
                history.clear()
                obs, _ = env.reset(seed=ep)

            # pace to ~100 Hz real-time (frame_skip=5 × dt=0.002 = 10 ms)
            elapsed = time.perf_counter() - t_start
            sleep   = max(0.0, 0.010 - elapsed)
            time.sleep(sleep)

    finally:
        env.close()
        print(f"\n{RESET}Closed after {total_steps} steps across {ep} episode(s).")

if __name__ == "__main__":
    main()
