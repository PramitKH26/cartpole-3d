# MuJoCo Cartpole — 2D and 3D RL

Custom MuJoCo cartpole environments wrapped with Gymnasium and trained with
PPO (stable-baselines3). Two tasks share the same training pipeline:

- **2D cartpole** — cart slides along one axis, pole tips in one plane (5-D obs, 1-D action)
- **3D cartpole** — cart slides on the floor in x and y, pole sits on a universal joint and can fall any direction (10-D obs, 2-D action)

## Results

| Env | Train time | Steps | Eval reward (max 500) |
| --- | --- | --- | --- |
| 2D | ~20 s | 200,000 | 499.99 ± 0.01 |
| 3D | ~50 s | 400,000 | 499.99 ± 0.01 |

(4 parallel workers on a laptop CPU.)

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+.

## Train

```bash
python3 train.py --env 2d                  # 2D cartpole
python3 train.py --env 3d --timesteps 400000   # 3D cartpole
```

Models are saved to `ppo_cartpole_2d.zip` / `ppo_cartpole_3d.zip`.
TensorBoard logs land in `./logs/`:

```bash
tensorboard --logdir ./logs
```

## Watch a trained policy

On macOS the passive MuJoCo viewer requires `mjpython`:

```bash
mjpython train.py --env 2d --eval-only --render
mjpython train.py --env 3d --eval-only --render
```

## Play the 2D version yourself

```bash
mjpython play.py
```

Arrow keys push the cart left/right. The terminal shows live angle, reward,
and a sparkline so you can feel the reward signal change as you balance.

## Tests

```bash
pytest test_env.py -v
```

## Files

| File | Role |
| --- | --- |
| `cartpole.xml` / `cartpole3d.xml` | MJCF physics models |
| `cartpole_env.py` / `cartpole3d_env.py` | Gymnasium env wrappers |
| `train.py` | PPO training and eval |
| `play.py` | Interactive 2D version (keyboard) |
| `test_env.py` | Pytest sanity checks |
