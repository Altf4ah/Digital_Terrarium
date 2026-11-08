"""Stage 2: teamwork. N pods use the walking brain from stage 1. A simple dispatcher gives each pod a spot along the back of a
heavy box; once a pod is there its target becomes the box itself, so it pushes. The box only moves when enough pods push together.
   python swarm_demo.py --pods 1 2 3 4 6 --gif team.gif
"""
import argparse, math, time
import numpy as np
import torch
from pod import Swarm, OBS_DIM, ACT_DIM
from ppo import ActorCritic, RunningNorm

ap = argparse.ArgumentParser()
ap.add_argument("--walk", default="walk.pt")
ap.add_argument("--pods", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--box-mass", type=float, default=0.4)
ap.add_argument("--box-mu", type=float, default=0.4)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--goal", type=float, nargs=2, default=[1.0, 0.0])
ap.add_argument("--gif", default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
ck = torch.load(args.walk, weights_only=False)
ac = ActorCritic(OBS_DIM, ACT_DIM); ac.load_state_dict(ck["model"])
norm = RunningNorm(OBS_DIM); norm.mean, norm.var = ck["mean"], ck["var"]
GOAL = np.array(args.goal)


def run(n, record=False, seed=0):
    env = Swarm(n, box=True, box_mass=args.box_mass, box_mu=args.box_mu, seed=seed)
    env.reset(spread=0.25, box_at=np.array([0.0, 0.0]))
    for i in range(n):   # start behind the box (west side)
        p = env.pos(i); env.d.qpos[env.qadr[i]:env.qadr[i] + 2] = [-0.55 + env.rng.uniform(-0.15, 0.15), env.rng.uniform(-0.35, 0.35)]
    import mujoco; mujoco.mj_forward(env.m, env.d)
    engaged = [False] * n; frames = []; d0 = np.linalg.norm(GOAL - env.box_pos()); hit = None
    for t in range(args.steps):
        b = env.box_pos(); u = GOAL - b; u /= max(np.linalg.norm(u), 1e-6); v = np.array([-u[1], u[0]])
        raw = []
        for i in range(n):
            slot = (i - (n - 1) / 2) * 0.075
            spot = b - u * (0.08 + 0.085) + v * slot
            if np.linalg.norm(env.pos(i) - spot) < 0.1: engaged[i] = True
            if engaged[i] and np.linalg.norm(env.pos(i) - spot) > 0.3: engaged[i] = False
            target = b + v * slot if engaged[i] else spot
            raw.append(env.obs(i, target))
        a, _, _ = ac.act(torch.as_tensor(norm(np.stack(raw))), deterministic=True)
        env.step(a.numpy())
        if record and t % 6 == 0: frames.append((env.box_pos(), [env.pos(i) for i in range(n)], t))
        if hit is None and np.linalg.norm(GOAL - env.box_pos()) < 0.12: hit = t
    d1 = np.linalg.norm(GOAL - env.box_pos())
    return dict(pods=n, moved=d0 - d1, frac=(d0 - d1) / d0, reached=hit, frames=frames)


def make_gif(res, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, io
    from PIL import Image
    imgs = []
    for box, pods, t in res["frames"][::2]:
        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=80); fig.patch.set_facecolor("#080c11"); ax.set_facecolor("#0b1118")
        ax.add_patch(plt.Circle(GOAL, 0.12, fill=False, ec="#5eead4", ls="--"))
        ax.add_patch(plt.Rectangle(box - 0.08, 0.16, 0.16, fc="#c7d4e0", ec="#fff", alpha=0.9))
        for p in pods: ax.add_patch(plt.Circle(p, 0.04, fc="#ffb347", ec="#1a1205"))
        ax.set_xlim(-0.9, 1.3); ax.set_ylim(-0.6, 0.6); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{res['pods']} pods   t={t * 0.02:4.1f} s   box to goal {np.linalg.norm(GOAL - box):.2f} m", color="#d5e1ec", fontsize=9, loc="left")
        for s in ax.spines.values(): s.set_color("#213043")
        fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=fig.get_facecolor()); plt.close(fig); buf.seek(0); imgs.append(Image.open(buf).convert("P"))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=70, loop=0)


rows = []
for n in args.pods:
    r = run(n, record=bool(args.gif) and n == args.pods[-1], seed=args.seed)
    rows.append(r); print(f"pods {n}: box moved {r['moved']:.2f} m ({r['frac']:.0%} of the way)  reached goal: {'step ' + str(r['reached']) if r['reached'] is not None else 'no'}", flush=True)
if args.gif:
    make_gif(rows[-1], args.gif); print("saved", args.gif)
