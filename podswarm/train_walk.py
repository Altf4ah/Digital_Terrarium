"""Stage 1: one pod learns to crawl toward a target point (PPO, from scratch).
   python train_walk.py --steps 600000
"""
import argparse, math, time
import numpy as np
import torch
from pod import Swarm, OBS_DIM, ACT_DIM
from ppo import ActorCritic, RunningNorm, gae, update

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=600_000)
ap.add_argument("--envs", type=int, default=8)
ap.add_argument("--horizon", type=int, default=128)
ap.add_argument("--out", default="walk.pt")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
DT = 0.02; EP = 400


def new_target(env):
    a = env.rng.uniform(-math.pi, math.pi); r = env.rng.uniform(0.6, 1.2)
    return env.pos(0) + r * np.array([math.cos(a), math.sin(a)])


envs = [Swarm(1, seed=args.seed * 100 + k) for k in range(args.envs)]
tgt, prev, t_ep, ep_prog, ep_hits = [], [], [], [], []
for e in envs:
    e.reset(); tgt.append(new_target(e)); prev.append(e.pos(0)); t_ep.append(0); ep_prog.append(0.0); ep_hits.append(0)
ac = ActorCritic(OBS_DIM, ACT_DIM); opt = torch.optim.Adam(ac.parameters(), lr=3e-4); norm = RunningNorm(OBS_DIM)
done_stats = []; total = 0; t0 = time.time(); upd = 0
while total < args.steps:
    O, A, LP, V, R, D = [], [], [], [], [], []
    for _ in range(args.horizon):
        raw = np.stack([e.obs(0, tgt[k]) for k, e in enumerate(envs)])
        norm.update(raw); o = norm(raw)
        a, lp, v = ac.act(torch.as_tensor(o)); a = a.numpy()
        rew = np.zeros(len(envs)); dn = np.zeros(len(envs))
        for k, e in enumerate(envs):
            e.step(a[k:k + 1]); p = e.pos(0); d_prev = np.linalg.norm(tgt[k] - prev[k]); d_now = np.linalg.norm(tgt[k] - p)
            dirv = (tgt[k] - prev[k]) / max(d_prev, 1e-6); vproj = float((p - prev[k]) @ dirv) / DT
            r = 2.0 * min(vproj, 0.5) + 0.02 * e.upright(0) - 0.002 * float((np.clip(a[k], -1, 1) ** 2).sum())
            ep_prog[k] += (d_prev - d_now); prev[k] = p; t_ep[k] += 1
            if d_now < 0.12:
                r += 3.0; ep_hits[k] += 1; tgt[k] = new_target(e)
            if e.upright(0) < 0.3:
                r -= 2.0; dn[k] = 1
            if t_ep[k] >= EP: dn[k] = 1
            rew[k] = r
            if dn[k]:
                done_stats.append((ep_prog[k], ep_hits[k])); e.reset(); tgt[k] = new_target(e); prev[k] = e.pos(0); t_ep[k] = 0; ep_prog[k] = 0.0; ep_hits[k] = 0
        O.append(o); A.append(a); LP.append(lp.numpy()); V.append(v.numpy()); R.append(rew); D.append(dn)
    raw = np.stack([e.obs(0, tgt[k]) for k, e in enumerate(envs)]); _, _, lv = ac.act(torch.as_tensor(norm(raw)))
    O, A, LP, V, R, D = [np.array(x) for x in (O, A, LP, V, R, D)]
    adv = np.zeros_like(R); ret = np.zeros_like(R)
    for k in range(len(envs)):
        adv[:, k], ret[:, k] = gae(R[:, k], V[:, k], D[:, k], lv.numpy()[k])
    update(ac, opt, O.reshape(-1, OBS_DIM), A.reshape(-1, ACT_DIM), LP.reshape(-1), adv.reshape(-1), ret.reshape(-1))
    total += args.horizon * len(envs); upd += 1
    if upd % 10 == 0:
        recent = done_stats[-16:]
        pr = np.mean([p for p, h in recent]) if recent else 0.0; hs = np.mean([h for p, h in recent]) if recent else 0.0
        print(f"steps {total:>7}  progress/episode {pr:5.2f} m  targets reached/episode {hs:4.2f}  reward/step {R.mean():5.3f}  {total / (time.time() - t0):.0f} steps/s", flush=True)
        torch.save(dict(model=ac.state_dict(), mean=norm.mean, var=norm.var, count=norm.count), args.out)
torch.save(dict(model=ac.state_dict(), mean=norm.mean, var=norm.var, count=norm.count), args.out)
print("saved", args.out)
