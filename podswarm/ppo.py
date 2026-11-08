"""Small PPO (policy gradient with clipping) with observation normalisation. Shared by every pod."""
import numpy as np
import torch
import torch.nn as nn


class RunningNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim); self.var = np.ones(dim); self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean; tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -8, 8).astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hid=128):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, act_dim))
        self.v = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))

    def dist(self, o):
        return torch.distributions.Normal(self.pi(o), self.log_std.exp())

    def act(self, o, deterministic=False):
        with torch.no_grad():
            d = self.dist(o)
            a = d.mean if deterministic else d.sample()
            return a, d.log_prob(a).sum(-1), self.v(o).squeeze(-1)


def gae(rew, val, done, last_val, gamma=0.99, lam=0.95):
    T = len(rew); adv = np.zeros_like(rew); last = 0.0
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        nd = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nd - val[t]
        last = delta + gamma * lam * nd * last
        adv[t] = last
    return adv, adv + val


def update(ac, opt, obs, act, logp, adv, ret, epochs=6, mb=1024, clip=0.2, ent=0.0):
    obs, act, logp, adv, ret = [torch.as_tensor(x, dtype=torch.float32) for x in (obs, act, logp, adv, ret)]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = obs.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n)
        for s in range(0, n, mb):
            j = idx[s:s + mb]
            d = ac.dist(obs[j]); lp = d.log_prob(act[j]).sum(-1)
            ratio = (lp - logp[j]).exp()
            l1 = ratio * adv[j]; l2 = ratio.clamp(1 - clip, 1 + clip) * adv[j]
            loss = -torch.min(l1, l2).mean() + 0.5 * (ac.v(obs[j]).squeeze(-1) - ret[j]).pow(2).mean() - ent * d.entropy().sum(-1).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
