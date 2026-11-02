"""The ecosystem: evolving plants, neural-net animals that learn in life, predation, climate, reproduction."""
import math
import torch
import torch.nn.functional as F

from .config import Config
from .genes import ANIMAL_GENES, PLANT_GENES, AG, PG, animal_phenotype, plant_phenotype
from .brain import BrainSpec

TWO_PI = 2 * math.pi
CAUSES = ["starvation", "predation", "exposure", "poison", "old age"]
INTENTS = ["forage", "hunt", "flee", "rest", "explore"]
INF = float("inf")


def wrap(d, L):
    return d - L * torch.round(d / L)


def wrap_ang(x):
    return torch.remainder(x + math.pi, TWO_PI) - math.pi


def take(d, m):
    return {k: v[m] for k, v in d.items()}


def cat(a, b):
    return {k: torch.cat([a[k], b[k]], 0) for k in a}


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class World:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        c = self.cfg
        torch.manual_seed(c.seed)
        self.dev = pick_device(c.device)
        self.S = c.sectors
        self.NSENS = self.S * 4                      # plant food, plant toxin, prey, threat
        self.NI = self.NSENS + 3 + len(INTENTS)      # + energy, comfort, pain, intent one-hot
        self.PXP = self.NSENS + 4                    # curiosity model input: senses + 3 actions + bias
        self.brain = BrainSpec(self.NI, c.hidden, 3)  # outputs: turn, thrust, bite
        self.tick = 0
        self.next_id = 1
        self.births = self.deaths = self.kills = 0
        self.causes = [0] * len(CAUSES)
        self.recent_ages = []
        self.lives = []
        self.champions = []
        self.cache = {}
        self.a = self._spawn_random_animals(c.init_animals, energy_frac=0.7)
        self.p = self._spawn_random_plants(c.init_plants)

    # ----- random helpers
    def rand(self, *shape):
        return torch.rand(*shape, device=self.dev)

    def randn(self, *shape):
        return torch.randn(*shape, device=self.dev)

    def zeros(self, *shape, dtype=torch.float32):
        return torch.zeros(*shape, device=self.dev, dtype=dtype)

    # ----- construction
    def _make_animals(self, genes, p0, gen, e, pos):
        n = genes.shape[0]
        ids = torch.arange(self.next_id, self.next_id + n, device=self.dev)
        self.next_id += n
        return dict(
            id=ids, pos=pos, ang=self.rand(n) * TWO_PI, e=e, age=self.zeros(n), gen=gen,
            genes=genes, p0=p0, params=p0.clone(), trace=self.zeros(n, self.brain.P),
            h=self.zeros(n, self.cfg.hidden), base=self.zeros(n),
            wp=self.zeros(n, self.PXP, self.NSENS), xprev=self.zeros(n, self.PXP), has_prev=self.zeros(n),
            eaten=self.zeros(n), kills=self.zeros(n), bitten=self.zeros(n), poisoned=self.zeros(n),
            hurt=self.zeros(n), expo=self.zeros(n), pois=self.zeros(n),
            intent=torch.full((n,), -1, device=self.dev, dtype=torch.long),
            llm=self.zeros(n, dtype=torch.bool), last_dp=self.zeros(n), last_dprey=self.zeros(n),
            last_dth=self.zeros(n), last_att=torch.full((n,), -1, device=self.dev, dtype=torch.long),
        )

    def _random_genes(self, n):
        g = self.rand(n, len(ANIMAL_GENES))
        g[:, AG["diet"]] = self.rand(n) ** 2 * 0.5
        g[:, AG["mutability"]] = 0.3 + 0.2 * self.rand(n)
        return g

    def _spawn_random_animals(self, n, energy_frac=0.7, carnivores=0.0):
        g = self._random_genes(n)
        if carnivores > 0:
            m = self.rand(n) < carnivores
            g[m, AG["diet"]] = 0.75 + 0.25 * self.rand(int(m.sum()))
            g[m, AG["aggression"]] = 0.6 + 0.4 * self.rand(int(m.sum()))
        ph = animal_phenotype(g)
        pos = self.rand(n, 2) * self.cfg.size
        return self._make_animals(g, self.brain.init(n, self.dev), self.zeros(n) + 1, ph["maxE"] * energy_frac, pos)

    def _spawn_random_plants(self, n):
        g = self.rand(n, len(PLANT_GENES))
        g[:, PG["toxin"]] = self.rand(n) * 0.5
        return dict(pos=self.rand(n, 2) * self.cfg.size, biomass=0.3 + 0.4 * self.rand(n), genes=g)

    def add_carnivores(self, n=20):
        extra = self._spawn_random_animals(n, carnivores=1.0)
        self.a = cat(take_nograd(self.a), extra)

    # ----- climate
    def temperature(self, x):
        theta = TWO_PI * self.tick / self.cfg.climate_period
        return 0.5 + 0.5 * torch.cos(TWO_PI * x / self.cfg.size - theta)

    # ----- one tick
    @torch.no_grad()
    def step(self):
        self.tick += 1
        self._plants_phase()
        self._animals_phase()

    # ----- plants: growth, seeding, death, evolution
    def _plants_phase(self):
        c, p = self.cfg, self.p
        ph = plant_phenotype(p["genes"])
        fit = torch.exp(-((self.temperature(p["pos"][:, 0]) - ph["warm"]) ** 2) / 0.06)
        p["fit"] = fit
        growth = ph["growth"] * fit * (1 - p["biomass"]) * (1 - 0.5 * ph["toxin"])
        p["biomass"] = (p["biomass"] + growth).clamp(max=1.0)
        dead = (self.rand(p["biomass"].shape[0]) < 0.0006 + 0.01 * (1 - fit)) | (p["biomass"] < 0.03)
        # seeding
        n = p["biomass"].shape[0]
        room = c.max_plants - int((~dead).sum())
        seed = (p["biomass"] > 0.7) & (self.rand(n) < ph["fecund"] * 0.03) & ~dead
        idx = seed.nonzero().squeeze(1)
        if room > 0 and idx.numel() > 0:
            idx = idx[:room]
            p["biomass"][idx] -= 0.3
            g = p["genes"][idx].clone()
            flip = self.rand(*g.shape) < 0.25
            g = (g + flip * self.randn(*g.shape) * 0.06).clamp(0, 1)
            pos = (p["pos"][idx] + self.randn(idx.numel(), 2) * ph["seed_range"][idx, None]) % c.size
            kid = dict(pos=pos, biomass=torch.full((idx.numel(),), 0.1, device=self.dev), genes=g,
                       fit=fit[idx])
        else:
            kid = None
        keep = ~dead
        self.p = {k: v[keep] for k, v in p.items()}
        if kid is not None:
            self.p = {k: torch.cat([self.p[k], kid[k]], 0) for k in self.p}
        if self.p["biomass"].shape[0] < 25:   # a few wild seeds so the plant world never ends
            w = self._spawn_random_plants(10)
            w["fit"] = torch.ones(10, device=self.dev)
            self.p = {k: torch.cat([self.p[k], w[k]], 0) for k in self.p}

    # ----- animals
    def _animals_phase(self):
        c, a, pp = self.cfg, self.a, self.p
        N, M, S, L = a["pos"].shape[0], pp["biomass"].shape[0], self.S, c.size
        half = c.fov / 2
        ph = animal_phenotype(a["genes"])
        a["age"] += 1
        comfort = self.temperature(a["pos"][:, 0]) - ph["warm"]
        idxN = torch.arange(N, device=self.dev)
        eye = torch.eye(N, dtype=torch.bool, device=self.dev)

        # --- senses: plants
        dpl = wrap(pp["pos"][None] - a["pos"][:, None], L)
        dp = dpl.norm(dim=-1)
        relp = wrap_ang(torch.atan2(dpl[..., 1], dpl[..., 0]) - a["ang"][:, None])
        vis = ph["vision"][:, None]
        inv = (dp < vis) & (relp.abs() < half)
        prox = (1 - dp / vis).clamp(min=0) * inv
        sec = ((relp + half) / c.fov * S).long().clamp(0, S - 1)
        ptox = plant_phenotype(pp["genes"])["toxin"]
        z = self.zeros(N, S)
        sens = torch.stack([
            z.scatter_reduce(1, sec, prox * pp["biomass"][None], "amax", include_self=True),
            z.scatter_reduce(1, sec, prox * ptox[None], "amax", include_self=True),
        ], 2)
        # --- senses: other animals
        dan = wrap(a["pos"][None] - a["pos"][:, None], L)
        da = dan.norm(dim=-1)
        rela = wrap_ang(torch.atan2(dan[..., 1], dan[..., 0]) - a["ang"][:, None])
        ina = (da < vis) & (rela.abs() < half) & ~eye
        proxa = (1 - da / vis).clamp(min=0) * ina
        seca = ((rela + half) / c.fov * S).long().clamp(0, S - 1)
        bigger = ph["size"][None, :] > ph["size"][:, None]
        sens = torch.cat([sens, torch.stack([
            z.scatter_reduce(1, seca, proxa * (~bigger), "amax", include_self=True),
            z.scatter_reduce(1, seca, proxa * bigger, "amax", include_self=True),
        ], 2)], 2)
        sflat = sens.reshape(N, -1)

        # nearest things (for eating, shaping, and for the LLM's view)
        dp_edible = dp.masked_fill((pp["biomass"] < 0.15)[None], INF)
        dpm, jp = dp_edible.min(1)
        hd = (a["genes"][:, AG["hue"]][None] - a["genes"][:, AG["hue"]][:, None]).abs()
        hd = torch.minimum(hd, 1 - hd)
        kin = hd < 0.06
        da_prey = da.masked_fill(eye | bigger | kin, INF)
        dprey, kprey = da_prey.min(1)
        da_thr = da.masked_fill(eye | ~bigger, INF)
        dthr, kthr = da_thr.min(1)

        # --- curiosity: a tiny world model predicts the next senses
        pred = torch.einsum("nk,nkc->nc", a["xprev"], a["wp"])
        err = (sflat - pred) * a["has_prev"][:, None]
        surprise = err.pow(2).mean(1).sqrt()
        a["wp"] += 0.02 * torch.einsum("nk,nc->nkc", a["xprev"], err)
        curious_r = ph["cur"] * surprise.clamp(max=0.6) * 0.25 if c.curiosity else self.zeros(N)

        # --- think
        intent_oh = F.one_hot(a["intent"].clamp(min=0), len(INTENTS)).float() * (a["intent"] >= 0)[:, None]
        x = torch.cat([sflat, (a["e"] / ph["maxE"]).clamp(0, 1.5)[:, None], (comfort * 2).clamp(-1, 1)[:, None],
                       a["hurt"][:, None], intent_oh], 1)
        a["params"] = a["params"].detach().requires_grad_(c.learn)
        with torch.enable_grad() if c.learn else torch.no_grad():
            out, h2 = self.brain.forward(a["params"], x, a["h"])
            noise = c.sigma * torch.randn_like(out) if c.learn else torch.zeros_like(out)
            act = (out.detach() + noise).clamp(-1, 1)
            logp = -0.5 * ((act - out) ** 2).sum(1)
            if c.learn:
                logp.sum().backward()
        a["h"] = h2.detach()
        a["xprev"] = torch.cat([sflat, act, torch.ones(N, 1, device=self.dev)], 1)
        a["has_prev"] = torch.ones(N, device=self.dev)

        # --- move on the torus
        a["ang"] = wrap_ang(a["ang"] + act[:, 0] * 0.18)
        thrust = (act[:, 1] + 1) / 2 * ph["speed"]
        step_v = torch.stack([torch.cos(a["ang"]), torch.sin(a["ang"])], 1) * thrust[:, None]
        a["pos"] = (a["pos"] + step_v) % L
        climate_cost = 0.25 * comfort ** 2
        a["e"] = a["e"] - (0.04 * ph["size"] + 0.0003 * ph["vision"] + thrust ** 2 * 0.04 * ph["size"]
                           + climate_cost + 0.02 * ph["resist"])
        a["expo"] = a["expo"] * 0.995 + climate_cost * 0.005
        a["hurt"] = a["hurt"] * 0.95
        a["pois"] = a["pois"] * 0.98
        reward = curious_r - climate_cost * 2

        # --- graze (automatic on contact) and suffer plant toxins
        reach = 8 + 5 * ph["size"]
        can = dpm < reach
        bite = torch.where(can, (pp["biomass"][jp] - 0.05).clamp(0, 0.3), self.zeros(N))
        eff = 1 - 0.85 * ph["diet"]
        gain = bite * c.food_energy * eff
        poison = bite * ptox[jp] * 70 * (1 - ph["resist"])
        a["e"] = a["e"] + gain - poison
        if can.any():
            pp["biomass"].index_add_(0, jp[can], -bite[can])
            pp["biomass"].clamp_(min=0)
        a["eaten"] += (bite > 0).float()
        a["poisoned"] += (poison > 2).float()
        a["pois"] += (poison > 2).float() * 0.3
        a["hurt"] = torch.maximum(a["hurt"], (poison > 2).float() * 0.5)
        reward = reward + gain / (0.3 * c.food_energy) - poison / 30

        # --- predation (a bite action against a non-kin neighbour in reach)
        hit = torch.zeros(N, dtype=torch.bool, device=self.dev)
        if c.predation:
            reach_a = 4 * (ph["size"][:, None] + ph["size"][None, :]) + 6
            cand = (da < reach_a) & ~eye & ~kin
            dmin_a, kt = da.masked_fill(~cand, INF).min(1)
            wants = (act[:, 2] > 0.3) & torch.isfinite(dmin_a)
            dmg = 8 * (0.5 + ph["aggr"]) * ph["size"] * wants
            if wants.any():
                a["e"].index_add_(0, kt[wants], -dmg[wants])
                a["bitten"].index_add_(0, kt[wants], torch.ones(int(wants.sum()), device=self.dev))
                hit[kt[wants]] = True
                a["last_att"][kt[wants]] = idxN[wants]
                a["hurt"][kt[wants]] = (a["hurt"][kt[wants]] + 0.6).clamp(max=1)
                pgain = dmg * (0.25 + 0.75 * ph["diet"])
                a["e"] = a["e"] + pgain - 0.6 * (0.5 + ph["aggr"]) * wants
                reward = reward + pgain / 25
            a["e"] = a["e"] - 0.04 * ((act[:, 2] > 0.3) & ~wants)
            victims = hit & (a["e"] <= 0)
            if victims.any():
                att = a["last_att"][victims]
                bonus = 0.45 * ph["maxE"][victims] * (0.25 + 0.75 * ph["diet"][att])
                a["e"].index_add_(0, att, bonus)
                a["kills"].index_add_(0, att, torch.ones_like(bonus))
                reward.index_add_(0, att, bonus / 25)
                self.kills += int(victims.sum())
        a["e"] = torch.minimum(a["e"], ph["maxE"])
        reward = reward - 0.05 * (a["hurt"] > 0.5)

        # --- intent shaping: LLM-guided animals are rewarded for following their own plan
        prog = lambda last, cur: torch.nan_to_num(((last - cur).clamp(-3, 3)), nan=0.0, posinf=0.0, neginf=0.0) * a["has_prev"]
        ip = a["intent"]
        shape = (ip == 0) * prog(a["last_dp"], dpm) * 0.05 + (ip == 1) * prog(a["last_dprey"], dprey) * 0.05 \
            - (ip == 2) * prog(a["last_dth"], dthr) * 0.05 + (ip == 3) * 0.05 * (1 - (act[:, 1] + 1) / 2) \
            + (ip == 4) * surprise * 0.25
        reward = reward + shape
        a["last_dp"], a["last_dprey"], a["last_dth"] = dpm, dprey, dthr

        # --- learn in life: policy gradient with eligibility traces
        if c.learn:
            g = a["params"].grad
            a["trace"] = a["trace"] * c.trace + g
            adv = (reward - a["base"]).clamp(-1, 1)
            a["base"] = a["base"] + 0.02 * (reward - a["base"])
            a["params"].add_(ph["lr"][:, None] * adv[:, None] * a["trace"]).clamp_(-3, 3)
            a["params"].grad = None

        # snapshot (indices refer to this tick's animals, before deaths and births reshuffle them)
        self.cache = dict(dpm=dpm, jp=jp, relp=relp, ptox=ptox, dprey=dprey, kprey=kprey, rela=rela, dthr=dthr,
                          kthr=kthr, comfort=comfort, surprise=surprise, phen=ph, id=a["id"].clone(),
                          efrac=(a["e"] / ph["maxE"]).clone(), hurt=a["hurt"].clone(), age=a["age"].clone(),
                          eaten=a["eaten"].clone(), kills=a["kills"].clone(), bitten=a["bitten"].clone(),
                          poisoned=a["poisoned"].clone())
        self._deaths_and_births(ph, hit, da, eye, hd)

    def _deaths_and_births(self, ph, hit, da, eye, hd):
        c, a = self.cfg, self.a
        N = a["pos"].shape[0]
        dead = (a["e"] <= 0) | (a["age"] > c.max_age)
        if dead.any():
            for i in dead.nonzero().squeeze(1).tolist():
                if a["age"][i] > c.max_age:
                    cause = 4
                elif bool(hit[i]):
                    cause = 1
                elif a["pois"][i] > 0.3:
                    cause = 3
                elif a["expo"][i] > 0.03:
                    cause = 2
                else:
                    cause = 0
                self.causes[cause] += 1
                self.deaths += 1
                age = float(a["age"][i])
                self.recent_ages.append(age)
                score = float(a["eaten"][i] + 4 * a["kills"][i])
                row = [self.tick, int(a["id"][i]), int(a["gen"][i]), age, CAUSES[cause], float(a["eaten"][i]), float(a["kills"][i])]
                row += [round(float(ph[k][i]), 4) for k in ("vision", "speed", "size", "diet", "aggr", "resist", "warm")]
                if len(self.lives) < 50000:
                    self.lives.append(row)
                self._maybe_champion(i, score)
            self.recent_ages = self.recent_ages[-60:]
        # births (before removal so parents index correctly)
        births = None
        room = c.max_animals - int((~dead).sum())
        elig = (a["e"] >= ph["thresh"]) & (a["age"] > 200) & ~dead
        par = elig.nonzero().squeeze(1)[:max(0, room)]
        if par.numel() > 0:
            births = self._make_children(par, ph, da, eye, hd, dead)
        keep = ~dead
        self.a = take(a, keep)
        if births is not None:
            self.a = cat(self.a, births)
        # settlers if the population is nearly gone
        if self.a["pos"].shape[0] < c.min_animals and self.tick % 20 == 0:
            self.a = cat(self.a, self._settler())

    def _maybe_champion(self, i, score):
        if score <= 0:
            return
        a = self.a
        if len(self.champions) < 10 or score > self.champions[-1]["score"]:
            self.champions.append(dict(score=score, genes=a["genes"][i].clone(), params=a["params"][i].detach().clone(),
                                       p0=a["p0"][i].clone(), gen=int(a["gen"][i]), age=float(a["age"][i])))
            self.champions.sort(key=lambda d: -d["score"])
            self.champions = self.champions[:10]

    def _settler(self):
        if self.champions and float(torch.rand(1)) < 0.6:
            ch = self.champions[int(torch.randint(len(self.champions), (1,)))]
            g = ch["genes"][None].clone()
            p0 = self._mutate_params(ch["p0"][None], 0.15)
            g = (g + self.randn(*g.shape) * 0.05).clamp(0, 1)
            ph = animal_phenotype(g)
            return self._make_animals(g, p0, self.zeros(1) + 1, ph["maxE"] * 0.7, self.rand(1, 2) * self.cfg.size)
        return self._spawn_random_animals(1)

    def _mutate_params(self, p, rate):
        rate = rate if torch.is_tensor(rate) else torch.tensor(rate, device=self.dev)
        m = self.rand(*p.shape) < (rate.reshape(-1, 1) if rate.ndim else rate)
        return (p + m * self.randn(*p.shape) * 0.25).clamp(-3, 3)

    def _make_children(self, par, ph, da, eye, hd, dead):
        c, a = self.cfg, self.a
        k = par.numel()
        # mate: nearest compatible neighbour
        compat = (hd[par] < 0.08) & (da[par] < 40) & ~eye[par] & ((a["e"] >= 0.4 * ph["maxE"]) & ~dead)[None, :]
        dm, mate = da[par].masked_fill(~compat, INF).min(1)
        has_mate = torch.isfinite(dm)
        mate = torch.where(has_mate, mate, par)
        inh = c.inherit

        def genome_params(i):
            return (1 - inh) * a["p0"][i] + inh * a["params"][i].detach()

        pa, pb = genome_params(par), genome_params(mate)
        cross = (self.rand(k, self.brain.P) < 0.5) & has_mate[:, None]
        p0 = torch.where(cross, pb, pa)
        rate = c.mutation * ph["mutab"][par]
        p0 = self._mutate_params(p0, rate)
        gx = (self.rand(k, len(ANIMAL_GENES)) < 0.5) & has_mate[:, None]
        g = torch.where(gx, a["genes"][mate], a["genes"][par])
        g[:, AG["hue"]] = a["genes"][par, AG["hue"]]
        mut = self.rand(*g.shape) < (rate * 3).clamp(max=1)[:, None]
        g = (g + mut * self.randn(*g.shape) * 0.07).clamp(0, 1)
        g[:, AG["hue"]] = (a["genes"][par, AG["hue"]] + self.randn(k) * 0.012) % 1
        cost = ph["cost"][par]
        a["e"][par] -= cost
        pos = (a["pos"][par] + self.randn(k, 2) * 6) % c.size
        kid = self._make_animals(g, p0, torch.maximum(a["gen"][par], a["gen"][mate]) + 1, cost * 0.85, pos)
        self.births += k
        return kid

    # ----- helpers for the LLM cortex and reporting
    def stats(self):
        a, ph = self.a, animal_phenotype(self.a["genes"])
        n = a["pos"].shape[0]
        pl = plant_phenotype(self.p["genes"])
        row = dict(tick=self.tick, animals=n, plants=int(self.p["biomass"].shape[0]),
                   hunters=int((ph["diet"] > 0.45).sum()), births=self.births, deaths=self.deaths, kills=self.kills,
                   max_gen=int(a["gen"].max()) if n else 0,
                   mean_lifespan=sum(self.recent_ages) / max(1, len(self.recent_ages)),
                   plant_biomass=float(self.p["biomass"].sum()))
        for k in ("vision", "speed", "size", "diet", "aggr", "resist", "warm", "cur", "lr", "breed"):
            row["a_" + k] = float(ph[k].mean()) if n else 0.0
        for k in ("growth", "seed_range", "fecund", "toxin", "warm"):
            row["p_" + k] = float(pl[k].mean())
        return row

    def describe(self, animal_id):
        """What the animal with this id sensed on the last tick, as plain numbers (the cortex turns it into words)."""
        cc = self.cache
        if "id" not in cc:
            return None
        hit = (cc["id"] == animal_id).nonzero()
        if hit.numel() == 0:
            return None
        i = int(hit[0])
        ph = cc["phen"]
        f = lambda t: float(t[i])
        d = dict(id=animal_id, age=f(cc["age"]), energy=f(cc["efrac"]), comfort=f(cc["comfort"]), pain=f(cc["hurt"]),
                 diet=f(ph["diet"]), size=f(ph["size"]), vision=f(ph["vision"]), eaten=f(cc["eaten"]),
                 kills=f(cc["kills"]), bitten=f(cc["bitten"]), poisoned=f(cc["poisoned"]),
                 surprise=f(cc["surprise"]), plant=None, prey=None, threat=None)
        if math.isfinite(f(cc["dpm"])):
            j = int(cc["jp"][i])
            d["plant"] = dict(dist=f(cc["dpm"]), angle=float(cc["relp"][i, j]), toxin=float(cc["ptox"][j]))
        if math.isfinite(f(cc["dprey"])):
            d["prey"] = dict(dist=f(cc["dprey"]), angle=float(cc["rela"][i, int(cc["kprey"][i])]))
        if math.isfinite(f(cc["dthr"])):
            d["threat"] = dict(dist=f(cc["dthr"]), angle=float(cc["rela"][i, int(cc["kthr"][i])]))
        return d

    def set_mind(self, animal_id, llm=True, intent=None):
        """Mark an animal as LLM-guided and/or set its current intent (index into INTENTS)."""
        m = self.a["id"] == animal_id
        if not m.any():
            return False
        self.a["llm"][m] = llm
        if intent is not None:
            self.a["intent"][m] = intent
        return True

    def state_dict(self):
        return dict(tick=self.tick, next_id=self.next_id, a={k: v.cpu() for k, v in self.a.items()},
                    p={k: v.cpu() for k, v in self.p.items()}, causes=self.causes, births=self.births,
                    deaths=self.deaths, kills=self.kills, cfg=self.cfg.to_dict())

    def load_state_dict(self, s):
        self.tick, self.next_id = s["tick"], s["next_id"]
        self.a = {k: v.to(self.dev) for k, v in s["a"].items()}
        self.p = {k: v.to(self.dev) for k, v in s["p"].items()}
        self.causes, self.births, self.deaths, self.kills = s["causes"], s["births"], s["deaths"], s["kills"]


def take_nograd(d):
    return {k: v.detach() for k, v in d.items()}
