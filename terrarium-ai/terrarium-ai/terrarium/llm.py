"""The cortex: a slow, language-based mind that sits on top of a few animals' fast neural brains.

Every `llm_interval` ticks each LLM-guided animal gets a short description of what it senses and remembers.
The model answers with an intent (forage, hunt, flee, rest, explore), a first-person thought, and a belief
about how its world works. The intent is fed into the animal's neural brain as an extra input, and the animal is
rewarded for following its own plan, so the reflex brain learns to carry out what the cortex decides.

Backends: mock (offline rules, free), ollama (local model), anthropic (Claude API).
"""
import json
import math
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .world import INTENTS

NAMES = ["Ash", "Briar", "Cinder", "Dune", "Ember", "Fern", "Gale", "Heron", "Ivy", "Juniper", "Kestrel", "Lark",
         "Moss", "Nettle", "Onyx", "Pebble", "Quill", "Reed", "Sorrel", "Thistle", "Umber", "Vale", "Wren", "Yarrow"]

SYSTEM = ("You are {name}, a living creature. Your whole world is what you can see, smell and feel. "
          "You know of nothing beyond it and assume it is the entire universe. Think briefly, in the first person, "
          "like an animal, not like a scientist. Each time you are asked, choose ONE intent from: "
          "forage, hunt, flee, rest, explore. Reply with JSON only, no other text: "
          '{{"intent": "...", "thought": "at most 25 words", "belief": "at most 20 words about how your world works, or empty"}}')


def direction(angle):
    a = abs(angle)
    side = "left" if angle > 0 else "right"
    if a < 0.35:
        return "straight ahead"
    if a < 1.2:
        return f"ahead and to the {side}"
    if a < 2.2:
        return f"to your {side}"
    return "behind you"


def describe(obs, last, memory):
    """Turn raw sensor numbers into the words a creature would 'feel'."""
    L = []
    e = obs["energy"]
    L.append("You feel " + ("faint with hunger" if e < 0.25 else "hungry" if e < 0.5 else "well fed" if e < 0.85 else "full of energy") + f" ({e:.0%} energy).")
    c = obs["comfort"]
    L.append("The air here feels " + ("too cold" if c < -0.25 else "a little cool" if c < -0.08 else "too hot" if c > 0.25
             else "a little warm" if c > 0.08 else "just right") + ".")
    if obs["pain"] > 0.4:
        L.append("You are hurting right now.")
    d = {k: obs[k] - last.get(k, 0) for k in ("eaten", "kills", "bitten", "poisoned")}
    ev = []
    if d["eaten"] > 0:
        ev.append(f"ate {int(d['eaten'])} time(s)")
    if d["poisoned"] > 0:
        ev.append(f"felt sick from bitter plants {int(d['poisoned'])} time(s)")
    if d["bitten"] > 0:
        ev.append(f"was bitten by another creature {int(d['bitten'])} time(s)")
    if d["kills"] > 0:
        ev.append(f"brought down prey {int(d['kills'])} time(s)")
    L.append("Since you last thought: " + (", ".join(ev) if ev else "nothing much happened") + ".")
    if obs["plant"]:
        p = obs["plant"]
        L.append(f"Nearest plant: {p['dist']:.0f} steps away, {direction(p['angle'])}, "
                 + ("it smells strongly bitter." if p["toxin"] > 0.55 else "it smells faintly bitter." if p["toxin"] > 0.3 else "it smells sweet."))
    else:
        L.append("No plant is near.")
    if obs["prey"]:
        L.append(f"A smaller creature is {obs['prey']['dist']:.0f} steps away, {direction(obs['prey']['angle'])}.")
    if obs["threat"]:
        L.append(f"A bigger creature is {obs['threat']['dist']:.0f} steps away, {direction(obs['threat']['angle'])}.")
    if memory["diary"]:
        L.append("Your recent thoughts: " + " | ".join(memory["diary"][-3:]))
    if memory["belief"]:
        L.append("What you believe so far: " + memory["belief"])
    return "\n".join(L)


def parse_reply(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in reply")
    j = json.loads(m.group(0))
    intent = str(j.get("intent", "")).strip().lower()
    if intent not in INTENTS:
        raise ValueError(f"bad intent {intent!r}")
    return dict(intent=intent, thought=str(j.get("thought", ""))[:200], belief=str(j.get("belief", ""))[:160])


# ---------------- backends ----------------
class MockBackend:
    """Offline stand-in for an LLM: simple rules plus templated first-person thoughts."""
    name = "mock"

    def decide(self, name, system, text, obs, memory):
        e, p, pr, th = obs["energy"], obs["plant"], obs["prey"], obs["threat"]
        belief = memory["belief"]
        if obs["poisoned"] - memory["last"].get("poisoned", 0) > 0 and "bitter" not in belief:
            belief = "The bitter plants make me sick. Sweet ones are safe."
        if th and th["dist"] < 60 and obs["size"] < 1.4:
            intent, thought = "flee", "Something big is close. I must get away."
        elif pr and obs["diet"] > 0.4 and e < 0.8 and pr["dist"] < 90:
            intent, thought = "hunt", "Meat is near, and I am hungry for it."
        elif e < 0.55 and p:
            intent, thought = "forage", "My belly is empty. I will go to the plants."
        elif e > 0.85:
            intent, thought = "explore", "I am strong. What lies over there?"
        elif obs["comfort"] > 0.25 or obs["comfort"] < -0.25:
            intent, thought = "explore", "This place does not feel right. I should move on."
        elif e > 0.6:
            intent, thought = "rest", "I have enough. I will rest a while."
        else:
            intent, thought = "forage", "I should find something to eat."
        return dict(intent=intent, thought=thought, belief=belief)


class _HTTPBackend:
    def __init__(self, cfg):
        self.cfg = cfg

    def _post(self, url, body, headers):
        req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout) as r:
            return json.loads(r.read().decode())


class OllamaBackend(_HTTPBackend):
    name = "ollama"

    def decide(self, name, system, text, obs, memory):
        model = self.cfg.llm_model or "llama3.2"
        j = self._post(self.cfg.llm_host.rstrip("/") + "/api/chat",
                       dict(model=model, stream=False, format="json",
                            messages=[dict(role="system", content=system), dict(role="user", content=text)]), {"Content-Type": "application/json"})
        return parse_reply(j["message"]["content"])


class AnthropicBackend(_HTTPBackend):
    name = "anthropic"

    def decide(self, name, system, text, obs, memory):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        j = self._post(base + "/v1/messages",
                       dict(model=self.cfg.llm_model or "claude-haiku-5-5", max_tokens=200, system=system,
                            messages=[dict(role="user", content=text)]),
                       {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        return parse_reply("".join(b.get("text", "") for b in j["content"]))


def make_backend(cfg):
    return {"mock": lambda: MockBackend(), "ollama": lambda: OllamaBackend(cfg),
            "anthropic": lambda: AnthropicBackend(cfg)}[cfg.llm_backend]()


# ---------------- the cortex ----------------
class Cortex:
    def __init__(self, cfg, diary_path=None):
        self.cfg = cfg
        self.backend = make_backend(cfg)
        self.fallback = MockBackend()
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.memory = {}          # animal id -> dict(name, diary, belief, last)
        self.pending = {}         # animal id -> Future
        self.calls = 0
        self.errors = 0
        self.diary_path = diary_path
        self._name_i = 0

    def _mem(self, aid):
        if aid not in self.memory:
            self.memory[aid] = dict(name=NAMES[self._name_i % len(NAMES)] + ("" if self._name_i < len(NAMES) else str(self._name_i // len(NAMES) + 1)),
                                    diary=[], belief="", last={})
            self._name_i += 1
        return self.memory[aid]

    def _job(self, aid, obs, text, mem):
        name = mem["name"]
        system = SYSTEM.format(name=name)
        try:
            r = self.backend.decide(name, system, text, obs, mem)
        except Exception as e:   # network down, bad JSON, no key: fall back so the animal still thinks
            self.errors += 1
            r = self.fallback.decide(name, system, text, obs, mem)
            r["error"] = f"{type(e).__name__}: {e}"[:160]
        return r

    def tick(self, world):
        cfg = self.cfg
        # 1) apply finished thoughts
        for aid in list(self.pending):
            f = self.pending[aid]
            if f.done():
                del self.pending[aid]
                r = f.result()
                mem = self.memory[aid]
                if world.set_mind(aid, True, INTENTS.index(r["intent"])):
                    mem["diary"] = (mem["diary"] + [r["thought"]])[-6:]
                    if r.get("belief"):
                        mem["belief"] = r["belief"]
                    self._write(world, aid, mem, r)
        if world.tick % cfg.llm_interval != 0 or cfg.llm_agents <= 0:
            return
        # 2) keep `llm_agents` animals wired to the cortex (oldest, most experienced first)
        a = world.a
        flagged = a["llm"].nonzero().squeeze(1)
        if flagged.numel() < cfg.llm_agents:
            elder = (a["age"] * (~a["llm"])).argsort(descending=True)
            for i in elder[: cfg.llm_agents - flagged.numel()].tolist():
                if a["age"][i] > 100:
                    world.set_mind(int(a["id"][i]), True)
        ids = world.a["id"][world.a["llm"]].tolist()
        # 3) ask for new thoughts
        for aid in ids:
            if aid in self.pending or self.calls >= cfg.llm_max_calls:
                continue
            obs = world.describe(aid)
            if obs is None:
                continue
            mem = self._mem(aid)
            text = describe(obs, mem["last"], mem)
            mem["last"] = {k: obs[k] for k in ("eaten", "kills", "bitten", "poisoned")}
            self.calls += 1
            self.pending[aid] = self.pool.submit(self._job, aid, obs, text, mem)

    def _write(self, world, aid, mem, r):
        if not self.diary_path:
            return
        rec = dict(tick=world.tick, id=aid, name=mem["name"], intent=r["intent"], thought=r["thought"], belief=mem["belief"])
        if r.get("error"):
            rec["error"] = r["error"]
        with open(self.diary_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
