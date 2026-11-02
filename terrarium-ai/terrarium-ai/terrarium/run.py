"""Command line: python -m terrarium.run --ticks 20000 --llm mock"""
import argparse
import csv
import json
import os
import time

import torch

from .config import Config
from .world import World, CAUSES
from .llm import Cortex
from . import viz


def parse():
    ap = argparse.ArgumentParser(description="Evolving ecosystem with neural-net animals, evolving plants and an LLM cortex.")
    ap.add_argument("--ticks", type=int, default=10000)
    ap.add_argument("--out", default=None, help="output folder (default runs/<timestamp>)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--llm", default="mock", choices=["mock", "ollama", "anthropic"])
    ap.add_argument("--model", default="")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--agents", type=int, default=4, help="animals with an LLM mind")
    ap.add_argument("--interval", type=int, default=40, help="ticks between thoughts")
    ap.add_argument("--max-calls", type=int, default=400, help="cap on LLM requests")
    ap.add_argument("--animals", type=int, default=80)
    ap.add_argument("--plants", type=int, default=250)
    ap.add_argument("--mutation", type=float, default=0.05)
    ap.add_argument("--inherit", type=float, default=0.0, help="0..1 share of learned weights passed to children")
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--no-curiosity", action="store_true")
    ap.add_argument("--no-predation", action="store_true")
    ap.add_argument("--hunters", type=float, default=0.0, help="fraction of starting animals that are hunters")
    ap.add_argument("--gif-every", type=int, default=0, help="save a frame every N ticks and build world.gif")
    ap.add_argument("--resume", default=None, help="path to a state.pt from an earlier run")
    ap.add_argument("--log-every", type=int, default=50)
    return ap.parse_args()


def main():
    args = parse()
    cfg = Config(seed=args.seed, device=args.device, init_animals=args.animals, init_plants=args.plants,
                 mutation=args.mutation, inherit=args.inherit, learn=not args.no_learn, curiosity=not args.no_curiosity,
                 predation=not args.no_predation, llm_backend=args.llm, llm_model=args.model, llm_host=args.host,
                 llm_agents=args.agents, llm_interval=args.interval, llm_max_calls=args.max_calls)
    out = args.out or os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    json.dump(cfg.to_dict(), open(os.path.join(out, "config.json"), "w"), indent=1)
    world = World(cfg)
    if args.hunters > 0:
        world.a = world._spawn_random_animals(cfg.init_animals, 0.7, args.hunters)
    if args.resume:
        world.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=False))
    cortex = Cortex(cfg, os.path.join(out, "diary.jsonl"))
    print(f"device={world.dev}  llm={cfg.llm_backend}  out={out}")
    stats_path = os.path.join(out, "stats.csv")
    frames = []
    t0 = time.time()
    with open(stats_path, "w", newline="") as fh:
        wr = None
        for _ in range(args.ticks):
            world.step()
            cortex.tick(world)
            if world.tick % args.log_every == 0:
                s = world.stats()
                if wr is None:
                    wr = csv.DictWriter(fh, fieldnames=list(s))
                    wr.writeheader()
                wr.writerow(s)
                fh.flush()
            if world.tick % 1000 == 0:
                s = world.stats()
                print(f"tick {world.tick:>6}  animals {s['animals']:>3}  plants {s['plants']:>3}  hunters {s['hunters']:>3}  "
                      f"gen {s['max_gen']:>3}  lifespan {s['mean_lifespan']:>6.0f}  toxin {s['p_toxin']:.2f}  resist {s['a_resist']:.2f}  "
                      f"llm calls {cortex.calls}  {world.tick / (time.time() - t0):.0f} ticks/s")
            if args.gif_every and world.tick % args.gif_every == 0:
                frames.append(viz.snapshot(world, cortex))
            if world.a["pos"].shape[0] == 0:
                break
    cortex.close()
    viz.snapshot(world, cortex, os.path.join(out, "world.png"))
    viz.report(stats_path, os.path.join(out, "report.png"))
    with open(os.path.join(out, "lives.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["tick", "id", "generation", "lifespan", "cause", "meals", "kills", "vision", "speed", "size", "diet", "aggression", "resist", "warm"])
        wr.writerows(world.lives)
    torch.save(dict(champions=world.champions, config=cfg.to_dict()), os.path.join(out, "champions.pt"))
    torch.save(world.state_dict(), os.path.join(out, "state.pt"))
    if frames:
        from PIL import Image
        imgs = [Image.open(f).convert("P") for f in frames]
        imgs[0].save(os.path.join(out, "world.gif"), save_all=True, append_images=imgs[1:], duration=120, loop=0)
    print("deaths by cause:", dict(zip(CAUSES, world.causes)))
    print(f"LLM requests: {cortex.calls}  failed (fell back to mock): {cortex.errors}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
