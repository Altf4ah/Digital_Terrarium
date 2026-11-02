"""Gene definitions. Every gene is stored in [0, 1] and mapped to a phenotype here."""
import torch

ANIMAL_GENES = ["vision", "speed", "size", "diet", "aggression", "resist", "warm",
                "mutability", "breeding", "curiosity", "learn_rate", "hue"]
PLANT_GENES = ["growth", "seed_range", "fecundity", "toxin", "warm"]
AG = {n: i for i, n in enumerate(ANIMAL_GENES)}
PG = {n: i for i, n in enumerate(PLANT_GENES)}


def animal_phenotype(g: torch.Tensor) -> dict:
    size = 0.7 + 1.1 * g[:, AG["size"]]
    maxE = 120.0 * size
    breed = g[:, AG["breeding"]]
    cost = maxE * (0.18 + 0.30 * breed)          # energy a parent pays for one child
    return dict(
        vision=60.0 + 120.0 * g[:, AG["vision"]],
        speed=1.0 + 2.0 * g[:, AG["speed"]],
        size=size,
        diet=g[:, AG["diet"]],                   # 0 grazer ... 1 hunter
        aggr=g[:, AG["aggression"]],
        resist=g[:, AG["resist"]],               # tolerance of plant toxins
        warm=g[:, AG["warm"]],                   # preferred temperature
        mutab=0.25 + 2.5 * g[:, AG["mutability"]],
        breed=breed,
        cur=g[:, AG["curiosity"]],
        lr=0.003 + 0.037 * g[:, AG["learn_rate"]],
        hue=g[:, AG["hue"]],
        maxE=maxE,
        cost=cost,
        thresh=cost + 0.35 * maxE,
    )


def plant_phenotype(g: torch.Tensor) -> dict:
    return dict(
        growth=0.004 + 0.02 * g[:, PG["growth"]],
        seed_range=20.0 + 120.0 * g[:, PG["seed_range"]],
        fecund=g[:, PG["fecundity"]],
        toxin=g[:, PG["toxin"]],
        warm=g[:, PG["warm"]],
    )
