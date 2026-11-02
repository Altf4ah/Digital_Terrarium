# Terrarium AI

An evolving ecosystem where **plants evolve**, **animals have neural-net brains that learn during their lives**, and a few animals also have an **LLM mind** that makes slow, language-based plans.

```
            LLM cortex (every ~40 ticks, a few animals)         <- slow, words, memory, beliefs
                 intent: forage / hunt / flee / rest / explore
                              |  (extra input + reward for following the plan)
                              v
   senses -> recurrent neural brain (every tick, every animal)  <- fast reflexes, learns by reinforcement
                              |
                  turn / thrust / bite
                              v
   world: plants (evolve), predators, toxins, drifting climate, reproduction with mutation
```

## What is real here

| Layer | What it is |
|---|---|
| Animal brain | Recurrent net (`terrarium/brain.py`), one set of weights per animal, the whole population runs as batched tensor ops. 835 weights, 32 inputs, 3 outputs. |
| Learning in a lifetime | Policy gradient with eligibility traces, computed with PyTorch autograd. Rewards: food, kills, curiosity; penalties: poison, bites, bad climate. |
| Curiosity | A small linear model per animal predicts its next senses. Prediction error is a reward. |
| Evolution | Weights and 12 genes mutate and cross over between compatible lineages. Mutation rate is itself a gene. `--inherit` passes learned weights to children. |
| Plants | Organisms with 5 genes (growth, seed range, fecundity, toxin, climate preference). They grow, seed, die and evolve. Toxin costs growth, so plants and animals coevolve (see `report.png`). |
| LLM cortex | Pluggable: `mock` (offline, free), `ollama` (local model), `anthropic` (Claude API). The model sees only what the creature senses, answers in JSON, and keeps a diary and beliefs (`diary.jsonl`). |

## Run it

```bash
pip install -r requirements.txt
python -m terrarium.run --ticks 10000 --llm mock              # free, offline
python -m terrarium.run --ticks 10000 --hunters 0.2 --gif-every 250   # predators from the start, makes world.gif
python -m pytest -q                                           # tests
```

Free GPU: open a Colab notebook, set the runtime to GPU, `!pip install -r requirements.txt`, then run the same command (`--device auto` picks CUDA).

### Real LLM minds

```bash
# local and free (install Ollama, then: ollama pull llama3.2)
python -m terrarium.run --llm ollama --model llama3.2 --agents 4

# Claude API (costs money; the request cap protects you)
export ANTHROPIC_API_KEY=...
python -m terrarium.run --llm anthropic --agents 4 --interval 60 --max-calls 300
```

Cost control: `--max-calls` is a hard cap (default 400). Each request is roughly 350 input and 60 output tokens, so a full default run is around 150k tokens. If a request fails, the animal falls back to the mock mind and the run continues.

### Output (in `runs/<timestamp>/`)

`report.png` (charts), `world.png` / `world.gif`, `stats.csv`, `lives.csv` (every death with genes and cause), `diary.jsonl` (what the LLM minds thought), `champions.pt` (best brains), `state.pt` (resume with `--resume`).

## Honest limits

- An LLM for *every* animal is not practical: hundreds of animals x thousands of ticks would be millions of calls. This design gives language minds to a few animals and lets their plans steer fast neural brains, which is how the cost stays sane.
- The `ollama` and `anthropic` code paths are tested against a local fake server that mimics their response formats, not against the live services. Check them with a small `--ticks 300` run first.
- GPU use is supported by the code but was not tested on a GPU.
- Whether following the LLM's plans helps animals survive has not been measured. The shaping reward teaches the brain to follow plans; it does not prove the plans are good.
- The world is a flat torus. The 3D planet is in the separate browser version.

## Ideas to extend

Communication signals between animals, plant-animal symbiosis, multiple species of plant, a memory summariser so LLM minds can live thousands of ticks, a web viewer that replays `stats.csv` and `diary.jsonl`.
