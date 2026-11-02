from dataclasses import dataclass, asdict


@dataclass
class Config:
    # world
    size: float = 1000.0            # side of the square torus world
    seed: int = 0
    device: str = "auto"            # auto | cpu | cuda
    init_animals: int = 80
    init_plants: int = 250
    max_animals: int = 250
    max_plants: int = 600
    min_animals: int = 14           # settlers arrive below this
    climate_period: int = 8000      # ticks for the warm band to circle the world
    max_age: int = 9000
    food_energy: float = 80.0       # energy in one full plant (per unit biomass eaten)
    # senses and brain
    sectors: int = 6
    fov: float = 2.6                # radians
    hidden: int = 16
    # learning inside a lifetime
    learn: bool = True
    curiosity: bool = True
    sigma: float = 0.3              # exploration noise on actions
    trace: float = 0.92             # eligibility trace decay
    # evolution
    mutation: float = 0.05
    inherit: float = 0.0            # 0 = genome only, 1 = child also inherits what the parent learned
    predation: bool = True
    # LLM cortex
    llm_backend: str = "mock"       # mock | ollama | anthropic
    llm_model: str = ""             # default depends on backend
    llm_host: str = "http://localhost:11434"
    llm_agents: int = 4             # how many animals get an LLM mind at a time
    llm_interval: int = 40          # ticks between thoughts
    llm_max_calls: int = 400        # hard cap on LLM requests per run (cost control)
    llm_timeout: float = 30.0

    def to_dict(self):
        return asdict(self)
