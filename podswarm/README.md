# podswarm: tiny crawler robots that learn to walk, then push a box together

Original 3-legged "pod" robot (not the film's microbot design). MuJoCo physics, PPO reinforcement learning.

    pip install -r requirements.txt
    python train_walk.py --steps 700000     # stage 1: learn to crawl toward a target (~3 min CPU). Produces walk.pt
    python swarm_demo.py --gif team.gif     # stage 2: N pods + box, reports box progress per team size

Included walk.pt is already trained. On Colab just run the same commands.

## Stage 1
Shared 128x128 PPO policy; 6 servos per pod; reward = speed toward target, stay upright, low effort.
Result: ~0.9 m progress and 0.75 targets reached per 8 s episode.

## Stage 2 (honest status)
The walking brain is reused unchanged by every pod. A scripted dispatcher gives each pod a slot behind the box, then sends it into the box.
Default box 0.4 kg, mu 0.4. Typical result: 1 pod moves it 0%, 2 pods ~37%, 4 pods ~73% of the way in 40 s. So teamwork matters.
Limits: no goal reached yet; 6 pods crowd (box face fits ~2 pods) and do worse; dispatcher is scripted, not learned.
Next steps: PPO fine-tune with a team reward, wider box or multi-side pushing, docking via weld constraints.
