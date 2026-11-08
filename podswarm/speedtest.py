import time, numpy as np
from pod import Swarm
e = Swarm(1)
e.reset(); t = time.time(); n = 0
for _ in range(400):
    e.step(np.random.uniform(-1, 1, (1, 6))); n += 1
print('steps/s', n / (time.time() - t), 'z', e.d.qpos[e.qadr[0] + 2], 'upright', e.upright(0))
