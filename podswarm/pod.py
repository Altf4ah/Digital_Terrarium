"""Original small crawler robot ("pod") and the physics environments around it, built on MuJoCo.

A pod is a low disc with three legs. Each leg has two servos: a swing about the vertical axis and a lift about the
horizontal one. That is 6 actuators per pod. Pods can be put in a scene alone (walking practice) or in a group with a
heavy box that no single pod can push (teamwork).
"""
import math
import numpy as np
import mujoco

R_BODY, LEG_LEN, LEG_DROP = 0.04, 0.05, 0.035
SWING = 0.7        # rad
LIFT_LO, LIFT_HI = -0.25, 0.75
DT_SIM, SKIP = 0.004, 5            # control at 50 Hz
OBS_DIM, ACT_DIM = 21, 6


def pod_xml(i, x, y, yaw):
    s = f'<body name="pod{i}" pos="{x} {y} 0.05" euler="0 0 {math.degrees(yaw):.2f}">\n<freejoint name="free{i}"/>\n'
    s += f'<geom name="disc{i}" type="cylinder" size="{R_BODY} 0.012" mass="0.25" rgba="0.2 0.75 0.7 1" friction="0.4 0.005 0.0001"/>\n'
    s += f'<geom type="box" size="0.012 0.012 0.006" pos="0.0 0 0.016" mass="0.02" rgba="1 0.7 0.28 1" contype="0" conaffinity="0"/>\n'
    for k in range(3):
        a = k * 120
        hx = R_BODY * math.cos(math.radians(a)); hy = R_BODY * math.sin(math.radians(a))
        s += f'<body name="hip{i}_{k}" pos="{hx:.4f} {hy:.4f} 0" euler="0 0 {a}">\n<geom type="sphere" size="0.004" mass="0.005" contype="0" conaffinity="0"/>\n'
        s += f'<joint name="swing{i}_{k}" type="hinge" axis="0 0 1" range="{-SWING} {SWING}" damping="0.02" armature="0.0005"/>\n'
        s += f'<body name="leg{i}_{k}">\n<joint name="lift{i}_{k}" type="hinge" axis="0 1 0" range="{LIFT_LO} {LIFT_HI}" damping="0.02" armature="0.0005"/>\n'
        s += f'<geom type="capsule" fromto="0 0 0 {LEG_LEN} 0 {-LEG_DROP}" size="0.006" mass="0.02" rgba="0.15 0.5 0.55 1" friction="1.4 0.005 0.0001"/>\n</body>\n</body>\n'
    return s + '</body>\n'


def build_xml(n_pods, box=False, box_mass=1.2, box_mu=0.5, seed=0):
    rng = np.random.default_rng(seed)
    x = '<mujoco model="podswarm"><option timestep="%g" gravity="0 0 -9.81"/>' % DT_SIM
    x += '<default><geom condim="3" solimp="0.95 0.99 0.001" solref="0.004 1"/></default>'
    x += '<worldbody><geom name="floor" type="plane" size="6 6 0.1" rgba="0.08 0.1 0.13 1" friction="1.0 0.005 0.0001"/>'
    for i in range(n_pods):
        x += pod_xml(i, 0.0, 0.0, 0.0)
    if box:
        h = 0.08
        x += f'<body name="box" pos="1 0 {h}"><freejoint name="boxfree"/><geom name="boxg" type="box" size="{h} {h} {h}" mass="{box_mass}" rgba="0.75 0.8 0.85 1" friction="{box_mu} 0.005 0.0001"/></body>'
    x += '</worldbody><actuator>'
    for i in range(n_pods):
        for k in range(3):
            x += f'<position joint="swing{i}_{k}" kp="0.8" ctrlrange="{-SWING} {SWING}" forcerange="-0.6 0.6"/>'
            x += f'<position joint="lift{i}_{k}" kp="0.8" ctrlrange="{LIFT_LO} {LIFT_HI}" forcerange="-0.6 0.6"/>'
    x += '</actuator></mujoco>'
    return x


class Swarm:
    """n pods (and optionally a box) in one physics scene. Observations and actions are per pod, in each pod's own frame."""

    def __init__(self, n_pods=1, box=False, box_mass=1.2, box_mu=0.5, seed=0):
        self.n, self.box = n_pods, box
        self.m = mujoco.MjModel.from_xml_string(build_xml(n_pods, box, box_mass, box_mu))
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        m = self.m
        self.qadr = [m.joint(f"free{i}").qposadr[0] for i in range(n_pods)]
        self.vadr = [m.joint(f"free{i}").dofadr[0] for i in range(n_pods)]
        self.jq = np.array([[m.joint(f"{t}{i}_{k}").qposadr[0] for k in range(3) for t in ("swing", "lift")] for i in range(n_pods)])
        self.jv = np.array([[m.joint(f"{t}{i}_{k}").dofadr[0] for k in range(3) for t in ("swing", "lift")] for i in range(n_pods)])
        lo = np.tile([-SWING, LIFT_LO], 3); hi = np.tile([SWING, LIFT_HI], 3)
        self.mid, self.half = (hi + lo) / 2, (hi - lo) / 2
        if box:
            self.bq = m.joint("boxfree").qposadr[0]; self.bv = m.joint("boxfree").dofadr[0]

    # ----- geometry helpers
    def pos(self, i):
        return self.d.qpos[self.qadr[i]:self.qadr[i] + 2].copy()

    def quat(self, i):
        return self.d.qpos[self.qadr[i] + 3:self.qadr[i] + 7]

    def rot(self, i):
        R = np.zeros(9); mujoco.mju_quat2Mat(R, self.quat(i)); return R.reshape(3, 3)

    def yaw(self, i):
        R = self.rot(i); return math.atan2(R[1, 0], R[0, 0])

    def to_body(self, i, v2):
        """world 2D vector -> pod frame"""
        c, s = math.cos(self.yaw(i)), math.sin(self.yaw(i)); return np.array([c * v2[0] + s * v2[1], -s * v2[0] + c * v2[1]])

    def upright(self, i):
        return float(self.rot(i)[2, 2])

    def box_pos(self):
        return self.d.qpos[self.bq:self.bq + 2].copy()

    # ----- reset / step
    def reset(self, spread=0.0, box_at=None):
        mujoco.mj_resetData(self.m, self.d)
        for i in range(self.n):
            if spread > 0:
                p = self.rng.uniform(-spread, spread, 2) + np.array([-0.8, 0.0])
            else:
                p = np.zeros(2)
            yaw = self.rng.uniform(-math.pi, math.pi)
            q = self.qadr[i]
            self.d.qpos[q:q + 3] = [p[0], p[1], 0.05]
            self.d.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        if self.box:
            b = box_at if box_at is not None else np.array([0.0, 0.0])
            self.d.qpos[self.bq:self.bq + 3] = [b[0], b[1], 0.08]; self.d.qpos[self.bq + 3] = 1.0
        mujoco.mj_forward(self.m, self.d)
        for _ in range(40):     # settle
            mujoco.mj_step(self.m, self.d)

    def obs(self, i, target):
        """21 numbers: leg joint angles (6), leg speeds (6), gravity direction in body (3), spin (3), target in body frame (3)."""
        d = self.d
        q = (d.qpos[self.jq[i]] - self.mid) / self.half
        v = d.qvel[self.jv[i]] * 0.1
        R = self.rot(i); g = R.T @ np.array([0, 0, -1.0])
        w = d.qvel[self.vadr[i] + 3:self.vadr[i] + 6] * 0.2
        t = self.to_body(i, target - self.pos(i)); dist = float(np.hypot(*t))
        tb = t / max(dist, 1e-6)
        return np.concatenate([q, v, g, w, [tb[0], tb[1], min(dist, 2.0) / 2.0]]).astype(np.float32)

    def step(self, actions):
        """actions: [n, 6] in [-1, 1]"""
        a = np.clip(actions, -1, 1) * self.half + self.mid
        for i in range(self.n):
            self.d.ctrl[i * 6:(i + 1) * 6] = a[i]
        for _ in range(SKIP):
            mujoco.mj_step(self.m, self.d)
