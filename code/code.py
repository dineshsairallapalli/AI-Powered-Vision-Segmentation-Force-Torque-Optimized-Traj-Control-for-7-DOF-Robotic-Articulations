import os
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional
import copy

import numpy as np
import robosuite as suite
from scipy.spatial.transform import Rotation as Rot
import imageio

@dataclass
class ForceTorqueLimits:
    f_soft: float
    f_hard: float
    t_soft: float
    t_hard: float

@dataclass
class CostConfig:
    pos: float
    ori: float
    hinge: float
    force: float
    torque: float
    hard: float

@dataclass
class ControlConfig:
    servo_gain: float
    ori_gain: float
    servo_dist: float
    approach_dist: float
    pre_grip_level: float
    full_grip_level: float
    contact_timeout: float
    mpc_horizon: int
    mpc_samples: int
    noise_scale: float
    backoff_dist: float = 0.06
    backoff_steps: int  = 40

def find_handle_geom(sim) -> Optional[int]:
    for name in sim.model.geom_names:
        if name == "Door_handle_base":
            return sim.model.geom_name2id(name)
    for name in sim.model.geom_names:
        if "handle" in name.lower():
            return sim.model.geom_name2id(name)
    return None

def read_wrench(env) -> Tuple[np.ndarray, np.ndarray]:
    robot = env.robots[0]
    f = next(iter(robot.ee_force.values()))
    t = next(iter(robot.ee_torque.values()))
    return f.copy(), t.copy()

def orientation_mismatch(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    z = Rot.from_quat(quat).as_matrix()[:, 2]
    u = vec / (np.linalg.norm(vec) + 1e-8)
    return 0.5 * np.cross(z, u)

class DoorController:
    def __init__(self, env, ft: ForceTorqueLimits, cost: CostConfig, ctrl: ControlConfig, *, seed=None):
        self.env = env
        self.ft = ft
        self.cost = cost
        self.ctrl = ctrl
        self.rng = np.random.default_rng(seed)

        self.prev_force = 0.0
        self.in_grasp = False
        self.last_touch = -np.inf
        self.grip_value = -1.0
        self.recover_count = 0
        self.yaw_jitter = 0.0
        self.prev_plan = None

        names = env.sim.model.geom_names
        self.finger_geoms = [n for n in names if "finger" in n.lower()]
        self.handle_geoms = [n for n in names if "handle" in n.lower()]
        if not self.finger_geoms or not self.handle_geoms:
            raise RuntimeError("Required finger or handle geoms not found")

    def _in_contact(self) -> bool:
        return self.env.check_contact(self.finger_geoms, self.handle_geoms)

    def act(self, obs) -> np.ndarray:
        global ACT_LOW, ACT_HIGH, ACTION_DIM, GRIP_IDX
        f, t = read_wrench(self.env)
        f_mag = float(np.linalg.norm(f))
        delta_f = f_mag - self.prev_force
        self.prev_force = f_mag

        spike = (f_mag > self.ft.f_hard) or (delta_f > self.ft.f_soft)
        if spike and self.recover_count == 0:
            self.recover_count = self.ctrl.backoff_steps
            self.in_grasp = False
            self.grip_value = -1.0
            self.yaw_jitter = self.rng.uniform(-0.44, 0.44)
            self.prev_plan = None

        if self.recover_count > 0:
            self.recover_count -= 1
            cmd = np.zeros(ACTION_DIM)
            cmd[2] = -self.ctrl.backoff_dist / self.ctrl.backoff_steps
            cmd[GRIP_IDX] = -1.0
            return np.clip(cmd, ACT_LOW, ACT_HIGH)

        now = time.time()
        if self._in_contact():
            self.last_touch = now
            if not self.in_grasp:
                self.in_grasp = True
                self.grip_value = ACT_HIGH[GRIP_IDX]
        elif self.in_grasp and (now - self.last_touch) > self.ctrl.contact_timeout:
            self.in_grasp = False
            self.grip_value = self.ctrl.pre_grip_level

        hid = find_handle_geom(self.env.sim)
        hpos = self.env.sim.data.geom_xpos[hid]
        p = obs['robot0_eef_pos']
        delta_pos = hpos - p
        dist = np.linalg.norm(delta_pos)

        if not self.in_grasp and abs(self.yaw_jitter) > 1e-3:
            c, s = np.cos(self.yaw_jitter), np.sin(self.yaw_jitter)
            delta_pos[:2] = np.array([[c, -s], [s, c]]) @ delta_pos[:2]

        if dist > self.ctrl.servo_dist and not self.in_grasp:
            cmd = np.zeros(ACTION_DIM)
            cmd[:3] = np.clip(self.ctrl.servo_gain * delta_pos, ACT_LOW[:3], ACT_HIGH[:3])
            cmd[3:6] = np.clip(
                self.ctrl.ori_gain * orientation_mismatch(obs['robot0_eef_quat'], delta_pos),
                ACT_LOW[3:6], ACT_HIGH[3:6]
            )
            cmd[GRIP_IDX] = -1.0
            return cmd

        base = (np.zeros((self.ctrl.mpc_horizon, ACTION_DIM))
                if self.prev_plan is None
                else np.vstack([self.prev_plan[1:], np.zeros((1, ACTION_DIM))]))
        if not self.in_grasp:
            step_vec = delta_pos / max(dist, 1e-6) * min(0.03, dist)
            base[:, :3] += step_vec
            self.grip_value = self.ctrl.pre_grip_level
        else:
            base[:, :3] += np.array([0, 0.08, 0]) / self.ctrl.mpc_horizon
        base[:, GRIP_IDX] = self.grip_value

        seqs = self.rng.normal(base, self.ctrl.noise_scale,
                               (self.ctrl.mpc_samples, self.ctrl.mpc_horizon, ACTION_DIM))
        seqs = np.clip(seqs, ACT_LOW, ACT_HIGH)
        seqs[:, :, GRIP_IDX] = self.grip_value
        snap = copy.deepcopy(self.env.sim.get_state())
        costs = []
        for s in seqs:
            cmd = s[0]
            sim = self.env.sim
            sim.set_state(snap)
            sim.data.ctrl[:ACTION_DIM] = cmd
            sim.step()
            f_sim, _ = read_wrench(self.env)
            costs.append(np.linalg.norm(f_sim))
        best = int(np.argmin(costs))
        self.prev_plan = seqs[best]
        return self.prev_plan[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Door")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--camera", default="robot0_eye_in_hand")
    parser.add_argument("--seg-level", choices=["element","instance"], default="element")
    parser.add_argument("--freq", type=int, default=20)
    parser.add_argument("--hinge-cutoff", type=float, default=0.085)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "glfw")
    env = suite.make(
        env_name=args.env_name,
        robots=[args.robot],
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=[args.camera],
        camera_segmentations=[args.seg_level],
        camera_heights=[512],
        camera_widths=[512],
        control_freq=args.freq,
        renderer="mujoco",
        use_latch=False,
    )

    print("Available geoms:", env.sim.model.geom_names)

    geom_id = find_handle_geom(env.sim)
    if geom_id is None:
        avail = [n for n in env.sim.model.geom_names if "handle" in n.lower()]
        raise RuntimeError(f"No handle geom found. Options: {avail}")
    HANDLE_PX = (geom_id + 1 if args.seg_level == "element"
                 else env.sim.model.geom_bodyid[geom_id] + 1)

    os.makedirs("segmentations", exist_ok=True)
    low, high = env.action_spec
    rng = np.random.default_rng()
    left_bias = -0.2 * (high[3] - low[3])
    frame = 0

    print("Searching for Door_handle_base segmentation…")
    while True:
        action = rng.uniform(low, high)
        action[3] = np.clip(action[3] + left_bias, low[3], high[3])
        obs, _, _, _ = env.step(action)
        env.render()

        seg = obs[f"{args.camera}_segmentation_{args.seg_level}"].squeeze(-1)
        imageio.imwrite(f"segmentations/seg_{frame:04d}.png", seg.astype(np.uint8))
        frame += 1
        if np.any(seg == HANDLE_PX):
            print(f"Handle detected at frame {frame}")
            break

    global ACT_LOW, ACT_HIGH, ACTION_DIM, GRIP_IDX
    ACT_LOW, ACT_HIGH = env.action_spec
    ACTION_DIM = ACT_LOW.shape[0]
    GRIP_IDX = ACTION_DIM - 1

    ft_cfg = ForceTorqueLimits(30., 60., 6., 12.)
    cost_cfg = CostConfig(12., 1.5, -60., 2., 0.8, 1e3)
    ctrl_cfg = ControlConfig(4., 1.5, 0.04, 0.18, 0.30, 0.60, 0.4, 15, 256, 0.4)
    controller = DoorController(env, ft_cfg, cost_cfg, ctrl_cfg)

    obs = env.reset()
    f_history = []
    for i in range(args.max_steps):
        action = controller.act(obs)
        obs, _, _, _ = env.step(action)
        env.render()

        seg = obs[f"{args.camera}_segmentation_{args.seg_level}"].squeeze(-1)
        hmask = (seg == HANDLE_PX)
        rgb = np.zeros((*seg.shape, 3), dtype=np.uint8)
        rgb[..., 0][hmask] = 255
        rgb[..., 1:][hmask] = 0
        imageio.imwrite(f"segmentations/seg_{frame:04d}.png", rgb)
        frame += 1

        if i % 150 == 0:
            f_val, t_val = read_wrench(env)
            f_norm = round(np.linalg.norm(f_val), 1)
            f_history.append(f_norm)
            if len(f_history) >= 3 and f_history[-1] == f_history[-2] == f_history[-3]:
                print("Force constant 3x; closing and pulling door.")
                controller.in_grasp = True
                controller.grip_value = ACT_HIGH[GRIP_IDX]
                controller.prev_plan = None
                f_history.clear()
            print(f"Step {i:5d} | Hinge {obs['hinge_qpos']:+.3f} | |F|{f_norm:.1f}N | |T|{np.linalg.norm(t_val):.1f}Nm"
                  + (" | GRASP" if controller.in_grasp else ""))

        if controller.in_grasp and np.any(seg == HANDLE_PX):
            controller.recover_count = ctrl_cfg.backoff_steps
            controller.in_grasp = False
            controller.grip_value = -1.0
            controller.prev_plan = None

    env.close()

if __name__ == "__main__":
    main()
