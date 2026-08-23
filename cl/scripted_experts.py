"""
scripted_experts.py - Scripted Expert Policies for ManiSkill3 Tasks

Provides deterministic scripted expert policies for 4 tasks:
  - PushCube-v1: Push cube to target position
  - PickCube-v1: Pick cube and place at target
  - StackCube-v1: Stack cubeA on top of cubeB
  - PegInsertionSide-v1: Insert peg into side hole

Action space (pd_ee_delta_pose, 7D):
- action[0:3] = delta position (dx, dy, dz)
- action[3:6] = delta rotation (axis-angle)
- action[6] = gripper action (+1 open, -1 close)
"""

import numpy as np
from typing import Dict, Any


# ============================================================
# Quaternion Utilities
# ============================================================

def _quat_to_rot(q):
    """Quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _quat_aa_error(q_from, q_to):
    """Axis-angle error vector from q_from to q_to. Both [w,x,y,z]."""
    qf = np.array(q_from, dtype=float)
    qt = np.array(q_to, dtype=float)
    c = np.array([qf[0], -qf[1], -qf[2], -qf[3]])
    w = qt[0]*c[0] - qt[1]*c[1] - qt[2]*c[2] - qt[3]*c[3]
    x = qt[0]*c[1] + qt[1]*c[0] + qt[2]*c[3] - qt[3]*c[2]
    y = qt[0]*c[2] - qt[1]*c[3] + qt[2]*c[0] + qt[3]*c[1]
    z = qt[0]*c[3] + qt[1]*c[2] - qt[2]*c[1] + qt[3]*c[0]
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    if angle < 1e-6:
        return np.zeros(3)
    s = np.sin(angle / 2.0)
    if s < 1e-6:
        return np.zeros(3)
    return np.array([x, y, z]) / s * angle


def _rot_to_quat(R):
    """3x3 rotation matrix to quaternion [w, x, y, z]."""
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def _extract_extra(obs_dict):
    """Extract 'extra' dict from obs, squeeze batch dim, convert to numpy."""
    extra = obs_dict.get('extra', obs_dict)
    result = {}
    for k, v in extra.items():
        if hasattr(v, 'cpu'):
            arr = v.cpu().numpy()
        else:
            arr = np.asarray(v, dtype=np.float32)
        if arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        result[k] = arr
    return result


class ScriptedExpertPolicy:
    """Base class for scripted expert policies."""

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.phase = "init"
        self.phase_steps = 0

    def reset(self):
        self.phase = "init"
        self.phase_steps = 0

    def get_action(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


class PushCubeExpert(ScriptedExpertPolicy):
    """Push cube to goal position."""

    def __init__(self):
        super().__init__("PushCube-v1")
        self.approach_offset = 0.08
        self.push_height = 0.03

    def reset(self):
        super().reset()
        self.phase = "approach_above"
        self.phase_steps = 0

    def get_action(self, obs_dict):
        extra = _extract_extra(obs_dict)
        tcp_pos = extra['tcp_pose'][:3].copy()
        obj_pos = extra['obj_pose'][:3].copy()
        goal_pos = extra['goal_pos'][:3].copy()

        action = np.zeros(7, dtype=np.float32)

        obj_to_goal = goal_pos - obj_pos
        obj_to_goal_norm = np.linalg.norm(obj_to_goal)
        if obj_to_goal_norm > 1e-6:
            obj_to_goal_dir = obj_to_goal / obj_to_goal_norm
        else:
            obj_to_goal_dir = np.array([1.0, 0, 0])

        approach_pos = obj_pos - obj_to_goal_dir * self.approach_offset
        approach_pos[2] = 0.15

        if self.phase == "approach_above":
            target_pos = approach_pos.copy()
            delta = target_pos - tcp_pos
            delta_norm = np.linalg.norm(delta)

            if delta_norm < 0.02:
                self.phase = "descend"
                self.phase_steps = 0
            else:
                action[:3] = np.clip(delta * 5.0, -0.5, 0.5)
                action[6] = 0

        elif self.phase == "descend":
            target_pos = approach_pos.copy()
            target_pos[2] = self.push_height
            delta = target_pos - tcp_pos

            if abs(delta[2]) < 0.01:
                self.phase = "push"
                self.phase_steps = 0
            else:
                action[:3] = np.clip(delta * 8.0, -0.3, 0.3)
                action[6] = 0

        elif self.phase == "push":
            target_pos = obj_pos + obj_to_goal_dir * 0.05
            target_pos[2] = self.push_height
            delta = target_pos - tcp_pos

            action[:3] = np.clip(delta * 5.0, -0.3, 0.3)
            action[6] = 0

            if obj_to_goal_norm < 0.03:
                action[:] = 0

        self.phase_steps += 1
        return action


class PickCubeExpert(ScriptedExpertPolicy):
    """Pick cube and place at goal."""

    def __init__(self):
        super().__init__("PickCube-v1")
        self.approach_height = 0.08
        self.grasp_height = 0.02
        self.lift_height = 0.20
        self.place_height = 0.10

    def reset(self):
        super().reset()
        self.phase = "approach_above"
        self.phase_steps = 0
        self.grasped = False

    def get_action(self, obs_dict):
        extra = _extract_extra(obs_dict)
        tcp_pos = extra['tcp_pose'][:3].copy()
        obj_pos = extra['obj_pose'][:3].copy()
        goal_pos = extra['goal_pos'][:3].copy()
        is_grasped = bool(extra.get('is_grasped', [False])[0])

        action = np.zeros(7, dtype=np.float32)

        if self.phase == "approach_above":
            target_pos = obj_pos.copy()
            target_pos[2] = obj_pos[2] + self.approach_height
            delta = target_pos - tcp_pos

            if np.linalg.norm(delta) < 0.02:
                self.phase = "descend_grasp"
                self.phase_steps = 0
            else:
                action[:3] = np.clip(delta * 5.0, -0.5, 0.5)
                action[6] = 1

        elif self.phase == "descend_grasp":
            target_pos = obj_pos.copy()
            target_pos[2] = obj_pos[2] + self.grasp_height
            delta = target_pos - tcp_pos

            if abs(delta[2]) < 0.01:
                self.phase = "grasp"
                self.phase_steps = 0
            else:
                action[:3] = np.clip(delta * 6.0, -0.3, 0.3)
                action[6] = 1

        elif self.phase == "grasp":
            action[6] = -1
            if is_grasped or self.phase_steps > 15:
                self.phase = "lift"
                self.phase_steps = 0
                self.grasped = True

        elif self.phase == "lift":
            target_pos = obj_pos.copy()
            target_pos[2] = self.lift_height
            delta = target_pos - tcp_pos

            if abs(delta[2]) < 0.02 or self.phase_steps > 50:
                self.phase = "move_to_goal"
                self.phase_steps = 0
            else:
                action[:3] = np.clip(delta * 5.0, -0.4, 0.4)
                action[6] = -1

        elif self.phase == "move_to_goal":
            target_pos = goal_pos.copy()
            target_pos[2] = self.lift_height
            delta = target_pos - tcp_pos

            if np.linalg.norm(delta[:2]) < 0.02:
                self.phase = "descend_place"
                self.phase_steps = 0
            else:
                action[:3] = np.clip(delta * 4.0, -0.4, 0.4)
                action[6] = -1

        elif self.phase == "descend_place":
            target_pos = goal_pos.copy()
            target_pos[2] = goal_pos[2] + self.place_height
            delta = target_pos - tcp_pos

            if abs(delta[2]) < 0.02 or self.phase_steps > 50:
                self.phase = "release"
                self.phase_steps = 0
            else:
                action[:3] = np.clip(delta * 5.0, -0.3, 0.3)
                action[6] = -1

        elif self.phase == "release":
            action[6] = 1
            if self.phase_steps > 10:
                action[:] = 0

        self.phase_steps += 1
        return action


class StackCubeExpert(ScriptedExpertPolicy):
    """Stack cubeA on top of cubeB."""

    def __init__(self):
        super().__init__("StackCube-v1")
        self.approach_height = 0.08
        self.grasp_height = 0.02
        self.lift_height = 0.25
        self.stack_height = 0.05

    def reset(self):
        super().reset()
        self.phase = "approach_cubeA"
        self.phase_steps = 0

    def get_action(self, obs_dict):
        extra = _extract_extra(obs_dict)
        tcp_pos = extra['tcp_pose'][:3].copy()
        cubeA_pos = extra['cubeA_pose'][:3].copy()
        cubeB_pos = extra['cubeB_pose'][:3].copy()

        action = np.zeros(7, dtype=np.float32)

        pick_pos = cubeA_pos
        target_pos_stack = cubeB_pos.copy()
        target_pos_stack[2] = cubeB_pos[2] + 0.04 + self.stack_height

        if self.phase == "approach_cubeA":
            target = pick_pos.copy()
            target[2] = pick_pos[2] + self.approach_height
            delta = target - tcp_pos

            if np.linalg.norm(delta) < 0.02:
                self.phase = "descend_grasp"
                self.phase_steps = 0
            else:
                action[:3] = np.clip(delta * 5.0, -0.5, 0.5)
                action[6] = 1

        elif self.phase == "descend_grasp":
            target = pick_pos.copy()
            target[2] = pick_pos[2] + self.grasp_height
            delta = target - tcp_pos

            if abs(delta[2]) < 0.01:
                self.phase = "grasp"
                self.phase_steps = 0
            else:
                action[:3] = np.clip(delta * 6.0, -0.3, 0.3)
                action[6] = 1

        elif self.phase == "grasp":
            action[6] = -1
            if self.phase_steps > 15:
                self.phase = "lift"
                self.phase_steps = 0

        elif self.phase == "lift":
            target = pick_pos.copy()
            target[2] = self.lift_height
            delta = target - tcp_pos

            if abs(delta[2]) < 0.02 or self.phase_steps > 50:
                self.phase = "move_to_cubeB"
                self.phase_steps = 0
            else:
                action[:3] = np.clip(delta * 5.0, -0.4, 0.4)
                action[6] = -1

        elif self.phase == "move_to_cubeB":
            target = target_pos_stack.copy()
            target[2] = self.lift_height
            delta = target - tcp_pos

            if np.linalg.norm(delta[:2]) < 0.02:
                self.phase = "descend_stack"
                self.phase_steps = 0
            else:
                action[:3] = np.clip(delta * 4.0, -0.4, 0.4)
                action[6] = -1

        elif self.phase == "descend_stack":
            target = target_pos_stack.copy()
            delta = target - tcp_pos

            if abs(delta[2]) < 0.02 or self.phase_steps > 50:
                self.phase = "release"
                self.phase_steps = 0
            else:
                action[:3] = np.clip(delta * 5.0, -0.3, 0.3)
                action[6] = -1

        elif self.phase == "release":
            action[6] = 1
            if self.phase_steps > 10:
                action[:] = 0

        self.phase_steps += 1
        return action


class PegInsertionSideExpert(ScriptedExpertPolicy):
    """Insert peg into side hole - v2 robust.

    Strategy (all positions read from obs each step, no caching):
    1. approach  - move above peg center
    2. descend   - lower gripper to peg
    3. grasp     - close gripper
    4. lift      - lift to safe height
    5. rotate    - rotate peg long axis to align with hole axis
    6. move      - move peg head to just outside hole entrance
    7. insert    - push peg into hole
    """

    def __init__(self):
        super().__init__("PegInsertionSide-v1")
        self.lift_z = 0.25
        self._prev_tcp_z = None
        self._stuck_count = 0

    def reset(self):
        super().reset()
        self.phase = "approach"
        self._grasp_timer = 0
        self._prev_tcp_z = None
        self._stuck_count = 0
        self._move_best_dist = None
        self._move_stuck_count = 0

    # ---- helpers ----
    @staticmethod
    def _aa_error(current_axis, target_axis):
        """Axis-angle error vector (world frame) to rotate current_axis onto target_axis."""
        cross = np.cross(current_axis, target_axis)
        dot = float(np.dot(current_axis, target_axis))
        dot = np.clip(dot, -1.0, 1.0)
        angle = np.arccos(dot)
        if angle < 1e-6:
            return np.zeros(3)
        sin_a = np.sin(angle)
        if sin_a < 1e-6:
            # Anti-parallel: pick arbitrary perpendicular axis
            perp = np.array([1, 0, 0]) if abs(current_axis[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(current_axis, perp)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            return axis * np.pi
        axis = cross / sin_a
        return axis * angle

    @staticmethod
    def _world_to_tcp(vec_pos, vec_rot, tcp_quat):
        """Transform position and rotation deltas from world frame to TCP frame.
        pd_ee_delta_pose expects deltas in the TCP (end-effector) frame."""
        # Build rotation matrix from quaternion (xyzw format)
        x, y, z, w = tcp_quat
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
        ])
        R_T = R.T  # world-to-TCP = transpose of TCP-to-world
        return R_T @ vec_pos, R_T @ vec_rot


    # ---- main policy ----
    def get_action(self, obs_dict):
        extra = _extract_extra(obs_dict)

        # --- read current state every step ---
        tcp_pos  = extra['tcp_pose'][:3].copy()
        tcp_quat = extra['tcp_pose'][3:7].copy()
        peg_pos  = extra['peg_pose'][:3].copy()
        peg_quat = extra['peg_pose'][3:7].copy()
        peg_half = extra['peg_half_size'].copy()
        hole_pos  = extra['box_hole_pose'][:3].copy()
        hole_quat = extra['box_hole_pose'][3:7].copy()

        peg_mat = _quat_to_rot(peg_quat)
        hole_mat = _quat_to_rot(hole_quat)
        long_idx = int(np.argmax(peg_half))
        peg_axis = peg_mat[:, long_idx].copy()       # current peg long axis (world)
        half_len = float(peg_half[long_idx])
        ins_dir  = hole_mat[:, 0].copy()             # hole axis = insert direction

        # ensure ins_dir points from peg toward hole (so we push in +ins_dir)
        to_hole = hole_pos - peg_pos
        if np.dot(ins_dir, to_hole) < 0:
            ins_dir = -ins_dir

        action = np.zeros(7, dtype=np.float32)

        # phase timeout: phase-dependent (move/insert need more steps)
        _PHASE_ORDER = ["approach", "descend", "grasp", "lift",
                        "rotate", "move", "insert", "done"]
        _phase_timeout = {"move": 200, "insert": 100}
        _timeout = _phase_timeout.get(self.phase, 120)
        if self.phase_steps >= _timeout and self.phase != "done":
            idx = _PHASE_ORDER.index(self.phase)
            self.phase = _PHASE_ORDER[idx + 1]
            self.phase_steps = 0
            if self.phase == "grasp":
                self._grasp_timer = 0

        # ======================== approach ========================
        if self.phase == "approach":
            target = peg_pos.copy()
            target[2] = self.lift_z          # go to safe height above peg
            delta = target - tcp_pos
            if np.linalg.norm(delta[:2]) < 0.025:
                self.phase = "descend"
                self.phase_steps = 0
            action[:3] = np.clip(3.0 * delta, -0.3, 0.3)
            action[6] = 1.0   # gripper open

        # ======================== descend ========================
        elif self.phase == "descend":
            # Target: get TCP as close to peg as possible
            target = peg_pos.copy()
            target[2] = peg_pos[2] + 0.005   # very close to peg center

            # Stuck detection: if tcp_z barely changing, we've hit the limit
            if self._prev_tcp_z is not None:
                dz = abs(tcp_pos[2] - self._prev_tcp_z)
                if dz < 0.001:
                    self._stuck_count += 1
                else:
                    self._stuck_count = 0
            self._prev_tcp_z = tcp_pos[2]

            delta = target - tcp_pos
            # Exit if: close enough OR stuck for 15 steps OR phase timeout approaching
            if abs(delta[2]) < 0.008 or self._stuck_count >= 15:
                self.phase = "grasp"
                self.phase_steps = 0
                self._grasp_timer = 0
                self._prev_tcp_z = None
                self._stuck_count = 0
            if self.phase == "descend":
                action[:3] = np.clip(2.5 * delta, -0.25, 0.25)
                action[6] = 1.0   # open

        # ======================== grasp ========================
        elif self.phase == "grasp":
            action[6] = -1.0   # close
            self._grasp_timer += 1
            # Check if TCP is close enough to actually grasp
            grasp_dist = np.linalg.norm(tcp_pos - peg_pos)
            if self._grasp_timer > 8 and grasp_dist > 0.04:
                # Too far, can't grasp - skip to lift (will fail but save steps)
                self.phase = "lift"
                self.phase_steps = 0
            elif self._grasp_timer > 20:
                self.phase = "lift"
                self.phase_steps = 0
            action[:3] = 0.0

        # ======================== lift ========================
        elif self.phase == "lift":
            target = tcp_pos.copy()
            target[2] = self.lift_z
            delta = target - tcp_pos
            if abs(delta[2]) < 0.02:
                self.phase = "rotate"
                self.phase_steps = 0
            action[:3] = np.clip(3.0 * delta, -0.3, 0.3)
            action[6] = -1.0

        # ======================== rotate ========================
        elif self.phase == "rotate":
            target_ax = ins_dir.copy()
            # peg head is at peg_pos + peg_axis * half_len
            # we want head to face the hole
            peg_head = peg_pos + peg_axis * half_len
            head_to_hole = hole_pos - peg_head
            if np.dot(peg_axis, head_to_hole) < 0:
                target_ax = -target_ax

            err = self._aa_error(peg_axis, target_ax)
            ang = np.linalg.norm(err)

            if ang < 0.20:  # relaxed for faster convergence
                self.phase = "move"
                self.phase_steps = 0

            action[3:6] = np.clip(-2.5 * err, -0.25, 0.25)
            action[:3] = 0.0
            action[6] = -1.0

        # ======================== move ========================
        elif self.phase == "move":
            # --- alignment quality check ---
            target_ax = ins_dir.copy()
            if np.dot(peg_axis, hole_pos - peg_pos) < 0:
                target_ax = -target_ax
            ang_err = np.linalg.norm(self._aa_error(peg_axis, target_ax))

            # --- position target using ins_dir (not peg_axis) ---
            target_peg = hole_pos - ins_dir * (half_len + 0.005)
            delta = target_peg - peg_pos
            dist = np.linalg.norm(delta)

            # --- exit condition: close enough, insert will finalize ---
            if dist < 0.050 and ang_err < 0.50:
                self.phase = "insert"
                self.phase_steps = 0

            # --- stuck detection: force advance if not improving ---
            if not hasattr(self, '_move_best_dist') or self._move_best_dist is None:
                self._move_best_dist = dist
                self._move_stuck_count = 0
            if dist < self._move_best_dist - 0.005:
                self._move_best_dist = dist
                self._move_stuck_count = 0
            else:
                self._move_stuck_count += 1
            if self._move_stuck_count >= 50 and dist < 0.15:
                self.phase = "insert"
                self.phase_steps = 0
                self._move_best_dist = None
                self._move_stuck_count = 0

            # --- CRITICAL: hard cap per-step delta at 5cm ---
            # IK solver fails on large deltas; small steps keep it solvable
            if dist > 0.001:
                max_step = 0.08  # 8cm per step max
                scale = min(1.0, max_step / dist)
                action[:3] = delta * scale
            else:
                action[:3] = 0.0
            # Freeze rotation during coarse move
            action[3:6] = 0.0
            action[6] = -1.0

        elif self.phase == "insert":
            # Closed-loop insert: correct perpendicular error + push along ins_dir
            peg_head = peg_pos + peg_axis * half_len
            head_to_hole = hole_pos - peg_head

            # Decompose error into push (along ins_dir) and correction (perpendicular)
            push_dist = np.dot(head_to_hole, ins_dir)
            perp_error = head_to_hole - push_dist * ins_dir
            perp_mag = np.linalg.norm(perp_error)

            # Push forward along ins_dir with moderate force
            push_action = ins_dir * 0.8

            # Correct perpendicular misalignment aggressively
            if perp_mag > 0.001:
                corr_action = np.clip(5.0 * perp_error, -0.4, 0.4)
            else:
                corr_action = np.zeros(3)

            action[:3] = push_action + corr_action
            action[:3] = np.clip(action[:3], -0.6, 0.6)

            # Fix rotation alignment during insert
            rot_err = self._aa_error(peg_axis, ins_dir)
            action[3:6] = np.clip(-4.0 * rot_err, -0.4, 0.4)
            action[6] = -1.0

            # Success check: peg head close to hole
            total_err = np.linalg.norm(head_to_hole)
            if total_err < 0.005:
                self.phase = "done"
                self.phase_steps = 0

        else:
            pass  # done

        self.phase_steps += 1
        return action


def get_scripted_expert(env_id: str) -> ScriptedExpertPolicy:
    """Factory function to get scripted expert by env_id."""
    experts = {
        "PushCube-v1": PushCubeExpert,
        "PickCube-v1": PickCubeExpert,
        "StackCube-v1": StackCubeExpert,
        "PegInsertionSide-v1": PegInsertionSideExpert,
    }

    if env_id not in experts:
        raise ValueError(f"No scripted expert for {env_id}. Available: {list(experts.keys())}")

    return experts[env_id]()
