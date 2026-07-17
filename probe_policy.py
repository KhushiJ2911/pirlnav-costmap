"""Greedy costmap-descent controller (HANDOVER §4 experiment 1).

Pure numpy — no habitat imports — so the decision logic is unit-testable on any
machine (see test_probe_policy.py). The habitat harness lives in
run_probe_greedy.py.

Policy: STOP when near-zero cost is visible; otherwise turn toward the direction
of the minimum visible cost and move forward. Two guards keep it from the
obvious scripted-policy failure modes: a scan mode when no valid costmap pixels
are visible (facing a wall / everything beyond the depth window), and a stall
breaker for collisions (forward didn't move the agent).
"""

import numpy as np

# Action indices must match TASK.POSSIBLE_ACTIONS order:
# ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3


class GreedyCostmapPolicy:
    def __init__(
        self,
        max_cost=30.0,
        stop_threshold=0.25,
        center_frac=1.0 / 3.0,
        stall_dist=0.05,
        invalid_frac=0.999,
    ):
        # Pixels at >= invalid_frac * max_cost are treated as invalid: the
        # sensor writes exactly max_cost for out-of-depth-window pixels and
        # clips real costs to max_cost, so both are unusable as gradient signal.
        self.max_cost = float(max_cost)
        self.stop_threshold = float(stop_threshold)
        self.center_frac = float(center_frac)
        self.stall_dist = float(stall_dist)
        self.invalid_cutoff = float(max_cost) * float(invalid_frac)
        self.reset()

    def reset(self):
        self._last_action = None
        self._last_position = None
        self._scan_steps = 0

    def act(self, costmap, position=None):
        """costmap: (H, W, 1) or (H, W) float array from CostmapSensor.
        position: agent world position (3,) if available, for stall detection.
        Returns an action index."""
        cm = np.asarray(costmap, dtype=np.float32)
        if cm.ndim == 3:
            cm = cm[:, :, 0]
        H, W = cm.shape

        stalled = (
            self._last_action == MOVE_FORWARD
            and position is not None
            and self._last_position is not None
            and float(np.linalg.norm(np.asarray(position) - self._last_position))
            < self.stall_dist
        )

        valid = cm < self.invalid_cutoff
        if not valid.any():
            # Nothing usable in view: scan in place. Cap the scan at a full
            # rotation's worth of turns; after that keep scanning anyway (there
            # is no better option without memory).
            action = TURN_LEFT
            self._scan_steps += 1
        else:
            self._scan_steps = 0
            min_cost = float(cm[valid].min())
            if min_cost <= self.stop_threshold:
                action = STOP
            else:
                masked = np.where(valid, cm, np.inf)
                _, col = np.unravel_index(int(np.argmin(masked)), (H, W))
                if col < W * self.center_frac:
                    action = TURN_LEFT
                elif col >= W * (1.0 - self.center_frac):
                    action = TURN_RIGHT
                else:
                    action = MOVE_FORWARD

            if stalled and action == MOVE_FORWARD:
                # Collision: goal-ward pixel is centered but blocked. Sidestep
                # toward the half of the image with lower mean valid cost.
                left = np.where(valid[:, : W // 2], cm[:, : W // 2], np.nan)
                right = np.where(valid[:, W // 2 :], cm[:, W // 2 :], np.nan)
                lmean = np.nanmean(left) if np.isfinite(left).any() else np.inf
                rmean = np.nanmean(right) if np.isfinite(right).any() else np.inf
                action = TURN_LEFT if lmean <= rmean else TURN_RIGHT

            # Oscillation breaker: L,R or R,L in a row means the min-cost pixel
            # straddles the center band; commit to forward instead of spinning.
            if (
                action in (TURN_LEFT, TURN_RIGHT)
                and self._last_action in (TURN_LEFT, TURN_RIGHT)
                and action != self._last_action
                and not stalled
            ):
                action = MOVE_FORWARD

        self._last_action = action
        if position is not None:
            self._last_position = np.asarray(position, dtype=np.float32).copy()
        return action
