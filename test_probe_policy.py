"""Unit tests for the greedy probe controller and SP-demo replay bookkeeping.

Runs anywhere with numpy — no habitat, no GPU:
    python3 test_probe_policy.py
"""

import numpy as np

from probe_policy import (
    GreedyCostmapPolicy,
    STOP,
    MOVE_FORWARD,
    TURN_LEFT,
    TURN_RIGHT,
)
from generate_sp_demos import replay_stats


def make_map(fill=30.0, H=64, W=64):
    return np.full((H, W, 1), fill, dtype=np.float32)


def test_stop_when_goal_visible():
    p = GreedyCostmapPolicy()
    cm = make_map(fill=5.0)
    cm[40, 32, 0] = 0.1  # near-zero cost visible
    assert p.act(cm) == STOP


def test_forward_when_min_ahead():
    p = GreedyCostmapPolicy()
    cm = make_map(fill=10.0)
    cm[40, 32, 0] = 5.0  # min cost dead ahead (center column)
    assert p.act(cm) == MOVE_FORWARD


def test_turn_left_when_min_left():
    p = GreedyCostmapPolicy()
    cm = make_map(fill=10.0)
    cm[40, 3, 0] = 5.0  # min cost far left
    assert p.act(cm) == TURN_LEFT


def test_turn_right_when_min_right():
    p = GreedyCostmapPolicy()
    cm = make_map(fill=10.0)
    cm[40, 60, 0] = 5.0  # min cost far right
    assert p.act(cm) == TURN_RIGHT


def test_scan_when_all_invalid():
    p = GreedyCostmapPolicy()
    cm = make_map(fill=30.0)  # sensor fill value: nothing valid in view
    assert p.act(cm) == TURN_LEFT


def test_stall_triggers_turn():
    p = GreedyCostmapPolicy()
    cm = make_map(fill=10.0)
    cm[40, 32, 0] = 5.0  # forward-looking gradient
    # make the left half clearly cheaper so the sidestep direction is deterministic
    cm[:, :32, 0] = 8.0
    cm[40, 32, 0] = 5.0
    a1 = p.act(cm, position=np.array([0.0, 0.0, 0.0]))
    assert a1 == MOVE_FORWARD
    # same position again => forward didn't move us => stall => turn
    a2 = p.act(cm, position=np.array([0.0, 0.0, 0.0]))
    assert a2 in (TURN_LEFT, TURN_RIGHT)


def test_no_oscillation():
    p = GreedyCostmapPolicy()
    cm_left = make_map(fill=10.0)
    cm_left[40, 20, 0] = 5.0  # just left of the center band boundary region
    cm_right = make_map(fill=10.0)
    cm_right[40, 44, 0] = 5.0
    a1 = p.act(cm_left, position=np.array([0.0, 0.0, 0.0]))
    assert a1 == TURN_LEFT
    a2 = p.act(cm_right, position=np.array([0.0, 0.0, 0.0]))
    # would be TURN_RIGHT greedily -> oscillation breaker commits to FORWARD
    assert a2 == MOVE_FORWARD


def test_reset_clears_state():
    p = GreedyCostmapPolicy()
    cm = make_map(fill=10.0)
    cm[40, 32, 0] = 5.0
    p.act(cm, position=np.array([0.0, 0.0, 0.0]))
    p.reset()
    assert p._last_action is None and p._last_position is None


def test_2d_and_3d_input_agree():
    p1, p2 = GreedyCostmapPolicy(), GreedyCostmapPolicy()
    cm = make_map(fill=10.0)
    cm[40, 60, 0] = 5.0
    assert p1.act(cm) == p2.act(cm[:, :, 0])


def test_replay_stats_matches_inflection_sensor_semantics():
    # replay: [placeholder STOP, F, F, L, F, STOP]
    eps = [
        {
            "reference_replay": [
                {"action": "STOP"},
                {"action": "MOVE_FORWARD"},
                {"action": "MOVE_FORWARD"},
                {"action": "TURN_LEFT"},
                {"action": "MOVE_FORWARD"},
                {"action": "STOP"},
            ]
        }
    ]
    total, infl = replay_stats(eps)
    # timesteps 1..5 -> total 5; changes at t=1 (STOP->F), t=3 (F->L),
    # t=4 (L->F), t=5 (F->STOP) -> 4 inflections
    assert total == 5, total
    assert infl == 4, infl


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(
        (k, v) for k, v in dict(globals()).items() if k.startswith("test_")
    ):
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            fails += 1
    raise SystemExit(fails)
