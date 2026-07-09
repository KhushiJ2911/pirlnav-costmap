#!/usr/bin/env python3

import py_compile

# ============================================================
# Patch pirlnav/algos/agent.py
# ============================================================

p = "pirlnav/algos/agent.py"

with open(p, "r") as f:
    s = f.read()

assert "bc_metrics" not in s, "agent.py already patched"

a1 = """
        total_loss_epoch = 0.0
        total_entropy = 0.0
        total_action_loss = 0.0"""

assert a1 in s, "agent init anchor missing"

s = s.replace(
    a1,
    a1
    + """
        total_accuracy = 0.0
        total_top3 = 0.0""",
    1,
)

a2 = """
            action_loss = cross_entropy_loss(
                logits.permute(0, 2, 1), actions_batch.squeeze(-1)
            )"""

assert a2 in s, "agent action_loss anchor missing"

s = s.replace(
    a2,
    a2
    + """
            with torch.no_grad():
                _target = actions_batch.squeeze(-1)
                _pred = logits.argmax(dim=-1)
                total_accuracy += (_pred == _target).float().mean().item()
                _k = min(3, logits.size(-1))
                _topk = logits.topk(_k, dim=-1).indices
                total_top3 += (
                    (_topk == _target.unsqueeze(-1)).any(dim=-1).float().mean().item()
                )""",
    1,
)

a3 = """
        total_action_loss /= self.num_mini_batch

        return ("""

assert a3 in s, "agent return anchor missing"

s = s.replace(
    a3,
    """
        total_action_loss /= self.num_mini_batch
        total_accuracy /= self.num_mini_batch
        total_top3 /= self.num_mini_batch

        self.bc_metrics = {
            "action_ce_loss": total_action_loss,
            "action_accuracy": total_accuracy,
            "top_1_action_accuracy": total_accuracy,
            "top_3_action_accuracy": total_top3,
        }

        return (""",
    1,
)

with open(p, "w") as f:
    f.write(s)

py_compile.compile(p, doraise=True)

# ============================================================
# Patch pirlnav/il_trainer.py
# ============================================================

p2 = "pirlnav/il_trainer.py"

with open(p2, "r") as f:
    t = f.read()

assert 'f"train/{k}"' not in t, "il_trainer.py already patched"

b1 = """
                for k, v in losses.items():
                    writer.add_scalar(f"losses/{k}", v, self.num_steps_done)"""

assert b1 in t, "trainer losses anchor missing"

t = t.replace(
    b1,
    b1
    + """
                if hasattr(self.agent, "bc_metrics"):
                    for k, v in self.agent.bc_metrics.items():
                        writer.add_scalar(f"train/{k}", v, self.num_steps_done)""",
    1,
)

with open(p2, "w") as f:
    f.write(t)

py_compile.compile(p2, doraise=True)

print("OK: patched agent.py + il_trainer.py, both compile cleanly")