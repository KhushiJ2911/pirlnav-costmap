#!/usr/bin/env python3
import py_compile

p = "pirlnav/il_trainer.py"
t = open(p).read()

assert "bc_metrics" not in t, "il_trainer already logs bc_metrics"

old = """
        losses = self._coalesce_post_step(
            dict(
                action_loss=action_loss,
                entropy=dist_entropy,
            ),
            count_steps_delta,
        )"""

new = """
        losses = self._coalesce_post_step(
            dict(
                action_loss=action_loss,
                entropy=dist_entropy,
                **getattr(self.agent, "bc_metrics", {}),
            ),
            count_steps_delta,
        )"""

assert old in t, "losses-dict anchor missing"

open(p, "w").write(t.replace(old, new, 1))
py_compile.compile(p, doraise=True)

print("OK: accuracy now logged (losses/action_accuracy, top_3, etc.)")