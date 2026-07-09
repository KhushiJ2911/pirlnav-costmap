#!/usr/bin/env python3
"""Patch PIRLNav for gradient accumulation (Dustin's fix, merged with accuracy patch).
Run from ~/pirlnav:  python patch_gradaccum.py
Then set NUM_ACCUM_STEPS via config/CLI (1 GPU + 4 envs -> 64)."""
import py_compile

# ===== agent.py =====
p = "pirlnav/algos/agent.py"
s = open(p).read()
assert "accumulate_gradients" not in s, "agent.py already patched"

# 1) signature
s = s.replace(
    "    def update(self, rollouts) -> Tuple[float, float, float]:",
    "    def update(self, rollouts, accumulate_gradients=False, num_accum_steps=None) -> Tuple[float, float, float]:",
    1,
)

# 2) zero_grad becomes conditional
s = s.replace(
    "            self.optimizer.zero_grad()\n            inflections_batch = batch[\"observations\"][",
    "            if not accumulate_gradients:\n                self.optimizer.zero_grad()\n            inflections_batch = batch[\"observations\"][",
    1,
)

# 3) reorder metrics + conditional scale/step
old_block = """            total_loss = action_loss_term - entropy_term

            self.before_backward(total_loss)
            total_loss.backward()
            self.after_backward(total_loss)

            self.before_step()
            self.optimizer.step()
            self.after_step()

            total_loss_epoch += total_loss.item()
            total_action_loss += action_loss_term.item()
            total_entropy += dist_entropy.item()
            hidden_states.append(rnn_hidden_states)"""
new_block = """            total_loss = action_loss_term - entropy_term

            total_loss_epoch += total_loss.item()
            total_action_loss += action_loss_term.item()
            total_entropy += dist_entropy.item()

            if accumulate_gradients:
                total_loss = total_loss / num_accum_steps

            self.before_backward(total_loss)
            total_loss.backward()
            self.after_backward(total_loss)

            if not accumulate_gradients:
                self.before_step()
                self.optimizer.step()
                self.after_step()

            hidden_states.append(rnn_hidden_states)"""
assert old_block in s, "agent.py backward/step block anchor missing"
s = s.replace(old_block, new_block, 1)

# 4) add apply_accumulated_gradients after the 4-tuple return
ret_block = """        return (
            total_loss_epoch,
            hidden_states,
            total_entropy,
            total_action_loss,
        )"""
assert ret_block in s, "agent.py return anchor missing"
s = s.replace(
    ret_block,
    ret_block + """

    def apply_accumulated_gradients(self) -> None:
        self.before_step()
        self.optimizer.step()
        self.after_step()
        self.optimizer.zero_grad()""",
    1,
)
open(p, "w").write(s)
py_compile.compile(p, doraise=True)

# ===== il_trainer.py =====
p2 = "pirlnav/il_trainer.py"
t = open(p2).read()
assert "apply_accumulated_gradients" not in t, "il_trainer.py already patched"

old_call = """        (
            action_loss,
            rnn_hidden_states,
            dist_entropy,
            _,
        ) = self.agent.update(self.rollouts)

        self.rollouts.after_update(rnn_hidden_states)"""
new_call = """        num_accum_steps = self.config.NUM_ACCUM_STEPS
        accumulate_gradients = num_accum_steps > 1

        (
            action_loss,
            rnn_hidden_states,
            dist_entropy,
            _,
        ) = self.agent.update(
            self.rollouts, accumulate_gradients, num_accum_steps
        )

        if accumulate_gradients and (
            (self.num_updates_done + 1) % num_accum_steps == 0
        ):
            self.agent.apply_accumulated_gradients()

        self.rollouts.after_update(rnn_hidden_states)"""
assert old_call in t, "il_trainer.py update-call anchor missing"
t = t.replace(old_call, new_call, 1)
open(p2, "w").write(t)
py_compile.compile(p2, doraise=True)

print("OK: gradient accumulation patched into agent.py + il_trainer.py")