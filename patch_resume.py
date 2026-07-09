import torch
p = "data/new_checkpoints/il_repro/.habitat-resume-state.pth"
st = torch.load(p, map_location="cpu")
cfg = st["config"]; cfg.defrost()
cfg.NUM_UPDATES = 80000        # 20k done -> train to 80k (~paper frames)
cfg.NUM_CHECKPOINTS = 40       # checkpoint ~every 2k updates
cfg.freeze()
torch.save(st, p)
print("patched: NUM_UPDATES =", st["config"].NUM_UPDATES,
      "| NUM_CHECKPOINTS =", st["config"].NUM_CHECKPOINTS,
      "| updates_done =", st["requeue_stats"]["num_updates_done"])