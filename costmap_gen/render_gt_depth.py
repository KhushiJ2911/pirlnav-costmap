"""Render GT depth for a session using Habitat-Sim (320x240, HFOV=120, sensor [0,0.4,0])."""

import argparse
import os
import numpy as np
import habitat_sim

WIDTH, HEIGHT, HFOV, SENSOR_HEIGHT = 320, 240, 120, 0.4


def make_cfg(scene_glb):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    sim_cfg.enable_physics = False

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [HEIGHT, WIDTH]
    depth_spec.position = [0.0, SENSOR_HEIGHT, 0.0]
    depth_spec.hfov = HFOV

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [depth_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_folder", required=True)
    ap.add_argument("--scene_glb", required=True)
    args = ap.parse_args()

    session_folder = os.path.expanduser(args.session_folder)

    states = np.load(
        os.path.join(session_folder, "agent_states.npy"),
        allow_pickle=True
    )

    out_dir = os.path.join(session_folder, "images_depth")
    os.makedirs(out_dir, exist_ok=True)

    sim = habitat_sim.Simulator(make_cfg(args.scene_glb))
    agent = sim.get_agent(0)

    n = len(states)
    print(f"Rendering depth for {n} frames -> {out_dir}")

    for i, st in enumerate(states):
        s = habitat_sim.AgentState()
        s.position = np.array(st.position, dtype=np.float32)
        s.rotation = st.rotation

        agent.set_state(s)

        obs = sim.get_sensor_observations()
        depth = np.asarray(
            obs["depth_sensor"],
            dtype=np.float32
        )

        np.save(
            os.path.join(out_dir, f"{i:05d}.npy"),
            depth
        )

    sim.close()
    print(f"Done. Saved {n} depth maps (float32 meters) to {out_dir}")


if __name__ == "__main__":
    main()