import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from gym import Space
from habitat import Config, logger
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net

from pirlnav.policy.policy import ILPolicy
from pirlnav.policy.visual_encoder import VisualEncoder

COSTMAP_UUID = "costmap"


class CostmapTransform:
    """Resize (N,H,W,1) costmap -> (N,2,S,S).

    Two channels because the two signals need different normalizations:
      ch0 absolute: cost / max_cost. Carries "how far is the goal" -> the STOP
          cue (goal reached <=> near-zero values visible).
      ch1 egocentric contrast: per-frame min-max stretched to [0,1]. Carries
          "which direction is downhill". With the global /max_cost scaling
          alone, the within-frame gradient spans only a few percent of the
          input range (e.g. costs 5.6-7.0 -> 0.19-0.23), which a from-scratch
          encoder fails to exploit - observed as the policy converging to the
          action-prior floor (loss ~1.0) while ignoring the costmap.
    """

    def __init__(self, size, max_cost):
        self.size = size
        self.max_cost = float(max_cost)

    def __call__(self, x):
        x = x.permute(0, 3, 1, 2).float()
        x = TF.resize(x, [self.size, self.size])
        absolute = x / self.max_cost
        mins = x.amin(dim=(2, 3), keepdim=True)
        maxs = x.amax(dim=(2, 3), keepdim=True)
        relative = (x - mins) / (maxs - mins).clamp(min=1e-6)
        return torch.cat([absolute, relative], dim=1)


class ObjectNavILCostmapNet(Net):
    def __init__(
        self,
        observation_space: Space,
        policy_config: Config,
        num_actions: int,
        run_type: str,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
    ):
        super().__init__()
        self.policy_config = policy_config
        rnn_input_size = 0

        cm_config = policy_config.COSTMAP_ENCODER
        self.visual_transform = CostmapTransform(
            size=cm_config.image_size, max_cost=cm_config.max_cost
        )

        self.visual_encoder = VisualEncoder(
            image_size=cm_config.image_size,
            backbone=cm_config.backbone,
            input_channels=2,  # [absolute, per-frame contrast] from CostmapTransform
            resnet_baseplanes=cm_config.resnet_baseplanes,
            resnet_ngroups=cm_config.resnet_baseplanes // 2,
            normalize_visual_inputs=cm_config.normalize_visual_inputs,
            avgpooled_image=cm_config.avgpooled_image,
            drop_path_rate=cm_config.drop_path_rate,
        )

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.visual_encoder.output_size, cm_config.hidden_size),
            nn.ReLU(True),
        )
        rnn_input_size += cm_config.hidden_size
        logger.info("Costmap encoder is {}".format(cm_config.backbone))

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]) + 1
            )
            logger.info("Object categories: {}".format(self._n_object_categories))
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")

        if policy_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.rnn_input_size = rnn_input_size
        logger.info(
            "State enc: {} - {} - {} - {}".format(
                rnn_input_size, hidden_size, rnn_type, num_recurrent_layers
            )
        )
        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )
        self._hidden_size = hidden_size
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        N = rnn_hidden_states.size(1)
        x = []

        cm = observations[COSTMAP_UUID]
        if len(cm.size()) == 5:
            cm = cm.contiguous().view(-1, cm.size(2), cm.size(3), cm.size(4))
        cm = self.visual_transform(cm)
        cm = self.visual_encoder(cm)
        cm = self.visual_fc(cm)
        x.append(cm)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(-1, object_goal.size(2))
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if self.policy_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(
            x, rnn_hidden_states.contiguous(), masks
        )
        return x, rnn_hidden_states


@baseline_registry.register_policy
class ObjectNavILCostmapPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        policy_config: Config,
        run_type: str,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
    ):
        super().__init__(
            ObjectNavILCostmapNet(
                observation_space=observation_space,
                policy_config=policy_config,
                num_actions=action_space.n,
                run_type=run_type,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_recurrent_layers=num_recurrent_layers,
            ),
            action_space.n,
            no_critic=policy_config.CRITIC.no_critic,
            mlp_critic=policy_config.CRITIC.mlp_critic,
            critic_hidden_dim=policy_config.CRITIC.hidden_dim,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            policy_config=config.POLICY,
            run_type=config.RUN_TYPE,
            hidden_size=config.POLICY.STATE_ENCODER.hidden_size,
            rnn_type=config.POLICY.STATE_ENCODER.rnn_type,
            num_recurrent_layers=config.POLICY.STATE_ENCODER.num_recurrent_layers,
        )