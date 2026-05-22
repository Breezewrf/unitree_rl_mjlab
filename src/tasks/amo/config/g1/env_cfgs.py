"""Unitree G1 AMO environment configurations."""

from src.assets.robots import (
    G1_ACTION_SCALE,
    get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from src.tasks.amo.mdp.amo_command import AmoCommandCfg
import src.tasks.amo.mdp as mdp
from src.tasks.amo.velocity_env_cfg import make_amo_env_cfg

# AMO uses lower-body-only actions; filter out upper-body scale entries.
_UPPER_BODY_PATTERNS = ("elbow", "shoulder", "wrist", "waist")
G1_LOWER_BODY_ACTION_SCALE = {
    k: v
    for k, v in G1_ACTION_SCALE.items()
    if not any(p in k for p in _UPPER_BODY_PATTERNS)
}


def _load_amo_module(env, env_ids, checkpoint_path: str):
    """Startup event: load AMO MLP checkpoint and build joint index mappings.

    Stores the following on ``env``:
        _amo_module: AmoModule instance
        _amo_upper_indices: indices into robot joint_pos for checkpoint upper_names
        _amo_lower_indices: indices into robot joint_pos for checkpoint lower_names
    """
    from src.tasks.amo.mdp.amo_module import AmoModule

    amo_module = AmoModule(checkpoint_path, device="cpu")

    asset = env.scene["robot"]
    robot_joint_names = list(asset.joint_names)

    # Build index mappings: checkpoint joint name -> index in robot joint_pos.
    upper_indices = []
    for name in amo_module.upper_names:
        if name not in robot_joint_names:
            raise ValueError(
                f"AMO checkpoint upper joint '{name}' not found in robot joints: "
                f"{robot_joint_names}"
            )
        upper_indices.append(robot_joint_names.index(name))

    lower_indices = []
    for name in amo_module.lower_names:
        if name not in robot_joint_names:
            raise ValueError(
                f"AMO checkpoint lower joint '{name}' not found in robot joints: "
                f"{robot_joint_names}"
            )
        lower_indices.append(robot_joint_names.index(name))

    env._amo_module = amo_module
    env._amo_upper_indices = upper_indices
    env._amo_lower_indices = lower_indices

    print(f"[AMO] Loaded checkpoint: {checkpoint_path}")
    print(f"[AMO] Upper joints ({len(upper_indices)}): {amo_module.upper_names}")
    print(f"[AMO] Lower joints ({len(lower_indices)}): {amo_module.lower_names}")


def unitree_g1_rough_env_cfg(
    play: bool = False,
    checkpoint_path: str = "src/tasks/amo/mdp/amo_module.pt",
) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 rough terrain AMO configuration."""
    cfg = make_amo_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 55

    cfg.scene.entities = {"robot": get_g1_robot_cfg()}

    # Set raycast sensor frame to G1 pelvis.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = "pelvis"

    site_names = ("left_foot", "right_foot")
    geom_names = tuple(
        f"{side}_foot{i}_collision"
        for side in ("left", "right")
        for i in range(1, 8)
    )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_LOWER_BODY_ACTION_SCALE

    cfg.viewer.body_name = "torso_link"

    amo_cmd = cfg.commands["amo"]
    assert isinstance(amo_cmd, AmoCommandCfg)
    amo_cmd.viz.z_offset = 1.15

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    # Add AMO module loading event.
    if checkpoint_path:
        cfg.events["load_amo_module"] = EventTermCfg(
            func=_load_amo_module,
            mode="startup",
            params={"checkpoint_path": checkpoint_path},
        )

    # disable default pose reward
    cfg.rewards["pose"].weight = 0.0

    # Disable height/RPY tracking rewards initially; ramp up via curriculum.
    cfg.rewards["track_height"].weight = 0.0
    cfg.rewards["track_rpy"].weight = 0.0

    cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["body_ang_vel"] = RewardTermCfg(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    )
    cfg.rewards["angular_momentum"] = RewardTermCfg(
        func=mdp.angular_momentum_penalty,
        weight=-0.025,
        params={"sensor_name": "robot/root_angmom"},
    )
    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )
    cfg.rewards["amo_ref_tracking"] = RewardTermCfg(
        func=mdp.amo_ref_tracking,
        weight=0.5,
        params={
            "command_name": "amo",
            "std": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Apply play mode overrides.
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def unitree_g1_flat_env_cfg(
    play: bool = False,
    checkpoint_path: str = "src/tasks/amo/mdp/amo_module.pt",
) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 flat terrain AMO configuration."""
    cfg = unitree_g1_rough_env_cfg(play=play, checkpoint_path=checkpoint_path)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove raycast sensor and height scan (no terrain to scan).
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    del cfg.observations["critic"].terms["height_scan"]

    # Disable terrain curriculum.
    cfg.curriculum.pop("terrain_levels", None)

    # AMO command curriculum: gradually expand height and RPY ranges.
    cfg.curriculum["command_amo"] = CurriculumTermCfg(
        func=mdp.commands_amo,
        params={
            "command_name": "amo",
            "amo_stages": [
                {"step": 5000*24, "height": (0.5, 0.785)},
                # {"step": 10000*24, "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
            ],
        },
    )

    # Ramp up tracking rewards alongside command curriculum.
    cfg.curriculum["track_height_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "track_height",
            "weight_stages": [
                {"step": 5000*24, "weight": 0.5},
            ],
        },
    )
    cfg.curriculum["track_rpy_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "track_rpy",
            "weight_stages": [
                {"step": 10000*24, "weight": 0.5},
            ],
        },
    )

    if play:
        amo_cmd = cfg.commands["amo"]
        assert isinstance(amo_cmd, AmoCommandCfg)
        amo_cmd.ranges.lin_vel_x = (-0.5, 1.0)
        amo_cmd.ranges.lin_vel_y = (-0.5, 0.5)
        amo_cmd.ranges.ang_vel_z = (-0.5, 0.5)

    return cfg
