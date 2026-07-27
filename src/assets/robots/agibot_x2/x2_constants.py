"""Agibot X2 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

X2_XML: Path = SRC_PATH / "assets" / "robots" / "agibot_x2" / "x2_simplified.xml"
assert X2_XML.exists()

X2_CSV_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_pitch_joint",
  "waist_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
  "head_yaw_joint",
  "head_pitch_joint",
)

# The deployed head is fixed, so tracking and control use the first 29 joints.
X2_JOINT_NAMES: tuple[str, ...] = X2_CSV_JOINT_NAMES[:-2]


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, X2_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """Load X2 and fix its two head joints for the 29-DoF control layout."""
  spec = mujoco.MjSpec.from_file(str(X2_XML))
  spec.assets = get_assets(spec.meshdir)

  for actuator in list(spec.actuators):
    spec.delete(actuator)

  fixed_head_joints = {"head_yaw_joint", "head_pitch_joint"}
  for sensor in list(spec.sensors):
    if sensor.objname in fixed_head_joints:
      spec.delete(sensor)
  for joint_name in fixed_head_joints:
    spec.delete(spec.joint(joint_name))

  # EntityCfg supplies the initial state, and SceneCfg supplies terrain and lighting.
  for key in list(spec.keys):
    spec.delete(key)
  for geom in list(spec.worldbody.geoms):
    spec.delete(geom)
  for light in list(spec.worldbody.lights):
    spec.delete(light)

  # Normalize the native sensor names to the convention used by mjlab tasks.
  spec.sensor("body-linear-vel").name = "imu_lin_vel"
  spec.sensor("body-angular-velocity").name = "imu_ang_vel"
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="pelvis",
  )

  spec.option.timestep = mujoco.MjOption().timestep
  return spec


X2_ACTUATOR_HIP_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint"),
  stiffness=40.0,
  damping=4.0,
  effort_limit=120.0,
)
X2_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=30.0,
  damping=3.0,
  effort_limit=120.0,
)
X2_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=80.0,
  damping=8.0,
  effort_limit=120.0,
)
X2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=40.0,
  damping=4.0,
  effort_limit=36.0,
)
X2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=20.0,
  damping=2.0,
  effort_limit=24.0,
)
X2_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=20.0,
  damping=4.0,
  effort_limit=120.0,
)
X2_ACTUATOR_WAIST_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=20.0,
  damping=4.0,
  effort_limit=48.0,
)
X2_ACTUATOR_SHOULDER_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"),
  stiffness=20.0,
  damping=2.0,
  effort_limit=36.0,
)
X2_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_yaw_joint",
  ),
  stiffness=20.0,
  damping=2.0,
  effort_limit=24.0,
)
X2_ACTUATOR_WRIST_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_roll_joint"),
  stiffness=20.0,
  damping=2.0,
  effort_limit=4.8,
)

X2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X2_ACTUATOR_HIP_PITCH_ROLL,
    X2_ACTUATOR_HIP_YAW,
    X2_ACTUATOR_KNEE,
    X2_ACTUATOR_ANKLE_PITCH,
    X2_ACTUATOR_ANKLE_ROLL,
    X2_ACTUATOR_WAIST_YAW,
    X2_ACTUATOR_WAIST_PITCH_ROLL,
    X2_ACTUATOR_SHOULDER_PITCH_ROLL,
    X2_ACTUATOR_ARM,
    X2_ACTUATOR_WRIST_PITCH_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)

X2_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.68),
  joint_pos={
    ".*_hip_pitch_joint": -0.3,
    ".*_knee_joint": 0.7,
    ".*_ankle_pitch_joint": -0.18,
    "waist_pitch_joint": 0.2,
    ".*_shoulder_pitch_joint": 0.15,
    ".*_elbow_joint": -1.0,
    "left_wrist_yaw_joint": -0.8,
    "right_wrist_yaw_joint": 0.8,
  },
  joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)


def get_x2_robot_cfg() -> EntityCfg:
  """Return a fresh 29-DoF X2 configuration with a physically fixed head."""
  return EntityCfg(
    init_state=X2_HOME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=X2_ARTICULATION,
    sort_actuators=True,
  )


X2_ACTION_SCALE: dict[str, float] = {}
for actuator in X2_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  assert actuator.effort_limit is not None
  for name_expr in actuator.target_names_expr:
    X2_ACTION_SCALE[name_expr] = (
      0.25 * actuator.effort_limit / actuator.stiffness
    )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_x2_robot_cfg())
  viewer.launch(robot.spec.compile())
