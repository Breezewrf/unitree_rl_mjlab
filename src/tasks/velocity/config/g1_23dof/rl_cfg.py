"""RL configuration for Unitree G1-23DOF velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
  RslRlSymmetryCfg,
)


def unitree_g1_23dof_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1-23DOF velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      # Symmetry augmentation + mirror-consistency loss for bipedal locomotion.
      symmetry_cfg=RslRlSymmetryCfg(
        use_data_augmentation=True,
        data_augmentation_func=(
          "src.tasks.velocity.rl.symmetry:g1_23dof_symmetry_augmentation"
        ),
        use_mirror_loss=True,
        mirror_loss_coeff=0.1,
      ),
    ),
    experiment_name="g1_23dof_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
