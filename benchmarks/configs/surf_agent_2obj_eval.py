import os
from dataclasses import asdict

import numpy as np

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    EvalLoggingConfig,
    MontyArgs,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    PatchAndViewSOTAMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM
)
from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    FeatureChangeSM,
)
from tbp.monty.simulators.habitat.configs import (
    SurfaceViewFinderMountHabitatDatasetArgs,
)

"""
Basic setup
-----------
"""
# Specify the directory where an output directory will be created.
project_dir = os.path.expanduser("~/tbp/results/monty/projects")

# Specify the model name. This needs to be the same name as used for pretraining.
model_name = "surf_agent_1lm_2obj"

# Where to find the pretrained model.
model_path = os.path.join(project_dir, model_name, "pretrained")

# Where to save eval logs.
output_dir = os.path.join(project_dir, model_name)
run_name = "eval"

# Specify objects to test and the rotations in which they'll be presented.
object_names = ["mug", "banana"]
test_rotations = [
    np.array([0.0, 15.0, 30.0]),
    np.array([7.0, 77.0, 2.0]),
    np.array([81.0, 33.0, 90.0]),
]

# Let's add some noise to the sensor module outputs to make the task more challenging.
sensor_noise_params = dict(
    features=dict(
        pose_vectors=2,  # rotate by random degrees along xyz
        hsv=np.array([0.1, 0.2, 0.2]),  # add noise to each channel (the values here specify std. deviation of gaussian for each channel individually)
        principal_curvatures_log=0.1,
        pose_fully_defined=0.01,  # flip bool in 1% of cases
    ),
    location=0.002,  # add gaussian noise with 0.002 std (0.2cm)
)

sensor_module_0 = dict(
    sensor_module_class=FeatureChangeSM,
    sensor_module_args=dict(
        sensor_module_id="patch",
        # Features that will be extracted and sent to LM
        # note: don't have to be all the features extracted during pretraining.
        features=[
            "pose_vectors",
            "pose_fully_defined",
            "on_object",
            "object_coverage",
            "min_depth",
            "mean_depth",
            "hsv",
            "principal_curvatures",
            "principal_curvatures_log",
        ],
        save_raw_obs=False,
        # FeatureChangeSM will only send an observation to the LM if features or location
        # changed more than these amounts.
        delta_thresholds={
            "on_object": 0,
            "n_steps": 20,
            "hsv": [0.1, 0.1, 0.1],
            "pose_vectors": [np.pi / 4, np.pi * 2, np.pi * 2],
            "principal_curvatures_log": [2, 2],
            "distance": 0.01,
        },
        surf_agent_sm=True,  # for surface agent
        noise_params=sensor_noise_params,
    ),
)
sensor_module_1 = dict(
    sensor_module_class=DetailedLoggingSM,
    sensor_module_args=dict(
        sensor_module_id="view_finder",
        save_raw_obs=False,
    ),
)
sensor_module_configs = dict(
    sensor_module_0=sensor_module_0,
    sensor_module_1=sensor_module_1,
)

# Tolerances within which features must match stored values in order to add evidence
# to a hypothesis.
tolerances = {
    "patch": {
        "hsv": np.array([0.1, 0.2, 0.2]),
        "principal_curvatures_log": np.ones(2),
    }
}

# Features where weight is not specified default to 1.
feature_weights = {
    "patch": {
        # Weighting saturation and value less since these might change under different
        # lighting conditions.
        "hsv": np.array([1, 0.5, 0.5]),
    }
}

learning_module_0 = dict(
    learning_module_class=EvidenceGraphLM,
    learning_module_args=dict(
        # Search the model in a radius of 1cm from the hypothesized location on the model.
        max_match_distance=0.01,  # =1cm
        tolerances=tolerances,
        feature_weights=feature_weights,
        # Most likely hypothesis needs to have 20% more evidence than the others to 
        # be considered certain enough to trigger a terminal condition (match).
        x_percent_threshold=20,
        # Update all hypotheses with evidence > x_percent_threshold (faster)
        evidence_threshold_config="x_percent_threshold",
        # Config for goal state generator of LM which is used for model-based action
        # suggestions, such as hypothesis-testing actions.
        gsg_class=EvidenceGoalStateGenerator,
        gsg_args=dict(
            # Tolerance(s) when determining goal-state success
            goal_tolerances=dict(
                location=0.015,  # distance in meters
            ),
            # Number of necessary steps for a hypothesis-testing action to be considered
            min_post_goal_success_steps=5,
        ),
        hypotheses_updater_args=dict(
            # Look at features associated with (at most) the 10 closest learned points.
            max_nneighbors=10,
        )
    ),
)
learning_module_configs = dict(learning_module_0=learning_module_0)

# The config dictionary for the evaluation experiment.
surf_agent_2obj_eval = dict(
    # Set up experiment
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,  # load the pre-trained models from this path
        n_eval_epochs=len(test_rotations),
        max_total_steps=5000,
    ),
    logging_config=EvalLoggingConfig(
        output_dir=output_dir,
        run_name=run_name,
        wandb_handlers=[],  # remove this line if you, additionally, want to log to WandB.
    ),
    # Set up monty, including LM, SM, and motor system.
    monty_config=PatchAndViewSOTAMontyConfig(
        monty_args=MontyArgs(min_eval_steps=20),
        sensor_module_configs=sensor_module_configs,
        learning_module_configs=learning_module_configs,
        motor_system_config=MotorSystemConfigCurInformedSurfaceGoalStateDriven(),
    ),
    # Set up environment/data
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
    # Doesn't get used, but currently needs to be set anyways.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)