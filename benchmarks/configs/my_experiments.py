# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import asdict
from benchmarks.configs.names import MyExperiments
from benchmarks.configs.two_agent_monty_experiment import two_agent_monty_experiment
from tbp.monty.frameworks.models.motor_policies import NaiveScanPolicy

# from benchmarks.configs.first_experiment import first_experiment
# from benchmarks.configs.surf_agent_2obj_train import surf_agent_2obj_train
# from benchmarks.configs.surf_agent_2obj_eval import surf_agent_2obj_eval

# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)

from dataclasses import asdict

# from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    PatchAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    ExperimentArgs,
    get_env_dataloader_per_object_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.simulators.habitat.configs import (
    PatchViewFinderMountHabitatDatasetArgs,
)

#####
# To test your env and help you familiarize yourself with the code, we'll run the simplest possible
# experiment. We'll use a model with a single learning module as specified in
# monty_config. We'll also skip evaluation, train for a single epoch for a single step,
# and only train on a single object, as specified in experiment_args and train_dataloader_args.
#####

first_experiment = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    logging_config=LoggingConfig(),
    experiment_args=ExperimentArgs(
        do_eval=False,
        max_train_steps=1,
        n_train_epochs=1,
    ),
    monty_config=PatchAndViewMontyConfig(),
    # Data{set, loader} config
    # dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.EnvironmentDataLoaderPerObject,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
)
# --------------------------------------------------------------------------------

import os
from dataclasses import asdict

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MotorSystemConfigCurvatureInformedSurface,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.graph_matching import GraphLM
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatSurfacePatchSM,
)
from tbp.monty.simulators.habitat.configs import (
    SurfaceViewFinderMountHabitatDatasetArgs,
)

"""
Basic setup
-----------
"""
# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/tbp/results/monty/projects")

# Specify a name for the model.
model_name = "surf_agent_1lm_2obj"

"""
Training
----------------------------------------------------------------------------------------
"""
# Here we specify which objects to learn. 'mug' and 'banana' come from the YCB dataset.
# If you don't have the YCB dataset, replace with names from habitat (e.g.,
# 'capsule3DSolid', 'cubeSolid', etc.).
object_names = ["mug", "banana"]
# Get predefined object rotations that give good views of the object from 14 angles.
train_rotations = get_cube_face_and_corner_views_rotations()

# The config dictionary for the pretraining experiment.
surf_agent_2obj_train = dict(
    # Specify monty experiment and its args.
    # The MontySupervisedObjectPretrainingExperiment class will provide the model
    # with object and pose labels for supervised pretraining.
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(train_rotations),
        do_eval=False,
    ),
    # Specify logging config.
    logging_config=PretrainLoggingConfig(
        output_dir=project_dir,
        run_name=model_name,
        wandb_handlers=[],
    ),
    # Specify the Monty config.
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        # sensory module configs: one surface patch for training (sensor_module_0),
        # and one view-finder for initializing each episode and logging
        # (sensor_module_1).
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSurfacePatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    # a list of features that the SM will extract and send to the LM
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        # learning module config: 1 graph learning module.
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=GraphLM,
                learning_module_args=dict(),  # Use default LM args
            )
        ),
        # Motor system config: class specific to surface agent.
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
    # Set up the environment and agent
    # dataset_class=ED.EnvironmentDataset,
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
    ),
    # For a complete config we need to specify an eval_dataloader but since we only train here, this is unused
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
    ),
)

#----------------------------------------------------------------------------------------

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
#----------------------------------------------------------------------------------------

import os
from dataclasses import asdict

import numpy as np

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    CSVLoggingConfig,
    MontyArgs,
    SurfaceAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    RandomRotationObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM
)
from tbp.monty.simulators.habitat.configs import (
    SurfaceViewFinderMountHabitatDatasetArgs,
)

"""
Basic setup
-----------
"""

# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/tbp/results/monty/projects")

# Specify a name for the model.
model_name = "surf_agent_2obj_unsupervised"

# Here we specify which objects to learn. We are going to use the mug and bowl
# from the YCB dataset.
object_names = ["mug", "bowl"]

# Set up config for an evidence graph learning module.
learning_module_0 = dict(
    learning_module_class=EvidenceGraphLM,
    learning_module_args=dict(
        max_match_distance=0.01,
        # Tolerances within which features must match stored values in order to add
        # evidence to a hypothesis.
        tolerances={
            "patch": {
                "hsv": np.array([0.05, 0.1, 0.1]),
                "principal_curvatures_log": np.ones(2),
            }
        },
        feature_weights={
            "patch": {
                # Weighting saturation and value less since these might change
                # under different lighting conditions.
                "hsv": np.array([1, 0.5, 0.5]),
            }
        },
        x_percent_threshold=20,
        # Thresholds to use for when two points are considered different enough to
        # both be stored in memory.
        graph_delta_thresholds=dict(
            patch=dict(
                distance=0.01,
                pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                principal_curvatures_log=[1.0, 1.0],
                hsv=[0.1, 1, 1],
            )
        ),
        # object_evidence_th sets a minimum threshold on the amount of evidence we have
        # for the current object in order to converge; while we can also set min_steps
        # for the experiment, this puts a more stringent requirement that we've had
        # many steps that have contributed evidence.
        object_evidence_threshold=100,
        # Symmetry evidence (indicating possibly symmetry in rotations) increments a lot
        # after 100 steps and easily reaches the default required evidence. The below
        # parameter value partially addresses this, altough we note these are temporary
        # fixes and we intend to implement a more principled approach in the future.
        required_symmetry_evidence=20,
        hypotheses_updater_args=dict(
            max_nneighbors=5
        )
    ),
)
learning_module_configs = dict(learning_module_0=learning_module_0)

# The config dictionary for the unsupervised learning experiment.
surf_agent_2obj_unsupervised = dict(
    # Set up unsupervised experiment.
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=ExperimentArgs(
        # Not running eval here. The only difference between training and evaluation
        # is that during evaluation, no models are updated.
        do_eval=False,
        n_train_epochs=3,
        max_train_steps=2000,
        max_total_steps=5000,
    ),
    logging_config=CSVLoggingConfig(
        python_log_level="INFO",
        output_dir=project_dir,
        run_name=model_name,
    ),
    # Set up monty, including LM, SM, and motor system. We will use the default
    # sensor modules (1 habitat surface patch, one logging view finder), motor system,
    # and connectivity matrices given by `SurfaceAndViewMontyConfig`.
    monty_config=SurfaceAndViewMontyConfig(
        # Take 1000 exploratory steps after recognizing the object to collect more
        # information about it. Require at least 100 steps before recognizing an object
        # to avoid early misclassifications when we have few objects in memory.
        monty_args=MontyArgs(num_exploratory_steps=1000, min_train_steps=100),
        learning_module_configs=learning_module_configs,
    ),
    # Set up the environment and agent.
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
    # Doesn't get used, but currently needs to be set anyways.
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

#----------------------------------------------------------------------------------------

import os
from dataclasses import asdict

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    FiveLMMontyConfig,
    MontyArgs,
    MotorSystemConfigNaiveScanSpiral,
    PretrainLoggingConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    get_env_dataloader_per_object_by_idx,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.simulators.habitat.configs import (
    FiveLMMountHabitatDatasetArgs,
)

# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/tbp/results/monty/projects")

# Specify a name for the model.
model_name = "dist_agent_5lm_2obj"

# Specify the objects to train on and 14 unique object poses.
object_names = ["mug", "banana"]
train_rotations = get_cube_face_and_corner_views_rotations()

# The config dictionary for the pretraining experiment.
dist_agent_5lm_2obj_train = dict(
    # Specify monty experiment class and its args.
    # The MontySupervisedObjectPretrainingExperiment class will provide the model
    # with object and pose labels for supervised pretraining.
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(train_rotations),
    ),
    # Specify logging config.
    logging_config=PretrainLoggingConfig(
        output_dir=project_dir,
        run_name=model_name,
    ),
    # Specify the Monty model. The FiveLLMMontyConfig contains all of the sensor module
    # configs, learning module configs, and connectivity matrices we need.
    monty_config=FiveLMMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=dict(
                policy_class=NaiveScanPolicy,
                policy_args=make_naive_scan_policy_config(step_size=5),
            )
        ),
    ),
    # Set up the environment and agent.
    dataset_args=FiveLMMountHabitatDatasetArgs(),
    # Set up the training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
    ),
    # Set up the evaluation dataloader. Unused, but required.
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,  # just placeholder
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
)

#----------------------------------------------------------------------------------------

import copy
import os

import numpy as np

from tbp.monty.frameworks.config_utils.config_args import (
    EvalLoggingConfig,
    FiveLMMontyConfig,
    MontyArgs,
    MotorSystemConfigInformedGoalStateDriven,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
    get_env_dataloader_per_object_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM
)
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching
)
from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.simulators.habitat.configs import (
    FiveLMMountHabitatDatasetArgs,
)

"""
Basic Info
"""

# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/tbp/results/monty/projects")

# Specify a name for the model.
model_name = "dist_agent_5lm_2obj"

object_names = ["mug", "banana"]
test_rotations = [np.array([0, 15, 30])] # A previously unseen rotation of the objects

model_path = os.path.join(
    project_dir,
    model_name,
    "pretrained",
)

"""
Learning Module Configs
"""
# Create a template config that we'll make copies of.
evidence_lm_config = dict(
    learning_module_class=EvidenceGraphLM,
    learning_module_args=dict(
        max_match_distance=0.01,  # =1cm
        feature_weights={
            "patch": {
                # Weighting saturation and value less since these might change under
                # different lighting conditions.
                "hsv": np.array([1, 0.5, 0.5]),
            }
        },
        # Use this to update all hypotheses > x_percent_threshold (faster)
        evidence_threshold_config="x_percent_threshold",
        x_percent_threshold=20,
        gsg_class=EvidenceGoalStateGenerator,
        gsg_args=dict(
            goal_tolerances=dict(
                location=0.015,  # distance in meters
            ),  # Tolerance(s) when determining goal-state success
            min_post_goal_success_steps=5,  # Number of necessary steps for a hypothesis
        ),
        hypotheses_updater_args=dict(
            max_nneighbors=10,
        )
    ),
)
# We'll also reuse these tolerances, so we specify them here.
tolerance_values = {
    "hsv": np.array([0.1, 0.2, 0.2]),
    "principal_curvatures_log": np.ones(2),
}

# Now we make 5 copies of the template config, each with the tolerances specified for
# one of the five sensor modules.
learning_module_configs = {}
for i in range(5):
    lm = copy.deepcopy(evidence_lm_config)
    lm["learning_module_args"]["tolerances"] = {f"patch_{i}": tolerance_values}
    learning_module_configs[f"learning_module_{i}"] = lm

# The config dictionary for the pretraining experiment.
dist_agent_5lm_2obj_eval = dict(
    #  Specify monty experiment class and its args.
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations),
        min_lms_match=3,   # Terminate when 3 learning modules makes a decision.
    ),
    # Specify logging config.
    logging_config=EvalLoggingConfig(
        output_dir=os.path.join(project_dir, model_name),
        run_name="eval",
        monty_handlers=[BasicCSVStatsHandler],
        wandb_handlers=[],
    ),
    # Specify the Monty model. The FiveLLMMontyConfig contains all of the
    # sensor module configs and connectivity matrices. We will specify
    # evidence-based learning modules and MontyForEvidenceGraphMatching which
    # facilitates voting between evidence-based learning modules.
    monty_config=FiveLMMontyConfig(
        monty_args=MontyArgs(min_eval_steps=20),
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=learning_module_configs,
        motor_system_config=MotorSystemConfigInformedGoalStateDriven(),
    ),
    # Set up the environment and agent.
    dataset_args=FiveLMMountHabitatDatasetArgs(),
    # Set up the training dataloader. Unused, but must be included.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
    # Set up the evaluation dataloader.
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)

#----------------------------------------------------------------------------------------




experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    first_experiment=first_experiment,
    surf_agent_2obj_train=surf_agent_2obj_train,
    surf_agent_2obj_eval=surf_agent_2obj_eval,
    surf_agent_2obj_unsupervised=surf_agent_2obj_unsupervised,
    dist_agent_5lm_2obj_train=dist_agent_5lm_2obj_train,
    dist_agent_5lm_2obj_eval=dist_agent_5lm_2obj_eval,
    two_agent_monty_experiment = two_agent_monty_experiment,
)
CONFIGS = asdict(experiments)
