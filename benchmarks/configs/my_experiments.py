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

# from benchmarks.configs.first_experiment import first_experiment
from benchmarks.configs.surf_agent_2obj_train import surf_agent_2obj_train
from benchmarks.configs.surf_agent_2obj_eval import surf_agent_2obj_eval

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


experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    first_experiment=first_experiment,
    surf_agent_2obj_train=surf_agent_2obj_train,
    surf_agent_2obj_eval=surf_agent_2obj_eval,
)
CONFIGS = asdict(experiments)
