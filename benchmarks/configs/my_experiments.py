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

from benchmarks.configs.first_experiment import first_experiment
from benchmarks.configs.surf_agent_2obj_train import surf_agent_2obj_train
from benchmarks.configs.surf_agent_2obj_eval import surf_agent_2obj_eval

# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)


experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    first_experiment=first_experiment,
    surf_agent_2obj_train=surf_agent_2obj_train,
    surf_agent_2obj_eval=surf_agent_2obj_eval,
)
CONFIGS = asdict(experiments)
