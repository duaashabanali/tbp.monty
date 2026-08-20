# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Multi-agent-aware counterparts of the graph matching loggers.

`BasicGraphMatchingLogger`/`DetailedGraphMatchingLogger` (see
graph_matching_loggers.py) gather stats by calling ``get_stats_per_lm`` on a
single ``model``. `MultiAgentMontyExperiment` runs a list of independent
Monty instances (``self.monty_agents``), but the parent-class logging
machinery only ever sees ``self.model``, which is hardwired to
``monty_agents[0]``. The classes here gather stats from every agent in
``monty_agents`` instead, keying each learning module's stats as
``agent_<i>_LM_<j>`` so an arbitrary number of agents, each with an arbitrary
number of learning modules, land in the same data pool -- and so agent 0 is
named the same way as every other agent, instead of being the implicit,
unprefixed ``LM_<j>``.
"""

from __future__ import annotations

from typing import Any

from tbp.monty.frameworks.loggers.graph_matching_loggers import (
    BasicGraphMatchingLogger,
    DetailedGraphMatchingLogger,
)
from tbp.monty.frameworks.utils.logging_utils import (
    get_stats_per_lm,
    target_data_to_dict,
)

__all__ = [
    "MultiAgentBasicGraphMatchingLogger",
    "MultiAgentDetailedGraphMatchingLogger",
]


def _populate_multi_agent_basic_data(
    logger_obj: BasicGraphMatchingLogger,
    logger_args: dict[str, Any],
    model: Any,
    agents: list[Any],
) -> None:
    """Fill ``logger_obj.data["BASIC"]`` with per-(agent, LM) stats.

    Equivalent to ``BasicGraphMatchingLogger.update_episode_data``, except it
    calls ``get_stats_per_lm`` once per agent in ``agents`` instead of once
    for a single model, and keys each learning module's stats as
    ``agent_<i>_LM_<j>``.

    ``model`` is still used for the handful of fields that are tracked once
    per episode rather than per agent (experiment mode, action sequence,
    step counters) -- it is expected to be ``agents[0]``, the "primary"
    agent already used for this bookkeeping elsewhere in
    ``MultiAgentMontyExperiment`` (e.g. checkpointing, episode counters).
    """
    target = logger_args["target"]
    seed = logger_args["episode_seed"]

    performance_dict = {}
    for agent_idx, agent in enumerate(agents):
        agent_stats = get_stats_per_lm(agent, target, seed)
        for lm_key, lm_stats in agent_stats.items():
            if not lm_key.startswith("LM_"):
                continue
            performance_dict[f"agent_{agent_idx}_{lm_key}"] = lm_stats

    if len(logger_obj.lms) == 0:  # first time function is called
        logger_obj.lms = list(performance_dict.keys())

    target_dict = target_data_to_dict(target)
    mode = model.experiment_mode
    episode = logger_args[f"{mode}_episodes"]
    actions = model.motor_system.action_sequence
    logger_time = {k: v for k, v in logger_args.items() if k != "target"}

    logger_obj.data["BASIC"][f"{mode}_stats"][episode] = performance_dict
    logger_obj.update_overall_stats(
        mode, episode, model.episode_steps, model.matching_steps
    )
    overall_stats = logger_obj.get_formatted_overall_stats(mode, episode)

    logger_obj.data["BASIC"][f"{mode}_overall_stats"][episode] = overall_stats
    logger_obj.data["BASIC"][f"{mode}_actions"][episode] = actions
    logger_obj.data["BASIC"][f"{mode}_targets"][episode] = target_dict
    logger_obj.data["BASIC"][f"{mode}_timing"][episode] = logger_time
    logger_obj.data["BASIC"][f"{mode}_stats"][episode]["target"] = target_dict


class MultiAgentBasicGraphMatchingLogger(BasicGraphMatchingLogger):
    """BasicGraphMatchingLogger that gathers stats from every agent."""

    def __init__(self, handlers, monty_agents: list[Any]) -> None:
        super().__init__(handlers)
        self._agents = monty_agents

    def update_episode_data(self, logger_args, model) -> None:
        _populate_multi_agent_basic_data(self, logger_args, model, self._agents)


class MultiAgentDetailedGraphMatchingLogger(DetailedGraphMatchingLogger):
    """DetailedGraphMatchingLogger that gathers BASIC stats from every agent.

    The DETAILED portion (per-step buffers dumped to JSON) is copied from
    ``DetailedGraphMatchingLogger.update_episode_data`` essentially unchanged
    and is still scoped to ``model`` (agent 0) only -- the base class already
    flags that buffer path as not yet supporting multiple, independent
    sensor agents. Only the BASIC portion (what ends up in the CSV) is made
    multi-agent aware here.
    """

    def __init__(self, handlers, monty_agents: list[Any]) -> None:
        super().__init__(handlers)
        self._agents = monty_agents

    def update_episode_data(self, logger_args, model) -> None:
        _populate_multi_agent_basic_data(self, logger_args, model, self._agents)

        episodes = logger_args["train_episodes"] + logger_args["eval_episodes"]
        self.train_episodes_to_total[logger_args["train_episodes"]] = episodes
        self.eval_episodes_to_total[logger_args["eval_episodes"]] = episodes

        buffer_data = {}
        for i, lm in enumerate(model.learning_modules):
            lm_dict = {}
            lm_dict.update(logger_args)
            lm_dict.update({"locations": lm.buffer.locations})
            lm_dict.update(lm.buffer.features)
            lm_dict.update({"displacements": lm.buffer.displacements})
            lm_dict.update(lm.buffer.stats)
            lm_dict.update(mode=model.experiment_mode.value)
            lm_dict.update({"stepwise_targets_list": lm.stepwise_targets_list})
            buffer_data[f"LM_{i}"] = lm_dict  # NOTE: probably same for all LMs

        for i, sm in enumerate(model.sensor_modules):
            if len(sm.state_dict()["raw_observations"]) > 0:
                buffer_data[f"SM_{i}"] = sm.state_dict()

        # TODO ensure will work with multiple, independent sensor agents
        buffer_data["motor_system"] = {}
        buffer_data["motor_system"]["action_sequence"] = (
            model.motor_system.action_sequence
        )
        buffer_data["motor_system"]["action_details"] = dict(
            model.motor_system._telemetry_surface_action_details.__dict__
        )
        buffer_data["motor_system"]["policy_selector"] = {
            "selected_goals": model.motor_system._policy_selector._selected_goals,
        }

        self.data["DETAILED"][episodes] = buffer_data
