# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch, sentinel

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.experiments.multi_agent_experiment import (
    MultiAgentMontyExperiment,
)
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)


def _new_experiment() -> MultiAgentMontyExperiment:
    """Build an uninitialized instance so tests can set only what they need.

    __init__ pulls apart a full Monty config and builds real models, which is
    unnecessary for exercising the agent-looping logic in isolation.

    Returns:
        An uninitialized MultiAgentMontyExperiment instance.
    """
    return object.__new__(MultiAgentMontyExperiment)


class MultiAgentMontyExperimentTrainTest(TestCase):
    def test_sets_mode_on_extra_agents_before_delegating_to_super(self) -> None:
        experiment = _new_experiment()
        agent0, agent1, agent2 = MagicMock(), MagicMock(), MagicMock()
        experiment.monty_agents = [agent0, agent1, agent2]

        call_order = []
        agent1.set_experiment_mode.side_effect = lambda _mode: call_order.append(
            "agent1"
        )
        agent2.set_experiment_mode.side_effect = lambda _mode: call_order.append(
            "agent2"
        )

        with patch.object(MontyExperiment, "train") as mock_super_train:
            mock_super_train.side_effect = lambda: call_order.append("super")
            experiment.train()

        agent1.set_experiment_mode.assert_called_once_with(ExperimentMode.TRAIN)
        agent2.set_experiment_mode.assert_called_once_with(ExperimentMode.TRAIN)
        # Agent 0 is self.model; the base class sets its mode inside train().
        agent0.set_experiment_mode.assert_not_called()
        mock_super_train.assert_called_once_with()
        self.assertEqual(call_order, ["agent1", "agent2", "super"])

    def test_single_agent_does_not_error(self) -> None:
        experiment = _new_experiment()
        agent0 = MagicMock()
        experiment.monty_agents = [agent0]

        with patch.object(MontyExperiment, "train") as mock_super_train:
            experiment.train()

        agent0.set_experiment_mode.assert_not_called()
        mock_super_train.assert_called_once_with()


class MultiAgentMontyExperimentEvaluateTest(TestCase):
    def test_sets_mode_on_extra_agents_before_delegating_to_super(self) -> None:
        experiment = _new_experiment()
        agent0, agent1, agent2 = MagicMock(), MagicMock(), MagicMock()
        experiment.monty_agents = [agent0, agent1, agent2]

        call_order = []
        agent1.set_experiment_mode.side_effect = lambda _mode: call_order.append(
            "agent1"
        )
        agent2.set_experiment_mode.side_effect = lambda _mode: call_order.append(
            "agent2"
        )

        with patch.object(MontyExperiment, "evaluate") as mock_super_evaluate:
            mock_super_evaluate.side_effect = lambda: call_order.append("super")
            experiment.evaluate()

        agent1.set_experiment_mode.assert_called_once_with(ExperimentMode.EVAL)
        agent2.set_experiment_mode.assert_called_once_with(ExperimentMode.EVAL)
        agent0.set_experiment_mode.assert_not_called()
        mock_super_evaluate.assert_called_once_with()
        self.assertEqual(call_order, ["agent1", "agent2", "super"])


class MultiAgentMontyExperimentPreEpisodeTest(TestCase):
    def test_propagates_target_to_extra_agents_without_semantic_map(self) -> None:
        experiment = _new_experiment()
        agent0, agent1, agent2 = MagicMock(), MagicMock(), MagicMock()
        experiment.monty_agents = [agent0, agent1, agent2]
        experiment.env_interface = SimpleNamespace(primary_target=sentinel.target)

        with patch.object(
            MontyObjectRecognitionExperiment, "pre_episode"
        ) as mock_super_pre_episode:
            experiment.pre_episode()

        mock_super_pre_episode.assert_called_once_with()
        # Agent 0 is self.model; the base class's pre_episode() call handles it.
        agent0.pre_episode.assert_not_called()
        agent1.pre_episode.assert_called_once_with(sentinel.target)
        agent2.pre_episode.assert_called_once_with(sentinel.target)

    def test_propagates_target_and_semantic_map_when_available(self) -> None:
        experiment = _new_experiment()
        agent0, agent1 = MagicMock(), MagicMock()
        experiment.monty_agents = [agent0, agent1]
        experiment.env_interface = SimpleNamespace(
            primary_target=sentinel.target,
            semantic_id_to_label=sentinel.semantic_map,
        )

        with patch.object(MontyObjectRecognitionExperiment, "pre_episode"):
            experiment.pre_episode()

        agent0.pre_episode.assert_not_called()
        agent1.pre_episode.assert_called_once_with(
            sentinel.target, sentinel.semantic_map
        )

    def test_single_agent_does_not_error(self) -> None:
        experiment = _new_experiment()
        agent0 = MagicMock()
        experiment.monty_agents = [agent0]
        experiment.env_interface = SimpleNamespace(primary_target=sentinel.target)

        with patch.object(MontyObjectRecognitionExperiment, "pre_episode"):
            experiment.pre_episode()

        agent0.pre_episode.assert_not_called()
