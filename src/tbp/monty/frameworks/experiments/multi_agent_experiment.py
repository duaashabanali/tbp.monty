# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)

__all__ = ["MultiAgentMontyExperiment"]

logger = logging.getLogger(__name__)


class MultiAgentMontyExperiment(MontyObjectRecognitionExperiment):
    """Runs multiple distinct Monty agents together in a shared environment.

    Each agent has its own sensor modules, learning modules, and motor system.
    They share one environment and step simultaneously each episode.

    Config additions vs MontyObjectRecognitionExperiment:
        agent_monty_configs (list[dict]): One entry per agent. Each entry must contain:
            - monty_config (dict): The same structure as the single-agent monty_config.
            - model_name_or_path (str | None): Optional checkpoint for this agent.

    The key extension point for inter-agent communication is
    ``_communicate_between_agents()``. Override it in a subclass to implement
    Layer 3 (message passing between Monty instances).
    """

    def __init__(self, config) -> None:
        # MontyExperiment.__init__ reads config["monty_config"] to resolve
        # supervised_lm_ids="all". With multiple agents we don't have a single
        # monty_config, so we resolve it from the first agent's config instead.
        if config.get("supervised_lm_ids") == "all":
            first_agent_cfg = config["agent_monty_configs"][0]
            config["supervised_lm_ids"] = list(
                first_agent_cfg["monty_config"]["learning_module_configs"].keys()
            )
        # Inject a dummy monty_config so the parent __init__ doesn't KeyError.
        # We replace self.model in setup_experiment before it is used.
        if "monty_config" not in config:
            config["monty_config"] = config["agent_monty_configs"][0]["monty_config"]
        super().__init__(config)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_experiment(self, config: dict[str, Any]) -> None:
        self.init_loggers(config["logging"])
        logger.info(config)
        self._init_all_models(config)
        self.load_environment_interfaces(config)
        self.init_monty_data_loggers(config["logging"])
        self.init_counters()

    def _init_all_models(self, config: dict[str, Any]) -> None:
        """Build one MontyBase instance per agent config."""
        self.monty_agents = []
        for agent_cfg in config["agent_monty_configs"]:
            model_path = agent_cfg.get("model_name_or_path")
            model = self.init_model(
                monty_config=agent_cfg["monty_config"],
                model_path=Path(model_path) if model_path else None,
            )
            self.monty_agents.append(model)

        # Keep self.model pointing at agent 0 so that all parent-class code that
        # references self.model (loggers, counters, etc.) keeps working.
        self.model = self.monty_agents[0]

    # ------------------------------------------------------------------
    # Experiment mode propagation
    # ------------------------------------------------------------------

    def train(self) -> None:
        logger.info(f"running {self.n_train_epochs} train epochs")
        self.experiment_mode = ExperimentMode.TRAIN
        self.logger_handler.pre_train(self.logger_args)
        for agent in self.monty_agents:
            agent.set_experiment_mode(ExperimentMode.TRAIN)
        for _ in range(self.n_train_epochs):
            self.run_epoch()
        self.logger_handler.post_train(self.logger_args)

    def evaluate(self) -> None:
        logger.info(f"running {self.n_eval_epochs} eval epochs")
        self.experiment_mode = ExperimentMode.EVAL
        self.logger_handler.pre_eval(self.logger_args)
        for agent in self.monty_agents:
            agent.set_experiment_mode(ExperimentMode.EVAL)
        for _ in range(self.n_eval_epochs):
            self.run_epoch()
        self.logger_handler.post_eval(self.logger_args)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def pre_episode(self) -> None:
        if self.experiment_mode is ExperimentMode.TRAIN:
            logger.info(
                f"running train epoch {self.train_epochs} "
                f"train episode {self.train_episodes}"
            )
        else:
            logger.info(
                f"running eval epoch {self.eval_epochs} "
                f"eval episode {self.eval_episodes}"
            )

        self.reset_episode_rng()

        for agent in self.monty_agents:
            if hasattr(self.env_interface, "semantic_id_to_label"):
                agent.pre_episode(
                    self.env_interface.primary_target,
                    self.env_interface.semantic_id_to_label,
                )
            else:
                agent.pre_episode(self.env_interface.primary_target)

        self.env_interface.pre_episode(self.rng)

        self.max_steps = (
            self.max_train_steps
            if self.experiment_mode is ExperimentMode.TRAIN
            else self.max_eval_steps
        )

        self.logger_handler.pre_episode(self.logger_args)

        if self.show_sensor_output:
            self.live_plotter.initialize_online_plotting()

    def post_episode(self, steps: int) -> None:
        self.logger_handler.post_episode(self.logger_args)
        for agent in self.monty_agents:
            agent.post_episode()

        if self.experiment_mode is ExperimentMode.TRAIN:
            self.train_episodes += 1
            self.total_train_steps += steps
        else:
            self.eval_episodes += 1
            self.total_eval_steps += steps

        self.env_interface.post_episode()

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------

    def run_episode_steps(self) -> int:
        """Step all agents together until every agent is done or a limit is hit.

        Each agent receives the full observation dict from the environment. Each
        agent's sensor modules filter to their own agent_id observations via
        sm_to_agent_dict, so they naturally only process their own sensors.

        After every cognitive step, ``_communicate_between_agents()`` is called.
        Override that method to implement inter-agent message passing (Layer 3).

        Returns:
            Total number of steps taken in this episode.
        """
        step = 0
        ctx = RuntimeContext(rng=self.rng)
        all_actions: list[Action] = []

        while True:
            observations, proprioceptive_state = self.env_interface.step(all_actions)

            # Terminal check: any agent hit its max matching steps
            for agent in self.monty_agents:
                if agent.check_reached_max_matching_steps(self.max_steps):
                    logger.info(
                        f"Terminated: agent reached max matching steps {self.max_steps}"
                    )
                    return step

            # Terminal check: absolute step budget
            if step >= self.max_total_steps:
                logger.info(f"Terminated: max episode steps {step}")
                for agent in self.monty_agents:
                    agent.deal_with_time_out()
                return step

            # Step every agent and collect their actions
            all_actions = []
            try:
                for agent in self.monty_agents:
                    if agent.is_motor_only_step:
                        actions = agent.motor_only_step(
                            ctx, observations, proprioceptive_state
                        )
                    else:
                        actions = agent.step(ctx, observations, proprioceptive_state)
                    all_actions.extend(actions)
            except StopIteration:
                # NaiveScanPolicy signals episode end via StopIteration
                for agent in self.monty_agents:
                    if hasattr(agent, "set_is_done"):
                        agent.set_is_done()
                    elif hasattr(agent, "set_done"):
                        agent.set_done()
                return step

            # Inter-agent communication — override this method for Layer 3
            self._communicate_between_agents()

            # Episode ends when every agent has converged
            if all(agent.is_done for agent in self.monty_agents):
                return step

            step += 1

    def _communicate_between_agents(self) -> None:
        """Hook for inter-agent communication (Layer 3).

        Called once per step, after all agents have stepped but before the next
        environment step. Override this in a subclass to implement message
        passing between Monty instances.

        Example pattern::

            messages = [agent.send_agent_message() for agent in self.monty_agents]
            for i, agent in enumerate(self.monty_agents):
                others = [m for j, m in enumerate(messages) if j != i]
                agent.receive_agent_message(others)
        """
        pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state_dict(self, output_dir=None) -> None:
        output_dir = output_dir if output_dir is not None else self.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)

        if not (
            self.experiment_mode is ExperimentMode.EVAL
            and self.monty_logger.use_parallel_wandb_logging
        ):
            logger.info(f"saving models to {output_dir}")
            for i, agent in enumerate(self.monty_agents):
                torch.save(agent.state_dict(), output_dir / f"model_agent_{i}.pt")
            torch.save(self.state_dict(), output_dir / "exp_state_dict.pt")
            torch.save(self.config, output_dir / "config.pt")

    def load_state_dict(self, load_dir) -> None:
        load_dir = Path(load_dir)
        exp_state_dict = torch.load(load_dir / "exp_state_dict.pt")
        config = torch.load(load_dir / "config.pt")
        for i, agent in enumerate(self.monty_agents):
            state_dict = torch.load(load_dir / f"model_agent_{i}.pt")
            agent.load_state_dict(state_dict)
        self.config = config
        for k in self.state_dict().keys():
            setattr(self, k, exp_state_dict[k])
