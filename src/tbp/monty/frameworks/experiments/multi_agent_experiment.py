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
import copy
import numpy as np
from pathlib import Path
from typing import Any

import wandb
import torch

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.loggers.exp_logger import LoggingCallbackHandler
from tbp.monty.frameworks.loggers.multi_agent_loggers import (
    MultiAgentBasicGraphMatchingLogger,
    MultiAgentDetailedGraphMatchingLogger,
)
from tbp.monty.frameworks.utils.logging_utils import get_stats_per_lm
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_orthonormal_vectors,
)

# Multi-agent-aware replacements for the loggers in
# MontyForGraphMatching.LOGGING_REGISTRY. Only BASIC and DETAILED are
# covered -- SILENT logs nothing, and SELECTIVE is a niche
# object-similarity-analysis logger not used by any multi-agent experiment.
_MULTI_AGENT_LOGGER_REGISTRY = {
    "BASIC": MultiAgentBasicGraphMatchingLogger,
    "DETAILED": MultiAgentDetailedGraphMatchingLogger,
}
def _merge_vote_into(
    merged: dict[str, list[Any]],
    sender_vote: dict[str, Any],
    receiver_pose: np.ndarray,
) -> None:
    """Transform one sender's vote into the receiver's reference frame.

    Mirrors MontyForEvidenceGraphMatching._combine_votes()'s per-(sender,
    receiver) transform exactly, merging the result into `merged` (a
    dict[graph_id, list[Message]]) in place. Works unmodified across
    different agents (not just LMs on one body) because each agent's
    DepthTo3DLocations transform already bakes that agent's own absolute
    world pose into every sensed location/orientation before Monty ever
    sees it -- so two agents' `sensed_pose_rel_body` values are already in
    the same shared world frame.

    Args:
        merged: Accumulator of already-transformed votes for one receiving
            learning module, keyed by graph_id. Updated in place.
        sender_vote: One learning module's send_out_vote() output.
        receiver_pose: The receiving learning module's own
            `sensed_pose_rel_body`, shape (4, 3).
    """
    sender_pose = sender_vote["sensed_pose_rel_body"]
    sensor_disp = np.array(receiver_pose[0]) - np.array(sender_pose[0])
    sensor_rotation_disp, _ = align_orthonormal_vectors(
        sender_pose[1:],
        receiver_pose[1:],
        as_scipy=False,
    )
    for graph_id, states in sender_vote["possible_states"].items():
        transformed = []
        for state in states:
            new_state = copy.deepcopy(state)
            rotated_displacement = new_state.get_pose_vectors().dot(sensor_disp)
            new_state.transform_morphological_features(
                translation=rotated_displacement,
                rotation=sensor_rotation_disp,
            )
            transformed.append(new_state)
        merged.setdefault(graph_id, []).extend(transformed)
        
__all__ = ["MultiAgentMontyExperiment"]

logger = logging.getLogger(__name__)


class MultiAgentMontyExperiment(MontyObjectRecognitionExperiment):
    """Runs multiple distinct Monty agents together in a shared environment.

    Each agent has its own sensor modules, learning modules, and motor system.
    They share one environment and step simultaneously each episode.

    Config additions vs MontyObjectRecognitionExperiment:
        agent_monty_configs (dict[str, dict]): One entry per agent, keyed by an
            arbitrary agent name (e.g. "agent0"). Each entry must contain:
            - monty_config (dict): The same structure as the single-agent monty_config.
            - model_name_or_path (str | None): Optional checkpoint for this agent.

        cross_agent_vote_matrix (list[dict] | None): Optional. One entry per
            sender-to-receiver cross-agent vote link:
                - sender: {agent: <index into agent_monty_configs>,
                           lm: <index into that agent's learning_modules>}
                - receiver: same shape as sender.
            Links are directional -- for bidirectional voting between two
            LMs, list both directions explicitly. `None` or an empty list
            (the default) disables cross-agent voting, same convention as
            `lm_to_lm_vote_matrix`: no entry means no voting.

        share_final_vote (bool): Optional, default False. If True, each
            agent runs to its own independent conclusion (no per-step
            cross-agent influence -- orthogonal to, and can be used
            instead of or alongside, `cross_agent_vote_matrix`), and the
            moment each agent finishes, its final decision is recorded:
            whichever of its own learning modules currently has the
            highest evidence for its most-likely hypothesis, taken as that
            agent's {object, confidence}. Once every agent has finished,
            the joint decision is the shared object if all agents agree,
            or the object of whichever agent had the higher confidence if
            they disagree. See `_share_final_vote()`.

    Cross-agent voting (Layer 3) is implemented in
    ``_communicate_between_agents()`` below, reusing the same
    sender/receiver transform MontyForEvidenceGraphMatching._combine_votes()
    already implements within one agent. Voting only happens between LMs
    whose owning agent is currently in a "matching_step" -- exploratory
    steps are skipped, exactly as MontyForGraphMatching._vote() only runs
    from _matching_step(), never _exploratory_step(). This matters once
    agents can be mid-training and on different step schedules from each
    other, or have multiple LMs each.
    """

    def __init__(self, config) -> None:
        # MontyExperiment.__init__ reads config["monty_config"] to resolve
        # supervised_lm_ids="all". With multiple agents we don't have a single
        # monty_config, so we resolve it from the first agent's config instead.
        if config.get("supervised_lm_ids") == "all":
            first_agent_cfg = next(iter(config["agent_monty_configs"].values()))
            config["supervised_lm_ids"] = list(
                first_agent_cfg["monty_config"]["learning_module_configs"].keys()
            )
        # Inject a dummy monty_config so the parent __init__ doesn't KeyError.
        # We replace self.model in setup_experiment before it is used.
        if "monty_config" not in config:
            config["monty_config"] = next(
                iter(config["agent_monty_configs"].values())
            )["monty_config"]
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
        for agent_cfg in config["agent_monty_configs"].values():
            model_path = agent_cfg.get("model_name_or_path")
            model = self.init_model(
                monty_config=agent_cfg["monty_config"],
                model_path=Path(model_path) if model_path else None,
            )
            self.monty_agents.append(model)

        # Keep self.model pointing at agent 0 so that all parent-class code that
        # references self.model (loggers, counters, etc.) keeps working.
        self.model = self.monty_agents[0]

    def init_monty_data_loggers(self, config: dict[str, Any]) -> None:
        """Build data loggers, then swap in multi-agent-aware equivalents.

        The parent implementation only knows about `self.model` (agent 0), so
        after it builds the normal single-agent logger + handlers, we replace
        the logger with one that gathers stats from every agent in
        `self.monty_agents`, reusing the same handler instances (e.g.
        MultiAgentCSVStatsHandler, DetailedJSONHandler) the parent already
        constructed from the config.
        """
        super().init_monty_data_loggers(config)

        multi_agent_logger_class = _MULTI_AGENT_LOGGER_REGISTRY.get(
            self.monty_log_level
        )
        if multi_agent_logger_class is not None:
            self.monty_logger = multi_agent_logger_class(
                handlers=self.monty_logger.handlers, monty_agents=self.monty_agents
            )
            self.logger_handler = LoggingCallbackHandler(
                self.monty_logger, self.model, output_dir=self.output_dir
            )

        # The parent implementation only flags self.model's (agent 0's)
        # learning modules as detailed-logged; do the same for every other
        # agent so their buffers behave consistently under DETAILED logging.
        has_detailed_logger = self.monty_log_level == "DETAILED"
        for agent in self.monty_agents[1:]:
            for lm in agent.learning_modules:
                lm.has_detailed_logger = has_detailed_logger

    # ------------------------------------------------------------------
    # Experiment mode propagation
    # ------------------------------------------------------------------

    def train(self) -> None:
        # super().train() sets agent 0 (== self.model)'s mode via the base
        # class; only the extra agents need to be set here.
        for agent in self.monty_agents[1:]:
            agent.set_experiment_mode(ExperimentMode.TRAIN)
        super().train()

    def evaluate(self) -> None:
        # super().evaluate() sets agent 0 (== self.model)'s mode via the base
        # class; only the extra agents need to be set here.
        for agent in self.monty_agents[1:]:
            agent.set_experiment_mode(ExperimentMode.EVAL)
        super().evaluate()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def pre_episode(self) -> None:
        # super().pre_episode() handles agent 0 (== self.model): RNG reset,
        # env/logger setup, and its own pre_episode(target, ...) call. Only
        # the extra agents need to be stepped through pre_episode here.
        super().pre_episode()

        # Reset final-vote tracking (see _share_final_vote()) for the new
        # episode, regardless of whether share_final_vote is enabled.
        self._final_votes: dict[int, dict[str, Any]] = {}
        self._joint_decision_logged = False

        target = self.env_interface.primary_target
        if hasattr(self.env_interface, "semantic_id_to_label"):
            for agent in self.monty_agents[1:]:
                agent.pre_episode(target, self.env_interface.semantic_id_to_label)
        else:
            for agent in self.monty_agents[1:]:
                agent.pre_episode(target)

    def post_episode(self, steps: int) -> None:
        # Buffer extra-agent metrics before the main logger commits so they land
        # on the same WandB step as agent 0's data.
        self._log_extra_agents_buffered()

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

    def _log_extra_agents_buffered(self) -> None:
        """Buffer per-episode metrics for agents beyond agent 0 into WandB.

        Called BEFORE the main logger commits so that extra-agent data lands on
        the same WandB step as agent 0's LM_0 metrics. Uses commit=False so the
        framework's final wandb.log({}) commit flushes everything together.
        """
        if not hasattr(self, "monty_agents") or len(self.monty_agents) <= 1:
            return
        target = getattr(self.env_interface, "primary_target", None)
        if target is None:
            return
        mode = self.experiment_mode
        episode = self.eval_episodes if mode is ExperimentMode.EVAL else self.train_episodes
        seed = self._rng_seed_history[-1] if self._rng_seed_history else self.config["seed"]
        metrics = {}
        for agent_idx in range(1, len(self.monty_agents)):
            agent = self.monty_agents[agent_idx]
            try:
                stats = get_stats_per_lm(agent, target, seed)
            except Exception:
                continue
            prefix = f"agent_{agent_idx}"
            for lm_key, lm_stats in stats.items():
                if not lm_key.startswith("LM_"):
                    continue
                metrics[f"{prefix}/{lm_key}/episode/steps_to_individual_ts"] = (
                    lm_stats.get("individual_ts_reached_at_step")
                )
                metrics[f"{prefix}/{lm_key}/episode/individual_ts_rotation_error"] = (
                    lm_stats.get("individual_ts_rotation_error")
                )
                metrics[f"{prefix}/{lm_key}/episode/avg_prediction_error"] = (
                    lm_stats.get("episode_avg_prediction_error")
                )
                metrics[f"{prefix}/{lm_key}/episode/primary_performance"] = (
                    lm_stats.get("primary_performance")
                )
        if metrics and wandb.run is not None:
            wandb.log(metrics, step=episode, commit=False)

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

            # Inter-agent communication (Layer 3) — no-op unless
            # cross_agent_vote_matrix is set in config.
            self._communicate_between_agents()

            # Episode ends when every agent has converged
            if all(agent.is_done for agent in self.monty_agents):
                return step

            step += 1

    def _communicate_between_agents(self) -> None:
        """Cross-agent voting (Layer 3), and/or final-vote reconciliation.

        Called once per step, after all agents have stepped but before the
        next environment step. Each piece no-ops unless its own config key
        is set, and the two are independent of each other.
        """
        self._vote_cross_agent()
        if self.config.get("share_final_vote"):
            self._share_final_vote()

    def _vote_cross_agent(self) -> None:
        """Cross-agent voting (Layer 3).

        No-ops if `cross_agent_vote_matrix` isn't set.
        """
        vote_links = self.config.get("cross_agent_vote_matrix")
        if not vote_links:
            return

        participants = set()
        for link in vote_links:
            participants.add((link["sender"]["agent"], link["sender"]["lm"]))
            participants.add((link["receiver"]["agent"], link["receiver"]["lm"]))

        # Ask every participating LM for its outgoing vote once. An LM's own
        # vote payload doubles as its current sensed pose, used below as the
        # reference frame for any votes IT receives -- so an LM that isn't
        # in a matching step, or has nothing new to report this step,
        # participates as neither a sender nor a receiver.
        votes: dict[tuple[int, int], dict[str, Any] | None] = {}
        for agent_idx, lm_idx in participants:
            agent = self.monty_agents[agent_idx]
            if agent.step_type != "matching_step":
                votes[(agent_idx, lm_idx)] = None
                continue
            votes[(agent_idx, lm_idx)] = agent.learning_modules[
                lm_idx
            ].send_out_vote()

        combined_by_receiver: dict[tuple[int, int], dict[str, list[Any]]] = {}
        for link in vote_links:
            sender = (link["sender"]["agent"], link["sender"]["lm"])
            receiver = (link["receiver"]["agent"], link["receiver"]["lm"])
            sender_vote = votes[sender]
            receiver_vote = votes[receiver]
            if sender_vote is None or receiver_vote is None:
                continue
            merged = combined_by_receiver.setdefault(receiver, {})
            _merge_vote_into(
                merged, sender_vote, receiver_vote["sensed_pose_rel_body"]
            )

        for (agent_idx, lm_idx), combined in combined_by_receiver.items():
            agent = self.monty_agents[agent_idx]
            lm = agent.learning_modules[lm_idx]
            lm.receive_votes(combined)
            agent.update_stats_after_vote(lm)

    def _share_final_vote(self) -> None:
        """Record each agent's final decision the moment it finishes.

        Reconciles into one joint decision once every agent has finished.
        Unlike `_vote_cross_agent()`, agents get zero cross-agent influence
        while they're still running -- each one's learning modules only
        ever vote with each other (intra-agent, via that agent's own
        `lm_to_lm_vote_matrix`). Only once an agent is fully done (`is_done`)
        is its result captured, using whichever of its own learning modules
        currently has the highest evidence for its most-likely hypothesis
        as that agent's {object, confidence} -- there's no existing
        Monty-level notion of "this agent's overall confidence" to reuse
        (`min_lms_match` is a pure count of independently-converged LMs, not
        an evidence comparison), so this picks the single strongest signal
        among the agent's LMs rather than e.g. averaging across them.

        Once every agent has a recorded vote: if they agree, that object is
        the joint decision; if they disagree, the higher-confidence agent's
        object wins. Stored on `self.joint_decision` and logged once per
        episode.
        """
        for agent_idx, agent in enumerate(self.monty_agents):
            if agent_idx in self._final_votes or not agent.is_done:
                continue
            best_object = None
            best_confidence = None
            for lm in agent.learning_modules:
                mlh = lm.get_current_mlh()
                if best_confidence is None or mlh["evidence"] > best_confidence:
                    best_confidence = mlh["evidence"]
                    best_object = mlh["graph_id"]
            self._final_votes[agent_idx] = {
                "object": best_object,
                "confidence": best_confidence,
            }
            logger.info(
                f"Agent {agent_idx} final vote: {best_object!r} "
                f"(confidence {best_confidence:.3f})"
            )

        if self._joint_decision_logged or len(self._final_votes) < len(
            self.monty_agents
        ):
            return

        votes = list(self._final_votes.values())
        objects = {v["object"] for v in votes}
        agreed = len(objects) == 1
        self.joint_decision = votes[0] if agreed else max(
            votes, key=lambda v: v["confidence"]
        )
        logger.info(
            f"Joint decision: {self.joint_decision['object']!r} "
            f"(agreed={agreed}, votes={self._final_votes})"
        )
        self._joint_decision_logged = True

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
