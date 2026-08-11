# Multi-Agent Monty Implementation

This document describes the implementation of multiple distinct Monty agents sharing a single Habitat environment. The goal is to support Theory of Mind research by allowing agents to perceive the same scene independently and eventually communicate with each other.

---

## Architecture Overview

The implementation is organized into three layers:

| Layer | What it does | Status |
|-------|-------------|--------|
| **Layer 1** — Environment | Multiple physical agents in one Habitat sim | Done |
| **Layer 2** — Experiment orchestration | One `MontyBase` instance per agent, stepped together | Done |
| **Layer 3** — Inter-agent communication | Message passing between Monty instances | Stub (ready to implement) |

Each agent has its own:

- Sensor modules (CameraSM + Probe)
- Learning module (EvidenceGraphLM)
- Motor system and policy
- Pretrained object model checkpoint

They share one environment and step simultaneously each episode. An episode ends when **all** agents have converged (or the step budget is exhausted).

---

## Files Changed

### Created
| File | Purpose |
|------|---------|
| `src/tbp/monty/frameworks/experiments/multi_agent_experiment.py` | Layer 2 orchestration class |
| `src/tbp/monty/conf/environment/habitat_ycb_two_agents.yaml` | Two-agent Habitat environment config |
| `src/tbp/monty/conf/experiment/two_agent_surf.yaml` | Full self-contained experiment config |

### Modified
| File | Change |
|------|--------|
| `src/tbp/monty/simulators/habitat/environment.py` | Changed `agents` parameter type to `list[dict \| AgentConfig]` |
| `src/tbp/monty/frameworks/experiments/__init__.py` | Removed import of `MultiAgentMontyExperiment` to fix circular import |

---

## Layer 1 — Environment

**File:** [`src/tbp/monty/simulators/habitat/environment.py`](src/tbp/monty/simulators/habitat/environment.py)

`HabitatSim` already natively supported multiple agents. The only change needed was in `HabitatEnvironment.__init__` — its type annotation constrained `agents` to a single config instead of a list.

**Change:**
```python
# Before
agents: dict | AgentConfig

# After
agents: list[dict | AgentConfig]
```

The loop that builds agent objects from the config list was already in place (the single-agent wrapping line was already commented out).

**Key fact about sensor naming:** Habitat scopes sensors per physical agent internally. Two agents can both have a sensor named `patch` without collision — each agent gets its own `AgentConfiguration` in the sim. This matters because pretrained model checkpoints store graph features under the sensor name (e.g., `patch`), so sensor IDs must match what the model was trained with.

---

## Layer 2 — Experiment Orchestration

**File:** [`src/tbp/monty/frameworks/experiments/multi_agent_experiment.py`](src/tbp/monty/frameworks/experiments/multi_agent_experiment.py)

`MultiAgentMontyExperiment` subclasses `MontyObjectRecognitionExperiment`. It replaces the single `self.model` with `self.monty_agents: list[MontyBase]` while keeping `self.model = self.monty_agents[0]` for backwards compatibility with all parent-class logger and counter code.

### Config structure

Instead of a single `monty_config`, the experiment expects:

```yaml
agent_monty_configs:
  - model_name_or_path: /path/to/pretrained/
    monty_config:
      monty_class: ...
      monty_args: ...
      sm_to_agent_dict:
        patch: agent_id_0        # must match this agent's agent_id
        view_finder: agent_id_0
      sensor_module_configs: ...
      learning_module_configs: ...
      motor_system_config: ...

  - model_name_or_path: /path/to/pretrained/
    monty_config:
      ...
      sm_to_agent_dict:
        patch: agent_id_1        # must match this agent's agent_id
        view_finder: agent_id_1
      motor_system_config:
        ...
        agent_id: agent_id_1     # motor policy also needs its agent_id
```

`sm_to_agent_dict` determines which observations each Monty instance reads from the shared environment observation dict. Each agent only processes sensors belonging to its own `agent_id`.

### Episode step loop

```
while True:
    observations, proprioceptive_state = env.step(all_actions)

    # terminal checks (max steps, step budget)

    all_actions = []
    for agent in monty_agents:
        actions = agent.step(...)
        all_actions.extend(actions)

    _communicate_between_agents()   # Layer 3 hook

    if all(agent.is_done for agent in monty_agents):
        break
```

### Checkpointing

Each agent is saved separately:
```
output_dir/
  model_agent_0.pt
  model_agent_1.pt
  exp_state_dict.pt
  config.pt
```

---

## Layer 3 — Inter-Agent Communication (Stub)

**Method:** `MultiAgentMontyExperiment._communicate_between_agents()`

Called once per step, after all agents have stepped but before the next environment step. Currently a no-op. Override in a subclass to implement message passing.

```python
def _communicate_between_agents(self) -> None:
    pass
```

**Example pattern for a subclass:**
```python
def _communicate_between_agents(self) -> None:
    messages = [agent.send_agent_message() for agent in self.monty_agents]
    for i, agent in enumerate(self.monty_agents):
        others = [m for j, m in enumerate(messages) if j != i]
        agent.receive_agent_message(others)
```

What agents could communicate:
- Current best hypothesis (object ID + pose)
- Evidence scores / uncertainty
- Goal state (where the agent is headed)
- Attention signals (which part of the object is most informative)

---

## Config Files

### Environment config
**File:** [`src/tbp/monty/conf/environment/habitat_ycb_two_agents.yaml`](src/tbp/monty/conf/environment/habitat_ycb_two_agents.yaml)

Defines two `MultiSensorAgent` instances:
- `agent_id_0` at position `[0.0, 1.5, 0.1]`
- `agent_id_1` at position `[0.3, 1.5, 0.1]` (offset 30 cm to the right)

Both have identical sensor setups: `patch` (zoom 10×) and `view_finder` (zoom 1×), resolution 64×64.

### Experiment config
**File:** [`src/tbp/monty/conf/experiment/two_agent_surf.yaml`](src/tbp/monty/conf/experiment/two_agent_surf.yaml)

Self-contained YAML (cannot reuse single-agent sub-configs because they hardcode single-agent sensor names). Key settings:

```yaml
experiment:
  _target_: tbp.monty.frameworks.experiments.multi_agent_experiment.MultiAgentMontyExperiment
  config:
    do_train: false
    do_eval: true
    n_eval_epochs: 10
    max_eval_steps: 500
    max_total_steps: 5000
```

Both agents point to the same pretrained checkpoint:
```
${constants.pretrained_dir}/surf_agent_1lm_10distinctobj/pretrained/
```

---

## Running the Experiment

```bash
cd /Users/duaaali/tbp/tbp.monty
python run.py experiment=two_agent_surf
```

> **Note:** Use `experiment=` without a leading `+`. The main `experiment.yaml` already has `experiment: ??` as a required placeholder — using `+experiment=` would try to add a second value and fail with "Multiple values for experiment".

### To see live sensor output
In `two_agent_surf.yaml`, set:
```yaml
show_sensor_output: true
```
A matplotlib window will show what each agent's `patch` sensor sees in real time.

---

## Outputs

### WandB dashboard
Each run syncs automatically to WandB. Project: `duaaali-monty/Monty`.

Key metrics logged per episode:
| Metric | Meaning |
|--------|---------|
| `episode/correct` | Agent recognized the object correctly |
| `episode/confused` | Agent could not disambiguate between objects |
| `LM_0/episode/steps_to_individual_ts` | Steps taken to reach recognition threshold |
| `LM_0/episode/avg_prediction_error` | Mean pose prediction error |
| `episode/goal_state_success_rate` | How often the motor system reached its goal |
| `overall/percent_correct` | Accuracy across all episodes in the epoch |

### CSV output
```
/Users/duaaali/tbp/results/monty/projects/monty_runs/two_agent_surf/eval_stats.csv
```

The logging system keeps only two CSVs at a time (`eval_stats.csv` = current run, `eval_stats_old.csv` = previous run). Use WandB for permanent per-run storage.

### Loading the CSV
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "/Users/duaaali/tbp/results/monty/projects/monty_runs/two_agent_surf/eval_stats.csv"
)
print(df.columns.tolist())

df["num_steps"].plot(title="Steps to recognition per episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.tight_layout()
plt.savefig("steps_per_episode.png")
plt.show()
```

---

## Known Issues / Gotchas

1. **Sensor IDs must match pretrained checkpoint keys.** The `EvidenceGraphLM` stores graph features under sensor names (e.g., `patch`). If you rename sensors in the config (e.g., to `patch_0`), loading the pretrained checkpoint will fail with `ConfigKeyError: Missing key patch`. Both agents can safely share the sensor name `patch` because Habitat scopes sensors per physical agent.

2. **Circular import if `MultiAgentMontyExperiment` is exported from `__init__.py`.** The import chain `environment → transforms → abstract_monty_classes → experiments/__init__ → multi_agent_experiment → object_recognition_experiments → environment` causes a partially-initialized module error. Solution: do not import `MultiAgentMontyExperiment` in `experiments/__init__.py`. The YAML config references the full module path directly via `_target_`, so the convenience import is not needed.

3. **Only agent 0's metrics are logged** by the parent-class logger (which reads `self.model`). Agent 1's per-step internals are not captured in WandB. To fix this properly, the logger would need to be extended to iterate over all agents.

---

## Next Steps

- **Implement Layer 3** — define what agents share (hypotheses, uncertainty, goals) and how the receiving agent incorporates that into its evidence update.
- **Extend logging** to capture per-agent metrics for all agents, not just agent 0.
- **Experiment with asymmetric agents** — different pretrained models, different sensor configurations, or one agent acting as a "teacher" and one as a "learner".
