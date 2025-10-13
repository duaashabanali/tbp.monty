"""
Two-Agent Communication Experiment for Monty
=============================================

This file demonstrates a complete Monty experiment with two agents that 
can
communicate and coordinate their exploration of objects.

File: two_agent_monty_experiment.py
Location: Save this in your project directory (e.g., ~/tbp/tbp.monty/)

To run this experiment:
    python two_agent_monty_experiment.py
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

  # Monty framework imports
from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    MontyArgs,
    MotorSystemConfigInformedNoTrans,
    PatchAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environment_utils.transforms import (    
    DepthTo3DLocations,                                            
    MissingToMaxDepth,                                             
    )  
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.models.graph_matching import GraphLM
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatDistantPatchSM,
)
from tbp.monty.frameworks.models.motor_policies import InformedPolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem

  # Simulator imports
from tbp.monty.simulators.habitat import (
    HabitatSim,
    SingleSensorAgent,
    MultiSensorAgent,
)
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgs,
)
from tbp.monty.simulators.habitat.environment import (
    HabitatEnvironment,
    AgentConfig,
)

# Actions
from tbp.monty.frameworks.actions.actions import Action


# ========================================================================
# Communication Infrastructure
# ========================================================================
  

@dataclass
class AgentMessage:
    """Message passed between agents."""
    from_agent_id: str
    to_agent_id: str
    message_type: str
    content: Dict
    timestep: int


class MultiAgentCommunicationManager:
    """
    Manages communication between multiple Monty agents.
      
    This class is designed to integrate with Monty's architecture and can 
  be
    called during the experiment's step() or pre_episode() methods.
    """

    def __init__(self):
        self.message_queues: Dict[str, List[AgentMessage]] = {}
        self.agent_states: Dict[str, Dict] = {}
        self.shared_beliefs: Dict[str, any] = {}
        self.timestep = 0
        self.logger = logging.getLogger(__name__)

    def register_agent(self, agent_id: str):
        """Register an agent in the communication system."""
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = []
            self.agent_states[agent_id] = {}
            self.logger.info(f"Registered agent: {agent_id}")

    def send_message(
        self, 
        from_agent_id: str, 
        to_agent_id: str, 
        message_type: str, 
        content: Dict
    ):
        """Send a message from one agent to another."""
        if to_agent_id not in self.message_queues:
            self.logger.warning(f"Agent {to_agent_id} not registered")
            return

        message = AgentMessage(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            message_type=message_type,
            content=content,
            timestep=self.timestep
        )
        self.message_queues[to_agent_id].append(message)
        self.logger.debug(f"Message sent from {from_agent_id} to {to_agent_id}: {message_type}")

    def broadcast(self, from_agent_id: str, message_type: str, content: Dict):
        """Broadcast a message to all other agents."""
        for agent_id in self.message_queues.keys():
            if agent_id != from_agent_id:
                self.send_message(from_agent_id, agent_id, message_type, content)

    def get_messages(self, agent_id: str) -> List[AgentMessage]:
        """Get all messages for an agent and clear the queue."""
        messages = self.message_queues.get(agent_id, [])
        self.message_queues[agent_id] = []
        return messages

    def update_agent_state(self, agent_id: str, state: Dict):
        """Update the state information for an agent."""
        self.agent_states[agent_id] = state

    def get_agent_state(self, agent_id: str) -> Dict:
        """Get the current state of another agent."""
        return self.agent_states.get(agent_id, {})

    def share_belief(self, key: str, value: any):
        """Share a belief/hypothesis in shared memory."""
        self.shared_beliefs[key] = value

    def get_belief(self, key: str) -> any:
        """Retrieve a shared belief."""
        return self.shared_beliefs.get(key)

    def step(self):
        """Increment timestep."""
        self.timestep += 1


# ========================================================================
  
# Custom Multi-Agent Dataset Configuration
# ========================================================================

@dataclass              
class EnvInitArgsTwoAgents(EnvInitArgs):                         
    """Environment initialization args for two agents."""        
    agents: List[AgentConfig] = field(                           
        default_factory=lambda: [                                
            # Agent 1 configuration                              
            AgentConfig(                                         
                SingleSensorAgent,                               
                dict(                                            
                    agent_id="agent_id_0",                       
                    sensor_id="sensor_0",                        
                    agent_position=(0.0, 1.5, 0.0),              
                    resolution=(64, 64),                         
                    semantic=False,                              
                    rotation_step=5.0,                           
                    translation_step=0.1,                                              
                    action_space_type="distant_agent",           
                  ),                                               
              ),                                                   
              # Agent 2 configuration                              
              AgentConfig(                                         
                  SingleSensorAgent,                               
                  dict(                                            
                      agent_id="agent_id_1",                       
                      sensor_id="sensor_1",                        
                      agent_position=(0.5, 1.5, 0.5),              
                      resolution=(64, 64),                         
                      semantic=False,                              
                      rotation_step=5.0,                           
                      translation_step=0.1,                        
                      action_space_type="distant_agent",           
                ),                                               
            ),                                                   
        ]                                                        
    )                                                            
                                                                
                                                            
@dataclass                                                
class TwoAgentHabitatDatasetArgs: 
    env_init_func: field = field(default=HabitatEnvironment)     
    env_init_args: Dict = field(                                 
        default_factory=lambda: EnvInitArgsTwoAgents().__dict__  
    )                                                            
    transform: Optional[List] = None
    def __post_init__(self):                              
        """Set up transforms for both agents to process de
 images."""                                                 
        # Agent 0 transforms                              
        self.transform = [                                             
            MissingToMaxDepth(agent_id="agent_id_0",      
 max_depth=1),                                              
            DepthTo3DLocations(                           
                agent_id="agent_id_0",                    
                sensor_ids=["sensor_0"],                  
                resolutions=[(64, 64)],                   
                world_coord=True,                         
                zooms=[1.0],                              
                get_all_points=True,                      
                use_semantic_sensor=False,                
            ),                                            
            # Agent 1 transforms                          
            MissingToMaxDepth(agent_id="agent_id_1",      
 max_depth=1),                                              
            DepthTo3DLocations(                           
                agent_id="agent_id_1",                    
                sensor_ids=["sensor_1"],                  
                resolutions=[(64, 64)],                   
                world_coord=True,                         
                zooms=[1.0],                              
                get_all_points=True,                      
                use_semantic_sensor=False,                
            ),                                            
        ] 

# ========================================================================
# Custom Experiment with Multi-Agent Communication
# ========================================================================


class TwoAgentCommunicationExperiment(MontyObjectRecognitionExperiment):
    """
    Custom Monty experiment with two agents that can communicate.
      
    This experiment extends the standard MontyObjectRecognitionExperiment
    to add communication capabilities between agents.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comm_manager = MultiAgentCommunicationManager()

        # Register agents with communication manager
        for agent_id in self.get_agent_ids():
            self.comm_manager.register_agent(agent_id)

    def get_agent_ids(self) -> List[str]:
        """Get list of agent IDs from the environment."""
        # This extracts agent IDs from the sm_to_agent_dict
        if hasattr(self, 'model') and hasattr(self.model, 'sm_to_agent_dict'):
            return list(set(self.model.sm_to_agent_dict.values()))
        return ["agent_id_0", "agent_id_1"]  # Default for two agents

    def pre_episode(self):
        """Called before each episode starts."""
        super().pre_episode()
        # Reset communication for new episode
        self.comm_manager.timestep = 0
        self.logger.info("Starting new episode with multi-agent communication")

    def process_agent_observations(self, observations: Dict, states: Dict):
        """
        Process observations from all agents and facilitate communication.
          
        Args:
            observations: Dictionary of observations keyed by agent_id
            states: Dictionary of agent states keyed by agent_id
        """
        # Update agent states in communication manager
        for agent_id in self.get_agent_ids():
            if agent_id in states:
                self.comm_manager.update_agent_state(agent_id, {
                    'position': states[agent_id].get('position'),
                    'rotation': states[agent_id].get('rotation'),
                })

        # Example: Share observations between agents
        # In a real scenario, you'd process learning module hypotheses here
        for agent_id in self.get_agent_ids():
            if agent_id in observations:
                # Broadcast that this agent has new observations
                self.comm_manager.broadcast(
                    from_agent_id=agent_id,
                    message_type="observation_update",
                    content={
                        "has_observations": True,
                        "timestep": self.comm_manager.timestep
                    }
                )

    def coordinate_agents(self):
        """
        Coordinate actions between agents.
          
        This method can be called to implement coordinated exploration
        strategies between agents.
        """
        agent_ids = self.get_agent_ids()

        # Example: Check if agents should share hypotheses
        for agent_id in agent_ids:
            messages = self.comm_manager.get_messages(agent_id)

            for msg in messages:
                if msg.message_type == "observation_update":
                    self.logger.debug(
                        f"Agent {agent_id} received observation update from {msg.from_agent_id}"
                      )
                elif msg.message_type == "hypothesis":
                    self.logger.info(
                          f"Agent {agent_id} received hypothesis: {msg.content.get('object_name')}"
                          )

    def post_episode(self):
        """Called after each episode ends."""
        super().post_episode()

        # Log communication statistics
        self.logger.info(f"Episode completed after {self.comm_manager.timestep} timesteps")
        self.logger.info(f"Shared beliefs: {self.comm_manager.shared_beliefs}")


  # ========================================================================
  # Multi-Agent Monty Configuration
  # ========================================================================


@dataclass
class TwoAgentMontyConfig(PatchAndViewMontyConfig):
    """
    Monty configuration for two agents.
      
    This configuration sets up two learning modules, two sensor modules,
    and maps them to two different agents.
    """

    learning_module_configs: Dict = field(
        default_factory=lambda: dict(
              learning_module_0=dict(
                learning_module_class=GraphLM,
                learning_module_args=dict(
                #     k=5,
                #     match_attribute="displacement",
                ),
            ),
            learning_module_1=dict(
                learning_module_class=GraphLM,
                learning_module_args=dict(
                    # k=5,
                    # match_attribute="displacement",
                ),
            ),
        )
    )

    sensor_module_configs: Dict = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_0",
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "object_coverage",
                        "hsv",
                        "principal_curvatures_log",
                    ],
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_1",
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "object_coverage",
                        "hsv",
                        "principal_curvatures_log",
                    ],
                    save_raw_obs=False,
                ),
            ),
            sensor_module_2=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                    features=["pose_vectors", "pose_fully_defined", "on_object"],
                ),
            ),
        )
    )

    motor_system_config: Dict = field(
        default_factory=MotorSystemConfigInformedNoTrans
    )

    # Map sensor modules to agents
    sm_to_agent_dict: Dict = field(
        default_factory=lambda: dict(
            patch_0="agent_id_0",  # First sensor on first agent
            patch_1="agent_id_1",  # Second sensor on second agent
            view_finder="agent_id_0",  # View finder on first agent
        )
    )

    # Map sensor modules to learning modules
    sm_to_lm_matrix: List = field(
        default_factory=lambda: [
            [0],  # patch_0 -> learning_module_0
            [1],  # patch_1 -> learning_module_1
            # view_finder not connected to any LM
        ]
    )

    # No hierarchical connections between LMs
    lm_to_lm_matrix: Optional[List] = None

    # LMs can vote/communicate with each other
    lm_to_lm_vote_matrix: List = field(
        default_factory=lambda: [
            [1],  # LM 0 votes with LM 1
            [0],  # LM 1 votes with LM 0
        ]
    )

    monty_args: MontyArgs = field(
        default_factory=lambda: MontyArgs(
            num_exploratory_steps=500,
            min_eval_steps=10,
            max_total_steps=1000,
        )
    )


# ========================================================================
  
# Experiment Configurations
# ========================================================================
  
# Directory for outputs
project_dir = os.path.expanduser("~/tbp/results/monty/projects")
model_name = "two_agent_communication"

# Objects to use (using primitive Habitat objects that don't require YCB dataset)
object_names = ["cubeSolid", "cylinderSolid"]

# Simple rotations for testing
test_rotations = [
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 45.0, 0.0]),
    np.array([0.0, 90.0, 0.0]),
]

# Configuration for two-agent communication experiment
two_agent_monty_experiment = dict(
    experiment_class=TwoAgentCommunicationExperiment,

    experiment_args=ExperimentArgs(
        do_eval=True,
        n_train_epochs=2,
        n_eval_epochs=len(test_rotations),
        max_train_steps=100,
        max_eval_steps=100,
    ),

    logging_config=LoggingConfig(
        output_dir=project_dir,
        run_name=model_name,
        python_log_level="INFO",
        monty_log_level="BASIC",
        wandb_handlers=[],  # Disable wandb for simplicity
    ),

    monty_config=TwoAgentMontyConfig(),

    # Use custom dataset with two agents
    dataset_args=TwoAgentHabitatDatasetArgs(),

    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations[:1]),
      ),

    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,

object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)


# ========================================================================
# Simplified Test Script
# ========================================================================

def run_simple_two_agent_test():
    """
    Simple test to verify two agents can operate in the same Habitat environment.
    This doesn't require the full Monty framework.
    """
    print("="*70)
    print("Simple Two-Agent Test (Without Full Monty Framework)")
    print("="*70)

    # Create two agents
    agent_1 = SingleSensorAgent(
        agent_id="agent_0",
        sensor_id="sensor_0",
        agent_position=(0.0, 1.5, 0.0),
        resolution=(64, 64),
        semantic=True,
        rotation_step=5.0,
        translation_step=0.1,
    )

    agent_2 = SingleSensorAgent(
        agent_id="agent_1",
        sensor_id="sensor_1",
        agent_position=(1.0, 1.5, 0.0),
        resolution=(64, 64),
        semantic=True,
        rotation_step=5.0,
        translation_step=0.1,
    )

    print("\nCreated two agents:")
    print(f"  - {agent_1.agent_id} at position {agent_1.position}")
    print(f"  - {agent_2.agent_id} at position {agent_2.position}")

    # Create communication manager
    comm_manager = MultiAgentCommunicationManager()
    comm_manager.register_agent(agent_1.agent_id)
    comm_manager.register_agent(agent_2.agent_id)

    # Create environment with both agents
    print("\nInitializing Habitat environment...")
    with HabitatSim(agents=[agent_1, agent_2]) as sim:
        # Add objects
        print("\nAdding objects to environment...")
        cube = sim.add_object(
            name="cubeSolid",
            position=(0.5, 1.5, -0.5),
            semantic_id=1
        )
        cylinder = sim.add_object(
            name="cylinderSolid",
            position=(-0.5, 1.5, -0.7),
            semantic_id=2
        )
        print(f"  - Cube (semantic_id={cube.semantic_id})")
        print(f"  - Cylinder (semantic_id={cylinder.semantic_id})")

        # Get initial observations
        print("\nGetting initial observations...")
        obs = sim.observations
        states = sim.states

        print(f"  Agent 0 position: {states['agent_0']['position']}")
        print(f"  Agent 1 position: {states['agent_1']['position']}")

        # Simulate communication
        print("\nSimulating agent communication...")

        # Agent 0 detects something and broadcasts
        comm_manager.broadcast(
            from_agent_id="agent_0",
            message_type="detection",
            content={"object": "cube", "confidence": 0.95}
        )
        print("  [agent_0] Broadcast: Detected cube with 95% confidence")

        # Agent 1 receives and responds
        messages = comm_manager.get_messages("agent_1")
        for msg in messages:
            print(f"  [agent_1] Received: {msg.message_type} from {msg.from_agent_id}")
            print(f"           Content: {msg.content}")

        # Agent 1 sends targeted message to Agent 0
        comm_manager.send_message(
            from_agent_id="agent_1",
            to_agent_id="agent_0",
            message_type="confirmation",
            content={"confirmed": True, "additional_info": "Also see cylinder nearby"}
        )
        print("  [agent_1] Sent confirmation to agent_0")

        # Agent 0 receives confirmation
        messages = comm_manager.get_messages("agent_0")
        for msg in messages:
            print(f"  [agent_0] Received: {msg.message_type} from {msg.from_agent_id}")
            print(f"           Content: {msg.content}")

        print("\n" + "="*70)
        print("Test completed successfully!")
        print("="*70)


def run_full_monty_experiment():
    """
    Run the full Monty experiment with two communicating agents.
      
    Note: This requires the full Monty framework to be properly set up.
    """
    print("\n" + "="*70)
    print("Running Full Two-Agent Monty Experiment")
    print("="*70)

    from tbp.monty.frameworks.run import run_experiments

    # Run the experiment
    run_experiments(
        experiments=[two_agent_comm_experiment],
        project_name="two_agent_communication_test"
    )


# ========================================================================
# Main Entry Point
# ========================================================================
  

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Run full Monty experiment
        run_full_monty_experiment()
    else:
        # Run simple test
        run_simple_two_agent_test()

        print("\n" + "="*70)
        print("To run the full Monty experiment, use:")
        print("  python two_agent_monty_experiment.py --full")
        print("="*70)
