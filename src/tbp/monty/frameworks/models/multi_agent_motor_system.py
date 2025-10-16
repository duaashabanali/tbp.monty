class MultiAgentMotorSystem:
    """
    Motor system that coordinates multiple independent agents.
      
    Each agent has its own motor policy and can move independently
    based on its local observations and global coordination signals.
    """

    def __init__(self, agent_configs: Dict[str, Dict]):
        """
        Args:
            agent_configs: Dict mapping agent_id to motor system config
                        e.g., {
                            "agent_id_0": {
                                "policy_class": InformedPolicy,
                                "policy_args": {...}
                            },
                            "agent_id_1": {...}
                        }
        """
        self.agents = {}
        self.policies = {}

        for agent_id, config in agent_configs.items():
            self.agents[agent_id] = {
                "policy_class": config["policy_class"],
                "policy": config["policy_class"](**config["policy_args"])
            }

    def propose_actions(self, observations: Dict, agent_states: Dict) -Dict[str, Action]:
        """
        Propose actions for all agents based on their observations.
          
        Returns:
            Dict mapping agent_id to Action
        """
        actions = {}
        for agent_id, policy_info in self.agents.items():
            policy = policy_info["policy"]
            # Each agent decides its own action based on local observations
            agent_obs = observations.get(agent_id, {})
            actions[agent_id] = policy.select_action(agent_obs,
agent_states[agent_id])

        return actions

    def coordinate_actions(self, proposed_actions: Dict, 
                            coordination_strategy: str = "independent") -> Dict:
        """
        Optionally coordinate actions between agents.
          
        Strategies:
            - "independent": Each agent acts independently
            - "turn_taking": Agents take turns moving
            - "synchronized": All agents move together
            - "coordinated": Use communication to avoid conflicts
        """
        if coordination_strategy == "independent":
            return proposed_actions

        elif coordination_strategy == "turn_taking":
            # Only one agent moves per step
            active_agent = self._select_active_agent()
            coordinated = {aid: None for aid in proposed_actions}
            coordinated[active_agent] = proposed_actions[active_agent]
            return coordinated

        # Add more coordination strategies as needed
        return proposed_actions