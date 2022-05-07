class MaSTNU():
    """MaSTNU class."""

    def __init__(self, agent2network, external_requirements, external_contingents, ref_event):
        self.agent2network = agent2network
        self.external_requirements = external_requirements
        self.external_contingents = external_contingents
        self.ref_event = ref_event

        self.event2agent = compile_event_to_agent(agent2network, ref_event)
        # Check that events for external constraints have assigned agents
        for c in self.external_contingents + self.external_requirements:
            assert(c.s in self.event2agent)
            assert(c.e in self.event2agent)

    @property
    def num_agents(self):
        return len(self.agent2network)

    def event_to_agent(self, event):
        if event not in self.event2agent:
            return None
        return self.event2agent[event]

    def get_shared_events(self):
        """Obtain the set of events that must be shared."""
        shared = set()
        for c in self.external_contingents + self.external_requirements:
            shared.add(c.s)
            shared.add(c.e)
        shared.add(self.ref_event)
        return list(shared)

def compile_event_to_agent(agent2network, ref_event):
    event2agent = {}
    for agent, network in agent2network.items():
        events = network.get_events()
        # Check that local network includes ref_event
        assert(ref_event in events)
        for e in events:
            if not e == ref_event:
                event2agent[e] = agent
    return event2agent
