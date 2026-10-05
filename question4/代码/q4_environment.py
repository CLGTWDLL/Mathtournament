"""Local mixed-source extension; hidden type/orientation never go to policy."""
import numpy as np
from q3_environment import Environment as OmniEnvironment


class Environment(OmniEnvironment):
    def __init__(self, seed, distribution='uniform'):
        super().__init__(seed)
        rng = np.random.default_rng(seed+14000000)
        channels = list(self.sources)
        # At least one of each type; count otherwise uniform 1..N-1.
        directional = set(rng.choice(channels, int(rng.integers(1, self.n)), replace=False))
        for c, source in self.sources.items():
            source['directional'] = c in directional
            source['theta'] = float(rng.uniform(0, 2*np.pi))
            if distribution == 'outward':
                a = rng.uniform(0, 2*np.pi)
                source['position'] = (rng.uniform(1750, 1800)*np.array([np.cos(a), np.sin(a)])).tolist()
                source['theta'] = float(a)
                source['radius'] = 1000.
            elif distribution == 'cluster':
                p = np.array([1050., 650.])+rng.normal(0, 160, 2)
                p *= min(1., 1790/max(np.linalg.norm(p), 1.))
                source['position'] = p.tolist()
            elif distribution != 'uniform':
                raise ValueError(distribution)

    def act(self, action, channel, q):
        # Reuse exact Q3 movement/clock/noise/clear behaviour. RF visibility is
        # changed temporarily in the environment only, never in planner state.
        source = self.sources.get(channel)
        radius = source['radius'] if source else None
        if action == 'measure' and source and source['directional']:
            delta = np.asarray(q)-source['position']
            u = np.array([np.cos(source['theta']), np.sin(source['theta'])])
            if u@delta < -1e-10:
                source['radius'] = -1.
        try:
            return super().act(action, channel, q)
        finally:
            if source:
                source['radius'] = radius
