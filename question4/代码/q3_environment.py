"""Local offline environment. Hidden truth never enters the belief API."""
import hashlib
import numpy as np


class Environment:
    def __init__(self, seed):
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.n = int(rng.integers(10, 17))
        channels = rng.choice(np.arange(1, 21), self.n, replace=False)
        angle = rng.uniform(0, 2*np.pi, self.n)
        radius = 1800*np.sqrt(rng.random(self.n))
        points = radius[:, None]*np.column_stack((np.cos(angle), np.sin(angle)))
        reception = rng.uniform(1000, 1500, self.n)
        self.sources = {int(c): dict(position=p.tolist(), radius=float(r))
                        for c, p, r in zip(channels, points, reception)}
        self.cleared = set()
        self.position = np.zeros(2)
        self.channel = 1
        self.distance = 0.
        self.measurements = self.switches = self.attempts = self.successes = 0

    def noise(self, channel, q):
        coords=[0. if x == 0 else float(x) for x in q]
        key = f'{self.seed}:{channel}:{coords[0].hex()}:{coords[1].hex()}'
        raw = hashlib.sha256(key.encode()).digest()[:8]
        return 2*((int.from_bytes(raw, 'big')+.5)/2**64)-1

    @property
    def time(self):
        return self.distance/5+5*self.measurements+self.switches+3*self.attempts+2*self.successes

    def act(self, action, channel, q):
        q = np.asarray(q, float)
        if action not in ('measure', 'clear') or channel not in range(1, 21):
            raise ValueError('Invalid action')
        self.distance += float(np.linalg.norm(q-self.position))
        self.position = q.copy()
        source = self.sources.get(channel) if channel not in self.cleared else None
        d = np.linalg.norm(q-source['position']) if source else np.inf
        result = dict(accepted=True)
        if action == 'measure':
            self.measurements += 1
            self.switches += int(channel != self.channel)
            self.channel = channel
            kind = 'no_signal' if not source or d > source['radius'] else 'near' if d <= 5 else 'direction'
            result['kind'] = kind
            if kind == 'direction':
                delta = np.asarray(source['position'])-q
                angle = np.rad2deg(np.arctan2(delta[1], delta[0]))+self.noise(channel, q)
                result['angle'] = float(np.round(angle, 2) % 360)
        else:
            self.attempts += 1
            result['kind'] = 'success' if d <= 20 else 'no_target_in_range'
            if d <= 20:
                self.successes += 1
                self.cleared.add(channel)
        result['virtual_time_s'] = self.time
        return result

    def truth_for_evaluation(self):
        return dict(seed=self.seed, source_count=self.n, sources=self.sources)


def fixed_stations():
    rho = 1800/(2*np.cos(np.pi/7))
    angles = np.arange(7)*2*np.pi/7
    return np.vstack((np.zeros(2), rho*np.column_stack((np.cos(angles), np.sin(angles)))))
