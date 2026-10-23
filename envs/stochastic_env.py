import numpy as np

class StochasticEnv:

    def __init__(self, env, noise = 0.1, sim_noise=False):
        self.env = env
        self.noise = noise
        self.sim_noise = sim_noise
        self.noise_bound = 2 * noise  # Noise is truncated at 2 sigma: SMPS needs bounded disturbances

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _noisy(self,act):
        w = np.random.normal(0., self.noise, size = np.shape(act))
        # Truncated Gaussian: redraw until every component is within the bound
        while np.any(np.abs(w) > self.noise_bound):
            w = np.random.normal(0., self.noise, size = np.shape(act))
        return np.clip(act + w, -1.,1.)

    def step(self, act):
        return self.env.step(self._noisy(act) if self.sim_noise else act)

    def real_step(self, act):
        return self.env.step(self._noisy(act))