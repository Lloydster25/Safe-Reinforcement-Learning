import numpy as np
import torch
from statrec import H, stat_rec  # Step limit for the recoverability check; SMPS statistical check


class MPSPlanner:
    def __init__(self, env, n_samples=0):
        self.env = env
        self.n_samples = n_samples  # 0: original noise-free check (MPS); N > 0: statistical check with N samples (SMPS)
        self.obs = None
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    # Executed actions use the noisy real_step (if the env has one); checks keep using noise-free env.step
    def _real_step(self,act):
        return getattr(self.env, "real_step", self.env.step)(act)

    def take_step(self, base_act):
        ok = stat_rec(self, base_act, self.n_samples) if self.n_samples else self.check_action(base_act)
        if ok:
            self.obs, rew, done, info = self._real_step(base_act)
            return [(self.obs, rew, done, info)], ([base_act], 0)

        backup_act = self.env.backup()
        self.obs, rew, done, info = self._real_step(backup_act)
        return [(self.obs, rew, done, info)], ([backup_act], 1)

    def check_action(self, action):
        state = self.env.get_state()
        is_recoverable = self.check_action_internal(action)
        self.env.set_state(state)
        return is_recoverable

    def check_action_internal(self, action):
        self.env.step(action)
        if not self.env.safe():
            return False

        for _ in range(H):  # Was "while True": with the hold backup a rollout can fail to come to rest
            backup_act = self.env.backup()
            self.env.step(backup_act)
            if not self.env.safe():
                return False
            if self.env.stable():
                return True
        return False
