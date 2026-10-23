import numpy as np
from envs.stochastic_env import StochasticEnv

# Safety code for SDMPS: the hold controller of the backup and the SMPS statistical check.

H = 30  # Max backup steps in a recoverability check; a rollout that has not reached the resting set by then fails
EPS, DELTA = 0.05, 0.05  # SMPS safety parameters
ALTERNATIVES = 3  # SDMPS: next-best root actions tried when a0* fails the check, before falling back to the backup
SDMPS_CHECKS = 2 + ALTERNATIVES  # Max checks per SDMPS step: a_RL, a0* and the alternatives (SMPS: 1)


def n_samples(eps, delta, checks):
    # SMPS eq. (1) with delta/checks: by a union bound, `checks` checks per step together err with probability <= delta
    return int(np.ceil(np.log(checks / delta) / np.log(1 / (1 - eps))))


def stat_rec(planner, action, n):
    """SMPS statistical check: the planner's existing recoverability check, run n times with noise.

    Accepts only if all n noisy rollouts pass.
    """
    planner.env.sim_noise = True                               # Simulated steps now sample noise
    ok = all(planner.check_action(action) for _ in range(n))  # all() stops at the first failed rollout
    planner.env.sim_noise = False                              # Back to noise-free simulation
    return ok


def hold_gains(sigma, dt, max_act):
    """Strongest hold controller ("spring") whose worst-case action stays within the action limit.

    Per axis the hold controller is a = -kp * (p - p_rest) - kd * v.
    Returns kp, kd and D, the worst-case 2D distance the agent can drift from its rest point.
    """
    w_max = 2 * sigma                          # Noise bound (noise is truncated at 2 sigma)
    A = np.array([[1., dt], [0., 1.]])         # Per-axis env dynamics: p += v * dt, then v += a * dt
    B = np.array([0., dt])
    x0 = np.array([0., w_max * dt])            # Holding starts at the rest point, with leftover speed <= w_max * dt

    for lam in np.arange(0., 1., 0.01):        # lam = how slowly the spring pulls back (0 = fastest)
        kd = (2 - 2 * lam) / dt                # Gains that make the closed loop return with rate lam
        kp = (1 - lam) ** 2 / dt ** 2
        K = np.array([-kp, -kd])
        M = A + np.outer(B, K)                 # Closed loop with noise: x_next = M x + B w

        # Worst case over all noise sequences: the effect of the start state plus,
        # for every future step, the largest effect the noise can have on that step.
        drift_start = act_start = drift_noise = act_noise = 0.
        Mk = np.eye(2)                         # M^k
        for _ in range(1000):
            drift_start = max(drift_start, abs((Mk @ x0)[0]))
            act_start = max(act_start, abs(K @ Mk @ x0))
            drift_noise += abs((Mk @ B)[0]) * w_max
            act_noise += abs(K @ Mk @ B) * w_max
            Mk = M @ Mk

        if act_start + act_noise <= max_act:   # Strongest spring the actuator can always deliver
            drift = drift_start + drift_noise  # Per axis; x and y are independent
            return kp, kd, np.sqrt(2) * drift

    raise ValueError(f"No hold controller fits the action limit for sigma={sigma}")


class HoldEnv(StochasticEnv):
    """StochasticEnv whose backup ends by holding the agent at a remembered rest point.

    Changes the meaning of backup() and stable(), so the existing DMPS/MPS checks use it unchanged.
    Only for DoubleGatePoint and ThickDoubleGatePoint (it uses their ring radii).
    """

    def __init__(self, env, noise):
        super().__init__(env, noise)
        self.kp, self.kd, self.D = hold_gains(noise, env.dt, env.max_act)
        self.rest = None                               # Remembered rest point; None while not holding

    def reset(self):
        self.rest = None                               # New episode: nothing remembered
        return self.env.reset()

    def far_from_rings(self, pos):
        # More than D outside every ring band, so drifting while holding can never reach a ring
        dist = np.linalg.norm(pos)
        return all(dist < inner - self.D or dist > outer + self.D
                   for inner, outer in zip(self.env.inner_radius_list, self.env.outer_radius_list))

    def hold_action(self):
        return (-self.kp * (self.env.pos - self.rest) - self.kd * self.env.vel) / self.env.max_act

    def backup(self):
        stopped = np.all(np.abs(self.env.vel) <= self.noise_bound * self.env.dt)  # Speed left after braking under noise
        if self.rest is None and stopped and self.far_from_rings(self.env.pos):
            self.rest = self.env.pos.copy()            # Start holding here
        if self.rest is None:
            return self.env.backup()                   # Original DMPS backup: leave the ring or brake
        return self.hold_action()

    def stable(self):
        return self.rest is not None                   # Resting set X_inv: the backup is holding

    def get_state(self):
        return self.env.get_state(), self.rest         # The rest point is part of the state

    def set_state(self, state):
        env_state, self.rest = state
        self.env.set_state(env_state)

    def forget_if_not_holding(self, act):
        # Any action other than the hold action means another controller took over
        if self.rest is not None and not np.allclose(act, self.hold_action()):
            self.rest = None

    def step(self, act):
        self.forget_if_not_holding(act)
        return super().step(act)

    def real_step(self, act):
        self.forget_if_not_holding(act)
        return super().real_step(act)
