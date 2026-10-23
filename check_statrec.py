import time
import numpy as np
from envs import get_env
from mps import MPSPlanner
from statrec import HoldEnv, EPS, DELTA, SDMPS_CHECKS, n_samples, stat_rec

# Checks for statrec.py. Run: python check_statrec.py


def check_hold(sigma=0.2, steps=5000):
    env = HoldEnv(get_env("DoubleGatePoint"), sigma)
    env.reset()
    inner = env.env  # Set and read pos/vel on the inner env: the wrapper only forwards reads

    # 1. Moving fast far outside the rings: the backup brakes, then holds within the drift bound D forever
    inner.pos, inner.vel = np.array([3.5, 0.]), np.array([1., -0.5])
    for t in range(30):
        env.real_step(env.backup())
        if env.stable():
            break
    assert env.stable(), "backup did not start holding within 30 steps"
    rest, max_drift, max_act = env.rest.copy(), 0., 0.
    for _ in range(steps):
        act = env.backup()
        env.real_step(act)
        assert env.stable() and np.array_equal(env.rest, rest), "rest point changed while holding"
        max_drift = max(max_drift, np.linalg.norm(inner.pos - rest))
        max_act = max(max_act, np.abs(act).max())
    assert max_drift <= env.D and max_act <= 1., (max_drift, max_act)
    print(f"hold: started after {t + 1} steps, max drift {max_drift:.4f} <= D={env.D:.4f}, max action {max_act:.2f} <= 1")

    # 2. Stopped closer than D to a ring: the backup does not start holding there (it keeps braking)
    inner.pos, inner.vel, env.rest = np.array([2.6 + env.D / 2, 0.]), np.zeros(2), None
    env.backup()
    assert not env.stable(), "backup held too close to a ring"
    print("hold: does not start holding closer than D to a ring")

    # 3. The rest point is saved and restored with the state; any other action forgets it
    inner.pos, inner.vel, env.rest = np.array([3.5, 0.]), np.zeros(2), None
    env.backup()
    saved = env.get_state()
    env.real_step(np.array([1., 0.]))
    assert not env.stable(), "rest point not forgotten after a non-hold action"
    env.set_state(saved)
    assert env.stable(), "rest point not restored by set_state"
    print("hold: rest point saved/restored with the state, forgotten after another action")


def check_stat_rec(sigma=0.2):
    # 1. Sample counts from SMPS eq. (1), with delta split over the checks per step
    assert [n_samples(e, 0.05, SDMPS_CHECKS) for e in (0.1, 0.05, 0.01)] == [44, 90, 459]  # SDMPS: delta/5 per check
    assert [n_samples(e, 0.05, 1) for e in (0.1, 0.05, 0.01)] == [29, 59, 299]               # SMPS: delta per check
    n = n_samples(EPS, DELTA, SDMPS_CHECKS)

    env = HoldEnv(get_env("DoubleGatePoint"), sigma)
    planner = MPSPlanner(env)
    planner.reset()
    inner = env.env

    # 2. Clearly safe: standing still far outside the rings, doing nothing
    inner.pos, inner.vel = np.array([3.5, 0.]), np.zeros(2)
    start = time.perf_counter()
    assert stat_rec(planner, np.zeros(2), n), "safe action rejected"
    seconds = time.perf_counter() - start

    # 3. Clearly unsafe: racing into the outer ring's wall (opening on the other side), accelerating inward
    inner.pos, inner.vel, inner.theta_list = np.array([2.8, 0.]), np.array([-1., 0.]), np.array([np.pi, np.pi])
    assert not stat_rec(planner, np.array([-1., 0.]), n), "unsafe action accepted"

    assert not env.sim_noise, "noise left switched on after the check"
    print(f"stat_rec: N={n}, safe action accepted ({seconds * 1000:.0f} ms), unsafe action rejected")


if __name__ == "__main__":
    check_hold()
    check_stat_rec()
