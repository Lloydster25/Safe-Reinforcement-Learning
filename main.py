from envs import get_env
import argparse
import numpy as np
import torch
from copy import deepcopy
import logging


class Normal:
    def __init__(self, size):
        self.size = size
    def gen(self):
        return np.random.normal(0., 0.2, size=self.size)

class OU:
    def __init__(self, size, sigma=0.2):
        self.theta = 0.15
        self.size = size
        self.mu = np.zeros(size)
        self.sigma = sigma * np.ones(size)
        self.dt = 1e-2
        self.prev = np.zeros(size)

    def gen(self):
        self.prev = (
            self.prev 
            + self.theta * (self.mu - self.prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        return self.prev


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="GatePoint")
    parser.add_argument("--mode", type=str, default="dmps")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--log_to", type=str)
    parser.add_argument("--save_to", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--hold", action="store_true")  # Backup ends with the hold controller (HoldEnv); always on for smps/sdmps
    parser.add_argument("--eps", type=float)  # SMPS epsilon for smps/sdmps (default: EPS in statrec.py)
    parser.add_argument("--resume", action="store_true")  # Continue training from the checkpoint in --save_to
    parser.add_argument("--load", type=str)  # Evaluate this saved model for 100 episodes instead of training (frozen-policy sweeps)


    args = parser.parse_args()

    torch.set_num_threads(1)  # One CPU thread per run, so parallel runs don't compete for cores
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = get_env(args.env)
    test_env = get_env(args.env)
    env.action_space.seed(args.seed)

    if args.hold or args.mode in ["smps", "sdmps"]:
        from statrec import HoldEnv
        env = HoldEnv(env, args.noise)  # Noisy env whose backup ends by holding at a rest point
        test_env = HoldEnv(test_env, args.noise)
    elif args.noise > 0:
        from envs.stochastic_env import StochasticEnv
        env = StochasticEnv(env, args.noise)
        test_env = StochasticEnv(test_env,args.noise)

    device = torch.device(f"cuda:{args.gpu}") if args.gpu is not None else torch.device("cpu")

    kwargs = deepcopy(env.kwargs)
    kwargs.update(eval_log_file=args.log_to, model_save_dir=args.save_to, act_noise=Normal(env.action_space.shape[-1]))
    kwargs.update(eval_freq=int(2e4))  # Evaluate every 20k steps instead of 10k: evaluations with the planner are expensive
    if args.load:
        kwargs.update(eval_num_episodes=100)  # Sweep point: enough episodes to measure violation rates

    if args.env in ["MountainCar", "Obstacle", "Obstacle2", "Road", "Road2d", "Pendulum"]:
        sigma = 0.5 if args.env == "MountainCar" else 0.2
        kwargs.update(act_noise=OU(env.action_space.shape[-1], sigma=sigma))

    if "td3_unsafe_penalty" in kwargs and args.mode == "td3":
        kwargs["unsafe_penalty"] = kwargs["td3_unsafe_penalty"]

    kwargs.pop("td3_unsafe_penalty", None)

    if args.mode == "dmps":
        from mcts_trainer import MCTSTrainer
        trainer = MCTSTrainer(env, test_env, device, **kwargs)
    elif args.mode == "sdmps":
        from mcts_trainer import MCTSTrainer
        from statrec import EPS, DELTA, SDMPS_CHECKS, n_samples
        n = n_samples(args.eps or EPS, DELTA, SDMPS_CHECKS)  # Up to SDMPS_CHECKS checks per step share delta
        trainer = MCTSTrainer(env, test_env, device, n_samples=n, **kwargs)  # DMPS with the statistical check
    elif args.mode == "mps":
        from mps_trainer import MPSTrainer
        trainer = MPSTrainer(env, test_env, device, **kwargs)
    elif args.mode == "smps":
        from mps_trainer import MPSTrainer
        from statrec import EPS, DELTA, SDMPS_CHECKS, n_samples
        n = n_samples(args.eps or EPS, DELTA, SDMPS_CHECKS)  # Same N as SDMPS (was M=1, N=59): same check on a_RL, so only the recovery differs
        trainer = MPSTrainer(env, test_env, device, n_samples=n, **kwargs)  # MPS with the statistical check
    else:
        from td3_trainer import Trainer
        trainer = Trainer(env, test_env, device, **kwargs)

    if args.load:  # Frozen-policy sweep: evaluate a saved model instead of training
        trainer.model = torch.load(args.load, weights_only=False)  # Saved with torch.save(model), so load the whole object
        if args.mode in ["dmps", "sdmps"]:
            trainer.test_planner.critic = trainer.model.critic1  # The planner keeps its own reference to the Q-network
        trainer.evaluate(0, 0)
    else:
        from utils import load_checkpoint
        start = load_checkpoint(trainer) if args.resume else (0, 1)  # (first step, episode number)
        trainer.run(200001, *start)  # 200k steps as in DMPS (+1 so the evaluation and model save at step 200k still happen); was 210k
