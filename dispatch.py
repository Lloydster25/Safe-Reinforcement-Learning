import os
import subprocess
import sys
import time
from utils import CHECKPOINT

# Runs the whole experiment grid on this machine, a few jobs at a time (CPU only).
# Usage: python dispatch.py [parallel jobs, default 7] [noise, default 0.2] [methods, default all]
# Example (sigma=0.1 runs of two methods): python dispatch.py 7 0.1 sdmps smps

parallel_procs = int(sys.argv[1]) if len(sys.argv) > 1 else 7
noise = sys.argv[2] if len(sys.argv) > 2 else "0.2"
modes = sys.argv[3:] or ["sdmps", "dmps", "smps", "td3"]
tag = "" if noise == "0.2" else f"_noise{noise}"  # The main grid keeps its names; other noise levels get a suffix

jobs = []  # (name, extra arguments for main.py)

# Main grid. Heaviest methods first, so the long runs start immediately and short ones fill the gaps
envs = ["ThickDoubleGatePoint", "DoubleGatePoint"]
# Main grid: method by method. Other noise levels: DoubleGatePoint first, one environment at a time, so its runs finish first
grid = [(mode, env) for mode in modes for env in envs] if noise == "0.2" else [(mode, env) for env in envs[::-1] for mode in modes]
for mode, env in grid:
    for seed in range(3):
        # DMPS uses the hold backup under noise (smps/sdmps always do); without noise it runs as published,
        # because the hold controller needs a noise bound
        hold = ["--hold"] if mode == "dmps" and float(noise) > 0 else []
        jobs.append((f"{env}_{mode}{tag}_s{seed}", ["--env", env, "--mode", mode, *hold, "--noise", noise, "--seed", str(seed)]))

# Restarting dispatch.py after a crash: skip finished runs, resume runs that have a checkpoint
commands = [(name, [sys.executable, "main.py", *args, "--log_to", f"results/{name}.log", "--save_to", f"models/{name}",
                    *(["--resume"] if os.path.exists(f"models/{name}/{CHECKPOINT}") else [])])
            for name, args in jobs if not os.path.exists(f"models/{name}/save_final.pth")]


def log_path(cmd):
    return cmd[cmd.index("--log_to") + 1]


def run_jobs(commands, parallel_procs):
    """Run (name, command) jobs, parallel_procs at a time."""
    # Logs are appended to, so an existing log would mix two runs: refuse to start
    existing = [log_path(cmd) for _, cmd in commands if os.path.exists(log_path(cmd)) and "--resume" not in cmd]
    if existing:
        sys.exit(f"These logs already exist, move or delete them first: {existing}")

    processes = []

    try:
        while True:
            terminated_processes = [(name, proc) for name, proc in processes if proc.poll() is not None]
            processes = [(name, proc) for name, proc in processes if proc.poll() is None]

            for name, proc in terminated_processes:
                print(f"{time.strftime('%H:%M')} FINISHED: {name} (exit code {proc.returncode})")

            while len(processes) < parallel_procs and len(commands) != 0:
                name, cmd = commands[0]
                log = log_path(cmd)
                os.makedirs(os.path.dirname(log), exist_ok=True)
                # stdout is not needed (the log file has everything); stderr is kept so a crash can be diagnosed
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(log[:-len(".log")] + ".err", "a" if "--resume" in cmd else "w"))
                print(f"{time.strftime('%H:%M')} STARTED: {name}")
                processes.append((name, proc))
                commands = commands[1:]

            if len(processes) == 0 and len(commands) == 0:
                break

            time.sleep(10)

    except KeyboardInterrupt:
        print("\n")
        for name, proc in processes:
            print(f"KILLING: {name}")
            proc.kill()


if __name__ == "__main__":
    run_jobs(commands, parallel_procs)
