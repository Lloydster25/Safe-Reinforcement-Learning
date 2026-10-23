import os
import numpy as np
import torch


def polyak_update(net_params, targ_params, tau):
    with torch.no_grad():
        for param, target_param in zip(net_params, targ_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


# Resumable checkpoints, shared by the three trainers (they have the same model, optimizers and buffer).
CHECKPOINT = "checkpoint.pth"  # In the run's model_save_dir
CHECKPOINT_FREQ = 5000         # Steps between checkpoints (saved at the first episode end after each multiple)


def save_checkpoint(trainer, t, ep_num):
    """Save everything needed to continue training after step t. Called at the end of an episode."""
    buf = trainer.buf
    state = dict(
        t=t, ep_num=ep_num, num_updates=trainer.num_updates,
        model=trainer.model.state_dict(),
        opts=[opt.state_dict() for opt in (trainer.actor_opt, trainer.critic1_opt, trainer.critic2_opt)],
        buf=(buf.size, buf.idx, [b[:buf.size] for b in (buf.obs_buf, buf.act_buf, buf.next_obs_buf, buf.rew_buf, buf.done_buf)]),
        act_noise=trainer.act_noise,  # OU noise has internal state
        rng=(np.random.get_state(), torch.get_rng_state(), trainer.env.action_space.np_random),
        log_size=os.path.getsize(trainer.eval_log_file) if trainer.eval_log_file else None,  # Log length at this point
    )
    path = os.path.join(trainer.model_save_dir, CHECKPOINT)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)  # Atomic: a crash while saving leaves the previous checkpoint intact


def load_checkpoint(trainer):
    """Restore a checkpoint saved by save_checkpoint. Returns (first step, episode number) to pass to run()."""
    ckpt = torch.load(os.path.join(trainer.model_save_dir, CHECKPOINT), weights_only=False)
    trainer.model.load_state_dict(ckpt["model"])  # In place, so optimizers and planners keep their references
    for opt, opt_state in zip((trainer.actor_opt, trainer.critic1_opt, trainer.critic2_opt), ckpt["opts"]):
        opt.load_state_dict(opt_state)
    buf = trainer.buf
    buf.size, buf.idx, data = ckpt["buf"]
    for b, d in zip((buf.obs_buf, buf.act_buf, buf.next_obs_buf, buf.rew_buf, buf.done_buf), data):
        b[:buf.size] = d  # Training stays under the buffer capacity, so the buffer never wraps
    trainer.num_updates = ckpt["num_updates"]
    trainer.act_noise = ckpt["act_noise"]
    np_state, torch_state, trainer.env.action_space._np_random = ckpt["rng"]  # gym Box has no np_random setter
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    # Drop log lines written after the checkpoint (that work is redone). Only ever shorten: if the log is shorter
    # (e.g. it was deleted), truncate() would extend it with NUL bytes
    if ckpt["log_size"] is not None and os.path.getsize(trainer.eval_log_file) > ckpt["log_size"]:
        with open(trainer.eval_log_file, "r+") as f:
            f.truncate(ckpt["log_size"])
    return ckpt["t"] + 1, ckpt["ep_num"]
