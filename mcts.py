import numpy as np
import torch
from statrec import H, ALTERNATIVES, stat_rec  # Step limit; alternatives tried after a0* fails; SMPS statistical check


class Node:
    def __init__(self, state, o, r, d, acts, q_vals):
        self.state = state
        self.o = o
        self.r = r 
        self.d = d
        self.acts = acts
        self.q_vals = q_vals
        self.next_nodes = [None] * len(self.acts)
        self.counts = np.ones(len(self.acts))

    def uct_select(self, c, t):
        return np.argmax(self.q_vals + c * np.sqrt(np.log(t + 1) / self.counts))
    
    def add_estimate(self, act_id, new_q):
        self.q_vals[act_id] = (self.counts[act_id] * self.q_vals[act_id] + new_q) / (self.counts[act_id] + 1)
        self.counts[act_id] += 1


class MCTSPlanner:
    def __init__(self, env, gamma, num_iters, branch_factor, path_len, device, critic, single_action=False, n_samples=0):
        self.env = env
        self.n_samples = n_samples  # 0: original noise-free checks (DMPS); N > 0: statistical check on a_RL and a0* (SDMPS)
        assert single_action or not n_samples, "SDMPS certifies one planned action per step, so it needs single_action"
        self.gamma = gamma
        self.num_iters = num_iters
        self.branch_factor = branch_factor
        self.path_len = path_len
        self.device = device
        self.critic = critic
        self.obs = None
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.single_action = single_action
        self.only_backup = False  # Set at every planner call: whether the root had only the backup action

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def _real_step(self,act):
        return getattr(self.env, "real_step", self.env.step)(act)

    def _execute_plan(self, acts, root):
        # SDMPS: the first planned action a0* must pass the statistical check. If it fails, try the next-best
        # root actions (by the planner's value, skipping the backup), and only then use the backup.
        # Codes: 1 = planned action a0* executed, 3 = an alternative root action executed, 2 = fell back to the backup
        if not self.n_samples or stat_rec(self, acts[0], self.n_samples):
            return [self._real_step(act) for act in acts], acts, 1
        backup_i = len(root.acts) - 1  # expand() adds the backup as the last root action
        alternatives = [i for i in np.argsort(-root.q_vals) if i != backup_i and not np.array_equal(root.acts[i], acts[0])]
        for i in alternatives[:ALTERNATIVES]:
            if stat_rec(self, root.acts[i], self.n_samples):
                return [self._real_step(root.acts[i])], [root.acts[i]], 3
        backup_act = self.env.backup()
        return [self._real_step(backup_act)], [backup_act], 2

    def take_step(self, base_act, c=0.25):
        ok = stat_rec(self, base_act, self.n_samples) if self.n_samples else self.check_action(base_act)
        if ok:
            self.obs, rew, done, info = self._real_step(base_act)
            return [(self.obs, rew, done, info)], ([base_act], 0)

    
        save_state = self.env.get_state()
        root = self.expand(save_state, self.obs, None, False)
        self.only_backup = len(root.acts) == 1  # For logging: the planner had nothing to offer except the backup

        for t in range(self.num_iters):
            nodes = []
            curr_node = root
            act_ids = []
            acts = []
            rews = []
            while curr_node is not None:
                rews.append(curr_node.r)
                nodes.append(curr_node)
                act_id = curr_node.uct_select(c, t)
                act_ids.append(act_id)
                acts.append(curr_node.acts[act_id])
                curr_node = curr_node.next_nodes[act_id]

            if len(acts) == self.path_len:
                # We can return this path
                self.env.set_state(save_state)
                if self.single_action:
                    acts = acts[:1]
                data, acts, code = self._execute_plan(acts, root)  # Was: execute acts directly (code 1)
                self.obs = data[-1][0]
                return data, (acts, code)
            
            if not nodes[-1].d:
                self.env.set_state(nodes[-1].state)
                o, r, d, _ = self.env.step(acts[-1])
                if d:
                    nodes[-1].q_vals[act_ids[-1]] = r
                new_node = self.expand(self.env.get_state(), o, r, d)
                nodes[-1].next_nodes[act_ids[-1]] = new_node
                nodes.append(new_node)
                new_act_id = new_node.uct_select(c, t)
                act_ids.append(new_act_id)
                acts.append(new_node.acts[new_act_id])
                rews.append(r)

            targ_val = nodes[-1].q_vals[act_ids[-1]] if not nodes[-1].d else 0.
            for t_to_upd in range(len(acts) - 2, -1, -1):
                targ_val *= self.gamma
                targ_val += rews[t_to_upd + 1]
                nodes[t_to_upd].add_estimate(act_ids[t_to_upd], targ_val)

        self.env.set_state(save_state)
        if self.n_samples:  # SDMPS: certify and execute only the best first action a0*
            data, acts, code = self._execute_plan([root.acts[np.argmax(root.q_vals)]], root)
            self.obs = data[-1][0]
            return data, (acts, code)
        data = []
        acts = []
        while root is not None:
            act_id = np.argmax(root.q_vals)
            act = root.acts[act_id]
            acts.append(act)
            self.obs, rew, done, info = self._real_step(act)
            data.append((self.obs, rew, done, info))
            root = root.next_nodes[act_id]
            if done or self.single_action:
                return data, (acts, 1)
        return data, (acts, 1)

    def expand(self, state, o, r, d):
        if d:
            return Node(state, o, r, d, np.array([self.env.backup()]), np.zeros(1))

        self.env.set_state(state)
        acts = 2 * np.random.rand(self.branch_factor, self.act_dim) - 1.
        acts = np.stack([act for act in acts if self.check_action(act)] + [self.env.backup()])
        broadcasted_o = np.broadcast_to(o, (len(acts), len(o))).copy()
        tensor_o = torch.as_tensor(broadcasted_o, dtype=torch.float32, device=self.device)
        tensor_acts = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_vals = self.critic(tensor_o, tensor_acts).cpu().squeeze(-1).numpy()
        return Node(state, o, r, d, acts, q_vals)

    def check_action(self, action):
        state = self.env.get_state()
        is_recoverable = self.check_action_internal(action)
        self.env.set_state(state)
        return is_recoverable

    def check_action_internal(self, action):
        self.env.step(action)
        if not self.env.safe():
            return False
        if self.env.stable():
            return True

        for _ in range(H):  # Was "while True": with the hold backup a rollout can fail to come to rest
            backup_act = self.env.backup()
            self.env.step(backup_act)
            if not self.env.safe():# or not self.env.can_recover():
                return False
            if self.env.stable():
                return True
        return False
