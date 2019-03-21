import numpy as np
from helper_func import Mapping, Callable, Sequence, Tuple, S, A, SAf, get_vf_func_approx_obj, get_qvf_func_approx_obj
from mdp_rl_fa import MDPRepForRLFA
from td_fa import TD0

class PolicyGradient():


    # initialize all the parameters
    def __init__(
        self,
        mdp_rep_for_rl_fa: MDPRepForRLFA,
        reinforce: bool,
        batch_size: int,
        num_batches: int,
        num_action_samples: int,
        max_steps: int,
        actor_lambda: float,
        critic_lambda: float,
        score_func: Callable[[A, Sequence[float]], Sequence[float]],
        sample_actions_gen_func: Callable[[Sequence[float], int], Sequence[A]],
        fa_spec: TD0,
        pol_fa_spec: Sequence[TD0]
    ) -> None:

        self.mdp_rep: MDPRepForRLFA = mdp_rep_for_rl_fa
        self.rein: bool = rein
        self.batch_size: int = batch_size
        self.num_batches: int = num_batches
        self.num_action_samples: int = num_action_samples
        self.max_steps: int = max_steps
        self.actor_lambda: float = actor_lambda
        self.critic_lambda: float = critic_lambda
        self.score_func: Callable[[A, Sequence[float]], Sequence[float]] = score_func
        self.sample_actions_gen_func: Callable[[Sequence[float], int], Sequence[A]] = sample_actions_gen_func
        self.vf_fa: FuncApproxBase = fa_spec.get_vf_func_approx_obj()
        self.qvf_fa: FuncApproxBase = fa_spec.get_qvf_func_approx_obj()
        self.pol_fa: Sequence[FuncApproxBase] = [s.get_vf_func_approx_obj() for s in pol_fa_spec]

    def get_value_func(self, pol_func: PolicyType) -> VFType:
        mo = self.mdp_rep
        for _ in range(self.num_batches * self.batch_size):
            state = mo.init_state_gen_func()
            steps = 0
            terminate = False
            states = []
            targets = []

            while not terminate:
                action = pol_func(state)(1)[0]
                next_state, reward = mo.state_reward_gen_func(
                    state,
                    action
                )
                target = reward + mo.gamma * self.vf_fa.get_func_eval(next_state)
                states.append(state)
                targets.append(target)
                steps += 1
                terminate = steps >= self.max_steps or mo.terminal_state_func(state)
                state = next_state

            self.vf_fa.update_params_from_gradient(
                [g / len(states) for g in
                 self.vf_fa.get_el_tr_sum_loss_gradient(
                     states,
                     targets,
                     mo.gamma * self.critic_lambda
                 )
                 ]
            )

        return self.vf_fa.get_func_eval

    def get_act_value_func(self, pol_func: PolicyType) -> QFType:
        mo = self.mdp_rep
        for _ in range(self.num_batches * self.batch_size):
            state = mo.init_state_gen_func()
            steps = 0
            terminate = False
            states_actions = []
            targets = []

            while not terminate:
                action = pol_func(state)(1)[0]
                next_state, reward = mo.state_reward_gen_func(
                    state,
                    action
                )
                target = reward + mo.gamma * self.vf_fa.get_func_eval(next_state)
                states_actions.append((state, action))
                targets.append(target)
                steps += 1
                terminate = steps >= self.max_steps or mo.terminal_state_func(state)
                state = next_state

            self.vf_fa.update_params_from_gradient(
                [g / len(states_actions) for g in
                 self.qvf_fa.get_el_tr_sum_loss_gradient(
                     states_actions,
                     targets,
                     mo.gamma * self.critic_lambda
                 )
                 ]
            )

        return lambda s: lambda a, s=s: self.qvf_fa.get_func_eval((s, a))

    def get_policy_as_policy_type(self) -> PolicyType:

        def pol(s: S) -> Callable[[int], Sequence[A]]:
            def gen_func(samples: int, s=s) -> Sequence[A]:
                return self.sample_actions_gen_func(
                    [f.get_func_eval(s) for f in self.pol_fa],
                    samples
                )

            return gen_func

        return pol

    def get_path(
        self,
        start_state: S
    ) -> Sequence[Tuple[S, Sequence[float], A, float]]:
        res = []
        state = start_state
        steps = 0
        terminate = False

        while not terminate:
            pdf_params = [f.get_func_eval(state) for f in self.pol_fa]
            action = self.sample_actions_gen_func(pdf_params, 1)[0]
            next_state, reward = self.mdp_rep.state_reward_gen_func(state, action)
            res.append((
                state,
                pdf_params,
                action,
                reward
            ))
            steps += 1
            terminate = steps >= self.max_steps or\
                self.mdp_rep.terminal_state_func(state)
            state = next_state
        return res

    def get_optimal_reinforce_func(self) -> PolicyType:
        mo = self.mdp_rep
        sc_func = self.score_func

        for _ in range(self.num_batches):
            pol_grads = [
                [np.zeros_like(layer) for layer in this_pol_fa.params]
                for this_pol_fa in self.pol_fa
            ]
            for _ in range(self.batch_size):
                states = []
                disc_return_scores = []
                return_val = 0.
                init_state = mo.init_state_gen_func()
                this_path = self.get_path(init_state)

                for i, (s, pp, a, r) in enumerate(this_path[::-1]):
                    i1 = len(this_path) - i - 1
                    states.append(s)
                    return_val = return_val * mo.gamma + r
                    disc_return_scores.append(
                        [return_val * mo.gamma ** i1 * x for x in sc_func(a, pp)]
                    )

                pg_arr = np.vstack(disc_return_scores)
                for i, pp_fa in enumerate(self.pol_fa):
                    this_pol_grad = pp_fa.get_sum_objective_gradient(
                        states,
                        - pg_arr[:, i]
                    )
                    for j in range(len(pol_grads[i])):
                        pol_grads[i][j] += this_pol_grad[j]

            for i, pp_fa in enumerate(self.pol_fa):
                pp_fa.update_params_from_gradient(
                    [pg / self.batch_size for pg in pol_grads[i]]
                )

        return self.get_policy_as_policy_type()

    def get_optimal_tdl_func(self) -> PolicyType:
        mo = self.mdp_rep
        sc_func = self.score_func

        for _ in range(self.num_batches):
            pol_grads = [
                [np.zeros_like(layer) for layer in this_pol_fa.params]
                for this_pol_fa in self.pol_fa
            ]
            for _ in range(self.batch_size):
                gamma_pow = 1.
                states = []
                deltas = []
                disc_scores = []
                init_state = mo.init_state_gen_func()
                this_path = self.get_path(init_state)

                for i, (s, pp, a, r) in enumerate(this_path):
                    fut_return = mo.gamma * self.vf_fa.get_func_eval(this_path[i + 1][0])\
                        if i < len(this_path) - 1 else 0.
                    delta = r + fut_return - self.vf_fa.get_func_eval(s)
                    states.append(s)
                    deltas.append(delta)
                    disc_scores.append([gamma_pow * x for x in sc_func(a, pp)])
                    gamma_pow *= mo.gamma

                self.vf_fa.update_params_from_gradient(
                    self.vf_fa.get_el_tr_sum_objective_gradient(
                        states,
                        np.power(mo.gamma, np.arange(len(states))),
                        - np.array(deltas),
                        mo.gamma * self.critic_lambda
                    )
                )

                pg_arr = np.vstack(disc_scores)
                for i, pp_fa in enumerate(self.pol_fa):
                    this_pol_grad = pp_fa.get_el_tr_sum_objective_gradient(
                        states,
                        pg_arr[:, i],
                        - np.array(deltas),
                        mo.gamma * self.actor_lambda
                    )
                    for j in range(len(pol_grads[i])):
                        pol_grads[i][j] += this_pol_grad[j]

            for i, pp_fa in enumerate(self.pol_fa):
                pp_fa.update_params_from_gradient(
                    [pg / self.batch_size for pg in pol_grads[i]]
                )

        return self.get_policy_as_policy_type()

    def get_optimal_stoch_policy_func(self) -> PolicyType:
        return self.get_optimal_reinforce_func() if self.reinforce \
            else self.get_optimal_tdl_func()

    def get_optimal_det_policy_func(self) -> Callable[[S], A]:
        papt = self.get_optimal_stoch_policy_func()

        def opt_det_pol_func(s: S) -> A:
            return tuple(np.mean(
                papt(s)(self.num_action_samples),
                axis=0
            ))

        return opt_det_pol_func