from mdp_rl_fa import MDPRepForRLFA
from helper_func import SAf, get_nt_return_eval_steps, S, A, get_returns_from_rewards_terminating

class MonteCarlo():

    def __init__(
        self,
        mdp_rep_for_rl: MDPRepForRLFA,
        epsilon: float,
        epsilon_half_life: float,
        num_episodes: int,
        max_steps: int,
        fa_spec: FuncApproxSpec
    ) -> None:

        self.mdp_rep = mdp_rep_for_rl
        self.epsilon = epsilon
        self.epsilon_half_life = epsilon_half_life
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.fa_spec = fa_spec
        self.nt_return_eval_steps = get_nt_return_eval_steps(
            max_steps,
            mdp_rep.gamma,
            1e-4
        )

    def get_mc_path(
        self,
        polf: PolicyActDictType,
        start_state: S,
        start_action: Optional[A] = None
    ) -> Sequence[Tuple[S, A, float]]:

        res = []
        state = start_state
        steps = 0
        terminate = False

        while not terminate:
            action = get_rv_gen_func_single(polf(state))() if (steps > 0 or start_action is None) else start_action
            next_state, reward = self.mdp_rep.state_reward_gen_func(state, action)
            res.append((state, action, reward))
            steps += 1
            terminate = steps >= self.max_steps or self.mdp_rep.terminal_state_func(state)
            state = next_state
        return res
    
    def get_value_func_fa(self, polf: PolicyActDictType) -> VFType:
        episodes = 0

        while episodes < self.num_episodes:
            start_state = self.mdp_rep.init_state_gen()
            mc_path = self.get_mc_path(
                polf,
                start_state,
                start_action=None
            )

            rew_arr = np.array([x for _, _, x in mc_path])
            if self.mdp_rep.terminal_state_func(mc_path[-1][0]):
                returns = get_returns_from_rewards_terminating(
                    rew_arr,
                    self.mdp_rep.gamma
                )
            else:
                returns = get_returns_from_rewards_non_terminating(
                    rew_arr,
                    self.mdp_rep.gamma,
                    self.nt_return_eval_steps
                )

            sgd_pts = [(mc_path[i][0], r) for i, r in enumerate(returns)]
            self.vf_fa.update_params(*zip(*sgd_pts))

            episodes += 1

        return self.vf_fa.get_func_eval