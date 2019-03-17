from mdp_rl_fa import MDPRepForRLFA
from helper_func import SAf, get_nt_return_eval_steps, S, A, get_returns_from_rewards_terminating

class TD0():
    def __init__(
            self,
            mdp_rep_for_rl: MDPRepForRLFA,
            exploring_start: bool,
            algorithm: TDAlgorithm,
            softmax: bool,
            epsilon: float,
            epsilon_half_life: float,
            num_episodes: int,
            max_steps: int,
            fa_spec: FuncApproxSpec
    ) -> None:

        self.mdp_rep_for_rl=mdp_rep_for_rl,
        self.exploring_start=exploring_start,
        self.softmax=softmax,
        self.epsilon=epsilon,
        self.epsilon_half_life=epsilon_half_life,
        self.num_episodes=num_episodes,
        self.max_steps=max_steps,
        self.fa_spec=fa_spec
        self.algorithm: TDAlgorithm = algorithm

    def get_value_func_fa(self, polf: PolicyActDictType) -> VFType:
        episodes = 0

        while episodes < self.num_episodes:
            state = self.mdp_rep.init_state_gen()
            steps = 0
            terminate = False

            while not terminate:
                action = get_rv_gen_func_single(polf(state))()
                next_state, reward = self.mdp_rep.state_reward_gen_func(state, action)
                target = reward + self.mdp_rep.gamma *self.vf_fa.get_func_eval(next_state)
                self.vf_fa.update_params([state], [target])
                steps += 1
                terminate = steps >= self.max_steps or self.mdp_rep.terminal_state_func(state)
                state = next_state

            episodes += 1

        return self.vf_fa.get_func_eval