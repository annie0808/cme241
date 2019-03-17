from typing import Optional, Tuple, Sequence, Mapping, Callable
from mdp_rl_tabular import MDPForRLTabular
from helper_funcs import S, SAf, get_rv_gen_func_single
from mdp_refined import MDPRefined

class TD0():

    def __init__(
            self,
            mdp_rep_for_rl: MDPRLTabular,
            epsilon: float,
            epsilon_half_life: float,
            learning_rate: float,
            learning_rate_decay: float,
            num_episodes: int,
            max_steps: int
    ) -> None:

        self.mdp_rep: MDPRForLTabular=mdp_rep_for_rl
        self.epsilon=epsilon
        self.epsilon_half_life=epsilon_half_life
        self.learning_rate=learning_rate
        self.learning_rate_decay=earning_rate_decay
        self.num_episodes=num_episodes
        self.max_steps=max_steps
        
    
    def get_value_func_dict(self, pol: SAf) -> Mapping[S, float]:
        sa_dict = self.mdp_rep.state_action_dict
        vf_dict = {s: 0.0 for s in sa_dict.keys()}
        action_gen_dic = {s: get_rv_gen_func_single(pol[s]) \
                      for s in self.mdp_rep.state_action_dict.keys()}
        episodes = 0
        updates = 0
        
        while episodes < self.num_episodes:
             state = self.mdp_rep.init_state_gen()
             steps = 0
             terminate = False
             
             while not terminate:
                 action = action_gen_dic[state]()
                 next_state, reward = \
                    self.mdp_rep.state_reward_gen_dict[state][action]()
                 vf_dict[state] += self.learning_rate * \
                     (updates / self.learning_rate_decay + 1) ** -0.5 *\
                    (reward + self.mdp_rep.gamma * vf_dict[next_state] -
                     vf_dict[state])
                 updates += 1
                 steps += 1
                 terminate = steps >= self.max_steps or \
                    state in self.mdp_rep.terminal_states
                 state = next_state
            
             episodes += 1
             
        return vf_dict

if __name__ == '__main__':
    mdp_refined_data = {
        1: {
            'a': {1: (0.3, 9.2), 2: (0.6, 4.5), 3: (0.1, 5.0)},
            'b': {2: (0.3, -0.5), 3: (0.7, 2.6)},
            'c': {1: (0.2, 4.8), 2: (0.4, -4.9), 3: (0.4, 0.0)}
        },
        2: {
            'a': {1: (0.3, 9.8), 2: (0.6, 6.7), 3: (0.1, 1.8)},
            'c': {1: (0.2, 4.8), 2: (0.4, 9.2), 3: (0.4, -8.2)}
        },
        3: {
            'a': {3: (1.0, 0.0)},
            'b': {3: (1.0, 0.0)}
        }
    }
    gamma_val = 1.0
    mdp_ref_obj1 = MDPRefined(mdp_refined_data, gamma_val)
    mdp_rep_obj = mdp_ref_obj1.get_mdp_rep_for_rl_tabular()

    exploring_start_val = False
    algorithm_type = TDAlgorithm.ExpectedSARSA
    softmax_flag = False
    epsilon_val = 0.1
    epsilon_half_life_val = 1000
    learning_rate_val = 0.1
    learning_rate_decay_val = 1e6
    episodes_limit = 10000
    max_steps_val = 1000

    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    pol_obj = Policy(policy_data)

    this_qf_dict = sarsa_obj.get_act_value_func_dict(pol_obj)
    print(this_qf_dict)
    this_vf_dict = sarsa_obj.get_value_func_dict(pol_obj)
    print(this_vf_dict)