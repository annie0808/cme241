from typing import Optional, Tuple, Sequence, Mapping, Callable
from mdp_rl_tabular import MDPForRLTabular
from TD_zero import TD0
from helper_func import S, SAf, get_rv_gen_func_single, get_expected_action_value, get_epsilon_greedy_action

class TD_control(TD0):
    
    def __init__(
            self,
            mdp_for_rl: MDPForRLTabular,
            epsilon: float,
            epsilon_half_life: float,
            learning_rate: float,
            learning_rate_decay: float,
            num_episodes: int,
            max_steps: int,
            choice: bool
    ) -> None:
        super().__init__(
                mdp_for_rl,
                epsilon,
                epsilon_half_life,
                learning_rate,
                learning_rate_decay,
                num_episodes,
                max_steps)
        self.choice = choice
    
    # get both for Sarsa and Q-learning depends on the choice
    def get_qv_func_dict(self) -> SAf:
        sa_dict = self.mdp_rep.state_action_dict
        qf_dict = {s: {a: 0.0 for a in v} for s, v in sa_dict.items()}
        episodes = 0
        updates = 0
        
        while episodes < self.num_episodes:
            state = self.mdp_rep.init_state_gen()
            action = get_epsilon_greedy_action(qf_dict[state], self.epsilon)
            steps = 0
            terminate = False
            
            while not terminate:
                next_state, reward = self.mdp_rep.state_reward_gen_dict[state][action]()
                # Sarsa
                if self.choice == 0:
                    next_qv = get_expected_action_value(qf_dict[next_state], \
                                                        self.epsilon)
                    next_action = get_epsilon_greedy_action(qf_dict[next_state], \
                                                        self.epsilon)
                # Q-learning
                else:
                    next_qv = max(qf_dict[next_state][a] for a in
                                  qf_dict[next_state])
                    next_action, next_qv = max(qf_dict[next_state].items(), \
                                          key = lambda l:l[1])

                qf_dict[state][action] += self.learning_rate *(updates / self.learning_rate_decay + 1) ** -0.5 *\
                    (reward + self.mdp_rep.gamma * next_qv -
                     qf_dict[state][action])
                updates += 1
                steps += 1
                terminate = steps >= self.max_steps or \
                    state in self.mdp_rep.terminal_states
                state = next_state
                action = next_action
                
            episodes += 1
             
        return qf_dict