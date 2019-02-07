import numpy as np
from mp.py import MP
from mp_funcs.py import SSf, S
from typing import Mapping, Sequence, Set

# Markov Reward Process
# Add on data structures based on MP
class MRP(MP):
    
    # Initialization of MRP 
    # Input: transition: dictionary of dictionary of float
    #        s_r: dictionary of dictionary of float
    #        gamma (discounted rate): float
    def __init__(self, transition:SSf, s_r: dict, gamma: float) -> None:
        super().__init__(transition)
        self.gamma = gamma
        self.s_r: = s_r
        self.terminal_states: Set[S] = self.get_terminal_states()
        self.nt_states_list: Sequence[S] = self.get_nt_states_list()
        self.reward: np.ndarray = np.array([s_r[s] for s in self.nt_states_list])
        self.trans_matrix: np.ndarray = self.get_trans_matrix()


    # Get states
    def get_nt_states_list(self) -> Sequence[S]:
        return [i for i in self.states if i not in self.terminal_states]
    
    # Get transition matrix
    def get_trans_matrix(self) -> np.ndarray:
        s = self.nt_states_list   
        matrix = np.zeros((len(s), len(s)))
        for i in range(len(s)):
            for s, d in self.transitions[self.nt_states_list[i]].items():
                if s in self.nt_states_list:
                    matrix[i, self.nt_states_list.index(s)] = d
        return matrix
    
    # Get value function
    # Only for non-terminal
    def get_value_func(self) -> np.ndarray:
        return np.linalg.inv(np.identity(len(self.nt_states_list)) - \
                self.gamma*self.trans_matrix).dot(self.reward)

if __name__ == '__main__':
    data = {
        1: {1: 0.6, 2: 0.3, 3: 0.1},
        2: {1: 0.1, 2: 0.2, 3: 0.7},
        3: {3: 1.0}
    }
    s_r = {1: 7.0, 2:10.0, 3:0.0}
    mrp_obj = MRP(data, s_r, 1.0)
    print('States:', mrp_obj.get_states(), '\n')
    print('Transition Matrix:\n', mrp_obj.get_trans_matrix, '\n')
    print('MRP Value Function:\n', mrp_obj.get_value_func())
        
        