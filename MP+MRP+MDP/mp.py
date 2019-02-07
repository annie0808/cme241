import numpy as np
from scipy.linalg import eig
from typing import Mapping, Set, Generic
from mp_funcs.py import get_state, get_transition, verify_mp, S, SSf

# Markov Process
class MP:
    # Initialization of MP
    # Input: state: set, 
    #        transition probability matrix: dictionary of dictionary of float
    def __init__(self, transitions: SSf) -> None:
        if self.verify_mp(transitions):
            self.transitions = transitions
            self.states = self.get_states()
            self.transition_matrix = self.get_transition(transitions)
        else:
            raise ValueError

    # Get states     
    def get_states(self) -> Set:
        return set(self.states)

    # Get transition matrix    
    def get_matrix(self) -> np.array:
        return self.transition_matrix

    # Get terminal states
    def get_sink_states(self) -> Set[S]:
        return {k for k, v in self.transitions.items()
                if len(v) == 1 and k in v.keys()}

    # Get stationary distribution    
    def get_stationary_distribution(self) -> Mapping[S, float]:
        eig_vals, eig_vecs = eig(self.transitions.T)
        stat = np.array(
            eig_vecs[:, np.where(np.abs(eig_vals - 1.) < 1e-8)[0][0]].flat
        ).astype(float)
        stat_norm = stat / sum(stat)
        return {s: stat_norm[i] for i, s in enumerate(self.states)}
    

if __name__ == '__main__':
    transitions = {
        1: {1: 0.1, 2: 0.6, 3: 0.1, 4: 0.2},
        2: {1: 0.25, 2: 0.22, 3: 0.24, 4: 0.29},
        3: {1: 0.7, 2: 0.3},
        4: {1: 0.3, 2: 0.5, 3: 0.2}
    }
    
    mp_obj = MP(transitions)
    print('States:', mp_obj.states, '\n')
    print('Transition Matrix:\n', mp_obj.transition_matrix, '\n')
    print('Stationary distribution:\n', mp_obj.get_stationary_distribution())
