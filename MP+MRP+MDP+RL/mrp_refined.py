import numpy as np
from mp_funcs.py import SSf, get_states, get_matrix
from mp.py import MP
from mrp.py import MRP

# Refined MRP with another definition 
class MRP_refined(MP):
    def __init__(self, transition: SSf, rewards: SSf, gamma: float) -> None:
        super().__init__(transition)
        self.R = get_matrix(rewards)
        self.gamma = gamma

    # Change the definition in the form of MRP's definition    
    def refined(self) -> Tuple[SSf, dict, float]:
        probability = self.transition_matrix
        reward = list(np.diag(probability.dot((self.R).T)))
        return {j:reward[i] for i, j in enumerate(self.states)}

    # Get value function with the new definition according to MRP definition    
    def get_value_func(self) -> float:
        mrp_obj = MRP(self.transitions, self.refined(), self.gamma)
        return mrp_obj.get_value_func()
