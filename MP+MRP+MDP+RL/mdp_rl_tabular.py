from typing import Mapping, Set, Callable, Tuple, Generic
from helper_funcs import S, A
import random

Type1 = Mapping[S, Mapping[A, Callable[[], Tuple[S, float]]]]

class MDPForRLTabular(Generic[S, A]):
    
    def __init__(
        self, 
        state_action_dict: Mapping[S, Set[A]],
                 terminal_states: Set[S], state_reward_gen_dict: Type1,
                 gamma: float) -> None:
        self.state_action_dict: Mapping[S, Set[A]] = state_action_dict
        self.terminal_states: Set[S] = terminal_states
        self.state_reward_gen_dict: Type1 = state_reward_gen_dict
        self.gamma = gamma
        
    def init_state_gen(self) -> S:
        return [s for s in self.state_action_dict.keys()][random.randint(0,len(self.state_action_dict.keys())-1)]
               
    def init_state_action_gen(self) -> Tuple[S, A]:
        state = self.init_state_gen()
        A = [a for a in self.state_action_dict[state]][random.randint(0,len(self.state_action_dict[state])-1)]
        return state, A