import numpy as np
from typing import Mapping, List, Any, TypeVar
from scipy.stats import rv_discrete

# basic type of state and action
S = TypeVar('S')
A = TypeVar('A')

SSf = Mapping[S, Mapping[S, float]] # state
SAf = Mapping[S, Mapping[A, float]] # action 
SASf = Mapping[S, Mapping[A, Mapping[S, float]]] # policy

# Get all the states
def get_states(d: Mapping[S, Any]) -> List[S]:
    return list(d.keys())

# Get transition probability: P[s][s'] = P[S_t+1 = s'| S_t = s]
def get_transition(transition: SSf) -> np.ndarray:
    all_s = get_states(transition)
    matrix = np.zeros(len(all_s), len(all_s))

    for i in range(len(all_s)):
        for j in range(len(all_s)):
            matrix[i][j] = tr[all_s[i]][all_s[j]]

    return matrix

# Validate MP
# Check if sum of probability is 1
def verify_mp(mp_data: SSf) -> bool:
    val_seq = [i for i in mp_data.values()]
    r = [sum([value for value in v.values()]) for v in val_seq]

    for i in r:
        if np.abs(1-i) > 1e-8:
            return False
    return True

# Validate MDP
def verify_mdp(mp_data: SASf) -> bool:
    r = [sum(list(j.values())) for i in mp_data.values() for j in i.values()]

    for i in r:
        if np.abs(1-i) > 1e-8:
            return False
    return True

# get states for action
def get_states_action(mdp_data: Mapping[S, Mapping[A, Any]])\
        -> Mapping[S, Set[A]]:
    return {k: set(v.keys()) for k, v in mdp_data.items()}

def get_rv_gen_func_single(prob_dict: Mapping[S, float])\
        -> Callable[[], S]:
    outcomes, probabilities = zip(*prob_dict.items())
    rvd = rv_discrete(values=(range(len(outcomes)), probabilities))
    
    return lambda rvd=rvd, outcomes=outcomes: outcomes[rvd.rvs(size=1)[0]]

def get_epsilon_greedy_action(a_qv: Mapping[A, float], epsilon: float) -> float:
    a_opt, val_opt = max(a_qv.items(), key = lambda l:l[1])
    m = len(a_qv.keys())
    prob_dict = {a: epsilon/m + 1 - epsilon if a == a_opt else epsilon/m for a,v in a_qv.items()}
    gf = get_rv_gen_func_single(prob_dict)
    
    return gf()

def get_rv_gen_func(prob_dict: Mapping[S, float])\
        -> Callable[[int], Sequence[S]]:
    outcomes, probabilities = zip(*prob_dict.items())
    rvd = rv_discrete(values=(range(len(outcomes)), probabilities))
    # noinspection PyShadowingNames
    return lambda n, rvd=rvd, outcomes=outcomes: [outcomes[k]
                                                  for k in rvd.rvs(size=n)]
def get_state_reward_gen_func(
    prob_dict: Mapping[S, float],
    rew_dict: Mapping[S, float]
) -> Callable[[], Tuple[S, float]]:
    gf = get_rv_gen_func_single(prob_dict)
    
    def ret_func(gf=gf, rew_dict=rew_dict) -> Tuple[S, float]:
        state_outcome = gf()
        reward_outcome = rew_dict[state_outcome]
        return state_outcome, reward_outcome

    return ret_func

def get_state_reward_gen_dict(tr: SASf, rr: SASf) -> Mapping[S, Mapping[A, Callable[[], Tuple[S, float]]]]:
    return {s: {a: get_state_reward_gen_func(tr[s][a], rr[s][a])
                for a, _ in v.items()}
            for s, v in rr.items()}

def get_expected_action_value(a_qv: Mapping[A, float], epsilon: float) -> float:
    _, val_opt = max(a_qv.items(), key = lambda l:l[1])
    m = len(action_qv.keys())
    expected_a_val = sum([val*epsilon/m for val in a_qv.values()])+val_opt*(1-epsilon)
    return expected_a_val
               
