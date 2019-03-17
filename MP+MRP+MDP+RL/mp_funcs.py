import numpy as np
from typing import Mapping, List, Any, TypeVar

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