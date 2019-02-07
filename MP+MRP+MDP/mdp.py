import numpy as np
from typing import Mapping, Set, Sequence, List, Tuple
from mrp.py import MRP
from mp_funcs.py import S, A, SAf, SASf, get_states, verify_mdp


# Markov Decision Process
# Add on data structures based on MRP
class MDP(MRP):

    # Initialization of MDP 
    # Input: transition: dictionary of dictionary of float
    #        s_r: dictionary of dictionary of dictionary of float
    #        gamma (discounted rate): float 
    def __init__(self, transition:SASf, s_r: SAf, gamma: float) -> None:
        if verify_mdp(transition):
            self.transitions = transition
            self.states: List[S] = get_states(transition)
            self.s_r = s_r
            self.sink_states = self.get_sink_states()
            self.nt_states_list = self.get_nt_states_list()
            self.reward = np.array([s_r[i] for i in self.nt_states_list])
            self.transition_matrix: Mapping[S,np.ndarray] = self.get_trans_matrix()
            self.gamma = gamma
        else:
            raise ValueError
    
    
    # Get sink state
    def get_sink_states(self) -> Set[S]:

        return {k for k, v in self.transitions.items() if
                all(len(v1) == 1 and k in v1.keys() for _, v1 in v.items())
                }
        
    
    # Get transition matrix, where it is a dictionary of s of dictionary: s': action
    def get_trans_matrix(self) -> Mapping[S,np.ndarray]:
        dic = {}

        for i in range(len(self.states)):
            s_prime = self.transitions[states[i]]
            matrix = np.zeros((len(s_prime.keys()),len(self.states)))
            for j, (s, v) in enumerate(s_prime.items()):
                for k, l in enumerate(v.keys()):
                    matrix[j][k] = v[l]
            dic[states[i]] = matrix
        
        return dic
    
    
    # Get MRP with the given policy
    # Policy is a dictionary of dictionary
    def get_mrp(self, policy: Mapping[S,Mapping[A, float]]) -> MRP:
        transition = {}
        rewards = {}
        for s in self.states:

            transition[s] = {}
            reward[s] = 0
            s_p = policy[s]
            s_probability = self.transitions[s]
            s_reward = self.s_r[s]

            # with policy of actions
            for a, p in s_p.items():
                reward[s] += s_reward[a]*p

                for state in s_probability[a].keys():

                    if state in transition[s].keys():
                        transition[s][state] += s_probability[a][state]*p
                    else:
                        transition[s][state] = pro[a][state]*p
                    
        return MRP(transition, reward, self.gamma)
    
    
    # To find best policy pi            
    def iter_val(self) -> Tuple[Mapping[S,Mapping[A, float]], Mapping[S, float]]:
        value = {s: 0 for s in self.states}
        new_v = value
        count = 0
        policy = {}

        while count <= 100 and thres>1e-8:
            count = count+1

            for s in self.states:
                sa_q = {}
                for a in self.transitions[s].keys():
                    trans = self.transitions[s][a]
                    sa_q[a] = self.s_r[s][a]+self.gamma*sum([trans[s_prime]*value[s_prime] for s_prime in trans.keys()])
                new_val[s] = (max(q_sa.items(), key=lambda k: k[1]))[1]
                policy[s] = {max(q_sa.items(), key=lambda k: k[1])[0]: 1}
            thres = max([np.abs(new_v[s] - value[s]) for s in self.states])

            # update value
            value = new_val.copy()
        
        if thres >1e-8:
            print("Fail to converge")            
            
        return policy, value

    # Get policy evaluation
    def eval_pol(self, policy: Mapping[S,Mapping[A, float]]) -> Mapping[S, float]:
        return {state: self.get_mrp(policy).get_val_func()[i] for i, s in enumerate(self.nt_states_list)}

    # Policy iteration 
    def iter_pol(self, policy: Mapping[S,Mapping[A, float]]) -> Tuple[Mapping[S,Mapping[A, float]], Mapping[S, float]]:
        policy = None
        new_p = policy
        count = 0

        while count<=100 and policy != new_p:

            # update policy and value function
            count+=1
            policy = new_p
            value = self.val_pol(policy)

            # start from the terminal states to backward states
            for s in self.sink_states:
                value[s] = [s for s in self.s_r[s].values()][0]
            new_policy = {}

            # start from the rest states which are non-terminal
            for s in self.nt_states_list:
                sa_r = self.s_r[s]
                trans = self.transitions[s]
                a_v = {}

                # using the definition 
                for a in trans.keys():
                    a_v[a] = self.gamma*sum([trans[a][s_prime]*value[s_prime] for s_prime in trans[a].keys()])
                sa_q = {a: sa_r[a] + a_v[ac] for ac in trans.keys()}       
                new_p[s] = {max(sa_q.items(),key=lambda k:k[1])[0]:1}

            # update sink state
            for s in self.sink_states:
                new_p[s] = policy[s]

        
        if pol_new != pol:
            print("Fail to converge")
        
        return policy, value
    
if __name__ == '__main__':
    data = {
        1: {
            'a': {1: 0.2, 2: 0.6, 3: 0.2},
            'b': {1: 0.6, 2: 0.3, 3: 0.1},
            'c': {1: 0.1, 2: 0.2, 3: 0.7}
        },
        2: {
            'a': {1: 0.1, 2: 0.6, 3: 0.3},
            'c': {1: 0.6, 2: 0.2, 3: 0.2}
        },
        3: {
            'b': {3: 1.0}
        }
    }

    reward = {
        1: {
            'a': 7.0,
            'b': -2.0,
            'c': 10.0
        },
        2: {
            'a': 1.0,
            'c': -1.2
        },
        3: {
            'b':  0.0
        }
    }

    policy = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }

    mdp_obj = MDP(data, reward, policy, gamma = 0.9)

    print('States:', mdp_obj.get_states(), '\n')
    print('Transition Matrix:\n', mrp_obj.get_trans_matrix, '\n')
    print('MRP Value Function:\n', mrp_obj.get_value_func())

    mrp_obj = mdp_obj.get_mrp(policy_data)
    print("MRP with policy: \n")
    print('MRP transition: \n', mrpobj.transitions)
    print('MRP State and Reward: \n' mrp_obj.s_r)

