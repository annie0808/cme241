import numpy as np
from typing import Sequence,Tuple
from mdp_rl_fa import MDPRepForRLFA
from helper_func import SAf, get_nt_return_eval_steps, S, A, get_returns_from_rewards_terminating
from policy_grad_td import PolicyGradient
from td_fa import TD0
from betaD import BetaDistribution

def state_initialization():
    t = 0
    w_t= 1.0
    return t, w_t

def CRRA(x: float, gamma=gamma) -> float:
    g = 1.0 - gamma
    if g !=0:
        u = x**g/g
    # avoid divide by zero
    else:
        u = np.log(x)
    return u

# return is gaussian distribution
def ret_gen_func(s: int,mean=mean,sigma=sigma) -> np.ndarray:
    return np.vstack((np.random.normal(loc=mean,scale=sigma,size=s),))

# get the consumption and allocation from action and get the score function from mu and nu
def score_func(action:Sequence[float], params: Sequence[float]) -> Sequence[float]:
        alpha_1, beta_1, alpha_2, beta_2 = params
        score_1, score_1_n = BetaDistribution(alpha_1, beta_1).get_mu_nu_scores(action[0])
        score_2, score_2_n = BetaDistribution(alpha_2, nu2).get_mu_nu_scores(action[1])
        return (score_1, score_1_n, score_2, score_2_n)

# generate samples of action,which includes consumption and allocation
def gen_sample_a(params: Sequence[float],ls: int) -> Sequence[float]:
        alpha_1, beta_1, alpha_2, beta_2 = params
        consumption = BetaDistribution(alpha_1, beta_1).get_samples(ls)
        allocation= BetaDistribution(alpha_1, beta_1).get_samples(ls)
        return [tuple(x) for x in np.vstack([consumption]+[allocation])]

# generate state and reward   
def gen_s_r(state: [int,float], action:[float,float], threshold: float, ls: int, t: int, T:int, discount_rate: float) -> Sequence[Tuple[[int,float], float]]:
    t, W_t = state
    consumption = action[0]
    allocation = action[1]
    risky_frac = 1.0 - allocation
    allocated = np.insert(np.array(risky_frac), 0, allocation)
    ret = np.hstack((np.full((ls, 1), allocation),ret_gen_func(ls)))
    terminate = [W_t*(1-consumption)*max(threshold, allocated.dot(np.exp(sample))) for sample in ret]

    s_r=[((t+1,s),np.exp(-discount_rate)*CRRA(x) if t==T-1 else 0) for x in terminate]

    return s_r

if __name__ == '__main__':
    # continue with the policy gradient
    # initialize parameters
    r=0.03
    thres = 10e-5
    mu = 0.08
    sig = 0.03
    discount_rate=0.04
    T=5
    gamma=0.2
    mdp_rl_pg=MDPRepForRLFA(np.exp(discount_rate), lambda: init_state, lambda s, a: state_reward_gen(s,a,1)[0], lambda s:s[0]==T-1)
    fa_spec=FuncApproxSpec([],[])
    policy_spec = [fa_spec]*4
    rein = True
    num_batches_val = 1000
    batch_size_val = 10
    num_action_samples_val = 100
    max_steps_val = 100
    num_state_samples_val = 400
    num_next_state_samples_val = 30
    actor_lambda_val = 0.95
    critic_lambda_val = 0.95

    policy_gradient = PolicyGradient(model_for_pg,rein,num_state_samples_val,
                num_batches_val,num_action_samples_val,
                T,actor_lambda_val,critic_lambda_val,
                score_func,sample_actions_gen,
                critic_spec=fa_spec,policy_spec)
