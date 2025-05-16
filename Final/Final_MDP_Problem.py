#Andrew Aquino Final MDP Problem 

import numpy as np

# these are the parameters for the MDP Problem
#Used these arbitrary values for alpha, beta and the rewards 
gamma = 0.9
alpha = 0.6
beta = 0.4
r_search = 1.0
r_wait = 0.5

#these are the states and the possible actions the robot can take 
states = ['high', 'low']
actions = {
    'high': ['search', 'wait'],
    'low': ['search', 'wait', 'recharge']
}

# here I haved defined the transition model for each state based on picture in the book 
transition_model = {
    'high': { 'search': [ ('high', alpha, r_search), ('low', 1 - alpha, r_search)], 'wait': [ ('high', 1.0, r_wait) ] },
    'low': { 'search': [ ('low', beta, r_search), ('high', 1 - beta, -3.0) ], 'wait': [ ('low', 1.0, r_wait) ], 'recharge': [ ('high', 1.0, 0.0) ] }
}

#here I defineded the value iteration model with break points so the function doesn't geet stuck 
#we want the function to try all possible actions, and compute the expected value for each action 
#pick the action that gives the maximum value
def value_iteration(theta=1e-6, max_iterations=1000):
    V = {s: 0.0 for s in states}
    for _ in range(max_iterations):
        delta = 0
        new_V = V.copy() #copy is used to not change the original value
        for s in states:
            action_values = []
            for a in actions[s]:
                value = 0
                for next_state, prob, reward in transition_model[s][a]:
                    value += prob * (reward + gamma * V[next_state])
                action_values.append(value)
            new_V[s] = max(action_values)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < theta:
            break
    return V

#this is where I defined the policy iteration, again define max iterations to limit loop
#I am assuming we are following a fixed policy
#after checking the current policy we check if another action returns a higher value, if so update the policy 
def policy_iteration(max_iterations=1000):
    policy = {s: actions[s][0] for s in states}
    V = {s: 0.0 for s in states}

    for _ in range(max_iterations):
        while True:
            delta = 0
            new_V = V.copy()
            for s in states:
                a = policy[s]
                value = sum(prob * (reward + gamma * V[next_state])
                            for next_state, prob, reward in transition_model[s][a])
                new_V[s] = value
                delta = max(delta, abs(new_V[s] - V[s]))
            V = new_V
            if delta < 1e-6:
                break

        # this is where we apply the policy improvment, if policy does not change we stop 
        policy_stable = True
        for s in states:
            old_action = policy[s]
            action_values = {}
            for a in actions[s]:
                value = sum(prob * (reward + gamma * V[next_state])
                            for next_state, prob, reward in transition_model[s][a])
                action_values[a] = value
            best_action = max(action_values, key=action_values.get)
            policy[s] = best_action
            if old_action != best_action:
                policy_stable = False
        if policy_stable:
            break
    return V, policy


value_result = value_iteration()
policy_result, policy_action = policy_iteration()


print("Value Iteration Result:")
for s in states:
    print(f"V*({s}) = {value_result[s]:.4f}")

print("\nPolicy Iteration Result:")
for s in states:
    print(f"V*({s}) = {policy_result[s]:.4f}, best action: {policy_action[s]}")
