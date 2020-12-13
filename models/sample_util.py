import torch
import numpy as np

epsilon = 1e-8

# issues to be finished
# 1. here we assume 1 order is generated each time now
# 2. distinguish between pickup and money pair
# 3. where to limit the pickup 
# 4. should we generate eposide with tensor rather than numpy array?


# some sample params
sample_params = {
    "h": 64,
    "w": 64,
    "abs": 100.0,
    "num_delivers": 3,
    "gas_cost": -1,
    "reject_cost": -3,
    "min_reward": 5,
    "max_reward": 20,
    "num_obstacle": int(5),
    "obstacle_ratio_h": 0.5,
    "obstacle_ratio_w": 0.5
}


# utility functions
# draw obstacles
def addObstacle(m, ys, xs, params):
    h, w = m.shape
    mask = np.zeros_like(m)
    for y,x in zip(ys, xs):
        poly_h = np.random.randint(params["obstacle_ratio_h"]*h)+1
        poly_w = np.random.randint(params["obstacle_ratio_w"]*w)+1
        h1 = int(max(0, y-0.5*poly_h))
        h2 = int(min(h-1, y+0.5*poly_h))
        w1 = int(max(0, x-0.5*poly_w))
        w2 = int(min(w-1, x+0.5*poly_w))
        m[h1:h2, w1:w2] = np.minimum(m[h1:h2, w1:w2], -params["abs"]*np.random.uniform())
        mask[h1:h2, w1:w2] = 1
    return m, mask


# this function returns a matrix of map with random scores at each tile
def generateMap(params=sample_params):
    # initalize the map
    h, w = params['h'], params['w']
    m = np.zeros([h, w], dtype=np.float32)
    m = m + params["gas_cost"]

    # add in number of obstacle
    ys = np.random.randint(0, h, size=(params["num_obstacle"]))
    xs = np.random.randint(0, w, size=(params["num_obstacle"]))
    # obs_mask is a mask indicate the position of obstacles, 
    # where 1 indicate the existence of obstacles
    m, obs_mask = addObstacle(m, ys, xs, params)

    # return normalized map
    return m/params["abs"], obs_mask

# generate the map for pickup
def generatePickupMap(params):
    h, w = params['h'], params['w']
    m = np.zeros((h, w), dtype=np.float32)
    y = np.random.randint(0, h)
    x = np.random.randint(0, w)
    # q1 here, what reward to assign to the pickup location
    m[y, x] = 1
    return m/params["abs"]

# generate the map for money delievery
def generateMoneyMap(params):
    h, w = params['h'], params['w']
    m = np.zeros((h, w), dtype=np.float32)
    y = np.random.randint(0, h)
    x = np.random.randint(0, w)
    m[y, x] = np.random.randint(params["min_reward"], params["max_reward"])
    return m/params["abs"]

# generate position map
def generatePosMap(mask, params):
    h, w = params['h'], params['w']
    m = np.zeros((h, w), dtype=np.float32)
    while(1):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        if mask[y, x] != 1:
            m[y, x] = 1
            break
    return m


# ------------------------ map related finished -----------------------------------
# ------------------------ reward related start -----------------------------------
# this function returns the reward and state in the next step based on
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def oneStepMove(pre_pos_map, people_action):
    h, w = pre_pos_map.shape
    pre_pos_ind = np.array(np.where(pre_pos_map==1)).reshape([2])
    new_pos_ind = pre_pos_ind
    # cast people action to prob 
    people_action = softmax(people_action.numpy().reshape([4,1]))
    prob = np.random.uniform(0,1)

    if prob <= people_action[0]: # move up
        new_pos_ind[0] = np.max([new_pos_ind[0]-1, 0])
    elif prob <= np.sum(people_action[0:2]): # move down
        new_pos_ind[0] = np.min([new_pos_ind[0]+1, h-1])
    elif prob <= np.sum(people_action[0:3]): # move left
        new_pos_ind[1] = np.max([new_pos_ind[1]-1, 0])
    else: # move right
        new_pos_ind[1] = np.min([new_pos_ind[1]+1, w-1])
    new_pos_map = np.zeros_like(pre_pos_map)
    new_pos_map[tuple(new_pos_ind)] = 1
    return new_pos_map, new_pos_ind

def getNextStateReward(last_state, pickup_controls, people_actions, params):
    h, w = params['h'], params['w']
    # we generate the initial state
    # the channel order of states are
    # map 0
    # pickup map 1
    # money map 2
    # iternate: position of deliever, pickup position of deliever, money position of deliever 3*(i+1)
    last_state = last_state.numpy()
    states = np.zeros((1, 3+3*params["num_delivers"], h, w), np.float32)
    states[0, 0] = last_state[0, 0]     # map doesn't change
    states[0, 1] = generatePickupMap(sample_params).reshape([h, w]) # generate new pickup
    states[0, 2] = generateMoneyMap(sample_params).reshape([h, w]) # generate new money
    rewards = 0

    # we need to determine pickup controls
    control = params["num_delivers"]
    pickup_controls = softmax(pickup_controls.numpy().reshape([params["num_delivers"]+1, 1]))
    prob = np.random.uniform(0, 1)
    if prob<pickup_controls[0]:
        control = 0
    for i in range(len(pickup_controls)):
        if prob < np.sum(pickup_controls[:params["num_delivers"]-i+1]):
            continue
        else:
            control = params["num_delivers"]-i+1
            break
    # print("the control is")
    # print(control)
    # print(prob)
    # print(pickup_controls)

    for i in range(params["num_delivers"]):
        # update pos map
        new_pos_map, new_pos_ind = oneStepMove(last_state[0, 3*(i+1)], people_actions[i])
        # print(new_pos_ind)
        states[0, 3*(i+1)] = new_pos_map

        # update pick up m
        pre_pickup_m = last_state[0, 1+3*(i+1)]
        if control == i:
            last_pickup_m = last_state[0, 1]
            pre_pickup_m = pre_pickup_m + last_pickup_m
            pre_pickup_m[new_pos_ind] = 0
            states[0, 1+3*(i+1)] = pre_pickup_m
        if pre_pickup_m[tuple(new_pos_ind)] != 0:
            rewards = rewards + pre_pickup_m[tuple(new_pos_ind)]
            # print(pre_pickup_m[tuple(new_pos_ind)])
            pre_pickup_m[tuple(new_pos_ind)] = 0
        states[0, 1+3*(i+1)] = pre_pickup_m

        # update money m
        pre_money_m = last_state[0, 2+3*(i+1)]
        if control == i:
            pre_money_m = last_state[0, 2+3*(i+1)]
            last_money_m = last_state[0, 2]
            pre_money_m = pre_money_m + last_money_m
        if pre_money_m[tuple(new_pos_ind)] != 0:
            rewards = rewards + pre_money_m[tuple(new_pos_ind)]
            # print(pre_money_m[tuple(new_pos_ind)])
            pre_money_m[tuple(new_pos_ind)] = 0
        states[0, 2+3*(i+1)] = pre_money_m
        
        # calculate rewards
        rewards = rewards + states[0, 0, new_pos_ind[0], new_pos_ind[1]]
        # print(states[0, 0, new_pos_ind[0], new_pos_ind[1]])

    # when the order is reject
    if control == params["num_delivers"]:
        rewards = rewards + params["reject_cost"]
    return states, rewards.reshape([1, 1])


# this function returns the whole eposide of samples, given the 
def SampleEpisode(model, params=sample_params, duration=250):
    eposide = []
    # we generate the initial state
    # the channel order of states are
    # map
    # pickup map
    # money map
    # iternate: position of deliever, pickup position of deliever, money position of deliever
    states = []
    h, w = params['h'], params['w']
    m, obs_mask = generateMap(sample_params)
    m = m.reshape([1, h, w])
    states.append(m)
    pickup_m = generatePickupMap(sample_params).reshape([1, h, w])
    states.append(pickup_m)
    money_m = generateMoneyMap(sample_params).reshape([1, h, w])
    states.append(money_m)

    for i in range(params["num_delivers"]):
        pos_map = generatePosMap(obs_mask, params).reshape([1, h, w])
        pre_pickup_m = np.zeros([1, h, w], dtype=np.float32)
        pre_money_m = np.zeros([1, h, w], dtype=np.float32)
        states.append(pos_map)
        states.append(pre_pickup_m)
        states.append(pre_money_m)

    states = np.concatenate(states, axis=0).reshape([1, 3+3*params["num_delivers"], h, w])
    all_states = []
    # states has size duration+1, while all other ones has size duration
    all_states.append(states)
    all_rewards = []
    # all_pickup_controls = []
    # all_people_actions = []

    for i in range(duration):
        # convert states from np into tensor
        last_state = torch.from_numpy(all_states[-1])
        last_state = last_state.cuda()
        pickup_controls, people_actions = model.actor(last_state)
        # pickup_controls = np.random.randn(1, 4)
        # pickup_controls = torch.from_numpy(pickup_controls)
        # people_actions = np.random.randn(4, 1, 4)
        # people_actions = torch.from_numpy(people_actions)

        # get the rewards and next state in np structure
        states, rewards = getNextStateReward(last_state, pickup_controls, people_actions, params)
        print('cut:{reward}'.format(reward=rewards[0]))
        all_states.append(states)
        all_rewards.append(rewards)
        # all_pickup_controls.append(pickup_controls.numpy())
        # all_people_actions.append(people_actions.numpy())
        

    # construct the eposide
    eposide = {}
    eposide['states'] = torch.from_numpy(np.concatenate(all_states, axis=0))
    eposide['rewards'] = torch.from_numpy(np.concatenate(all_rewards, axis=0))
    # eposide['pickup_controls'] = torch.from_numpy(np.concatenate(all_pickup_controls, axis=0))
    # eposide['people_actions'] = torch.from_numpy(np.concatenate(all_people_actions, axis=0))
    # eposide['donws'] = torch.from_numpy(np.concatenate(all_dones, axis=0))
    return eposide

# SampleEpisode(0)