import torch
import numpy as np

epsilon = 1e-8

# issues to be finished
# 1. here we assume 1 order is generated each time now, if more orders at each time, need to modify
# OrderPack
# 2. distinguish between pickup and money pair
# 3. where to limit the pickup 
# 4. should we generate eposide with tensor rather than numpy array?


# some sample params
sample_params = {
    "h": 16,
    "w": 16,
    "abs": 100.0,
    "num_delivers": 3,
    "gas_cost": -1,
    "reject_cost": -30,
    "min_reward": 1000,
    "max_reward": 2000,
    "pickup_reward": 200,
    "num_obstacle": int(5),
    "obstacle_min_ratio": 0.5,
    "obstacle_ratio_h": 0.5,
    "obstacle_ratio_w": 0.5,
    "max_order": 5
}


# utility functions
# draw obstacles
def addObstacle(m, ys, xs, params):
    h, w = m.shape
    mask = np.zeros_like(m)
    for y, x in zip(ys, xs):
        poly_h = np.random.randint(params["obstacle_ratio_h"] * h) + 1
        poly_w = np.random.randint(params["obstacle_ratio_w"] * w) + 1
        h1 = int(max(0, y - 0.5 * poly_h))
        h2 = int(min(h - 1, y + 0.5 * poly_h))
        w1 = int(max(0, x - 0.5 * poly_w))
        w2 = int(min(w - 1, x + 0.5 * poly_w))
        m[h1:h2, w1:w2] = np.minimum(m[h1:h2, w1:w2], -params["abs"] * np.random.uniform(
            low=params["obstacle_min_ratio"], high=1))
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
    return m / params["abs"], obs_mask


# generate the map for pickup
def generatePickupMap(params):
    h, w = params['h'], params['w']
    m = np.zeros((h, w), dtype=np.float32)
    y = np.random.randint(0, h)
    x = np.random.randint(0, w)
    # q1 here, what reward to assign to the pickup location
    m[y, x] = params["pickup_reward"]
    return m / params["abs"]


# generate the map for money delievery
def generateMoneyMap(params):
    h, w = params['h'], params['w']
    m = np.zeros((h, w), dtype=np.float32)
    y = np.random.randint(0, h)
    x = np.random.randint(0, w)
    m[y, x] = np.random.randint(params["min_reward"], params["max_reward"])
    return m / params["abs"]


# generate position map
def generatePosMap(mask, params):
    h, w = params['h'], params['w']
    m = np.zeros((h, w), dtype=np.float32)
    while (1):
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
    pre_pos_ind = np.array(np.where(pre_pos_map == 1)).reshape([2])
    new_pos_ind = pre_pos_ind
    # cast people action to prob 
    np_people_action = softmax(people_action.detach().cpu().numpy().reshape([4, 1]))
    prob = np.random.uniform(0, 1)
    out_people_action = np.zeros((1, 1, 4), dtype=np.int32)

    if prob <= np_people_action[0]:  # move up
        new_pos_ind[0] = np.max([new_pos_ind[0] - 1, 0])
        out_people_action[0, 0, 0] = 1
    elif prob <= np.sum(np_people_action[0:2]):  # move down
        new_pos_ind[0] = np.min([new_pos_ind[0] + 1, h - 1])
        out_people_action[0, 0, 1] = 1
    elif prob <= np.sum(np_people_action[0:3]):  # move left
        new_pos_ind[1] = np.max([new_pos_ind[1] - 1, 0])
        out_people_action[0, 0, 2] = 1
    else:  # move right
        new_pos_ind[1] = np.min([new_pos_ind[1] + 1, w - 1])
        out_people_action[0, 0, 3] = 1
    new_pos_map = np.zeros_like(pre_pos_map)
    new_pos_map[tuple(new_pos_ind)] = 1
    return new_pos_map, new_pos_ind, out_people_action


def getNextStateReward(last_state, pickup_controls, people_actions, order_recorder, params):
    h, w = params['h'], params['w']
    # we generate the initial state
    # the channel order of states are
    # map 0
    # pickup map 1
    # money map 2
    # iternate: position of deliever, pickup position of deliever, money position of deliever 3*(i+1)
    last_state = last_state.cpu().numpy()
    states = np.zeros((1, 3 + 3 * params["num_delivers"], h, w), np.float32)
    states[0, 0] = last_state[0, 0]  # map doesn't change
    states[0, 1] = generatePickupMap(sample_params).reshape([h, w])  # generate new pickup
    states[0, 2] = generateMoneyMap(sample_params).reshape([h, w])  # generate new money
    rewards = 0

    # we need to determine pickup controls
    control = params["num_delivers"]
    np_pickup_controls = softmax(pickup_controls.detach().cpu().numpy().reshape([params["num_delivers"] + 1, 1]))
    prob = np.random.uniform(0, 1)
    if prob < np_pickup_controls[0]:
        control = 0
    for i in range(len(np_pickup_controls)):
        if prob < np.sum(np_pickup_controls[:params["num_delivers"] - i + 1]):
            continue
        else:
            control = params["num_delivers"] - i + 1
            break
    # create control one hot
    out_pickup_controls = np.zeros([1, params["num_delivers"] + 1], dtype=np.int32)

    # create people actions
    out_people_actions = []

    # print("the control is")
    # print(control)
    # print(prob)
    # print(pickup_controls)

    for i in range(params["num_delivers"]):
        # update pos map
        new_pos_map, new_pos_ind, out_people_action = oneStepMove(last_state[0, 3 * (i + 1)], people_actions[i])
        out_people_actions.append(out_people_action)
        # print(new_pos_ind)
        states[0, 3 * (i + 1)] = new_pos_map

        # update pick up m
        picked = False
        pre_pickup_m = last_state[0, 1 + 3 * (i + 1)]
        if control == i and np.sum(pre_pickup_m) / params["pickup_reward"] < params["max_order"]:
            out_pickup_controls[0, control] = 1
            picked = True
            last_pickup_m = last_state[0, 1]
            last_money_m = last_state[0, 2]
            pre_pickup_m = pre_pickup_m + last_pickup_m
            states[0, 1 + 3 * (i + 1)] = pre_pickup_m
            # add new order to order pack
            order_ind = np.array(np.where(last_pickup_m > 0)).reshape([2])
            money_ind = np.array(np.where(last_money_m > 0)).reshape([2])
            order_recorder.add_order(i, order_ind, money_ind, np.sum(last_money_m))

        if pre_pickup_m[tuple(new_pos_ind)] > 0:
            pick_reward = order_recorder.pick_order(i, new_pos_ind)
            rewards = rewards + pick_reward
            pre_pickup_m[tuple(new_pos_ind)] = pre_pickup_m[tuple(new_pos_ind)] - pick_reward
            # print(pre_pickup_m[tuple(new_pos_ind)])
        states[0, 1 + 3 * (i + 1)] = pre_pickup_m

        # update money m
        pre_money_m = last_state[0, 2 + 3 * (i + 1)]
        if picked:  # means the order is accept
            pre_money_m = last_state[0, 2 + 3 * (i + 1)]
            last_money_m = last_state[0, 2]
            pre_money_m = pre_money_m + last_money_m
        if pre_money_m[tuple(new_pos_ind)] > 0:  # possible to finish an order
            money_reward = order_recorder.finish_order(i, new_pos_ind)
            rewards = rewards + money_reward
            # print(pre_money_m[tuple(new_pos_ind)])
            pre_money_m[tuple(new_pos_ind)] = pre_money_m[tuple(new_pos_ind)] - money_reward
        states[0, 2 + 3 * (i + 1)] = pre_money_m

        # calculate gas cost rewards
        rewards = rewards + states[0, 0, new_pos_ind[0], new_pos_ind[1]]
        # print(states[0, 0, new_pos_ind[0], new_pos_ind[1]])

    # when the order is reject
    if control == params["num_delivers"]:
        rewards = rewards + params["reject_cost"]

    # concate people actions
    out_people_actions = np.concatenate(out_people_actions, axis=0)

    return states, rewards.reshape([1, 1]), order_recorder, out_pickup_controls, out_people_actions


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

    # here we create a order pack to record whether the food of a order has already been collected
    # h,w,reward
    order_recorder = OrderPack(params["num_delivers"], params["max_order"], params)

    for i in range(params["num_delivers"]):
        pos_map = generatePosMap(obs_mask, params).reshape([1, h, w])
        pre_pickup_m = np.zeros([1, h, w], dtype=np.float32)
        pre_money_m = np.zeros([1, h, w], dtype=np.float32)
        states.append(pos_map)
        states.append(pre_pickup_m)
        states.append(pre_money_m)

    states = np.concatenate(states, axis=0).reshape([1, 3 + 3 * params["num_delivers"], h, w])
    all_states = []
    # states has size duration+1, while all other ones has size duration
    all_states.append(states)
    all_rewards = []
    # duration, num_people+1
    all_pickup_controls = []
    # people number, 1, 4
    all_people_actions = []

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
        states, rewards, order_recorder, out_pickup_controls, out_people_actions = getNextStateReward(
            last_state, pickup_controls, people_actions, order_recorder, params)
        # print('cut:{reward}'.format(reward=rewards[0]))
        all_states.append(states)
        all_rewards.append(rewards)
        all_pickup_controls.append(out_pickup_controls)
        all_people_actions.append(out_people_actions)

    # construct the eposide
    eposide = {}
    eposide['states'] = torch.from_numpy(np.concatenate(all_states, axis=0))
    eposide['rewards'] = torch.from_numpy(np.concatenate(all_rewards, axis=0).astype(np.float32))
    eposide['pickup_controls'] = torch.from_numpy(np.concatenate(all_pickup_controls, axis=0))
    eposide['people_actions'] = torch.from_numpy(np.concatenate(all_people_actions, axis=1))
    # eposide['donws'] = torch.from_numpy(np.concatenate(all_dones, axis=0))
    return eposide


# here we define a class to manage order
class OrderPack(object):
    def __init__(self, num_delivers, max_order, params):
        # order_h,order_w, money_h, money_w, reward, picked
        # reward=0 indicates empty
        self.data = np.zeros([num_delivers, max_order, 6], dtype=np.float32)
        self.num_delivers = num_delivers
        self.max_order = max_order
        self.params = params
        return

    # we assume no order has reward smaller than 0
    def add_order(self, deliver_id, order_ind, money_ind, money):
        for i in range(self.max_order):
            if self.data[deliver_id, i, 4] == 0:
                self.data[deliver_id, i] = [
                    order_ind[0], order_ind[1], money_ind[0], money_ind[1], money, False]
                break
            else:
                continue

    def pick_order(self, deliver_id, order_ind):
        reward = 0
        for i in range(self.max_order):
            if self.data[deliver_id, i, 4] != 0 and self.check_pos(self.data[deliver_id, i, 0:2], order_ind):
                reward = reward + self.params["pickup_reward"]
                self.data[deliver_id, i, 5] = True
            else:
                continue
        return reward

    def finish_order(self, deliver_id, money_ind):
        reward = 0
        for i in range(self.max_order):
            if self.data[deliver_id, i, 4] != 0 and self.check_pos(self.data[deliver_id, i, 2:4], money_ind) and \
                    self.data[deliver_id, i, 5]:
                reward = reward + self.data[deliver_id, i, 4]
                self.data[deliver_id, i, 4] = 0
            else:
                continue
        return reward

    def check_pos(self, a, b):
        ans = True
        for i in range(len(a)):
            if a[i] == b[i]:
                continue
            else:
                ans = False
                break
        return ans

# SampleEpisode(0)