import torch
import numpy as np
from actor_critic import ActorCritic


# some sample params
sample_params = {
    "abs": 100.0,
    "num_delivers": 3,
    "gas_cost": -1,
    "min_reward": 5,
    "max_reward": 20,
    "num_obstacle": int(5),
    "obstacle_ratio_h": 0.5,
    "obstacle_ratio_w": 0.5
}
sample_h, sample_w = [64, 64]


# utility functions
# draw obstacles
def addObstacle(m, ys, xs, params):
    h, w = m.shape
    for y,x in zip(ys, xs):
        poly_h = np.random.randint(params["obstacle_ratio_h"]*h)+1
        poly_w = np.random.randint(params["obstacle_ratio_w"]*w)+1
        h1 = int(max(0, y-0.5*poly_h))
        h2 = int(min(h-1, y+0.5*poly_h))
        w1 = int(max(0, x-0.5*poly_w))
        w2 = int(min(w-1, x+0.5*poly_w))
        m[h1:h2, w1:w2] = np.minimum(m[h1:h2, w1:w2], -params["abs"])
    return m


# this function returns a matrix of map with random scores at each tile
def generateMap(h, w, params=sample_params):
    # initalize the map
    m = np.zeros([h, w])
    m = m + params["gas_cost"]

    # add in number of obstacle
    ys = np.random.randint(0, h, size=(params["num_obstacle"]))
    xs = np.random.randint(0, w, size=(params["num_obstacle"]))
    m = addObstacle(m, ys, xs, params)

    # return normalized map
    return m/params["abs"]

# generate the map for pickup
def generatePickupMap(h, w, params):
    m = np.zeros(h, w)
    y = np.random.randint(0, h)
    x = np.random.randint(0, w)
    # q1 here, what reward to assign to the pickup location
    m[y, x] = 1
    return m/params["abs"]

# generate the map for money delievery
def generateMoneyMap(h, w, params):
    m = np.zeros(h, w)
    y = np.random.randint(0, h)
    x = np.random.randint(0, w)
    m[y, x] = np.random.randint(params["min_reward"], params["max_reward"])
    return m/params["abs"]

# generate position map
def generateMoneyMap(h, w, params):
    m = np.zeros(h, w)
    y = np.random.randint(0, h)
    x = np.random.randint(0, w)
    m[y, x] = 1
    return m/params["abs"]


# ------------------------ map related finished -----------------------------------
# ------------------------ reward related start -----------------------------------
# this function returns the reward and state in the next step based on
def getOneBatch(batch, params):

    return batch


# this function returns the whole eposide of samples, given the 
def getOneEposide(model, params=sample_params, duration=250,):
    eposide = []
    # we generate the initial state
    # the channel order of states are
    # map
    # pickup map
    # money map
    # iternate: position of deliever, pickup position of deliever, money position of deliever
    states = []
    m = generateMap(sample_h, sample_w, sample_params).reshape([1, sample_h, sample_w])
    states.append(m)
    pickup_m = generatePickupMap(sample_h, sample_w, sample_params).reshape([1, sample_h, sample_w])
    states.append(pickup_m)
    money_m = generateMoneyMap(sample_h, sample_w, sample_params).reshape([1, sample_h, sample_w])
    states.append(money_m)
    for i in range(params["num_delivers"]):
        pickup_map = 



    # finished generating initial batch
    batch = {}

    for i in range(duration):
        # we first add new pickup
        batch["pickup_controls"], batch["people_actions"] = model.actor(batch["states"][:-1])
        batch = getOneBatch(batch, params)
        eposide.append(batch)
    return eposide

# samples to generate one eposide


