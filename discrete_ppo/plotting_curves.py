import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# with open('jsons/cartpole_seed_1.json') as f:
#     seed_1 = json.load(f)
#
# with open('jsons/cartpole_seed_2.json') as f:
#     seed_2 = json.load(f)
#
# with open('jsons/cartpole_seed_3.json') as f:
#     seed_3 = json.load(f)
#
# with open('jsons/cartpole_seed_4.json') as f:
#     seed_4 = json.load(f)
#
# with open('jsons/cartpole_seed_5.json') as f:
#     seed_5 = json.load(f)

#
# with open('jsons/lunar_seed_1.json') as f:
#     seed_1 = json.load(f)
#


with open('jsons/mountain_car_seed_1.json') as f:
    seed_1 = json.load(f)



def tsplot(data,**kw):
    x = np.arange(len(data[0]))*seed_1["batch size"]
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    plt.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    plt.plot(x,est,**kw)
    plt.margins(x=0)

sns.set()
fig = plt.figure()
# rewards = [seed_1["rewards list"], seed_2["rewards list"], seed_3["rewards list"], seed_4["rewards list"], seed_5["rewards list"]]
rewards = [seed_1["rewards list"]]


xdata = np.arange(len(seed_1["rewards list"]))


tsplot(rewards)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ylabel("Reward")
plt.xlabel("Number of env interacts")
plt.title(seed_1["environment name"])
plt.show()
