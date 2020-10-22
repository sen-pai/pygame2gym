import gym
import simple_discrete_game
import matplotlib.pyplot as plt
from array2gif import write_gif


def random_agent(episodes=40):
    env = gym.make("GoalGrid-v0")
    all_states = []
    for i in range(5):
        env.reset()
        # env.render()
        for e in range(episodes):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            all_states.append(state)
            print(reward)
            if done:
                break
        write_gif(all_states, str(i) + ".gif", fps=10)
        all_states = []


if __name__ == "__main__":
    random_agent()
