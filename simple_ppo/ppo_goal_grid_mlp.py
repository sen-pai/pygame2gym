import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils import tensorboard

import argparse
import numpy as np
from statistics import mean
import gym
import simple_discrete_game

from models.mlp_agent import mlp_policy_net, mlp_value_net
from utils.memory import MainMemory
from algo.ppo_step import calc_ppo_loss_gae

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="GoalGrid-v0")
parser.add_argument("--batch-size", type=int, default=4000, help="batch_size")
parser.add_argument("--full-ppo-iters", type=int, default=100, help="num times whole thing is run")
args = parser.parse_args()


cuda = torch.device("cuda")
cpu = torch.device("cpu")


##Helper function
def flat_tensor(t):
    return torch.from_numpy(t).float().view(-1)


def preprocess_obs(obs):
    #convert to grayscale by averaging channels
    obs = np.mean(obs, axis=2)/255.0
    #convert to tensor
    return torch.from_numpy(obs).float().view(-1)


#### Hyper parameters
num_value_updates = 5
num_policy_updates = 5
num_evaluate = 20


def calculate_gae(memory, gamma=0.99, lmbda=0.95):
    gae = 0
    for i in reversed(range(len(memory.rewards))):
        delta = (
            memory.rewards[i]
            + gamma * memory.values[i + 1] * (not memory.is_terminals[i])
            - memory.values[i]
        )
        gae = delta + gamma * lmbda * (not memory.is_terminals[i]) * gae
        memory.returns.insert(0, gae + memory.values[i])

    adv = np.array(memory.returns) - memory.values[:-1]
    memory.advantages = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


def collect_exp_single_actor(env, actor, memory, iters):
    obs = env.reset()
    current_timestep = 0
    for _ in range(iters):
        # obs = flat_tensor(obs)
        current_timestep += 1
        obs = preprocess_obs(obs)

        memory.states.append(np.array(obs))

        action, log_prob = actor.act(obs)
        next_obs, reward, done, info = env.step(action.item())

        memory.is_terminals.append(done)
        memory.actions.append(action.item())
        memory.logprobs.append(log_prob.item())
        memory.rewards.append(reward)

        obs = next_obs

        if done or current_timestep > 50:
            obs = env.reset()
            current_timestep = 0

    return memory


if __name__ == "__main__":

    # creating environment
    env = gym.make(args.env_name)
    state_size = 128*128
    n_actions = env.action_space.n

    # create nn's
    main_actor = mlp_policy_net(state_size, 32, n_actions)
    critic = mlp_value_net(state_size, hidden_size=32)

    optim_actor = torch.optim.Adam(main_actor.parameters(), lr=3e-4, betas=(0.9, 0.999))
    optim_critic = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.9, 0.999))

    # create memory
    main_memory = MainMemory(batch_size=args.batch_size)

    # logging
    tb_summary = tensorboard.SummaryWriter()

    for iter in range(args.full_ppo_iters + 1):

        main_memory = collect_exp_single_actor(env, main_actor, main_memory, args.batch_size)
        critic.to(cuda)
        main_actor.to(cuda)
        main_memory.critic_values(critic, cuda)

        calculate_gae(main_memory)
        print(main_memory.memory_size())

        for k in range(num_policy_updates):
            optim_actor.zero_grad()
            ppo_loss = calc_ppo_loss_gae(main_actor, main_memory)
            ppo_loss.backward()
            optim_actor.step()

            tb_summary.add_histogram(
                "gradients/actor",
                torch.cat([p.grad.view(-1) for p in main_actor.parameters()]),
                global_step=num_policy_updates * iter + k,
            )

        # value loss
        value_loss_list = []
        for j in range(num_value_updates):
            batch_states, batch_returns = main_memory.get_value_batch()
            batch_states, batch_returns = batch_states.to(cuda), batch_returns.to(cuda)
            optim_critic.zero_grad()
            pred_returns = critic(batch_states)
            value_loss = F.mse_loss(pred_returns.view(-1), batch_returns.view(-1))
            value_loss.backward()
            optim_critic.step()

            value_loss_list.append(value_loss.item())

            tb_summary.add_histogram(
                "gradients/critic",
                torch.cat([p.grad.view(-1) for p in critic.parameters()]),
                global_step=num_value_updates * iter + j,
            )

        tb_summary.add_scalar("loss/value_loss", mean(value_loss_list), global_step=iter)

        main_actor.to(cpu)
        main_memory.clear_memory()

        # evaluation
        eval_ep = 0
        obs = env.reset()

        eval_rewards = []
        eval_timesteps = []

        ep_reward = 0
        ep_timestep = 0
        while num_evaluate > eval_ep:
            ep_timestep += 1
            # obs = flat_tensor(obs)

            obs = preprocess_obs(obs)
            action, log_prob = main_actor.act(obs)
            obs, reward, done, info = env.step(action.item())
            ep_reward += reward

            if done or ep_timestep > 50:
                if done:
                    print("entered done")
                obs = env.reset()
                eval_ep += 1
                eval_timesteps.append(ep_timestep)
                eval_rewards.append(ep_reward)
                ep_reward = 0
                ep_timestep = 0

        tb_summary.add_scalar("reward/eval_reward", mean(eval_rewards), global_step=iter)
        tb_summary.add_scalar("time/eval_traj_len", mean(eval_timesteps), global_step=iter)
        print("eval_reward ", mean(eval_rewards), " eval_timesteps ", mean(eval_timesteps))

        if iter % 50 == 0:
            torch.save(
                main_actor.state_dict(), "ppo_" + args.env_name + "_actor" + str(iter) + ".pth"
            )
            torch.save(critic.state_dict(), "ppo_" + args.env_name + "_critic" + str(iter) + ".pth")
