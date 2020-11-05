import torch
import torch.nn.functional as F
from torch.utils import tensorboard

import argparse
import numpy as np
import os
from statistics import mean, stdev
from tqdm import tqdm
import json

import gym
import simple_discrete_game

from models.cnn_agent import cnn_value_net, cnn_policy_net
from utils.memory import MainMemory
from utils.reproducibility import set_seed, log_params
from algo.ppo_step import calc_ppo_loss_gae

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="GoalGrid-v0")
parser.add_argument("--exp-name", default="goalgrid_seed_1")
parser.add_argument("--batch-size", type=int, default=1000, help="batch_size")
parser.add_argument("--full-ppo-iters", type=int, default=500, help="num times whole thing is run")
parser.add_argument("--seed", type=int, default=1, help="set random seed for reproducibility ")
parser.add_argument("--num-value-updates", type=int, default=4, help="update critic per epoch")
parser.add_argument("--num-policy-updates", type=int, default=4, help="update agent per epoch")
parser.add_argument("--num-evaluate", type=int, default=20, help="eval per epoch")
parser.add_argument(
    "--episode-max-lenght", type=int, default=100, help="max lenght to run an episode"
)
parser.add_argument("--save-interval", type=int, default=100, help="save weights every x episodes")
parser.add_argument("--agent-lr", type=int, default=0.002, help="agent learning rate")
parser.add_argument("--critic-lr", type=int, default=0.001, help="critic learing rate")


args = parser.parse_args()
json_log = log_params(args)

cuda = torch.device("cuda")
cpu = torch.device("cpu")


##Helper function
def flat_tensor(t):
    return torch.from_numpy(t).float().view(-1)


def preprocess_obs_img(obs):
    #channels first
    obs = np.moveaxis(obs, 2, 0)
    #normalize
    obs =  obs/255.0

    return torch.from_numpy(obs).float()


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

    # normalize advantages
    memory.advantages = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


def collect_exp_single_actor(env, actor, memory, iters):
    obs = env.reset()
    time_step = 0
    for _ in range(iters):
        obs = preprocess_obs_img(obs)
        memory.states.append(np.array(obs))

        action, log_prob = actor.act(obs.unsqueeze(0))
        next_obs, reward, done, info = env.step(action.item())

        memory.is_terminals.append(done)
        memory.actions.append(action.item())
        memory.logprobs.append(log_prob.item())
        memory.rewards.append(reward)

        obs = next_obs
        time_step += 1
        if done or time_step >= args.episode_max_lenght:
            obs = env.reset()

    # normalize rewards
    m = mean(memory.rewards)
    # print(memory.rewards)
    std = stdev(memory.rewards) + 1e-5
    memory.rewards = [(i + m) / std for i in memory.rewards]
    # (memory.rewards - mean(memory.rewards)) / (stdev(memory.rewards) + 1e-5)

    return memory


def save_episode_as_gif(agent, env, episode_max_lenght, gif_name):
    obs_to_vis = []
    obs = env.reset()
    for timestep in range(0, episode_max_lenght):
        obs = preprocess_obs_img(obs)
        action, _ = main_actor.act(obs.unsqueeze(0))
        obs, _, done, _ = env.step(action.item())
        obs_to_vis.append(env.render(mode="rgb_array"))
        if done:
            break
    write_gif(obs_to_vis, gif_name + ".gif", fps=30)


if __name__ == "__main__":

    # creating environment
    env = gym.make(args.env_name)
    set_seed(args.seed, env)

    n_actions = env.action_space.n
    n_channels = 3
    # create nn's
    main_actor = cnn_policy_net(n_channels, n_actions)
    critic = cnn_value_net(n_channels)

    optim_actor = torch.optim.Adam(main_actor.parameters(), lr=args.agent_lr, betas=(0.9, 0.999))
    optim_critic = torch.optim.Adam(critic.parameters(), lr=args.agent_lr, betas=(0.9, 0.999))

    # create memory
    main_memory = MainMemory(batch_size=args.batch_size)

    # logging
    tb_summary = tensorboard.SummaryWriter()

    for iter in tqdm(range(args.full_ppo_iters + 1)):

        main_memory = collect_exp_single_actor(env, main_actor, main_memory, args.batch_size)
        critic.to(cuda)
        main_actor.to(cuda)
        main_memory.critic_values(critic, cuda)

        calculate_gae(main_memory)
        # print(main_memory.memory_size())

        for k in range(args.num_policy_updates):
            optim_actor.zero_grad()
            ppo_loss = calc_ppo_loss_gae(main_actor, main_memory)
            ppo_loss.backward()
            optim_actor.step()

        # value loss
        value_loss_list = []
        for j in range(args.num_value_updates):
            batch_states, batch_returns = main_memory.get_value_batch()
            batch_states, batch_returns = batch_states.to(cuda), batch_returns.to(cuda)
            optim_critic.zero_grad()
            pred_returns = critic(batch_states)
            value_loss = F.mse_loss(pred_returns.view(-1), batch_returns.view(-1))
            value_loss.backward()
            optim_critic.step()

            value_loss_list.append(value_loss.item())

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
        num_done = 0
        while args.num_evaluate > eval_ep:
            ep_timestep += 1
            obs = preprocess_obs_img(obs)

            action, log_prob = main_actor.act(obs.unsqueeze(0))
            obs, reward, done, info = env.step(action.item())
            ep_reward += reward

            if done or ep_timestep >= args.episode_max_lenght:
                obs = env.reset()
                eval_ep += 1
                eval_timesteps.append(ep_timestep)
                eval_rewards.append(ep_reward)
                ep_reward = 0
                ep_timestep = 0
                num_done += 1

        tb_summary.add_scalar("reward/eval_reward", mean(eval_rewards), global_step=iter)
        tb_summary.add_scalar("time/eval_traj_len", mean(eval_timesteps), global_step=iter)
        tb_summary.add_scalar("reward/prob_done", num_done / args.num_evaluate, global_step=iter)

        json_log["rewards list"].append(mean(eval_rewards))
        json_log["avg episode timesteps"].append(mean(eval_timesteps))
        json_log["prob done"].append(num_done / args.num_evaluate)

        print("eval_reward ", mean(eval_rewards), " eval_timesteps ", mean(eval_timesteps), "prob_done", num_done / args.num_evaluate)

        if iter % args.save_interval == 0 and iter > 0:
            torch.save(
                main_actor.state_dict(), "ppo_" + args.exp_name + "_actor" + str(iter) + ".pth"
            )
            torch.save(critic.state_dict(), "ppo_" + args.exp_name + "_critic" + str(iter) + ".pth")

    os.chdir(os.path.join(os.getcwd(), "jsons"))
    with open(str(args.exp_name) + ".json", "w") as fp:
        json.dump(json_log, fp, sort_keys=True, indent=4)
