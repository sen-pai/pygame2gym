import torch
import numpy as np


def set_seed(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)


def log_params(args):
    json_log = {}
    json_log["seed"] = args.seed
    json_log["experiment name"] = args.exp_name
    json_log["batch size"] = args.batch_size
    json_log["environment name"] = args.env_name
    json_log["num_value_updates"] = args.num_value_updates
    json_log["num_policy_updates"] = args.num_policy_updates
    json_log["num evaluations per epoch"] = args.num_evaluate
    json_log["total epochs"] = args.full_ppo_iters
    json_log["episode_max_lenght"] = args.episode_max_lenght
    json_log["agent lr"] = args.agent_lr
    json_log["critic lr"] = args.critic_lr
    json_log["prob done"] = []
    json_log["rewards list"] = []
    json_log["avg episode timesteps"] = []

    return json_log
