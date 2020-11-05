import torch
import numpy as np


def calc_ppo_loss_gae(actor, memory, use_cuda=True, normalize_returns=False):
    eps_clip = 0.2
    cuda = torch.device("cuda")
    batch_states, batch_actions, old_logprobs, batch_advantages = memory.get_batch()
    if use_cuda:
        batch_states, batch_actions, old_logprobs, batch_advantages = (
            batch_states.to(cuda),
            batch_actions.to(cuda),
            old_logprobs.to(cuda),
            batch_advantages.to(cuda),
        )
    entropy, new_logprobs = actor.evaluate(batch_states, batch_actions)
    ratios = torch.exp(new_logprobs - old_logprobs.detach())

    surr1 = ratios * batch_advantages
    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * batch_advantages
    loss = -torch.min(surr1, surr2) - 0.01 * entropy

    return loss.mean()
