from muzero import MuZeroConfig
from util import SharedStorage, ReplayBuffer
from network import Network

import torch
import torch.optim as optim
from torch import nn
import numpy as np
import copy



def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
  
  network = copy.deepcopy(storage.latest_network())
  network.train()
  optimizer = optim.SGD(network.parameters(), lr=config.lr_init, momentum=config.momentum, weight_decay=config.weight_decay ,nesterov=True)


  for _ in range(config.training_steps):
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    update_weights(optimizer, network, batch)
  
  return network


def update_weights(optimizer: optim, network: Network, batch):
  loss = 0
  mseloss = nn.MSELoss()

  images = np.stack([sample[0] for sample in batch])

  for i in range(len(batch[0][2])):
    if i == 0:
      value, _, policy_logits, hidden_state = network.initial_inference(images)
    else:
      actions = [sample[1][i-1] for sample in batch]
      value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, actions)
    
    # targets
    target_value  = [sample[2][i][0] for sample in batch]
    target_reward = [sample[2][i][1] for sample in batch]
    target_policy = [sample[2][i][2] for sample in batch]
    terminal      = [sample[2][i][3] for sample in batch]

    # value loss
    value_loss = mseloss(value, torch.tensor(target_value, dtype=torch.float32).unsqueeze(1).cuda())
    loss += value_loss

    # reward loss
    if i!= 0:
      reward_loss = mseloss(reward, torch.tensor(target_reward, dtype=torch.float32).unsqueeze(1).cuda())
      loss += reward_loss

    # policy loss
    num_valid = 0
    entropy = 0
    for j in range(len(batch)):
      if not terminal[j]:
        num_valid += 1
        valid = np.array(target_policy[j])
        valid = valid[valid > 0]
        entropy += -(valid*np.log(valid)).sum() 
    lpol = - (torch.tensor(target_policy).cuda()*policy_logits).sum(dim=1)
    non_terminal = torch.tensor([not x for x in terminal], dtype=torch.float32).cuda()
    policy_loss = (lpol*non_terminal).mean()
    loss += policy_loss

    # if i==0:
    #   print(value_loss.item(), " ", policy_loss.item()-entropy/num_valid)
    # else:
    #   print(value_loss.item(), reward_loss.item(), policy_loss.item()-entropy/num_valid)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


def scalar_loss(prediction, target) -> float:
  # MSE in board games, cross entropy between categorical values in Atari.
  return -1