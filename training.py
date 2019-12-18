from muzero import MuZeroConfig
from util import SharedStorage, ReplayBuffer
from network import Network

import torch
import torch.optim as optim
from torch import nn
import numpy as np
import copy



def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer, tb_logger, iter_step):
  
  network = copy.deepcopy(storage.latest_network())
  network.train()
  optimizer = optim.SGD(network.parameters(), lr=config.lr_init, momentum=config.momentum, weight_decay=config.weight_decay ,nesterov=True)


  for batch_step in range(config.training_steps):
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    update_weights(optimizer, network, batch, tb_logger, config.training_steps*iter_step + batch_step)
  
  return network


def update_weights(optimizer: optim, network: Network, batch, tb_logger, step):
  loss = 0
  mseloss = nn.MSELoss()

  images = np.stack([sample[0] for sample in batch])
  total_value_loss = 0
  total_reward_loss = 0
  total_policy_loss = 0
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
    entropy = 0
    for j in range(len(batch)):
      if not terminal[j]:
        valid = np.array(target_policy[j])
        valid = valid[valid > 0]
        entropy += -(valid*np.log(valid)).sum() 
    lpol = - (torch.tensor(target_policy).cuda()*policy_logits).sum(dim=1)
    non_terminal = torch.tensor([not x for x in terminal], dtype=torch.float32).cuda()
    policy_loss = (lpol*non_terminal).mean()
    loss += policy_loss

    
    total_value_loss += value_loss.item()
    total_reward_loss += reward_loss.item() if i!= 0 else 0
    total_policy_loss += policy_loss.item()-entropy/len(batch)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  tb_logger.add_scalar("value_loss",  total_value_loss, step)
  tb_logger.add_scalar("reward_loss", total_reward_loss, step)
  tb_logger.add_scalar("policy_loss", total_policy_loss, step)


def scalar_loss(prediction, target) -> float:
  # MSE in board games, cross entropy between categorical values in Atari.
  return -1