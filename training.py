from muzero import MuZeroConfig
from util import SharedStorage, ReplayBuffer
from network import Network

import torch.optim as optim


def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
  
  network = storage.latest_network()
  network.train()
  optimizer = optim.SGD(network.parameters(), lr=config.lr_init, momentum=config.momentum, weight_decay=config.weight_decay ,nesterov=True)


  for _ in range(config.training_steps):
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    update_weights(optimizer, network, batch)


def update_weights(optimizer: optim, network: Network, batch):
  loss = 0
  for image, actions, targets in batch:

    # Initial step, from the real observation.
    value, reward, policy_logits, hidden_state = network.initial_inference(image)
    predictions = [(1.0, value, reward, policy_logits)]

    # Recurrent steps, from action and previous hidden state.
    for action in actions:
      # the action maybe not need to conver to torch tensor here.
      value, reward, policy_logits, hidden_state = network.recurrent_inference(
          hidden_state, [action])
      predictions.append((1.0 / len(actions), value, reward, policy_logits))

      #hidden_state = tf.scale_gradient(hidden_state, 0.5) Need to check how to deal with this. Maybe with a backward hook?

    for prediction, target in zip(predictions, targets):
      gradient_scale, value, reward, policy_logits = prediction
      target_value, target_reward, target_policy = target

      l = (
          scalar_loss(value, target_value) +
          scalar_loss(reward, target_reward) +
          tf.nn.softmax_cross_entropy_with_logits(
              logits=policy_logits, labels=target_policy))

      loss += tf.scale_gradient(l, gradient_scale)

  for weights in network.get_weights():
    loss += weight_decay * tf.nn.l2_loss(weights)

  optimizer.minimize(loss)


def scalar_loss(prediction, target) -> float:
  # MSE in board games, cross entropy between categorical values in Atari.
  return -1