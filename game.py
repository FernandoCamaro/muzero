from typing import List
import random

from util_leaf import Action, Node, Player, ActionHistory
from environment import Environment

import numpy as np


class Game(object):
  """A single episode of interaction with the environment."""

  def __init__(self, action_space_size: int, discount: float, Environment: Environment):
    self.environment = Environment()  # Game specific environment.
    self.players = [] # will store which player is acting in each time-step
    self.players.append(Player(self.environment.player))
    self.history = []
    self.rewards = [100.]
    self.observations = []
    self.child_visits = []
    self.root_values = []
    self.action_space_size = action_space_size
    self.discount = discount

  def terminal(self) -> bool:
    # Game specific termination rules.
    return self.environment.terminal()

  def legal_actions(self) -> List[Action]:
    # Game specific calculation of legal actions.
    return self.environment.getLegalActions()

  def apply(self, action: Action, network, netout):
    _ = self.environment.step(action)
    netout_pred = network.recurrent_inference(netout.hidden_state, action)
    current_observation = self.make_image(-1)
    netout_actual = network.initial_inference(current_observation)
    hidden_forward_error = netout_actual.hidden_state - netout_pred.hidden_state
    intrinsic_reward = np.mean(hidden_forward_error**2)
    self.players.append(Player(self.environment.player))
    self.rewards.append(intrinsic_reward)
    self.history.append(action)
    return netout_actual

  def store_search_statistics(self, root: Node):
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = (Action(index) for index in range(self.action_space_size))
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in action_space
    ])
    self.root_values.append(root.value())

  def make_image(self, state_index: int):
    # Game specific feature planes.
    if state_index == -1: # get new one
      observation = self.environment.state()
      self.observations.append(observation)
    if state_index >= len(self.observations):
      state_index = len(self.observations) - 1
    return self.observations[state_index]

  def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < len(self.root_values):
        sign = 1 if self.players[bootstrap_index] == self.players[current_index] else -1
        value = sign*(self.root_values[bootstrap_index] * self.discount**td_steps)
      else:
        value = 0

      for i, reward in enumerate(self.rewards[current_index+1:bootstrap_index]):
        sign = 1 if self.players[current_index+1+i] == self.players[current_index] else -1
        value += sign*(reward * self.discount**i)  # pytype: disable=unsupported-operands

      if current_index < len(self.root_values):
        targets.append((value, self.rewards[current_index],
                        self.child_visits[current_index], False))
      elif current_index == len(self.root_values):
        # States past the end of games are treated as absorbing states.
        targets.append((0, self.rewards[current_index], self.uniform_action_prob_dist(), True))
      else:
        targets.append((0, 0, self.uniform_action_prob_dist(), True))
        
    return targets

  def get_actions(self, i, num_unroll_steps):
    actual_actions_taken = self.history[i:i + num_unroll_steps]
    extra_random_actions = [self.get_random_action() for _ in range(num_unroll_steps - len(actual_actions_taken))]
    return actual_actions_taken + extra_random_actions

  def get_random_action(self):
    return Action(random.randint(0,self.action_space_size-1))

  def to_play(self) -> Player:
    return Player(self.environment.player)

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.history, self.action_space_size)

  def uniform_action_prob_dist(self):
    return [1/self.action_space_size for _ in range(self.action_space_size)]