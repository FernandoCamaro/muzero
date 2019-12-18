import typing
from typing import Dict, List
from util_leaf import Action

class NetworkOutput(typing.NamedTuple):
  value: float
  reward: float
  policy_logits: Dict[Action, float]
  hidden_state: List[float]


class Network(object):

  def  __init__(self, action_space_size: int):
    self.action_space_size = action_space_size
    self.tr_steps = 0

  def initial_inference(self, image) -> NetworkOutput:
    # representation + prediction function
    action_logits = self.default_action_logits()
    return NetworkOutput(0, 0, action_logits, [])

  def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
    # dynamics + prediction function
    action_logits = self.default_action_logits()
    return NetworkOutput(0, 0, action_logits, [])

  def get_weights(self):
    # Returns the weights of this network.
    return []

  def training_steps(self) -> int:
    # How many steps / batches the network has been trained for.
    return self.tr_steps

  def default_action_logits(self):
    action_logits = {}
    for i in range(self.action_space_size):
      action_logits[Action(i)] = 1.0
    return action_logits

def make_uniform_network(action_space_size: int):
  return Network(action_space_size)