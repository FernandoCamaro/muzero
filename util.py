import numpy

from network import Network, make_uniform_network
from muzero import MuZeroConfig
from game import Game
from util_leaf import Action, Node

class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      # policy -> uniform, value -> 0, reward -> 0
      return make_uniform_network()

  def save_network(self, step: int, network: Network):
    self._networks[step] = network


class ReplayBuffer(object):

  def __init__(self, config: MuZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, num_unroll_steps: int, td_steps: int):
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    return [(g.make_image(i), g.history[i:i + num_unroll_steps],
             g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
            for (g, i) in game_pos]

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    return self.buffer[0]

  def sample_position(self, game) -> int:
    # Sample position from game either uniformly or according to some priority.
    return -1

# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
  actions = list(node.children.keys())
  noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

def select_action(config: MuZeroConfig, num_moves: int, node: Node,
                  network: Network):
  visit_counts = [
      (child.visit_count, action) for action, child in node.children.items()
  ]
  t = config.visit_softmax_temperature_fn(
      num_moves=num_moves, training_steps=network.training_steps())
  _, action = softmax_sample(visit_counts, t)
  return action

def softmax_sample(distribution, temperature: float): #TODO
  return 0, 0