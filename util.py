import numpy

from network import Network, make_uniform_network
from muzero import MuZeroConfig
from game import Game
from util_leaf import Action, Node

class SharedStorage(object):

  def __init__(self, config: MuZeroConfig):
    self._networks = {}
    self.config = config

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      # policy -> uniform, value -> 0, reward -> 0
      return make_uniform_network(self.config.action_space_size)

  def save_network(self, step: int, network: Network):
    self._networks[step] = network


class ReplayBuffer(object):

  def __init__(self, config: MuZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.config= config
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, num_unroll_steps: int, td_steps: int):
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    return [(g.make_image(i), g.get_actions(i,num_unroll_steps),
             g.make_target(i, num_unroll_steps, td_steps))
            for (g, i) in game_pos]

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    return numpy.random.choice(self.buffer)

  def sample_position(self, game) -> int:
    # Sample position from game either uniformly or according to some priority.
    if len(game.history) == self.config.max_moves: # game.history and game.root_values should have the same length
      return numpy.random.randint(0, self.config.max_moves - (self.config.num_unroll_steps + self.config.td_steps))
    else:
      return numpy.random.randint(0, len(game.root_values))

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
  action = softmax_sample(visit_counts, t)
  return action

def select_action_pit(config: MuZeroConfig, num_moves: int, node: Node,
                  network: Network, t: float):
  visit_counts = [
      (child.visit_count, action) for action, child in node.children.items()
  ]
  action = softmax_sample(visit_counts, t)
  return action

def softmax_sample(visit_counts, temperature: float): #TODO
  p = numpy.array([x[0] for x in visit_counts])**(1/temperature)
  p = p/p.sum()
  actions = [x[1] for x in visit_counts]
  return numpy.random.choice(actions, p=p)