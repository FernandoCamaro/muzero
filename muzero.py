from typing import Optional
from util_leaf import KnownBounds

class MuZeroConfig(object):

  def __init__(self,
               action_space_size: int,
               max_moves: int,
               discount: float,
               dirichlet_alpha: float,
               num_simulations: int,
               batch_size: int,
               td_steps: int,
               num_actors: int,
               lr_init: float,
               lr_decay_steps: float,
               visit_softmax_temperature_fn,
               environment,
               zerosumgame: bool,
               known_bounds: Optional[KnownBounds] = None):
    ### Self-Play
    self.action_space_size = action_space_size
    self.num_actors = num_actors

    self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
    self.environment = environment
    self.zerosumgame = zerosumgame
    self.max_moves = max_moves
    self.num_simulations = num_simulations
    self.discount = discount

    # Root prior exploration noise.
    self.root_dirichlet_alpha = dirichlet_alpha
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    self.known_bounds = known_bounds

    ### Training
    self.training_steps = int(10000)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = batch_size
    self.num_unroll_steps = 5
    self.td_steps = td_steps

    self.weight_decay = 1e-4
    self.momentum = 0.9

    # Exponential learning rate schedule
    self.lr_init = lr_init
    self.lr_decay_rate = 0.1
    self.lr_decay_steps = lr_decay_steps

def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           environment,
                           lr_init: float) -> MuZeroConfig:

  def visit_softmax_temperature(num_moves, training_steps):
    return max(1.0 -( training_steps // 10000 ) / 10 , 1e-3)

  return MuZeroConfig(
      action_space_size=action_space_size,
      max_moves=max_moves,
      discount=1.0,
      dirichlet_alpha=dirichlet_alpha,
      num_simulations=400,
      batch_size=128,
      td_steps=20,  # Always use Monte Carlo return.
      num_actors=3000,
      lr_init=lr_init,
      lr_decay_steps=400e3,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      environment = environment,
      zerosumgame = True,
      known_bounds=KnownBounds(-1, 1))

def make_atari_config(environment) -> MuZeroConfig:

  def visit_softmax_temperature(num_moves, training_steps):
    if training_steps < 500e3:
      return 1.0
    elif training_steps < 750e3:
      return 0.5
    else:
      return 0.25

  return MuZeroConfig(
      action_space_size=18,
      max_moves=27000,  # Half an hour at action repeat 4.
      discount=0.997,
      dirichlet_alpha=0.25,
      num_simulations=50,
      batch_size=1024,
      td_steps=10,
      num_actors=350,
      lr_init=0.05,
      lr_decay_steps=350e3,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      environment = environment,
      zerosumgame = False)
