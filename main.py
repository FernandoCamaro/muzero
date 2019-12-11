from muzero import MuZeroConfig, make_board_game_config
from game import Game
from util import SharedStorage, ReplayBuffer, add_exploration_noise, select_action
from util_leaf import Node
from network import Network
from mcts import expand_node, run_mcts
from tictactoe.TicTacToeEnv import TicTacToeEnv

def muzero_training(config: MuZeroConfig):
  storage = SharedStorage(config)
  replay_buffer = ReplayBuffer(config)

  for i in range(1,10+1): # num iterations
    print("ITER:",i)
    run_selfplay(config, storage, replay_buffer, 10) # num episodes per iteration
    # train_network(config, storage, replay_buffer)
    

  return storage.latest_network()

def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer, num_episodes: int):
  network = storage.latest_network()
  for _ in range(num_episodes):
    game = play_game(config, network)
    replay_buffer.save_game(game)


def play_game(config: MuZeroConfig, network: Network) -> Game:
  game = Game(config.action_space_size, config.discount, config.environment)

  while not game.terminal() and len(game.history) < config.max_moves:
    # At the root of the search tree we use the representation function to
    # obtain a hidden state given the current observation.
    root = Node(0)
    current_observation = game.make_image(-1)
    expand_node(root, game.to_play(), game.legal_actions(),
                network.initial_inference(current_observation))
    add_exploration_noise(config, root)

    # We then run a Monte Carlo Tree Search using only action sequences and the
    # model learned by the network.
    run_mcts(config, root, game.action_history(), network)
    action = select_action(config, len(game.history), root, network)
    game.apply(action)
    game.store_search_statistics(root)
  return game

config = make_board_game_config(16+1, 16+1, 0.25, TicTacToeEnv, 0.01)
muzero_training(config)