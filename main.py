from muzero import MuZeroConfig, make_board_game_config
from game import Game
from util import SharedStorage, ReplayBuffer, add_exploration_noise, select_action
from util_leaf import Node
from network import Network
from mcts import expand_node, run_mcts
from tictactoe.TicTacToeEnv import TicTacToeEnv
from training import train_network



from models.tictactoe_model import tictactoeNetwork as myNetwork
import torch
from tensorboardX import SummaryWriter

def muzero_training(config: MuZeroConfig):
  storage = SharedStorage(config)
  storage.save_network(step  = 0, network =  myNetwork(config.action_space_size, cuda = True)) 
  replay_buffer = ReplayBuffer(config)
  
  for i in range(1,5+1): # num iterations
    print("ITER:",i)
    
    run_selfplay(config, storage, replay_buffer, 50) # num episodes per iteration
    # import pickle
    # pickle.dump( replay_buffer, open( "save.pkl", "wb" ) )
    # replay_buffer = pickle.load( open( "save.pkl", "rb" ) )
    trained_network = train_network(config, storage, replay_buffer, tb_logger, i-1)
    pwins, nwins, draws = pit_against(config, storage, trained_network)
    print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
    if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < 0.55:
        print('REJECTING NEW MODEL')
    else:
        print('ACCEPTING NEW MODEL')
        storage.save_network(i, trained_network)
        replay_buffer = ReplayBuffer(config)
  net = storage.latest_network()
  torch.save({
        'state_dict': net.model.state_dict()
  }, "model.tar")
  # checkpoint = torch.load("model.tar")
  # net.model.load_state_dict(checkpoint["state_dict"])
  # net.eval()
    

  return storage.latest_network()

def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer, num_episodes: int):
  network = storage.latest_network()
  network.eval()
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

def pit_against(config: MuZeroConfig, storage: SharedStorage, trained_network: Network):
  player  = 1
  num_games = 20
  not_trained_network = storage.latest_network()
  not_trained_network.eval()
  trained_network.eval()
  wins_not_trained = 0
  wins_trainied = 0
  draws = 0
  for _ in range(num_games):
    
    game = Game(config.action_space_size, config.discount, config.environment)
    
    while not game.terminal():
      cur_network = not_trained_network if player == 1 else trained_network
      root = Node(0)
      current_observation = game.make_image(-1)
      expand_node(root, game.to_play(), game.legal_actions(),
                cur_network.initial_inference(current_observation))
      run_mcts(config, root, game.action_history(), cur_network)
      action = select_action(config, len(game.history), root, cur_network) # review how it is salected
      game.apply(action)
      player *= -1
    whowins = game.environment.getGameEnded(game.environment.board, 1)
    if whowins == 1:
        wins_not_trained +=1
    elif whowins == -1:
        wins_trainied += 1
    else:
        draws += 1
  
  return wins_not_trained, wins_trainied, draws

tb_logger = SummaryWriter("/tmp/tb")
config = make_board_game_config(16+1, 16+1, 0.25, TicTacToeEnv, 0.001)
muzero_training(config)