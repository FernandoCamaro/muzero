from muzero import MuZeroConfig, make_board_game_config
from game import Game
from util import SharedStorage, ReplayBuffer, add_exploration_noise, select_action, select_action_pit
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

  num_updates = 0
  i = 1
  while (num_updates < 50) and (i<200):
    print("ITER:",i)
    network = storage.latest_network() if num_updates > 0 else Network(config.action_space_size)
    run_selfplay(config, network, storage, replay_buffer, 1000, num_updates) # num episodes per iteration
    # import pickle
    # pickle.dump( replay_buffer, open( "save.pkl", "wb" ) )
    # replay_buffer = pickle.load( open( "save.pkl", "rb" ) )
    trained_network = train_network(config, storage, replay_buffer, tb_logger, i-1)
    torch.save({'state_dict': trained_network.model.state_dict()}, "model_"+str(i)+".tar")
    pwins, nwins, draws = pit_against(config, network, trained_network)
    print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
    if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < 0.55:
        print('REJECTING NEW MODEL')
    else:
        num_updates += 1
        print('ACCEPTING NEW MODEL')
        storage.save_network(i, trained_network)
        replay_buffer = ReplayBuffer(config)
        
    i = i+1
    

  return storage.latest_network()

def run_selfplay(config: MuZeroConfig, network: Network, storage: SharedStorage,
                 replay_buffer: ReplayBuffer, num_episodes: int, num_updates: int):
  network.eval()
  
  for _ in range(num_episodes):
    network_op = storage.random_network() if num_updates > 0 else Network(config.action_space_size)
    network_op.eval()
    game = play_game(config, network, network_op)
    replay_buffer.save_game(game)


def play_game(config: MuZeroConfig, network: Network, network_op: Network) -> Game:
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
    

    if game.to_play().id == 1:
      action = select_action(config, len(game.history), root, network)
    else:
      root_op = Node(0)
      expand_node(root_op, game.to_play(), game.legal_actions(),
                network_op.initial_inference(current_observation))
      add_exploration_noise(config, root_op)
      run_mcts(config, root_op, game.action_history(), network_op)
      action = select_action(config, len(game.history), root_op, network_op)
      
    
    game.apply(action)
    game.store_search_statistics(root)
  return game

def pit_against(config: MuZeroConfig, not_trained_network: Network, trained_network: Network):
  player  = 1
  num_games = 20
  not_trained_network.eval()
  trained_network.eval()
  wins_not_trained = 0
  wins_trainied = 0
  draws = 0
  for _ in range(num_games):
    
    game = Game(config.action_space_size, config.discount, config.environment)
    first_move = True
    while not game.terminal():
      cur_network = not_trained_network if player == 1 else trained_network
      root = Node(0)
      current_observation = game.make_image(-1)
      expand_node(root, game.to_play(), game.legal_actions(),
                cur_network.initial_inference(current_observation))
      run_mcts(config, root, game.action_history(), cur_network)
      if first_move:
        action = select_action_pit(config, len(game.history), root, cur_network, 1.0) # review how it is salected
        first_move = False
      else:
        action = select_action_pit(config, len(game.history), root, cur_network, 0.2) # review how it is salected
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
config = make_board_game_config(16+1, 16+1, 0.25, TicTacToeEnv, 0.005)
muzero_training(config)
