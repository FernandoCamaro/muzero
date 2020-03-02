import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import pickle
import time

from muzero import MuZeroConfig, make_mario_config
from game import Game
from util import SharedStorage, ReplayBuffer, add_exploration_noise, select_action, select_action_pit, hard_update
from util_leaf import Node
from mcts import expand_node, run_mcts
from mario.MarioEnv import MarioEnv
from training import train_network

from network import Network, NetworkOutput
from mario.MarioNet import MarioNet as myNetwork

def muzero_training(config: MuZeroConfig):
  storage = SharedStorage(config)
  
  network =  myNetwork(config.action_space_size, cuda = True)
  target_network = myNetwork(config.action_space_size, cuda = True) # TODO: we don't need actually the representation network.
  hard_update(target_network, network)
  optimizer = optim.SGD(network.parameters(), lr=config.lr_init, momentum=config.momentum, weight_decay=config.weight_decay ,nesterov=True)
  storage.save_network(step  = 0, network = network) 
  replay_buffer = ReplayBuffer(config)

  for i in range(1,10000+1):
    print("ITER:",i)
    network = storage.latest_network()
    run_selfplay(config, network, target_network, replay_buffer, 1)
    #pickle.dump( replay_buffer.get_experience(), open( "repbuffer_"+str(i)+".pkl", "wb" ) )
    if i>=100:
      trained_network = train_network(config, network, target_network, optimizer, replay_buffer, tb_logger, i-1)
      storage.save_network(step = 0, network=trained_network)
      if i%100==0: 
        torch.save({'state_dict': {"pred": trained_network.pred_model.state_dict(),
                                  "rep" : trained_network.rep_model.state_dict(),
                                  "dyn" : trained_network.dyn_model.state_dict()}}
                    , "model_"+str(i)+".tar")
    

  return storage.latest_network()

# same as the board main
def run_selfplay(config: MuZeroConfig, network: Network, target_network: Network,
                 replay_buffer: ReplayBuffer, num_episodes: int):
  network.eval()
  for _ in range(num_episodes):
    game = play_game(config, network, target_network)
    replay_buffer.save_game(game)

def play_game(config: MuZeroConfig, network: Network, target_network: Network) -> Game:
  game = Game(config.action_space_size, config.discount, config.environment)
  current_observation = game.make_image(-1)
  netout = network.initial_inference(current_observation)
  
  while not game.terminal() and len(game.history) < config.max_moves:
    if network.training_steps() < 1:
      netout = NetworkOutput(netout.value, netout.reward, network.default_action_logits(), netout.hidden_state)
    # At the root of the search tree we use the representation function to
    # obtain a hidden state given the current observation.
    root = Node(0)
    root.visit_count += 1
    expand_node(root, game.to_play(), game.legal_actions(), netout)
    add_exploration_noise(config, root)

    # We then run a Monte Carlo Tree Search using only action sequences and the
    # model learned by the network.
    run_mcts(config, root, game.action_history(), network)
    action = select_action(config, len(game.history), root, network)
    netout = game.apply(action, network, target_network, netout)
    game.store_search_statistics(root)
  return game

tb_logger = SummaryWriter("/tmp/tb_mario")
config = make_mario_config(12, MarioEnv, 0.001)
muzero_training(config)
