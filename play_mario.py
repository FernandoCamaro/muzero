import torch
import time

from mario.MarioEnv import MarioEnv
from mario.MarioNet import MarioNet as myNetwork
from mcts import expand_node, run_mcts
from util import select_action_pit
from muzero import MuZeroConfig, make_mario_config
from network import Network
from game import Game
from util_leaf import Action, Node


# config
config = make_mario_config(12, MarioEnv, 0.005)

# load the trained model
net =  myNetwork(config.action_space_size, cuda = True)
checkpoint = torch.load("model_19.tar")
net.rep_model.load_state_dict(checkpoint["state_dict"]['rep'])
net.dyn_model.load_state_dict(checkpoint["state_dict"]['dyn'])
net.pred_model.load_state_dict(checkpoint["state_dict"]['pred'])
net.eval()

# main function
def play_game(config: MuZeroConfig, network: Network) -> Game:
  game = Game(config.action_space_size, config.discount, config.environment)
  game.environment.env.render()
  time.sleep(0.1)     
  
  step = 1
  while not game.terminal() and step < 200:

    current_observation = game.make_image(-1)
    netob  = network.initial_inference(current_observation)
    print("reward:",netob.reward)
    root = Node(0)
    root.visit_count += 1
    expand_node(root, game.to_play(), game.legal_actions(), netob)

    # We then run a Monte Carlo Tree Search using only action sequences and the
    # model learned by the network.
    run_mcts(config, root, game.action_history(), network)
    for action, child in root.children.items():
        print(action.index, child.visit_count, child.value(), child.prior)
    action = select_action_pit(config, len(game.history), root, network, 0.1)
    print(action.index)
    _ = game.apply(action, network, netob)
    game.store_search_statistics(root)
    game.environment.env.render()
    time.sleep(0.05)
    step += 1
  print(game.root_values)
  print(game.rewards)

play_game(config, net)