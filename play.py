import torch

from tictactoe.TicTacToeEnv import TicTacToeEnv
from models.tictactoe_model import tictactoeNetwork as myNetwork
from mcts import expand_node, run_mcts
from util import select_action_pit
from muzero import MuZeroConfig, make_board_game_config
from network import Network
from game import Game
from util_leaf import Action, Node


# config
config = make_board_game_config(16+1, 16+1, 0.25, TicTacToeEnv, 0.01)

# load the trained model
net =  myNetwork(config.action_space_size, cuda = True)
checkpoint = torch.load("model_10.tar")
net.model.load_state_dict(checkpoint["state_dict"])
net.eval()



# main function
def play_game(config: MuZeroConfig, network: Network) -> Game:
  game = Game(config.action_space_size, config.discount, config.environment)
  
  player = 1
  while not game.terminal():

    current_observation = game.make_image(-1)
    netob  = network.initial_inference(current_observation)
    print("value:",netob.value)
    if player == 1: # human player
        # show the board
        game.environment.display(game.environment.board)
        action = input("row column: ")
        row, column = [int(x) for x in action.split(' ')]
        action = Action(int(row*4+column))
        player = -1
    else:
        root = Node(0)
        expand_node(root, game.to_play(), game.legal_actions(),
                    netob)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        for action, child in root.children.items():
          print(action.index, child.visit_count, child.value(), child.prior)
        action = select_action_pit(config, len(game.history), root, network, 0.01)
        player = 1
    game.apply(action)
  whowins = game.environment.getGameEnded(game.environment.board, 1)
  if whowins != 0:
      game.environment.display(game.environment.board)
  if whowins == 1:
      print("Human player wins")
  elif whowins == -1:
      print("Computer wins")
  elif whowins != 0:
      print("Draw")
play_game(config, net)
