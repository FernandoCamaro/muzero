import torch
import torch.multiprocessing as multiprocessing
from multiprocessing import Queue
import numpy as np
import time

from tictactoe.TicTacToeEnv import TicTacToeEnv
from muzero import MuZeroConfig, make_board_game_config
from game import Game
from models.tictactoe_model import tictactoeNetwork as myNetwork

config = make_board_game_config(16+1, 16+1, 0.25, TicTacToeEnv, 0.01)

input_queues = []
output_queues = []
num_workers = 256
for i in range(num_workers):
    input_queues.append(Queue())
    output_queues.append(Queue())

def feeder(id, n):
        for i in range(n):
            game = Game(config.action_space_size, config.discount, config.environment)
            current_observation = game.make_image(-1)
            input_queues[id].put(current_observation)
            #qi1.put(current_observation)
            #print("feeder",id, "put", i )
            result = output_queues[id].get()
            #print("feeder",id, "get", i )

# network process
def network_process(num_feed):
    # input_queues = [qi1, qi2, qi3, qi4]
    # output_queues = [qo1, qo2, qo3, qo4]


    network =  myNetwork(config.action_space_size, cuda = True)
    # checkpoint = torch.load("model_tmp.tar")
    # network.model.load_state_dict(checkpoint["state_dict"])
    network.eval()
    for i in range(num_feed):
        inputs = []
        for j in range(len(input_queues)):
            while(input_queues[j].empty()):
                pass
            inputs.append(input_queues[j].get())
        
        inputs = np.stack(inputs)
        no = network.initial_inference(inputs)

        for j in range(len(output_queues)):
            output_queues[j].put((no.value[j], no.hidden_state[j] ))

if __name__ == '__main__':
    
    # try:
    #     multiprocessing.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     pass    

    # create 5 threads in which each one creates a game, puts the first observation in its queue
    # then check for the output queue
    t0 = time.time()
    processes = [ ]
    num_feed = 200
    for i in range(num_workers):
        t = multiprocessing.Process(target=feeder, args=(i,num_feed))
        processes.append(t)
        t.start()

    
    # network =  myNetwork(config.action_space_size, cuda = True)
    # torch.save({'state_dict': network.model.state_dict()}, "model_tmp.tar")
    net_process =  multiprocessing.Process(target=network_process, args=(num_feed,))
    net_process.start()

    # wait until all processes finish
    for one_process in processes:
        one_process.join()
    net_process.join()
    t1 = time.time()
    print(t1-t0)

    # sequential
    network =  myNetwork(config.action_space_size, cuda = True)
    network.eval()
    
    # current_observation = game.make_image(-1)
    t0 = time.time()
    for i in range(num_feed):
        for j in range(num_workers):
            game = Game(config.action_space_size, config.discount, config.environment)
            current_observation = game.make_image(-1)
            no = network.initial_inference(current_observation)
    t1 = time.time()
    print(t1-t0)