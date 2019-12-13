from typing import List
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from network import Network, NetworkOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util_leaf import Action

class tictactoeNetwork(Network):
    def __init__(self, action_space_size: int, cuda: bool=True):
        super(tictactoeNetwork, self).__init__(action_space_size)
        self.action_space_size = action_space_size
        self.cuda = cuda
        self.model = tictactoeModel(self.action_space_size,4,4,64,0.2)
        if cuda:
            self.model = self.model.cuda()
        self.training = False 
    
    def train(self):
        self.training = True
        self.model.train()

    def eval(self):
        self.training = False
        self.model.eval()


    def initial_inference(self, image) -> NetworkOutput:
        # image: numpy of array of shape BsxHxW or HxW: numpy array
        image = image.astype(np.float32)

        # prepare the image to be feed to the network
        if len(image.shape) == 2:
            H,W = image.shape
            Bs = 1
        elif len(image.shape) == 3:
            Bs, H, W = image.shape
        else:
            raise Exception("image shape not supported")
        image = torch.tensor(image).view(Bs, 1, H, W)
        if self.cuda:
            image = image.cuda()

        if self.training:
            print("training")
        v, _, pi, hidden_state = self.model.initial_inference(image)

        if not self.training:
            v = v.item()
            pi = pi.cpu().squeeze().detach().numpy()
            action_logits = {}
            for i,p in enumerate(pi):
                action_logits[Action(i)] = p
            pi = action_logits
            hidden_state = hidden_state.cpu().squeeze().detach().numpy()

        return NetworkOutput(v, 0, pi, hidden_state)

    def recurrent_inference(self, hidden_state, action: List[Action]) -> NetworkOutput:
        # hidden_state: Bs x num_channels x H x W or num_channels x H x W : numpy.ndarray
        #               Bs x num_channels x H x W torch.tensor
        # action: Bs or just one action

        # hidden state
        if len(hidden_state.shape) == 3:
            num_channels, H, W = hidden_state.shape
            Bs = 1
        elif len(hidden_state.shape) == 4:
            Bs, num_channels, H, W = hidden_state.shape
        else: Exception("hidden_state shape not supported")
        if not self.training:
            hidden_state = torch.tensor(hidden_state).view(Bs, num_channels, H, W)
            if self.cuda:
                hidden_state = hidden_state.cuda()
        
        # action
        action_plane = torch.zeros((Bs, self.action_space_size, H, W))
        if type(action) == Action:
            action = [action]
        for i,a in enumerate(action):
            action_plane[i,a.index,:,:] = 1.0
        if self.cuda:
            action_plane = action_plane.cuda()

        # main call
        v, reward, pi, hidden_state = self.model.recurrent_inference(hidden_state, action_plane)

        if not self.training:
            v = v.cpu().detach().numpy()
            reward = reward.cpu().detach().numpy()
            pi = pi.cpu().detach().numpy()
            hidden_state = hidden_state.cpu().detach().numpy()
            if Bs == 1:
                v = v[0]
                reward = reward[0]
                pi = pi[0]
                action_logits = {}
                for i,p in enumerate(pi):
                    action_logits[Action(i)] = p
                pi = action_logits
                hidden_state = hidden_state[0]
        
        return NetworkOutput(v, reward, pi, hidden_state)

    def parameters(self):
        return self.model.parameters()



class tictactoeModel(nn.Module):
    def __init__(self, action_size, board_x, board_y, num_channels, dropout):
        # game params
        self.board_x, self.board_y = board_x, board_y
        self.action_size = action_size
        self.num_channels, self.dropout = num_channels, dropout

        super(tictactoeModel, self).__init__()
        # conv1 to conv3 is the representation network. Its output is the hidden state
        # conv4 to conv6 + fc1 to fc4 is the prediction network. From the hidden state it ouputs the redicted state value function and the action logits.
        # conv 7 to conv 9 is the dynamics network. From a hidden state it produces the next hidden state for a given action
        # fc5 to fc fc8 returns for a given hidden state the associated predicted reward.

        self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)
        self.conv5 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.bn3 = nn.BatchNorm2d(self.num_channels)
        self.bn4 = nn.BatchNorm2d(self.num_channels)
        self.bn5 = nn.BatchNorm2d(self.num_channels)
        self.bn6 = nn.BatchNorm2d(self.num_channels)

        self.fc1 = nn.Linear(self.num_channels*(self.board_x-2)*(self.board_y-2), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

        # from hidden state to hidden state
        self.conv7 = nn.Conv2d(self.num_channels + self.action_size, self.num_channels, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(self.num_channels,                    self.num_channels, 3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(self.num_channels,                    self.num_channels, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(self.num_channels)
        self.bn8 = nn.BatchNorm2d(self.num_channels)
        self.bn9 = nn.BatchNorm2d(self.num_channels)

        # from hidden state to reward
        self.fc5 = nn.Linear(self.num_channels*(self.board_x)*(self.board_y), 1024)
        self.fc_bn5 = nn.BatchNorm1d(1024)

        self.fc6 = nn.Linear(1024, 512)
        self.fc_bn6 = nn.BatchNorm1d(512)

        self.fc7 = nn.Linear(512, 1)

    def initial_inference(self, s):
        
        hidden_state = self.representation_forward(s)
        v, pi = self.prediction_forward(hidden_state)

        return v, 0, pi, hidden_state

    def representation_forward(self, s):
        #                                                            s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x board_x x board_y
        hidden_state = s

        return hidden_state

    def prediction_forward(self, s):
        
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn5(self.conv5(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn6(self.conv6(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = s.view(-1, self.num_channels*(self.board_x-2)*(self.board_y-2))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return torch.tanh(v), F.log_softmax(pi, dim=1)

    def dynamics_forward(self, s):
        s = F.relu(self.bn7(self.conv7(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn8(self.conv8(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn9(self.conv9(s)))                          # batch_size x num_channels x board_x x board_y

        hidden_state = s

        s = s.view(-1, self.num_channels*(self.board_x)*(self.board_y))

        s = F.dropout(F.relu(self.fc_bn5(self.fc5(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn6(self.fc6(s))), p=self.dropout, training=self.training)  # batch_size x 512

        reward = self.fc7(s)                                                                         # batch_size x action_size
        
        return hidden_state, reward

    def recurrent_inference(self, hidden_state, action):
        # hidden state: batch_size x num_channels x board_x x board_y
        # action: batch_size x actions_size, H, W
        
        s = torch.cat((hidden_state, action), dim=1)
        hidden_state, reward = self.dynamics_forward(s)
        v, pi = self.prediction_forward(hidden_state)

        return v, reward, pi, hidden_state