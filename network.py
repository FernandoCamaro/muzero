from abc import ABC, abstractmethod
import torch
import numpy as np
import typing
from typing import Dict, List
from util_leaf import Action

class NetworkOutput(typing.NamedTuple):
  value: float
  reward: float
  policy_logits: Dict[Action, float]
  hidden_state: List[float]


class Network(object):

  def  __init__(self, action_space_size: int):
    self.action_space_size = action_space_size
    self.tr_steps = 0

  def initial_inference(self, image) -> NetworkOutput:
    # representation + prediction function
    action_logits = self.default_action_logits()
    return NetworkOutput(0, 0, action_logits, [])

  def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
    # dynamics + prediction function
    action_logits = self.default_action_logits()
    return NetworkOutput(0, 0, action_logits, [])

  def get_weights(self):
    # Returns the weights of this network.
    return []

  def training_steps(self) -> int:
    # How many steps / batches the network has been trained for.
    return self.tr_steps

  def default_action_logits(self):
    action_logits = {}
    for i in range(self.action_space_size):
      action_logits[Action(i)] = 1.0
    return action_logits

  def eval(self):
    return None

def make_uniform_network(action_space_size: int):
  return Network(action_space_size)

class canonicalNetwork(Network, ABC):
    def __init__(self, representation_model, dynamics_model, prediction_model, action_space_size: int, cuda: bool=True):
        super(canonicalNetwork, self).__init__(action_space_size)
        self.action_space_size = action_space_size
        self.cuda = cuda
        self.rep_model = representation_model
        self.dyn_model = dynamics_model
        self.pred_model = prediction_model
        self.models = [self.rep_model, self.dyn_model, self.pred_model]
        if cuda:
          self.rep_model = self.rep_model.cuda()
          self.dyn_model = self.dyn_model.cuda()
          self.pred_model = self.pred_model.cuda()
        self.training = False 
    
    def train(self):
        self.training = True
        for model in self.models:
          model.train()

    def eval(self):
        self.training = False
        for model in self.models:
          model.eval()


    def initial_inference(self, obs) -> NetworkOutput:
        # obs: numpy of array of shape BsxHxW or HxW: numpy array
        obs = obs.astype(np.float32)

        # prepare the observation to be feed to the network
        if len(obs.shape) == 2:
            H,W = obs.shape
            Bs = 1
        elif len(obs.shape) == 3:
            Bs, H, W = obs.shape
        else:
            raise Exception("observation shape not supported")
        obs = torch.tensor(obs).view(Bs, 1, H, W)
        if self.cuda:
            obs = obs.cuda()
        
        # main calls
        hidden_state = self.rep_model(obs)
        v, pi = self.pred_model(hidden_state)

        return self.process_output(v, pi, hidden_state, None)

    @abstractmethod
    def encode_action(self, action, H, W):
      action_plane = torch.zeros((H, W))
      if action.index < H*W:
        row = int(action.index / W)
        column = int(action.index % W)
        action_plane[row,column] = 1.0
      return action_plane

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
        action_plane = torch.zeros((Bs, 1, H, W))
        if type(action) == Action:
            action = [action]
        for i,a in enumerate(action):
          action_plane[i,0] =  self.encode_action(action, H, W)
        if self.cuda:
            action_plane = action_plane.cuda()

        # main calls
        s = torch.cat((hidden_state, action_plane), dim=1)
        hidden_state, reward = self.dyn_model(s)
        v, pi = self.pred_model(hidden_state)
        
        return self.process_output(v, pi, hidden_state, reward)

    def process_output(self, v, pi, hidden_state, reward = None):

        Bs = hidden_state.shape[0]
        
        if not self.training:
            v = v.cpu().detach().numpy()
            if reward != None:
                reward = reward.cpu().detach().numpy()
            pi = pi.cpu().detach().numpy()
            hidden_state = hidden_state.cpu().detach().numpy()
            if Bs == 1:
                v = v.item()
                if reward != None:
                    reward = reward.item()
                pi = pi[0]
                action_logits = {}
                for i,p in enumerate(pi):
                    action_logits[Action(i)] = p
                pi = action_logits
                hidden_state = hidden_state[0]
        
        return NetworkOutput(v, reward, pi, hidden_state)


    def parameters(self):
      params = list()
      for model in self.models():
        params += model.params()
      return params

class boardNetwork(canonicalNetwork):
  def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

  def encode_action(self, action, H, W):
    return super().encode_action(action, H, W)


class videogameNetwork(canonicalNetwork):
  def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

  def encode_action(self, action, H, W):
    action_plane = torch.ones((H, W))
    action_plane *= action.index/(self.action_space_size-1)
    return action_plane

