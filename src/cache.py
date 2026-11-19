import torch

from constant import *

class Cache():
    def __init__(self, ):
        self.seq_len = 0
        self.k_cache = [None] * num_hidden_layers
        self.v_cache = [None] * num_hidden_layers
        self.conv_state = [None] * num_hidden_layers
        self.recur_state = [None] * num_hidden_layers

        