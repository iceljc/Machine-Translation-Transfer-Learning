import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext import data, datasets
from data import rebatch
from model import make_model, SimpleLossCompute
from utils import print_data_info, print_examples, greedy_decode, lookup_words
from settings import params
import sacrebleu



saved_model = "mt_model.pt"

model = torch.load(saved_model)

print(model.state_dict().keys())

print(model.state_dict()['encoder.rnn.weight_ih_l0'].shape)
print(model.state_dict()['encoder.rnn.weight_hh_l0'].shape)

print(model.state_dict()['encoder.rnn.weight_ih_l0_reverse'].shape)
print(model.state_dict()['encoder.rnn.weight_hh_l0_reverse'].shape)










