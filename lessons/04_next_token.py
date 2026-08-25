import torch
import torch.nn as nn

torch.manual_seed(0)

VOCAB = ["<s>", "a", "b", "c", "</s>"]
V = len(VOCAB)

seq = [0, 1, 2, 3, 4]

inputs = torch.tensor([seq[:-1]])