import sys

import torch

env = "PongNoFrameskip-v4"
mode = "train"
xpid = None
disable_checkpoint = True
savedir = "~/logs/torchbeast"
num_actors = 1
total_steps = 100000
batch_size = 8
save_every = int(1e3)
unroll_length = 80
num_buffers = max(2 * num_actors, batch_size)  # change this to be larger than both but not smaller
num_learner_threads = 2
disable_cuda = True
use_lstm = True
seed = 33
entropy_cost = 0.0006
baseline_cost = 0.5
discounting = 0.99
reward_clipping = "abs_one"
learning_rate = 0.00048
alpha = 0.99
momentum = 0
epsilon = 0.01
grad_norm_clipping = 40.0
DEBUG = sys.gettrace() is not None
device = torch.device("cpu")
env_id = "SQL-v0"
#config_dict = locals()
