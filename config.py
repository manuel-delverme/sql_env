import torch
import sys

# lr = 7e-4  # _, help='learning rate (default: 7e_4)')
# eps = 1e-5  # _, help='RMSprop optimizer epsilon (default: 1e_5)')
# alpha = 0.99  # _, help='RMSprop optimizer apha (default: 0.99)')
gamma = 0.9
# use_gae = False  # _, help='use generalized advantage estimation')
# gae_lambda = 0.95  # _, help='gae lambda parameter (default: 0.95)')
# entropy_coef = 0.01  # _, help='entropy term coefficient (default: 0.01)')
# value_loss_coef = 0.5  # _, help='value loss coefficient (default: 0.5)')
# max_grad_norm = 0.5  # _, help='max norm of gradients (default: 0.5)')
seed = 1
num_processes = 1
# ppo_epoch = 2  # _, help='number of ppo epochs (default: 4)')
# num_mini_batch = 1  # _, help='number of batches for ppo (default: 32)')
# clip_param = 0.2  # _, help='ppo clip parameter (default: 0.2)')
# log_interval = 100
# log_query_interval = 1_000
# save_interval = 10000  # _, help='save interval, one save per n updates (default: 100)')
# eval_interval = None  # _, help='eval interval, one eval per n updates (default: None)')
num_env_steps = int(1e8)
# log_dir = '/tmp/gym/'  # _, help='directory to save agent logs (default: /tmp/gym)')
# save_dir = './trained_models/'  # _, help='directory to save agent logs (default: ./trained_models/)')
# no_cuda = False  # _, help='disables CUDA training')
# recurrent_policy = False  # _, help='use a recurrent policy')
# use_linear_lr_decay = False  # _, help='use a linear schedule on the learning rate')
#
# num_steps = 30  # episode length
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env_name = "SQL-v1"
# user = getpass.getuser()
complexity = 3
#
# # DQN
buffer_size = 10000
#
embedding_size = 64
encoder_rnn_hidden_size = 192
action_scorer_hidden_dim = 128
DEBUG = sys.gettrace() is not None
#
action_history_len = 10
num_tasks = 3
max_columns = 3  # 3
cheat_hidden_parameter = False
cheat_columns = False
#
#
# # replay memory
# replay_memory_capacity = 500000  # adjust this depending on your RAM size
# replay_memory_priority_fraction = 0.25
# update_per_k_game_steps = 4
replay_batch_size = 32
#
# # epsilon greedy
epsilon_anneal_episodes = 500
epsilon_anneal_from = .9
epsilon_anneal_to = .1
#
#
# training_batch_size = 16
learning_rate = 0.001
clip_grad_norm = 5
