import sys

import experiment_buddy
import getpass

lr = 7e-4  # _, help='learning rate (default: 7e_4)')
eps = 1e-5  # _, help='RMSprop optimizer epsilon (default: 1e_5)')
alpha = 0.99  # _, help='RMSprop optimizer apha (default: 0.99)')
gamma = 0.99  # _, help='discount factor for rewards (default: 0.99)')
use_gae = False  # _, help='use generalized advantage estimation')
gae_lambda = 0.95  # _, help='gae lambda parameter (default: 0.95)')
entropy_coef = 0.01  # _, help='entropy term coefficient (default: 0.01)')
value_loss_coef = 0.5  # _, help='value loss coefficient (default: 0.5)')
max_grad_norm = 0.5  # _, help='max norm of gradients (default: 0.5)')
seed = 1  # _, help='random seed (default: 1)')
num_processes = 1  # _, help='how many training CPU processes to use (default: 16)')
num_steps = 128  # _, help='number of forward steps in A2C (default: 5)')
ppo_epoch = 4  # _, help='number of ppo epochs (default: 4)')
num_mini_batch = 1  # _, help='number of batches for ppo (default: 32)')
clip_param = 0.2  # _, help='ppo clip parameter (default: 0.2)')
log_interval = 100
save_interval = 10000  # _, help='save interval, one save per n updates (default: 100)')
eval_interval = None  # _, help='eval interval, one eval per n updates (default: None)')
num_env_steps = 1e6  # _, help='number of environment steps to train (default: 10e6)')
log_dir = '/tmp/gym/'  # _, help='directory to save agent logs (default: /tmp/gym)')
save_dir = './trained_models/'  # _, help='directory to save agent logs (default: ./trained_models/)')
no_cuda = False  # _, help='disables CUDA training')
recurrent_policy = False  # _, help='use a recurrent policy')
use_linear_lr_decay = False  # _, help='use a linear schedule on the learning rate')
device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env_name = "SQL-v1"
user = getpass.getuser()

experiment_buddy.register(locals())
HOST = "mila" if user in ("d3sm0", "esac") else ""
DEBUG = sys.gettrace() is not None
PROC_NUM = 1
tb = experiment_buddy.deploy(host=HOST, sweep_yaml="",
                             wandb_kwargs={"mode": "disabled" if DEBUG else "online", "entity": "rl-sql"})
