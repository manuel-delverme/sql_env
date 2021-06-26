import getpass
import sys
# import experiment_buddy

seed = 1
num_buffers = 1
num_actors = 1
batch_size  = 1
total_steps = 1e5
learning_rate = 1e-4
mometum =  0.9
epsilon = 1
num_learner_threads = 1
save_every = 100



num_steps = 30  # episode length
device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env_name = "SQL-v1"
user = getpass.getuser()
complexity = 3

# experiment_buddy.register(locals())
DEBUG = sys.gettrace() is not None
PROC_NUM = 1
HOST = "mila" if user in ("d3sm0", "esac") else ""
YAML_FILE = "env_suite.yml"
# tb = experiment_buddy.deploy(host=HOST, sweep_yaml=YAML_FILE, proc_num=PROC_NUM, wandb_kwargs={"mode": "disabled" if DEBUG else "online", "entity": "rl-sql"})
