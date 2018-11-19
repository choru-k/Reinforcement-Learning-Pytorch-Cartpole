import torch

env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 32
lr = 0.1
initial_exploration = 1000
goal_score = 200
log_interval = 10
update_target = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
replay_memory_capacity = 1000


n_step = 3
