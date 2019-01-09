import torch

env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 1000
goal_score = 200
log_interval = 10
update_target = 100
replay_memory_capacity = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 32
burn_in_length = 4
eta = 0.9
local_mini_batch = 8