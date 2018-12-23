import torch

env_name = 'CartPole-v1'
gamma = 0.99
lr = 0.002
goal_score = 200
log_interval = 10
max_episode = 30000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



replay_memory_capacity = 10000
n_step = 3 
local_mini_batch = 32
batch_size = 32
alpha = 0.5
beta = 0.4