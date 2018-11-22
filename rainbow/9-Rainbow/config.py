import torch

env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 32
lr = 0.0001
initial_exploration = 1000
goal_score = 200
log_interval = 10
update_target = 100
replay_memory_capacity = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi_Step
n_step = 1

# PER
small_epsilon = 0.0001
alpha = 1
beta_start = 0.1

# Noisy Net
sigma_zero = 0.5

# Distributional
num_support = 8
V_max = 5
V_min = -5
