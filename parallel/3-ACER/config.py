import torch

env_name = 'CartPole-v1'
gamma = 0.99
lr = 0.001
goal_score = 200
log_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_episode = 30000


replay_memory_capacity = 1000
truncation_clip = 10
delta = 1
trust_region_decay = 0.99
replay_ratio = 4
max_gradient_norm = 40
