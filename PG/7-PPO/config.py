import torch

env_name = 'CartPole-v1'
gamma = 0.99
lr = 0.001
goal_score = 200
log_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda_gae = 0.96
epsilon_clip = 0.2
ciritic_coefficient = 0.5
entropy_coefficient = 0.01
batch_size = 8
epoch_k = 10
