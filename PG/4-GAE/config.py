import torch

env_name = 'CartPole-v1'
gamma = 0.99
lambda_gae = 0.96
lr = 0.0001
goal_score = 200
log_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ciritic_coefficient = 0.5 
entropy_coefficient = 0.01