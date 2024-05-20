import gymnasium as gym

from stable_baselines3 import A2C

# env = gym.make("CartPole-v1", render_mode="human")

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)

# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render()
#     # VecEnv resets automatically
#     # if done:
#     #   obs = env.reset()

# env.close()


import torch
import torch.nn as nn
import torch.optim as optim

class CustomModel(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim):
        super(CustomModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp_extractor = nn.Linear(input_dim, latent_dim).to(self.device)
        self.action_network = nn.Sequential(
                                    nn.Linear(latent_dim, output_dim),
                                    nn.Softmax(dim=-1)
                                    ).to(self.device)

    def forward(self, x):
        latent = self.mlp_extractor(x)
        action_probs = self.action_network(latent)
        return action_probs

# Instantiate the model
input_dim = 4
latent_dim = 128
output_dim = 2
model = CustomModel(input_dim, output_dim, latent_dim)

# Define an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define a loss function
criterion = nn.MSELoss()  # Use a suitable loss function for your task

# Dummy input and target
input = torch.randn(1, input_dim).to(model.device)
target = torch.randn(1, output_dim).to(model.device)

# Function to print model parameters
def print_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data}")

# Function to print model gradients
def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad: {param.grad}")
        else:
            print(f"{name} grad: None")

# Register hooks to check gradients during backpropagation
for name, param in model.named_parameters():
    if param.requires_grad:
        param.register_hook(lambda grad: print(f'Gradient of {name}: {grad}'))

# Initial parameters
print("Initial parameters:")
print_parameters(model)

# Training step
optimizer.zero_grad()  # Zero the gradients

output = model(input)  # Forward pass
loss = criterion(output, target)  # Compute loss

# print("Loss:", loss.item())

loss.backward()  # Backward pass to compute gradients

# Print gradients
# print("Gradients:")
# print_gradients(model)

optimizer.step()  # Update parameters

# Updated parameters
print("\nUpdated parameters:")
print_parameters(model)
a = 1
