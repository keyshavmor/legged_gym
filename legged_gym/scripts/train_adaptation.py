import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import wandb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class AdaptationModule(nn.Module):
    def __init__(self, num_obs_history, num_privileged_obs):
        super(AdaptationModule, self).__init__()
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        self.fc1 = nn.Linear(self.num_obs_history, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.num_privileged_obs)

    def forward(self, obs_history):
        x = F.relu(self.fc1(obs_history))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_all_transitions(filename):
    loaded_data = []
    with open(filename, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                loaded_data.append(data)
            except EOFError:
                break
    return loaded_data

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the data
    transitions = load_all_transitions('transition_data.pkl')

    # Flatten the transitions into a single dataset
    obs_history = []
    privileged_obs = []

    for data in transitions:
        obs_history.append(data["observation_histories"]) #893*[2048, 1440]
        privileged_obs.append(data["privileged_observations"])

    obs_history = torch.cat(obs_history)
    obs_history = obs_history.view(-1, 1440)  # Assuming each history has a flat shape of 1440
    privileged_obs = torch.cat(privileged_obs).view(-1, 1)  # Assuming privileged observations have shape of (N, 1)

    dataset = torch.utils.data.TensorDataset(obs_history, privileged_obs)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # Initialize the model
    adaptation_module = AdaptationModule(num_obs_history=1440, num_privileged_obs=1).to(device)
    optimizer = optim.Adam(adaptation_module.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Initialize the logger
    wandb.init(project="adaptation_module")
    # wandb.watch(adaptation_module, log="all")

    # Training loop
    num_epochs = 400  # Number of epochs to train

    for epoch in range(num_epochs):
        adaptation_module.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = adaptation_module(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            running_loss += loss
            if i % 500 == 499:  # Print every 10 batches
                # print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                wandb.log({"loss": running_loss/500})
                running_loss = 0.0


if __name__ == '__main__':
    train()
