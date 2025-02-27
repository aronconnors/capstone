import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class ConvDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ConvDQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.calc_conv_output(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def calc_conv_output(self, shape):
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.size()))

    def forward(self, x):
        conv_out = self.conv_layers(x).view(x.size()[0], -1)
        return self.fc_layers(conv_out)
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state) 
        state = torch.tensor(state, dtype=torch.float)
        next_state = np.array(next_state) 
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = np.array(action)  # Converts [0, 0, 1] to 2
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n,x)

        if len(state.shape) == 3:
            state = state.unsqueeze(0)       # (1, 1, H, W)
            next_state = next_state.unsqueeze(0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        #elif len(state.shape) == 4:
            # Batch of grayscale images (B, H, W) -> Convert to (B, 1, H, W)
            #state = state.unsqueeze(1)
            #next_state = next_state.unsqueeze(1)
            #pass
        #else:
            #print("fuck")

        #1: predicted Q values with current state
        pred = self.model(state)
        next = self.model(next_state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(next[idx])

            target[idx,action[idx]] = Q_new
        # 2: 
        # Q_new = r + y * max(next_predicted Q vaule) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        loss = self.criterion(target, pred)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()