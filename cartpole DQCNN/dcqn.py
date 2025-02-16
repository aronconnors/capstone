import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import random
from collections import deque
import gym
import torchvision.transforms as T

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
    
class ConvDQNAgent:
    def __init__(self, input_shape, num_actions, lr, gamma, epsilon, epsilon_decay, buffer_size):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.model = ConvDQN(input_shape, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def preprocess(self, state):
        transform = T.Compose([T.ToPILImage(), T.Grayscale(), T.Resize(self.input_shape[1:]), T.ToTensor()])
        return transform(state)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        state = self.preprocess(state)
        q_values = self.model(state.unsqueeze(0))
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = self.preprocess(next_state)
                target = reward + self.gamma * torch.max(self.model(next_state.unsqueeze(0))).item()
            state = self.preprocess(state)
            target_f = self.model(state.unsqueeze(0)).detach().numpy()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(torch.tensor(target_f), self.model(state.unsqueeze(0)))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

# Initialize environment and agent
env = gym.make('CartPole-v1', render_mode="rgb_array")
input_shape = (1, 84, 84)  # Stack of 1 grayscale frames, resized to 84x84
num_actions = env.action_space.n
agent = ConvDQNAgent(input_shape, num_actions, lr=0.00025, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=1000000)

# Train the ConvDQN agent
batch_size = 32
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = env.render()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = env.render()
        cv2.imshow('Game', next_state)
        cv2.waitKey(1)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.replay(batch_size)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")


# Evaluate the trained agent
total_rewards = []
num_episodes_eval = 10
for _ in range(num_episodes_eval):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    total_rewards.append(total_reward)
print(f"Average Total Reward (Evaluation): {np.mean(total_rewards)}")