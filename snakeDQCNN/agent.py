import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import QTrainer, ConvDQN # Linear_QNet
from helper import plot
import torchvision.transforms as T
import cv2

MAX_MEMORY = 100_000 #TUNABLE
BATCH_SIZE = 1000 #TUNABLE
LR = 0.0001 #TUNABLE

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.7 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.input_shape = (1, 84, 84)
        self.model = ConvDQN(self.input_shape, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def preprocess(self, state):
        transform = T.Compose([T.ToPILImage(), T.Grayscale(), T.Resize(self.input_shape[1:]), T.ToTensor()])
        return transform(state)
    
    def get_state(self, game):
        bgr_array = np.array(game.rgb_array[..., ::-1])
        bgr_array = np.transpose(bgr_array, (1, 0, 2))
        return self.preprocess(bgr_array)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEM reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        #print(actions)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(0, 0 - self.n_games)
        final_move = [0, 0, 0, 0]
        if random.randint(0, 50) < self.epsilon:
            move = random.randint(0, 3)
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            #state0 = self.preprocess(state)
            #print(state0.shape)
            #state = self.preprocess(state)
        
        # Add batch dimension (dim=0) and channel dimension (dim=1)
        # Now shape will be (1, 1, H, W)
           
            q_values = self.model(state0)
            print(q_values)
            move = torch.argmax(q_values).item()
        final_move[move] = 1  # Convert to one-hot encoded action
        return final_move 
        
        

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_rewards = []
    total_score = 0
    totalReward = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:

        #get old state 
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        #print(final_move)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        totalReward += reward
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #train long memory (replay memory or experience replay)
            #plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Reward", totalReward)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            #mean_reward = totalReward / agent.n_games
            #plot_rewards.append(mean_reward)
            #plot(plot_scores, plot_rewards)

            totalReward = 0


if __name__ == '__main__':
    train()