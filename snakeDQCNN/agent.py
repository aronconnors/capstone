import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import QTrainer, ConvDQN # Linear_QNet
from helper import plot
import torchvision.transforms as T
import cv2

MAX_MEMORY = 500_000 #TUNABLE
BATCH_SIZE = 2000 #TUNABLE
LR = 0.0075 #TUNABLE

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.99 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.input_shape = (1, 84, 84)
        #self.model = Linear_QNet(11, 256, 3)
        self.model = ConvDQN(self.input_shape, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def preprocess(self, state):
        transform = T.Compose([T.ToPILImage(), T.Grayscale(), T.Resize(self.input_shape[1:]), T.ToTensor()])
        return transform(state)
    
    def get_state(self, game):
        #print(np.array(game.rgb_array).shape)
        '''
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food location
            game.food.x < game.head.x, #food left
            game.food.x > game.head.x, #food right
            game.food.y < game.head.y, #food up
            game.food.y > game.head.y #food down
        ]'''
        #returns true/false to 0, 1
        #return np.array(state, dtype=int)
        #processed_image = self.preprocess(game.rgb_array)
        #processed_image = processed_image.squeeze(0).cpu().numpy()  # Remove batch dimension & convert to NumPy

        # Ensure correct dtype
        #processed_image = (processed_image * 255).astype(np.uint8)  # If values are normalized (0 to 1)

        #cv2.imshow("Preprocessed Image", processed_image)
        #cv2.waitKey(0)  # Wait for a key press
        return self.preprocess(game.rgb_array)

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
        self.epsilon = 2000 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 1000) < self.epsilon:
            move = random.randint(0, 2)
        else:
            #state0 = torch.tensor(state, dtype=torch.float)
            #state0 = self.preprocess(state)
            #print(state0.shape)
            #state = self.preprocess(state)
        
        # Add batch dimension (dim=0) and channel dimension (dim=1)
        # Now shape will be (1, 1, H, W)
           
            q_values = self.model(state.unsqueeze(0))
            print(q_values)
            move = torch.argmax(q_values).item()
        final_move[move] = 1  # Convert to one-hot encoded action
        return final_move 
        
        

def train():
    plot_scores = []
    plot_mean_scores = []
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

            totalReward = 0

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()