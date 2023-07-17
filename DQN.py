import random
from collections import deque
import numpy as np
import torch
import Q_Network


class DQN:
    def __init__(self, state_size, action_size, device):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0 # Exploration vs exploitation
        self.epsilon_decay_rate = 0.99
        self.min_epsilon = 0.15
        self.gamma = 0.99 # Discount factor
        self.update_rate = 100
        self.replay_buffer = deque(maxlen=2000)
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=0.001)
        self.loss_function = torch.nn.MSELoss()
        pass

    def build_network(self):
        model = Q_Network.Q_Network(self.device, self.state_size, self.action_size)
        return model

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))  # push it into the queue
        pass

    def epsilon_greedy(self, state):
        # Generate random number
        if random.uniform(0,1) < self.get_epsilon():
            # Below epsilon, explore
            Q_values = np.random.randint(self.action_size)
        else:
            # Otherwise, exploit using the main network
            self.main_network.eval()
            with torch.no_grad():
                Q_values = int(torch.argmax(self.main_network(state)[0]))
            pass
        return Q_values

    def train(self, batch_size):
        # Get a mini batch from the replay memory
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.target_network.eval()
            if not done:
                with torch.no_grad():
                    target_Q = reward + self.gamma*torch.max(self.target_network(next_state))
            else:
                target_Q = reward
                pass
            self.main_network.eval()
            with torch.no_grad():
                Q_values = self.main_network(state)
            Q_values[0][action] = target_Q  # batch size = 1
            self.train_NN(x=state, y=Q_values)
            pass
        pass

    def train_double_DQN(self, batch_size):
        list_of_states = []
        list_of_Q_values = []
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                with torch.no_grad():
                    self.main_network.eval()
                    # Select action with the maximum Q-value from the main network
                    next_action = np.argmax(self.main_network(next_state)[0])
                    # Evaluate the Q-value of the selected action using the target network
                    self.target_network.eval()
                    target_Q = reward + self.gamma * self.target_network(next_state)[0][next_action]

            else:
                target_Q = reward
            Q_values = self.main_network(state)
            Q_values[0][action] = target_Q
            list_of_states.append(state)
            list_of_Q_values.append(Q_values)

            pass
        state_tensor = torch.cat(list_of_states, dim=0)
        Q_value_tensor = torch.cat(list_of_Q_values, dim=0)
        self.train_NN(x=state_tensor, y=Q_value_tensor)
        pass

    def train_NN(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.main_network(x)
        loss = self.loss_function(prediction, y)
        loss.backward()
        self.optimizer.step()
        pass

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate
        pass

    def get_epsilon(self):
        return max(self.epsilon, self.min_epsilon)

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())
        pass


