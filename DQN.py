import random
from collections import deque, namedtuple
import numpy as np
import torch
import Q_Network
import math


class DQN:
    def __init__(self, state_size, action_size, device):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1 # Exploration vs exploitation
        self.epsilon_decay_rate = 0.99
        self.min_epsilon = 0.1
        self.gamma = 0.99 # Discount factor
        self.update_rate = 200
        self.model_save_rate = 1
        deque_len = 5000
        self.replay_buffer = deque(maxlen=deque_len)
        self.main_network = self.build_network().to(device)
        self.target_network = self.build_network().to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=0.00025)
        self.loss_function = torch.nn.MSELoss()
        self.Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'priority'])
    def build_network(self):
        model = Q_Network.Q_Network(self.device, self.state_size, self.action_size)
        return model

    def store_transition(self, state, action, reward, next_state, done, time_step):
        self.target_network.eval()
        self.main_network.eval()
        # Calculate the temporal difference error
        target_Q = reward + self.gamma * torch.max(self.target_network(next_state))
        Q_s_a = self.main_network(state)[0][action]
        td_error = target_Q - Q_s_a
        priority = math.pow(abs(td_error.item() + 1e-6), 0.4)

        total_priority = sum([experience.priority for experience in self.replay_buffer])
        priority_sum = priority + total_priority
        priority = priority/priority_sum
        x = math.pow(max(1, len(self.replay_buffer)), -1) * math.pow(priority, -1)
        beta = self.calculate_beta(time_step)
        weight = math.pow(x, beta)
        memory = self.Experience(state=state, action=action, reward=reward, next_state=next_state, done=done, priority=weight)
        self.replay_buffer.append(memory)  # push it into the queue
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
        for state, action, reward, next_state, done, td_error in minibatch:
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
        #minibatch = random.sample(self.replay_buffer, batch_size)
        minibatch = self.sample_replay_buffer(batch_size=batch_size)
        for state, action, reward, next_state, done, priority in minibatch:
            if not done:
                with torch.no_grad():
                    self.main_network.eval()
                    # Select action with the maximum Q-value from the main network
                    next_action = np.argmax(self.main_network(next_state)[0].to('cpu'))
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

    import random

    def sample_replay_buffer(self, batch_size):
        list_probabilities = [experience.priority for experience in self.replay_buffer]

        # Normalize the priorities to get probabilities
        total_priority = sum(list_probabilities)
        probabilities = [priority / total_priority for priority in list_probabilities]

        # Calculate cumulative probabilities for sampling
        cum_weights = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

        # Use random.choices with calculated cum_weights to sample from the buffer
        random_list = random.choices(self.replay_buffer, cum_weights=cum_weights, k=batch_size)

        return random_list

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

    def calculate_beta(self, x):
        # Anneal beta from 0.4 -> 1
        return 1 - 0.6*math.exp(-x/30)

    def save_model(self, path):
        torch.save(self.main_network.state_dict(), path)
        pass


