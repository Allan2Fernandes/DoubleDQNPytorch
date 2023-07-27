import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):
    def __init__(self, device, observation_space_size, action_space_size):
        super(Q_Network, self).__init__()
        self.device = device
        self.hidden_layer1 = nn.Linear(in_features=observation_space_size, out_features=512, device=self.device)
        self.hidden_layer2 = nn.Linear(in_features=512, out_features=512, device=self.device)
        self.hidden_layer3 = nn.Linear(in_features=512, out_features=256, device=self.device)
        self.final_layer = nn.Linear(in_features=256, out_features=action_space_size, device=self.device)

        self.hidden_layer_activation1 = nn.ReLU()
        self.hidden_layer_activation2 = nn.ReLU()
        self.hidden_layer_activation3 = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        pass

    def forward(self, state):
        x = self.hidden_layer1(state)
        x = self.hidden_layer_activation1(x)
        x = self.bn1(x)

        x = self.hidden_layer2(x)
        x = self.hidden_layer_activation2(x)
        x = self.bn2(x)

        x = self.hidden_layer3(x)
        x = self.hidden_layer_activation3(x)
        x = self.bn3(x)

        x = self.final_layer(x)
        return x

    def update_network_weights(self, x, y, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_function = torch.nn.MSELoss()
        for _ in epochs:
            optimizer.zero_grad()
            prediction = self.forward(x)
            loss = loss_function(prediction, y)
            loss.backward()
            optimizer.step()
            pass
        pass

    def copy_weights_from_main_network(self, main_network: nn.Module):
        self.load_state_dict(main_network.state_dict())
        pass

    pass