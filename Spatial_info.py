import torch
import torch.nn as nn
import torch.nn.functional as F


# This neural network is for mask map extraction

class EdgeDetectionNet(nn.Module):
    def __init__(self):
        super(EdgeDetectionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.edge_conv = nn.Conv2d(512, 1, kernel_size=1)  # Single-channel output for edge probability

    def forward(self, x):
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x),inplace=True)
        edge_prob_map = torch.sigmoid(self.edge_conv(x))  # Apply sigmoid to get probability map
        return edge_prob_map


# This part is for positional embedding
def generate_normalized_coordinates(height, width):
    # Generate a mesh grid of coordinates
    y_coords = torch.linspace(0, height - 1, height).view(1, height, 1).expand(1, height, width)
    x_coords = torch.linspace(0, width - 1, width).view(1, 1, width).expand(1, height, width)

    # Stack the coordinates to form a tensor of shape [2, height, width]
    coords = torch.stack((x_coords, y_coords), dim=1).squeeze(dim=0)  # Shape (2, height, width)

    # Normalize the coordinates to the range [-1, 1]
    coords[0] = (coords[0] / (width - 1)) * 2 - 1  # Normalize x coordinates
    coords[1] = (coords[1] / (height - 1)) * 2 - 1  # Normalize y coordinates

    # Add batch and channel dimensions to get shape [1, 2, height, width]
    coords = coords.unsqueeze(0)  # Shape (1, 2, height, width)

    return coords


class CoordTransform(nn.Module):
    def __init__(self, input_dim=2, output_dim=128):
        super(CoordTransform, self).__init__()
        self.conv1x1 = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        return self.conv1x1(x)








