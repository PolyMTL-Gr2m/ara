import torch
import torch.nn as nn

def print_shape(tensor):
    print(tensor.shape)
    return tensor

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(576, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        print_shape(x)
        out = print_shape(self.layer1(x))
        out = print_shape(self.layer2(out))
        out = print_shape(out.reshape(out.size(0), -1))
        out = print_shape(self.fc(out))
        out = print_shape(self.relu(out))
        out = print_shape(self.fc1(out))
        out = print_shape(self.relu1(out))
        out = print_shape(self.fc2(out))
        return out

if __name__ == "__main__":
    model = LeNet5()
    # import ipdb as pdb; pdb.set_trace()
    input_tensor = torch.rand(1,1,32,32)
    output = model(input_tensor)