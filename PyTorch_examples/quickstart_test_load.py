# This code corresponds to the PyTorch Quickstart tutorial where a model is loaded and tested.

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# We pass the Dataset as an argument to DataLoader. This wraps an iterable over our dataset,
# and supports automatic batching, sampling, shuffling and multiprocess data loading.
# Here we define a batch size of 64, i.e. each element in the dataloader iterable will return a batch
# of 64 features and labels.

batch_size = 64

# Create data loaders.
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# Get cpu or gpu device.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model (same as in quickstart.py).
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Loading Models

# Saved models can be used by loading them.  The process for loading a model includes re-creating
# the model structure and loading the state dictionary into it.

model2 = NeuralNetwork()  # Note that unlike the training case, .to(device) is not used here.
model2.load_state_dict(torch.load("model.pth"))
print(model2)

# This model can now be used to make predictions.

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model2.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model2(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
