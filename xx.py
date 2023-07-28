import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class LocalAdaptiveActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, slopes):
        ctx.save_for_backward(input, slopes)
        output = F.relu(input) * slopes
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, slopes = ctx.saved_tensors
        grad_input = grad_output * (input > 0).float() * slopes
        grad_slopes = (grad_output * (input > 0).float() * input).sum(dim=0)
        return grad_input, grad_slopes

class LocalAdaptiveReLU(nn.Module):
    def __init__(self, num_neurons):
        super(LocalAdaptiveReLU, self).__init__()
        self.slopes = nn.Parameter(torch.ones(num_neurons))

    def forward(self, input):
        return LocalAdaptiveActivation.apply(input, self.slopes)

# Create a toy dataset
torch.manual_seed(42)
X = torch.randn(100, 1)  # Input data
y = 3 * X + torch.randn(100, 1) * 0.5  # Output data with some noise

# Create the Multi-layer LocalAdaptiveNetwork
input_size = 1
hidden_sizes = [20, 20, 20]  # List of hidden layer sizes
output_size = 1

class MultiLayerLocalAdaptiveNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MultiLayerLocalAdaptiveNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([
            LocalAdaptiveReLU(hidden_sizes[i]) for i in range(len(hidden_sizes))
        ])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

# Create the MultiLayerLocalAdaptiveNetwork
model = MultiLayerLocalAdaptiveNetwork(input_size, hidden_sizes, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
losses = []

def reg_term(slopes):
    mean_slopes = slopes.mean()
    return torch.exp(mean_slopes).sum()

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Regularization term for each layer
    reg_loss = 0
    for layer in model.hidden_layers:
        slopes = layer.slopes
        reg_loss += reg_term(slopes)

    # Total loss = MSE loss + Regularization loss
    total_loss = loss + reg_loss

    # Backward and optimize
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Save the loss value for plotting
    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}')

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.show()

# Test the model
with torch.no_grad():
    test_input = torch.tensor([[2.0], [3.0], [4.0], [5.0]])
    predicted_output = model(test_input)
    print("Predicted Outputs:")
    print(predicted_output)