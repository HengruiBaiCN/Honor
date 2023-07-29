import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Create a toy dataset with boundary and initial conditions
torch.manual_seed(42)
X = torch.from_numpy(np.linspace(0, 1, 100)).float().requires_grad_()  # Spatial grid
t = torch.from_numpy(np.linspace(0, 0.2, 10)).float().requires_grad_()  # Time grid
print(X.dtype)
X, t = torch.meshgrid(X, t)
X = X.reshape(-1, 1)
t = t.reshape(-1, 1)
u_bc = torch.sin(t * 2 * torch.pi)  # Boundary condition: u(0, t) = sin(2*pi*t)
u_initial = torch.sin(2 * torch.pi * X)  # Initial condition: u(x, 0) = sin(2*pi*x)
y_bc = u_bc.reshape(-1, 1)

# Create the Local Adaptive Physics-Informed Neural Network
input_size = 2  # Spatial (x) and temporal (t) coordinates
hidden_sizes = [20, 20, 20]  # List of hidden layer sizes
output_size = 1

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

class LocalAdaptivePINN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LocalAdaptivePINN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([
            LocalAdaptiveReLU(hidden_sizes[i]) for i in range(len(hidden_sizes))
        ])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x, t):
        x = torch.cat((x, t), dim=1)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

# Define the heat conduction equation as a physics loss term
def physics_loss(u, x, t, alpha=0.1):
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, allow_unused=True)[0]
    heat_eq_loss = u_t - alpha * u_xx  # Heat conduction equation: d(u)/dt - alpha * d^2(u)/dx^2 = 0
    return torch.mean(heat_eq_loss**2)

# Create the Local Adaptive PINN model
model = LocalAdaptivePINN(input_size, hidden_sizes, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10000
losses = []

for epoch in range(num_epochs):
    # Forward pass
    u_pred = model(X, t)
    loss_data = criterion(u_pred, u_initial)  # Data fitting loss

    # Calculate gradients for the physics loss outside of the physics_loss function
    u_x = torch.autograd.grad(u_pred, X, torch.ones_like(u_pred), create_graph=True)[0]
    u_t = torch.autograd.grad(u_pred, t, torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True, allow_unused=True)[0]
    if u_xx is None:
        u_xx = 0
    heat_eq_loss = u_t - 0.1 * u_xx  # Heat conduction equation: d(u)/dt - alpha * d^2(u)/dx^2 = 0
    # return torch.mean(heat_eq_loss**2)

    total_loss = loss_data + torch.mean(heat_eq_loss**2)

    # Backward and optimize
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Save the loss value for plotting
    losses.append(loss_data.item())

    # if (epoch + 1) % 1000 == 0:
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Data Loss: {loss_data.item():.6f}, Physics Loss: {loss_physics.item():.6f}')

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Data Loss')
plt.title('Training Loss Curve')
plt.show()

# Test the model
with torch.no_grad():
    test_X = torch.linspace(0, 1, 100).unsqueeze(1)
    test_t = torch.linspace(0, 0.2, 10)
    test_X, test_t = torch.meshgrid(test_X, test_t)
    test_X = test_X.reshape(-1, 1)
    test_t = test_t.reshape(-1, 1)
    predicted_output = model(test_X, test_t)
    print("Predicted Outputs:")
    # print(predicted_output)