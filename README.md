# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.
<img width="1134" height="647" alt="image" src="https://github.com/user-attachments/assets/984a0889-df48-4b6f-8346-ea5ab1122df1" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:
### Register Number:
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10, 14)
        self.fc4 = nn.Linear(14, 1)
        self.relu = nn.ReLU()
        self.history={'loss':[]}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x



# Initialize the Model, Loss Function, and Optimizer

Malligesh=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(Malligesh.parameters(), lr=0.001)

def train_model(Malligesh, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(Malligesh(X_train), y_train)
        loss.backward()
        optimizer.step()

        Malligesh.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
train_model(Malligesh, X_train_tensor, y_train_tensor, criterion, optimizer)

h torch.no_grad():
    test_loss = criterion(Malligesh(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(Malligesh.history)
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = Malligesh(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')




```
## Dataset Information

<img width="615" height="361" alt="image" src="https://github.com/user-attachments/assets/dc7d07d9-b6f6-43ae-a7e2-16a189656aea" />


## OUTPUT
<img width="974" height="790" alt="image" src="https://github.com/user-attachments/assets/cc013295-c26d-4d4f-8159-ef4f58650f35" />


<img width="985" height="593" alt="image" src="https://github.com/user-attachments/assets/3d065a0c-7ff3-45b4-af47-2c745e879e8c" />


### Training Loss Vs Iteration Plot

<img width="974" height="790" alt="image" src="https://github.com/user-attachments/assets/cc013295-c26d-4d4f-8159-ef4f58650f35" />


### New Sample Data Prediction
<img width="808" height="120" alt="image" src="https://github.com/user-attachments/assets/f8ee96cc-094b-4412-840e-110be0a9f048" />


## RESULT

The program to develop a neural network regression model for the given dataset has been executed successively
