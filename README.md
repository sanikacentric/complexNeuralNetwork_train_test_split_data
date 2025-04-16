Building and Training a Neural Network from Scratch Using PyTorch: A Step-by-Step Guide
Subtitle: From architecture to accuracy — see how a simple 3-layer neural network can predict breast cancer diagnoses using PyTorch.

🧠 Neural Network Architecture
We begin by designing a simple yet powerful feedforward neural network with the following structure:

Input Layer: 5 neurons (features)

Hidden Layer: 3 neurons with ReLU activation

Output Layer: 1 neuron with Sigmoid activation (ideal for binary classification)

🎯 Parameter Calculation
Let's calculate the trainable parameters:

Input to Hidden Layer:
5 inputs * 3 hidden neurons + 3 biases = 18 parameters

Hidden to Output Layer:
3 inputs * 1 output neuron + 1 bias = 4 parameters

✅ Total Parameters: 18 + 4 = 22 trainable weights and biases.

🔄 Forward Pass
Once the model is defined and data is passed through the network, you’ll get predictions like:

python
Copy
Edit
tensor([[0.3681], [0.3740], ..., [0.3633]], grad_fn=<SigmoidBackward0>)
This indicates:

The model produces probabilities for binary classification.

The presence of grad_fn=<SigmoidBackward0> shows autograd is active, enabling backpropagation.

You can also use torch.nn.Sequential() to define the network more compactly and perform all steps in one go.

🧪 Dataset: Breast Cancer Detection
We'll use a real-world dataset for breast cancer prediction:

📂 Source:
https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv

🧹 Preprocessing Steps:
Remove unused columns:

python
Copy
Edit
data.drop(columns=['id', 'Unnamed: 32'], inplace=True)
Target analysis:

python
Copy
Edit
data["diagnosis"].value_counts()
Feature and label separation:

Input → all features except target

Output → diagnosis column

🛠 Data Standardization and Encoding
1. Feature Scaling with StandardScaler
python
Copy
Edit
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
🎯 Why?
Standardizing ensures faster convergence and prevents exploding/vanishing gradients.

2. Encoding the Target Labels
python
Copy
Edit
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
🎯 Why?
Neural networks require numerical labels, not strings.

🔄 Converting Data to Tensors
python
Copy
Edit
X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.astype(np.float32))
🏗 Model Training Pipeline
Hyperparameters:
Learning Rate: 0.1

Epochs: 1 (just to demonstrate one forward & backward pass)

Loss Function: BCELoss() (Binary Cross Entropy for binary output)

Training Loop:
python
Copy
Edit
for epoch in range(epochs):
    y_pred = model(X_train_tensor)
    loss = loss_function(y_pred, y_train_tensor.view(-1, 1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
✅ Model Evaluation
python
Copy
Edit
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = (y_pred > 0.5).float()
    accuracy = (y_pred == y_test_tensor).float().mean()
    print(f"Accuracy: {accuracy.item()}")
Explanation:
torch.no_grad() disables gradient tracking (for efficiency during testing)

Predictions are thresholded at 0.5

Accuracy is calculated by comparing predictions with true labels

📊 Summary Table

Step	Tool / Method	Purpose
Feature Scaling	StandardScaler()	Normalize input for stability
Label Encoding	LabelEncoder()	Convert categorical to numerical
Loss Function	BCELoss()	Binary classification error calculation
Optimizer	torch.optim.SGD()	Gradient-based weight update
Forward Pass	model(X) or forward()	Compute predictions
Evaluation	torch.no_grad()	Disable gradients during prediction
Accuracy Metric	(y_pred == y_true).mean()	Final performance metric
🧾 Bonus: Model Summary with torchinfo
Install:

bash
Copy
Edit
pip install torchinfo
Usage:

python
Copy
Edit
from torchinfo import summary
summary(model, input_size=(1, 5))  # Assuming 5 features
Gives a clean tabular summary of layer-wise parameter count and output shape.

💡 Final Thoughts
This end-to-end walkthrough shows how to build, train, and evaluate a neural network using PyTorch. From understanding layers and parameters to deploying a working training loop, you now have a clear blueprint for binary classification problems using deep learning.
