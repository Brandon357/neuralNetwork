#Ammar Qureshi, Subbarayudu Kanneganti, Brandon Williams
import numpy as np

input_size = 3 #input size
hidden_size1 = 16 #number of neurons in the first hidden layer
hidden_size2 = 12 #number of neurons in the second hidden layer
hidden_size3 = 8 #number of neurons in the third hidden layer
output_size = 1 #classificaion output size

np.random.seed(42)
w1 = np.random.rand(input_size, hidden_size1) #weihts from input to the first hidden layer
b1 = np.random.rand(1, hidden_size1) #bias for the first hidden layer
w2 = np.random.rand(hidden_size1, hidden_size2) #weights from the first hidden layer to the second hidden layer
b2 = np.random.rand(1, hidden_size2) #bias for the second hidden layer
w3 = np.random.rand(hidden_size2, hidden_size3) #weights from the second hidden layer to the third hidden layer
b3 = np.random.rand(1, hidden_size3) #bias for the third hidden layer
w4 = np.random.rand(hidden_size3, output_size) #weights from the third hidden layer to the output layer
b4 = np.random.rand(1, output_size) #bias for the output layer

def sigmoid(x): #activaion function
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_propagation(x):
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1) #activation of the first hidden layer

    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2) #activation of the second hidden layer

    z3 = np.dot(a2, w3) + b3
    a3 = sigmoid(z3) #activation of the third hidden layer

    z4 = np.dot(a3, w4) + b4
    a4 = sigmoid(z4) #activation of the output layer

    return a1, a2, a3, a4

def backward_propagation(x, y, a1, a2, a3, a4):
    m = x.shape[0] #sample size

    delta4 = a4 - y #calcuate the error of the output layer
    deltz4 = delta4 * sigmoid_derivative(a4)
    d_w4 = np.dot(a3.T, deltz4) / m #w4 & b4 gradients calculation
    d_b4 = np.sum(deltz4, axis=0, keepdims=True) / m

    delta3 = np.dot(deltz4, w4.T) #backpropagate the error of the third hidden layer
    deltz3 = delta3 * sigmoid_derivative(a3)
    d_w3 = np.dot(a2.T, deltz3) / m #w3 & b3 gradients calculation
    d_b3 = np.sum(deltz3, axis=0, keepdims=True) / m

    delta2 = np.dot(deltz3, w3.T) #backpropagate the error of the second hidden layer
    deltz2 = delta2 * sigmoid_derivative(a2)
    d_w2 = np.dot(a1.T, deltz2) / m #w2 & b2 gradients calculation
    d_b2 = np.sum(deltz2, axis=0, keepdims=True) / m

    delta1 = np.dot(deltz2, w2.T) #backpropagate the error of the first hidden layer
    deltz1 = delta1 * sigmoid_derivative(a1)
    d_w1 = np.dot(x.T, deltz1) / m #w1 & b1 gradients calculation
    d_b1 = np.sum(deltz1, axis=0, keepdims=True) / m

    return d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_w4, d_b4

#update the weights
def weights(w1, b1, w2, b2, w3, b3, w4, b4, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_w4, d_b4, learning_rate):
    w1 -= learning_rate * d_w1
    b1 -= learning_rate * d_b1
    w2 -= learning_rate * d_w2
    b2 -= learning_rate * d_b2
    w3 -= learning_rate * d_w3
    b3 -= learning_rate * d_b3
    w4 -= learning_rate * d_w4
    b4 -= learning_rate * d_b4

    return w1, b1, w2, b2, w3, b3, w4, b4

#train the neural network
def train_neural_network(x, y, epochs, learning_rate):
    global w1, b1, w2, b2, w3, b3, w4, b4

    for epoch in range(epochs):
        a1, a2, a3, a4 = forward_propagation(x)
        loss = np.mean((a4 - y) ** 2) #calculate the loss
        d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_w4, d_b4 = backward_propagation(x, y, a1, a2, a3, a4)
        w1, b1, w2, b2, w3, b3, w4, b4 = weights(w1, b1, w2, b2, w3, b3, w4, b4, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_w4, d_b4, learning_rate)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return w1, b1, w2, b2, w3, b3, w4, b4

def predict(x):
    a1, a2, a3, a4 = forward_propagation(x)
    return a4

#test the neural network
x = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]) #input data
y = np.array([[0], [1], [1], [0]]) #classification

hidden_size1 = 12 #number of neurons in the first hidden layer
hidden_size2 = 8 #number of neurons in the second hidden layer
hidden_size3 = 4 #number of neurons in the third hidden layer

#train the neural network
epochs = 10000
learning_rate = 0.1
train_neural_network(x, y, epochs, learning_rate)

#predict the output
predictions = predict(x)
print("Predictions:\n", predictions)