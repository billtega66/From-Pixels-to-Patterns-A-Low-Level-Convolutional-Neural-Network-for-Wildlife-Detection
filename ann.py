import numpy as np

class ANN:
    def __init__(self, input_size, hidden_size, output_size):
        # weight and bias
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size,output_size)
        self.bias_output = np.zeros((1,output_size))

    def sigmoid(self, x):
        return 1/(1+ np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return x * (1 - x)
    
    def forward_pass(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output,self.weights_hidden_output)+ self.bias_output
        self.output = self.sigmoid(self.output_layer_input)
        return self.output
    
    def back_propagation(self,X,y, learning_rate):
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        #Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T,output_delta) * learning_rate
        self.bias_output += np.sum( output_delta, axis = 0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis = 0, keepdims=True) * learning_rate

    def train(self,X,y,epochs,learning_rate):
        for epoch in range(epochs):
            self.forward_pass(X)
            self.back_propagation(X,y,learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((y-self.output)**2)
                print(f"Epoch:{epoch}, Loss:{loss}")

#dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize and train the ANN
ann = ANN(input_size=2, hidden_size=4, output_size=1)
ann.train(X, y, epochs=1000, learning_rate=0.1)           