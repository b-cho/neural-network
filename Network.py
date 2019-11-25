import numpy as np

class Network:
    def __init__(self, layers=[], activation=None, activation_deriv=None, cost=None, cost_deriv=None, learning_rate=10e-2):
        self.weights = []
        self.biases = []
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.cost = cost
        self.cost_deriv = cost_deriv
        self.layers = layers
        self.learning_rate = learning_rate

        self.biases = [np.random.randn(t, 1) for t in layers[1:]] # Initialize weights and biases with normal distribution
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]
        
    def predict(self, x):
        for w, b in zip(self.weights, self.biases):
            x = self.activation(np.matmul(w, x) + b) # Run through repeated matrix multiplication to get results
        
        return x

    def update(self, x, y_true):
        z = [] # an array of the weighted inputs for each neuron
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            _z = np.matmul(w, x) + b
            _a = self.activation(_z)
            x = _a

            z.append(_z) # add weighted inputs to z
            activations.append(_a)
        
        s_z = np.array([self.activation_deriv(q) for q in z]) # This is just sigmoid'(z^L) for every layer in the network, stored as a variable.

        error = [] # Initialize the error array as empty

        # We now compute the output layer's error, which is defined as grad_a C (.) activ'(z_L).
        error.append(self.cost_deriv(y_true, activations[-1].flatten()) * s_z[-1].flatten())

        for layer in range(2,len(self.layers)):
            # Use negative indices to iterate backwards over the error list.
            error.append(np.matmul(self.weights[-layer+1].T, error[-1]) * s_z[-layer].flatten())

        error = error[::-1] # Reverse it to be first to last.
        # We now update the weights and biases using the error.
        for layer in range(0,len(self.layers)-1):
            self.biases[layer] = self.biases[layer] + self.learning_rate * error[layer].reshape(error[layer].shape + (1,))
            self.weights[layer] = self.weights[layer] + self.learning_rate * np.matmul(error[layer].reshape(error[layer].shape + (1,)), activations[layer].T)
    
        return self.cost(y_true, activations[-1])
        


        



        
