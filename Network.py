import numpy as np
import json

class Network:
    def __init__(self, layers=[], activ=None, loss=None, learning_rate=10e-2):
        self.weights = []
        self.biases = []
        self.activation = getattr(activation, activ)
        self.activation_deriv = getattr(activation, activ+"_deriv")
        self.cost = getattr(cost, loss)
        self.cost_deriv = getattr(cost, loss+"_deriv")
        self.layers = layers
        self.learning_rate = learning_rate

        self.biases = [np.random.randn(t, 1) for t in layers[1:]] # Initialize weights and biases with normal distribution
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]

    def predict(self, x):
        for w, b in zip(self.weights, self.biases):
            x = self.activation(np.matmul(w, x) + b) # Run through repeated matrix multiplication to get results
        
        return x

    # This just calculates the gradient for a single input.
    def cost_gradient(self, x, y_true):
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
        
        nabla_w = []
        nabla_b = []
        # We now update the weights and biases using the error.
        for layer in range(0,len(self.layers)-1):
            nabla_w.append(np.matmul(error[layer].reshape(error[layer].shape + (1,)), activations[layer].T))
            nabla_b.append(error[layer].reshape(error[layer].shape + (1,)))
    
        return nabla_w, nabla_b
    
    # This function is for batch gradient descent
    def update(self, X, Y_true):
        assert len(X) == len(Y_true) # Make sure num. of labels is the num. of inputs

        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]

        for x, y in zip(X, Y_true):
            nw, nb = self.cost_gradient(x, y)
            for layer in range(len(self.layers)-1):
                dw[layer] = dw[layer] + (1 / len(X)) * self.learning_rate * nw[layer]
                db[layer] = db[layer] + (1 / len(X)) * self.learning_rate * nb[layer]
        
        self.weights = [self.weights[t] - dw[t] for t in range(len(self.layers)-1)]
        self.biases = [self.biases[t] - db[t] for t in range(len(self.layers)-1)]

    
    def save(self, filepath):
        save_obj = {}
        save_obj["weights"] = [t.tolist() for t in self.weights]
        save_obj["biases"] = [t.tolist() for t in self.biases]
        save_obj["learning_rate"] = self.learning_rate
        save_obj["activation"] = self.activation.__name__
        save_obj["cost"] = self.cost.__name__
        save_obj = json.dumps(save_obj)

        with open(filepath, "w+") as fl:
            fl.write(save_obj)
            fl.close()
    
    def load(self, filepath):
        with open(filepath, "r") as fl:
            raw_json = fl.read()
            save_obj = json.loads(raw_json)

            self.learning_rate = float(save_obj["learning_rate"])
            self.activation = getattr(activation, save_obj["activation"])
            self.activation_deriv = getattr(activation, save_obj["activation"]+"_deriv")
            self.cost = getattr(cost, save_obj["cost"])
            self.cost_deriv = getattr(cost, save_obj["cost"]+"_deriv")
            self.weights = [np.array(t) for t in list(save_obj["weights"])]
            self.weights = [np.array(t) for t in list(save_obj["biases"])]
            
            fl.close()

        
   
class activation:
    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x)) # Regular sigmoid function

    @staticmethod
    def sigmoid_deriv(x):
        return activation.sigmoid(x) * (1.0-activation.sigmoid(x))

class cost:
    @staticmethod
    def mse(y_true, y_pred):
        return (1/2) * np.linalg.norm(y_pred-y_true)**2 # Half of the squared norm of the distance between two vectors

    @staticmethod
    def mse_deriv(y_true, y_pred):
        return y_pred - y_true

        



        
