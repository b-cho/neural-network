import numpy as np
import json

class Network:
    def __init__(self, layers=None, activ=None, loss=None, learning_rate=None):
        self.weights = []
        self.biases = []
        if(activ is not None):
            self.activations = [getattr(Activation, t) for t in activ]
            self.activations_deriv = [getattr(Activation, t+"_deriv") for t in activ]
            # self.activation = getattr(Activation, activ)
            # self.activation_deriv = getattr(Activation, activ+"_deriv")
        if(loss is not None):
            self.cost = getattr(Cost, loss)
            self.cost_deriv = getattr(Cost, loss+"_deriv")
        if(layers is not None):
            assert len(layers) > 0
            self.layers = layers

            self.biases = [np.random.randn(t, 1) for t in layers[1:]] # Initialize weights and biases with normal distribution
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(layers[:-1], layers[1:])]

        if(learning_rate is not None):
            assert learning_rate > 0
            self.learning_rate = learning_rate

    def predict(self, x):
        for w, b, act in zip(self.weights, self.biases, self.activations):
            x = act(np.matmul(w, x) + b) # Run through repeated matrix multiplication to get results
        
        return x

    # This just calculates the gradient for a single input.
    def cost_gradient(self, x, y_true):
        z = [] # an array of the weighted inputs for each neuron
        activations = [x]
        for w, b, act in zip(self.weights, self.biases, self.activations):
            _z = np.matmul(w, x) + b
            _a = act(_z)
            x = _a

            z.append(_z) # add weighted inputs to z
            activations.append(_a)
        
        s_z = np.array([act_deriv(q) for q, act_deriv in zip(z, self.activations_deriv)]) # This is just activ'(z^L) for every layer in the network, stored as a variable.

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
        save_obj["activations"] = [t.__name__ for t in self.activations]
        save_obj["cost"] = self.cost.__name__
        save_obj["layers"] = self.layers
        save_obj = json.dumps(save_obj)

        with open(filepath, "w+") as fl:
            fl.write(save_obj)
            fl.close()
    
    def load(self, filepath):
        with open(filepath, "r") as fl:
            raw_json = fl.read()
            save_obj = json.loads(raw_json)

            self.learning_rate = float(save_obj["learning_rate"])
            self.activations = [getattr(Activation, a) for a in save_obj["activations"]]
            self.activations_deriv = [getattr(Activation, a_d+"_deriv") for a_d in save_obj["activations"]]
            self.cost = getattr(Cost, save_obj["cost"])
            self.cost_deriv = getattr(Cost, save_obj["cost"]+"_deriv")
            self.weights = [np.array(t) for t in list(save_obj["weights"])]
            self.biases = [np.array(t) for t in list(save_obj["biases"])]
            self.layers = list(save_obj["layers"])
            
            fl.close()

        
   
class Activation:
    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x)) # Regular sigmoid function

    @staticmethod
    def sigmoid_deriv(x):
        return Activation.sigmoid(x) * (1.0-Activation.sigmoid(x))

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)
    
    @staticmethod
    def relu_deriv(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))
    
    @staticmethod
    def softmax_deriv(x):
        exp = np.exp(x)
        r = np.sum(exp)
        
        sd = np.vectorize(lambda z: ((r - z) * z) / (r**2))
        return sd(exp)

    @staticmethod
    def linear(x):
        return x # Identity function
    
    @staticmethod
    def linear_deriv(x):
        return x.fill(1) # The derivative of a linear function with slope 1 is 1


class Cost:
    @staticmethod
    def mse(y_true, y_pred):
        return (1/2) * np.linalg.norm(y_pred-y_true)**2 # Half of the squared norm of the distance between two vectors

    @staticmethod
    def mse_deriv(y_true, y_pred):
        return y_pred - y_true

        



        
