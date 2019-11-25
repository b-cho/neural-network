import numpy as np
from Network import Network
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x)) # Regular sigmoid function

def sigmoid_deriv(x):
    return sigmoid(x) * (1.0-sigmoid(x))

def mse(y_true, y_pred):
    return (1/2) * np.linalg.norm(y_true-y_pred)**2 # Half of the squared norm of the distance between two vectors

def mse_deriv(y_true, y_pred):
    return y_true - y_pred

md = Network([784,10], activation=sigmoid, activation_deriv=sigmoid_deriv, cost=mse, cost_deriv=mse_deriv, learning_rate=5e-3)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

data = np.vstack([img.reshape(-1) for img in mnist.train.images])
data = data.reshape(data.shape + (1,))
labels = np.zeros((55000,10))
labels[np.arange(55000), mnist.train.labels] = 1

X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
X_test = X_test.reshape(X_test.shape + (1,))
y_test = mnist.test.labels

for epoch in tqdm(range(100)): # number of epochs
    random_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(random_state)
    np.random.shuffle(labels)

    train_attempted = 0
    train_cost = 0
    for dp in tqdm(range(55000)):
        train_attempted += 1
        train_cost += md.update(data[dp], labels[dp])
    
    # Now test.
    correct = 0
    test_attempted = 0
    total_cost = 0
    for i in range(200):
        test_attempted += 1
        raw_pred = md.predict(X_test[i])
        oh_label = np.zeros(10)
        oh_label[y_test[i]] = 1
        total_cost += mse(oh_label, raw_pred)
        pred = np.argmax(raw_pred)
        if(pred == y_test[i]):
            correct += 1
    print("\n\n", str(correct/test_attempted), str(total_cost/test_attempted), str(train_cost/train_attempted), "\n\n")
